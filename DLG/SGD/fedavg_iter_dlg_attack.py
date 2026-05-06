# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot


DEFAULT_PT_FILE = Path(__file__).resolve().parent / "fedavg_iter_outputs" / "fedavg_iter_gradients.pt"


parser = argparse.ArgumentParser(description='Deep Leakage from Gradients from a FedAvg .pt snapshot.')
parser.add_argument('--pt_file', type=str, default=str(DEFAULT_PT_FILE),
                    help='the path to the saved fedavg_iter_gradients.pt file.')
parser.add_argument('--round', type=int, required=True,
                    help='the FedAvg round whose saved gradient snapshot will be attacked.')
parser.add_argument('--client_id', type=int, required=True,
                    help='the client id whose saved gradient snapshot will be attacked.')
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

pt_path = Path(args.pt_file).expanduser().resolve()
if not pt_path.exists():
    raise FileNotFoundError(f"Gradient artifact not found: {pt_path}")

try:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(pt_path, map_location="cpu")

if not isinstance(payload, dict):
    raise ValueError("Loaded artifact is not a dictionary.")
if "gradient_snapshots" not in payload or not isinstance(payload["gradient_snapshots"], dict):
    raise KeyError("Artifact does not contain a valid 'gradient_snapshots' dictionary.")

available_rounds = sorted(int(round_key) for round_key in payload["gradient_snapshots"].keys())
round_snapshots = payload["gradient_snapshots"].get(args.round)
if round_snapshots is None:
    round_snapshots = payload["gradient_snapshots"].get(str(args.round))
if not isinstance(round_snapshots, dict):
    raise ValueError(f"round={args.round} was not found. Available rounds: {available_rounds}")

available_clients = sorted(int(client_key) for client_key in round_snapshots.keys())
snapshot = round_snapshots.get(args.client_id)
if snapshot is None:
    snapshot = round_snapshots.get(str(args.client_id))
if not isinstance(snapshot, dict):
    raise ValueError(
        f"client_id={args.client_id} was not found in round={args.round}. "
        f"Available clients for round {args.round}: {available_clients}"
    )

if isinstance(snapshot.get("sample_index"), (list, tuple)):
    raise ValueError("This script only supports single-sample snapshots, but sample_index is not scalar.")
if isinstance(snapshot.get("label"), (list, tuple)):
    raise ValueError("This script only supports single-sample snapshots, but label is not scalar.")

input_shape = tuple(int(dim) for dim in snapshot.get("input_shape", (1, 3, 32, 32)))
if len(input_shape) != 4 or input_shape[0] != 1:
    raise ValueError(
        f"This script only supports single-sample snapshots, but input_shape={input_shape}."
    )

img_index = int(snapshot["sample_index"])
artifact_label = int(snapshot["label"])
gt_data = tp(dst[img_index][0]).to(device)
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
if int(gt_label.item()) != artifact_label:
    raise ValueError(
        f"Snapshot label mismatch: artifact label={artifact_label}, dataset label={int(gt_label.item())}"
    )

gt_data = gt_data.view(1, *gt_data.size())
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

plt.imshow(tt(gt_data[0].cpu()))

from models.vision import LeNet
net = LeNet().to(device)


torch.manual_seed(1234)
np.random.seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

model_state_dict = snapshot.get("model_state_dict_before_step")
if not isinstance(model_state_dict, dict):
    raise KeyError("Snapshot does not contain 'model_state_dict_before_step'.")
net.load_state_dict(model_state_dict, strict=True)
criterion = cross_entropy_for_onehot

if "named_grads" not in snapshot or not isinstance(snapshot["named_grads"], dict):
    raise KeyError("Snapshot does not contain a valid 'named_grads' dictionary.")

original_dy_dx = []
missing_names = []
for name, _ in net.named_parameters():
    if name not in snapshot["named_grads"]:
        missing_names.append(name)
        continue
    original_dy_dx.append(snapshot["named_grads"][name].detach().clone().to(device))
if missing_names:
    raise KeyError(f"Snapshot gradients are missing model parameters: {missing_names}")

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


history = []
for iters in range(300):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()

        return grad_diff

    optimizer.step(closure)
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
