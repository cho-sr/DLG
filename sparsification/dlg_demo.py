import argparse
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from models.vision import LeNet
from utils import cross_entropy_for_onehot


DEFAULT_PT_FILE = Path(__file__).resolve().parent / "gradient_records" / "round_005.pt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "dlg_outputs"
DEFAULT_INPUT_SHAPE = (1, 3, 32, 32)
DEFAULT_NORM_MEAN = [0.4914, 0.4822, 0.4465]
DEFAULT_NORM_STD = [0.2023, 0.1994, 0.2010]


def parse_args():
    parser = argparse.ArgumentParser(description="Run DLG from a saved .pt gradient record.")
    parser.add_argument("--pt_file", type=str, default=str(DEFAULT_PT_FILE))
    parser.add_argument("--client_id", type=int, default=None)
    parser.add_argument("--use_avg", action="store_true")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes):
    model = LeNet()
    if num_classes != 10:
        model.fc[0] = nn.Linear(model.fc[0].in_features, num_classes)
    return model


def infer_num_classes(payload):
    if "num_classes" in payload:
        return int(payload["num_classes"])
    model_state = payload.get("model_state", {})
    if "fc.0.weight" in model_state:
        return int(model_state["fc.0.weight"].shape[0])
    return 10


def infer_input_shape(payload):
    if "input_shape" in payload:
        return tuple(payload["input_shape"])
    return DEFAULT_INPUT_SHAPE


def get_norm_stats(payload):
    mean = payload.get("norm_mean", DEFAULT_NORM_MEAN)
    std = payload.get("norm_std", DEFAULT_NORM_STD)
    return list(mean), list(std)


def select_gradient_source(payload, client_id, use_avg):
    if "clients" in payload:
        if use_avg:
            grads = payload.get("avg_grads")
            if not isinstance(grads, dict):
                raise KeyError("avg_grads is missing from the .pt file.")
            return grads, None, payload, "avg_grads"

        clients = payload["clients"]
        if not clients:
            raise ValueError("No client gradients found in the .pt file.")

        selected = None
        if client_id is None:
            selected = clients[0]
        else:
            for item in clients:
                if int(item.get("client_id", -1)) == int(client_id):
                    selected = item
                    break
        if selected is None:
            raise ValueError(f"client_id={client_id} not found in the .pt file.")
        attack = selected.get("attack")
        if isinstance(attack, dict) and "gradients" in attack:
            return (
                attack["gradients"],
                selected,
                attack,
                f"client_{int(selected.get('client_id', -1))}.attack.gradients",
            )
        return (
            selected["grads"],
            selected,
            selected,
            f"client_{int(selected.get('client_id', -1))}.grads",
        )

    if "grads" in payload and isinstance(payload["grads"], dict):
        return payload["grads"], None, payload, "grads"

    if "gradients" in payload:
        return payload["gradients"], None, payload, "gradients"

    if "attack" in payload and isinstance(payload["attack"], dict) and "gradients" in payload["attack"]:
        return payload["attack"]["gradients"], None, payload["attack"], "attack.gradients"

    raise KeyError("No gradient payload found in the .pt file.")


def gradients_to_list(gradient_source, model, device):
    if isinstance(gradient_source, dict):
        gradients = []
        missing = []
        for name, param in model.named_parameters():
            if name not in gradient_source:
                missing.append(name)
                continue
            gradients.append(gradient_source[name].detach().clone().to(device))
        if missing:
            raise KeyError(f"Missing gradients for model parameters: {missing}")
        return gradients

    if isinstance(gradient_source, (list, tuple)):
        gradients = [g.detach().clone().to(device) for g in gradient_source]
        expected = sum(1 for _ in model.parameters())
        if len(gradients) != expected:
            raise ValueError(
                f"Gradient count mismatch: got {len(gradients)}, expected {expected}."
            )
        return gradients

    raise TypeError("Unsupported gradient payload type.")


def denormalize(tensor, mean, std):
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    return tensor * std_t + mean_t


def save_progress_grid(history, out_path, mean, std):
    if not history:
        return

    snapshots = [denormalize(tensor, mean, std).clamp(0, 1) for _, tensor in history]
    stacked = torch.cat(snapshots, dim=0)
    nrow = min(5, max(1, int(math.ceil(math.sqrt(len(snapshots))))))
    save_image(stacked, out_path, nrow=nrow)


def dlg_attack(model, target_gradients, input_shape, num_classes, device, iters, log_interval):
    net = copy.deepcopy(model).to(device)
    net.eval()

    dummy_data = torch.randn(input_shape, device=device).requires_grad_(True)
    dummy_label = torch.randn((input_shape[0], num_classes), device=device).requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [dummy_data, dummy_label],
        lr=1.0,
        max_iter=20,
        history_size=100,
    )

    history = []
    loss_history = []

    for iteration in range(iters):

        def closure():
            optimizer.zero_grad()
            dummy_pred = net(dummy_data)
            dummy_onehot = F.softmax(dummy_label, dim=-1)
            dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, target_gradients):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        if iteration % log_interval == 0 or iteration == iters - 1:
            with torch.enable_grad():
                dummy_pred = net(dummy_data)
                dummy_onehot = F.softmax(dummy_label, dim=-1)
                dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=False)
                current_loss = 0
                for gx, gy in zip(dummy_dy_dx, target_gradients):
                    current_loss += ((gx - gy) ** 2).sum()
                current_loss = float(current_loss.item())

            print(f"iter {iteration:03d} | grad loss {current_loss:.6f}")
            history.append((iteration, dummy_data.detach().cpu().clone()))
            loss_history.append((iteration, current_loss))

    with torch.no_grad():
        pred_class = int(torch.argmax(net(dummy_data), dim=1).item())
        dummy_class = int(torch.argmax(F.softmax(dummy_label, dim=-1), dim=1).item())

    return dummy_data.detach().cpu(), pred_class, dummy_class, history, loss_history


def maybe_save_ground_truth(meta_source, out_dir, mean, std):
    if not isinstance(meta_source, dict):
        return None

    gt_data = meta_source.get("gt_data")
    if gt_data is None:
        return None

    gt_tensor = gt_data.detach().cpu()
    save_path = out_dir / "gt.png"
    save_image(denormalize(gt_tensor, mean, std).clamp(0, 1), save_path)
    return save_path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.pt_file, map_location="cpu", weights_only=False)
    gradient_source, selected_client, meta_source, source_name = select_gradient_source(
        payload,
        client_id=args.client_id,
        use_avg=args.use_avg,
    )
    state_source = meta_source if isinstance(meta_source, dict) and "model_state" in meta_source else payload
    num_classes = infer_num_classes(state_source)
    input_shape = infer_input_shape(meta_source if isinstance(meta_source, dict) else payload)
    mean, std = get_norm_stats(meta_source if isinstance(meta_source, dict) else payload)

    if "model_state" not in state_source:
        raise KeyError(
            "model_state is missing from the selected gradient payload. "
            "Regenerate the gradient record with the current server.py."
        )

    model = build_model(num_classes).to(device)
    model.load_state_dict(state_source["model_state"], strict=True)
    model.eval()
    target_gradients = gradients_to_list(gradient_source, model, device)

    print(f"Device: {device}")
    print(f"Loaded gradient file: {args.pt_file}")
    print(f"Gradient source: {source_name}")
    print(f"Input shape: {input_shape}")
    print(f"Iterations: {args.iters}")

    if selected_client is not None:
        print(
            "Selected client: "
            f"{int(selected_client.get('client_id', -1))} "
            f"(num_samples={int(selected_client.get('num_samples', 1))})"
        )
        if isinstance(meta_source, dict) and "global_index" in meta_source:
            print(f"Attack global index: {int(meta_source['global_index'])}")
        if source_name.endswith(".grads") and int(selected_client.get("num_samples", 1)) != 1:
            print("Warning: DLG on multi-sample client gradients is usually much harder.")
    elif args.use_avg:
        print("Warning: DLG on averaged client gradients is usually much harder.")

    recon, pred_class, dummy_class, history, loss_history = dlg_attack(
        model=model,
        target_gradients=target_gradients,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device,
        iters=args.iters,
        log_interval=max(1, args.log_interval),
    )

    recon_path = out_dir / "dlg_recon.png"
    save_image(denormalize(recon, mean, std).clamp(0, 1), recon_path)

    progress_path = out_dir / "dlg_progress_grid.png"
    save_progress_grid(history, progress_path, mean, std)

    gt_path = maybe_save_ground_truth(meta_source, out_dir, mean, std)

    summary_path = out_dir / "dlg_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"pt_file: {args.pt_file}\n")
        fh.write(f"gradient_source: {source_name}\n")
        fh.write(f"pred_class: {pred_class}\n")
        fh.write(f"dummy_class: {dummy_class}\n")
        if isinstance(meta_source, dict) and "global_index" in meta_source:
            fh.write(f"global_index: {int(meta_source['global_index'])}\n")
        if isinstance(meta_source, dict) and "gt_label" in meta_source:
            fh.write(f"gt_label: {int(meta_source['gt_label'])}\n")
        if loss_history:
            fh.write(f"final_grad_loss: {loss_history[-1][1]:.6f}\n")

    print("\nGenerated files:")
    print(f"  - {recon_path}")
    print(f"  - {progress_path}")
    print(f"  - {summary_path}")
    if gt_path is not None:
        print(f"  - {gt_path}")


if __name__ == "__main__":
    main()
