# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default=25,
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')

# ========================= [추가] .pt gradient 파일 경로 =========================
parser.add_argument('--pt_path', type=str, default="",
                    help='path to saved gradient .pt file')
# ==============================================================================

args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

# 0326.py와 동일하게 CIFAR-10 기준으로 공격 대상을 준비한다.
num_classes = 10
dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.tensor([dst[img_index][1]], dtype=torch.long).to(device)
gt_label = gt_label.view(1,)
gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)

plt.imshow(tt(gt_data[0].cpu()))

from models.vision import LeNet, weights_init
net = LeNet().to(device)

torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot

# ========================= [수정 시작] original gradient 준비 =========================
if len(args.pt_path) > 0:
    print(f"Load gradients from: {args.pt_path}")
    artifact = torch.load(args.pt_path, map_location=device)

    # 0) 모델 상태가 저장되어 있으면 그대로 로드해 gradient를 만든 시점의 모델과 맞춘다.
    if isinstance(artifact, dict) and "model_state_dict" in artifact:
        net.load_state_dict(artifact["model_state_dict"])

    # 1) label이 저장되어 있으면 사용
    if isinstance(artifact, dict) and "label" in artifact:
        gt_label = torch.tensor([artifact["label"]], dtype=torch.long, device=device).view(1,)
        gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)

    # 2) named_grads 형태인 경우
    if isinstance(artifact, dict) and "named_grads" in artifact:
        named_grads = artifact["named_grads"]

        # net.parameters() 순서에 맞게 gradient 리스트 생성
        original_dy_dx = []
        for name, param in net.named_parameters():
            if name not in named_grads:
                raise KeyError(f"Gradient for parameter '{name}' not found in pt file.")
            original_dy_dx.append(named_grads[name].detach().clone().to(device))

    # 3) grads 또는 dy_dx 리스트 형태인 경우
    elif isinstance(artifact, dict) and "grads" in artifact:
        original_dy_dx = [g.detach().clone().to(device) for g in artifact["grads"]]

    elif isinstance(artifact, dict) and "dy_dx" in artifact:
        original_dy_dx = [g.detach().clone().to(device) for g in artifact["dy_dx"]]

    # 4) 그냥 리스트/튜플로 저장된 경우
    elif isinstance(artifact, (list, tuple)):
        original_dy_dx = [g.detach().clone().to(device) for g in artifact]

    else:
        raise ValueError("Unsupported .pt format. Need 'named_grads', 'grads', 'dy_dx', or list/tuple.")

else:
    # 원래 코드 그대로: gt_data로부터 직접 gradient 계산
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = [_.detach().clone() for _ in dy_dx]
# ========================= [수정 끝] ============================================

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
        history.append(tt(dummy_data[0].detach().cpu()))

plt.figure(figsize=(12, 8))
for i in range(min(30, len(history))):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
