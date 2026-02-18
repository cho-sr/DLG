# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

from models.vision import LeNet, weights_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms

print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
print("Running on %s" % device)

dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# img_index = args.index
# gt_data = tp(dst[img_index][0]).to(device)
import random # 상단에 import 추가

# CIFAR10 데이터 개수 범위 내에서 랜덤하게 하나 뽑기
img_index = random.randint(0, len(dst) - 1)
print(f"Selected Image Index: {img_index}") # 몇 번을 뽑았는지 로그 출력
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

plt.imshow(tt(gt_data[0].cpu()))
plt.title("Ground Truth")
plt.show()

net = LeNet().to(device)
torch.manual_seed(1234)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

# --- 수정 전
# compute original gradient
# pred = net(gt_data)
# y = criterion(pred, gt_onehot_label)
# dy_dx = torch.autograd.grad(y, net.parameters())
#
# original_dy_dx = list((_.detach().clone() for _ in dy_dx))
# --- [수정 후] FedAvg 방식 (Local Epoch 수행) ---
# 1. 학습 전 초기 가중치 저장
original_weights = [p.detach().clone() for p in net.parameters()]

# 2. 클라이언트 로컬 학습 설정
local_epochs = 3
lr = 0.05
# 3. 로컬 학습 수행 (FedProx 시뮬레이션)
mu = 0.01
optimizer_client = torch.optim.SGD(net.parameters(), lr=lr)
print(f"Local Training (FedProx, mu={mu}): {local_epochs} epochs, lr={lr}")

net.train()
for epoch in range(local_epochs):
    optimizer_client.zero_grad()
    pred = net(gt_data)
    task_loss = criterion(pred, gt_onehot_label)
    # (2) Proximal Term 계산 (FedProx 핵심)
    # ||W - W_t||^2 : 현재 파라미터와 글로벌 파라미터(original_weights) 간의 차이 제곱합
    proximal_term = 0.0
    for w, w_t in zip(net.parameters(), original_weights):
        proximal_term += ((w - w_t) ** 2).sum()

    # (3) 최종 Loss = Task Loss + (mu / 2) * Proximal Term
    loss = task_loss + (mu / 2) * proximal_term
    loss.backward()
    optimizer_client.step()
    print(f"Epoch {epoch}: Task Loss {task_loss.item():.4f}, Prox Loss {proximal_term.item():.4f}")

# 4. 가중치 업데이트값(Weight Difference) 계산
# 공격자가 훔쳐보게 될 Target 값입니다.
target_grad = []
for w_init, w_trained in zip(original_weights, net.parameters()):
    # W_diff = W_init - W_trained (혹은 그 반대)
    # 이것을 "누적된 그래디언트"처럼 취급합니다.
    weight_diff = (w_init - w_trained).detach()
    effective_grad = weight_diff / (lr * local_epochs)
    target_grad.append(effective_grad)

# 모델을 다시 초기 상태로 되돌림 (공격자는 초기 상태 W_init을 알고 있다고 가정)
# FedAvg에서도 Global Model 파라미터는 공유되므로 공격자가 알고 있습니다.
with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
# dummy_label = gt_onehot_label.clone().to(device)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
plt.imshow(tt(dummy_data[0].cpu()))
plt.title("Initial Dummy")
plt.show()

optimizer = torch.optim.LBFGS([dummy_data,dummy_label], lr=0.005)
# optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

history = []
history_losses = []

# TV Loss 함수 정의
# def total_variation(x):
#     dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
#     dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
#     return dx + dy

for iters in range(301):
    def closure():
        optimizer.zero_grad()
        dummy_data.data.clamp_(0,1) # 픽셀 값

        dummy_pred = net(dummy_data)
        # dummy label이 one-hot 이므로 softmax 불필요
        # 만약 dummy label을 최적화 중이라면 sofrmax가 필요하지만 여기선 고정값
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, target_grad):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        # tv_reg = 0.001 * total_variation(dummy_data)

        # total_loss = grad_diff + tv_reg

        # total_loss.backward()
        # return  total_loss
        return grad_diff

    optimizer.step(closure)
    # scheduler.step()

    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))
        # cpu로 옮기는 이유 -> 시각화 라이브러리는 gpu 이미지 직접 못 읽음
        # clamp(0,1) -> 픽셀 값을 0~1로 강제로 맞춤
        # clamp_ 와 clamp 차이 in place, out place 원본을 건드리고 안건드리고 차이
        # history.append(tt(dummy_data.data.clone().clamp(0, 1)[0].cpu()))  # 저장 시 clamp 권장
        # history_losses.append(current_loss.item())


plt.figure(figsize=(15, 6))
num_plots = len(history)
cols = 10
rows = (num_plots // cols) + 1
for i in range(num_plots):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(history[i])
    plt.title(f"Iter {i * 10}")
    plt.axis('off')

plt.tight_layout() # 그래프 간격 자동 조절 (글자 겹침 방지)
plt.show()