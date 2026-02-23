# -*- coding: utf-8 -*-
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.vision import LeNet, weights_init
from utils import label_to_onehot, cross_entropy_for_onehot

print(f"Torch: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on {device}")

# ==============================================================================
# 1. 데이터 및 모델 준비
# ==============================================================================
dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# 랜덤 이미지 선택
img_index = random.randint(0, len(dst) - 1)
print(f"Selected Image Index: {img_index}")
gt_data = tp(dst[img_index][0]).to(device).view(1, 3, 32, 32)
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device).view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

# 정답 확인용 시각화
plt.figure(figsize=(4, 4))
plt.imshow(tt(gt_data[0].cpu()))
plt.title(f"Ground Truth (Label: {dst[img_index][1]})")
plt.show()

net = LeNet().to(device)
torch.manual_seed(1234)
net.apply(weights_init)
criterion = cross_entropy_for_onehot
original_weights = [p.detach().clone() for p in net.parameters()]

# ==============================================================================
# 2. 로컬 학습 시뮬레이션 (FedDyn/FedAvg)
# ==============================================================================
# 로컬 에포크 1로 설정하여 그래디언트의 선명도를 확보 (성공 검증용)
local_epochs = 1
lr = 0.01
alpha = 0.01  # FedDyn Coefficient

optimizer_client = torch.optim.SGD(net.parameters(), lr=lr)
print(f"Local Training: {local_epochs} epochs, lr={lr}")

# FedDyn용 이전 그라디언트 (첫 라운드는 0)
grad_prev = [torch.zeros_like(p) for p in net.parameters()]

net.train()
for epoch in range(local_epochs):
    optimizer_client.zero_grad()
    pred = net(gt_data)
    task_loss = criterion(pred, gt_onehot_label)

    # FedDyn Regularization
    dyn_loss = 0.0
    for w, w_t, g_prev in zip(net.parameters(), original_weights, grad_prev):
        linear_term = -torch.sum(g_prev * w)
        quad_term = (alpha / 2) * torch.sum((w - w_t) ** 2)
        dyn_loss += linear_term + quad_term

    loss = task_loss + dyn_loss
    loss.backward()
    optimizer_client.step()

# Target Gradient (공격자가 훔친 데이터)
target_grad = []
for w_init, w_trained in zip(original_weights, net.parameters()):
    weight_diff = (w_init - w_trained).detach()
    effective_grad = weight_diff / (lr * local_epochs)
    target_grad.append(effective_grad)

# 모델 초기화
with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)


# ==============================================================================
# 3. [핵심] 라벨 유추 (Analytical Label Recovery)
# ==============================================================================
def recover_label_from_grad(gradients, num_classes=10):
    """
    마지막 레이어(FC Layer)의 그래디언트를 분석하여 라벨을 찾아냅니다.
    이론: CrossEntropyLoss의 미분값은 (Prediction - Target) 입니다.
    - Target 클래스의 미분값은 (Probability - 1) 이므로 '음수'입니다.
    - 나머지 클래스의 미분값은 (Probability - 0) 이므로 '양수'입니다.
    따라서, 그래디언트 값이 가장 작은(음수인) 인덱스가 정답 라벨입니다.
    """
    # LeNet의 마지막 레이어 가중치 그래디언트 (보통 리스트의 뒤에서 2번째가 Weight, 1번째가 Bias)
    # 여기서는 안전하게 Weight를 사용합니다.
    last_weight_grad = gradients[-2]

    # 각 클래스별 그래디언트 합 계산 (Batch 단위라면 mean/sum)
    # last_weight_grad shape: [10, 84] (10개 클래스, 84개 입력 노드)
    grad_sum_per_class = torch.sum(last_weight_grad, dim=1)

    # 가장 작은 값을 가진 인덱스가 라벨
    estimated_label = torch.argmin(grad_sum_per_class).item()
    return estimated_label


print("\n[Step 1] Inferring Label from Gradient...")
inferred_label_idx = recover_label_from_grad(target_grad)
print(f" -> Estimated Label: {inferred_label_idx}")
print(f" -> Ground Truth   : {dst[img_index][1]}")

if inferred_label_idx == dst[img_index][1]:
    print(" -> Label Inference SUCCESS! ✅")
else:
    print(" -> Label Inference FAILED... ❌ (Proceeding with estimated label)")

# 유추한 라벨을 One-hot으로 변환하여 고정
dummy_label = label_to_onehot(torch.Tensor([inferred_label_idx]).long().to(device), num_classes=10).detach()

# ==============================================================================
# 4. 이미지 복원 (DLG Attack)
# ==============================================================================
print("\n[Step 2] Reconstructing Image with Inferred Label...")

# 더미 데이터 생성
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)

# 옵티마이저 (Adam 사용 - 수렴 속도 및 안정성 확보)
optimizer = torch.optim.Adam([dummy_data], lr=0.1)

history = []

for iters in range(501):
    def closure():
        optimizer.zero_grad()
        dummy_data.data.clamp_(0, 1)  # [필수] 픽셀 범위 제한

        dummy_pred = net(dummy_data)

        # [중요] 유추한 고정 라벨(dummy_label) 사용
        dummy_loss = criterion(dummy_pred, dummy_label)

        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, target_grad):
            grad_diff += ((gx - gy) ** 2).sum()

        grad_diff.backward()
        return grad_diff


    # Adam은 step()만 호출해도 됨 (closure 필요 없지만 구조 유지)
    optimizer.step(closure)

    if iters % 50 == 0:
        current_loss = closure().item()
        print(f"Iter {iters:3d}: Loss = {current_loss:.6f}")
        history.append(tt(dummy_data[0].detach().clone().clamp(0, 1).cpu()))

# ==============================================================================
# 5. 결과 시각화
# ==============================================================================
plt.figure(figsize=(15, 6))
num_plots = len(history)
cols = 10
rows = (num_plots // cols) + 1

for i in range(num_plots):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(history[i])
    plt.title(f"Iter {i * 50}")
    plt.axis('off')

plt.tight_layout()
plt.show()