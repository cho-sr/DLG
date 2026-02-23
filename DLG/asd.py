# -*- coding: utf-8 -*-
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim_func  # SSIM 계산용
import math

# 기존 사용자 정의 모듈 임포트
from models.vision import LeNet, weights_init
from utils import label_to_onehot, cross_entropy_for_onehot

import matplotlib
try:
    # Mac/Window/Linux 호환성이 가장 좋은 팝업창 모드
    matplotlib.use('TkAgg')
except:
    # Mac 전용 네이티브 모드 (TkAgg가 없을 경우 대비)
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

# --- [추가] Performance Metric 함수 정의 ---
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calculate_metrics(gt_tensor, dummy_tensor):
    # Tensor -> Numpy 변환 (C, H, W)
    gt_np = gt_tensor.detach().cpu().numpy().squeeze()
    dummy_np = dummy_tensor.detach().cpu().numpy().squeeze()

    # 1. MSE (Mean Squared Error)
    mse = np.mean((gt_np - dummy_np) ** 2)

    # 2. PSNR (Peak Signal-to-Noise Ratio)
    psnr = calculate_psnr(gt_np, dummy_np)

    # 3. SSIM (Structural Similarity Index)
    # channel_axis=0 은 (C, H, W) 형식을 의미함
    ssim = ssim_func(gt_np, dummy_np, data_range=1.0, channel_axis=0)

    return mse, psnr, ssim


# -----------------------------------------

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25")
parser.add_argument('--image', type=str, default="")
# args = parser.parse_args() # 실제 환경에 맞게 수정 필요 시 활용
# 주석: Jupyter나 스크립트 실행 환경에 따라 args 처리가 다를 수 있습니다.

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")

dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = 25  # args.index 대용
gt_data = tp(dst[img_index][0]).to(device)
gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

net = LeNet().to(device)
torch.manual_seed(1234)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

# --- FedAvg 방식 (Local Epoch 수행) ---
original_weights = [p.detach().clone() for p in net.parameters()]
local_epochs = 1
lr = 0.01
optimizer_client = torch.optim.SGD(net.parameters(), lr=lr)

net.train()
for _ in range(local_epochs):
    optimizer_client.zero_grad()
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    loss.backward()
    optimizer_client.step()

target_grad = []
for w_init, w_trained in zip(original_weights, net.parameters()):
    weight_diff = (w_init - w_trained).detach()
    effective_grad = weight_diff / (lr * local_epochs)
    target_grad.append(effective_grad)

with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)

# Dummy Data 초기화
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

history = []
mse_history, psnr_history, ssim_history = [], [], []  # 지표 저장용

print(f"{'Iter':<10} | {'Loss':<10} | {'MSE':<10} | {'PSNR':<10} | {'SSIM':<10}")
print("-" * 60)

for iters in range(101):
    def closure():
        optimizer.zero_grad()
        dummy_data.data.clamp_(0, 1)
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, target_grad):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        return grad_diff


    optimizer.step(closure)

    if iters % 10 == 0:
        current_loss = closure().item()

        # --- [추가] 성능 지표 계산 ---
        current_mse, current_psnr, current_ssim = calculate_metrics(gt_data, dummy_data)

        mse_history.append(current_mse)
        psnr_history.append(current_psnr)
        ssim_history.append(current_ssim)
        history.append(tt(dummy_data[0].cpu().clamp(0, 1)))

        print(
            f"{iters:<10d} | {current_loss:<10.6f} | {current_mse:<10.6f} | {current_psnr:<10.2f} | {current_ssim:<10.4f}")

# 결과 시각화
plt.figure(figsize=(15, 5))
for i in range(len(history)):
    plt.subplot(2, 11, i + 1)
    plt.imshow(history[i])
    plt.title(f"it={i * 10}")
    plt.axis('off')

# 성능 지표 그래프 추가
plt.subplot(2, 1, 2)
plt.plot(range(0, 101, 10), psnr_history, label='PSNR (Higher is better)', color='blue', marker='o')
plt.plot(range(0, 101, 10), ssim_history, label='SSIM (Max 1.0)', color='green', marker='s')
plt.xlabel('Iterations')
plt.ylabel('Score')
plt.legend()
plt.title('Reconstruction Performance over Iterations')
plt.tight_layout()
plt.show()