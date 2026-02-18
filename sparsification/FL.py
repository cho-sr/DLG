# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import copy
import time
import math
from skimage.metrics import structural_similarity as ssim_func

# Matplotlib 백엔드 설정
import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('MacOSX')

# models 폴더 경로 추가
sys.path.append(os.getcwd())
from models.vision import LeNet, weights_init

# 디바이스 설정
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if torch.cuda.is_available(): device = torch.device("cuda")
print(f" 사용 디바이스: {device}")

# 실험 파라미터
SPARSITY_RATIOS = [1.0, 0.1, 0.01]  # 100%, 10%, 1%
EPOCHS_FL = 300  # 빠른 테스트를 위해 조정
DLG_ITERATIONS = 300
BATCH_SIZE = 64


# ==============================================================================
# 1. Performance Metric 함수 (추가됨)
# ==============================================================================
def calculate_metrics(target_tensor, dummy_tensor):
    """
    원본과 복원 이미지 사이의 MSE, PSNR, SSIM을 계산합니다.
    Tensors는 [-1, 1] 범위(Normalize 기준)라고 가정하고 [0, 1]로 변환하여 계산합니다.
    """
    # 1. [0, 1] 범위로 역정규화 (Normalize((0.5,), (0.5,)) 대응)
    gt = (target_tensor.detach().cpu().clone() * 0.5 + 0.5).clamp(0, 1).numpy().squeeze()
    re = (dummy_tensor.detach().cpu().clone() * 0.5 + 0.5).clamp(0, 1).numpy().squeeze()

    # 2. MSE 계산
    mse = np.mean((gt - re) ** 2)

    # 3. PSNR 계산
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))

    # 4. SSIM 계산 (C, H, W 형식 대응을 위해 channel_axis=0)
    ssim = ssim_func(gt, re, data_range=1.0, channel_axis=0)

    return mse, psnr, ssim


# ==============================================================================
# 2. 유틸리티 함수 (Sparsification 등)
# ==============================================================================
def get_model():
    model = LeNet()
    model.fc = nn.Sequential(nn.Linear(768, 10))
    model.apply(weights_init)
    return model.to(device)


def sparsify_gradients(model, ratio):
    if ratio >= 1.0: return
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    if not grads: return
    all_grads = torch.cat(grads)
    num_params = all_grads.numel()
    k = max(1, int(num_params * ratio))
    top_values, _ = torch.topk(torch.abs(all_grads), k)
    threshold = top_values[-1]
    for param in model.parameters():
        if param.grad is not None:
            mask = torch.abs(param.grad) >= threshold
            param.grad.data *= mask.float()


# ==============================================================================
# 3. DLG 공격 시뮬레이션 (Metric 적용)
# ==============================================================================
def run_dlg_attack(target_img, target_label, ratio):
    print(f"   [DLG] 공격 시작 (Sparsity: {ratio * 100}%) ...")
    model = get_model()
    model.zero_grad()
    pred = model(target_img)
    loss = F.cross_entropy(pred, target_label)
    loss.backward()

    sparsify_gradients(model, ratio)
    original_dy_dx = [p.grad.detach().clone() for p in model.parameters()]

    dummy_data = torch.randn(target_img.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn((1, 10)).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    for iters in range(DLG_ITERATIONS):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = F.cross_entropy(dummy_pred, torch.softmax(dummy_label, dim=-1))
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

    # 성능 지표 계산
    mse, psnr, ssim = calculate_metrics(target_img, dummy_data)
    print(f"     -> [결과] PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}, MSE: {mse:.6f}")
    return (mse, psnr, ssim), dummy_data.detach().cpu()


# ==============================================================================
# 4. FL 학습 성능 시뮬레이션
# ==============================================================================
def run_fl_simulation(train_loader, test_loader, ratio):
    print(f"   [FL] 학습 시작 (Sparsity: {ratio * 100}%) ...")
    model = get_model()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS_FL):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            sparsify_gradients(model, ratio)
            optimizer.step()
            if batch_idx > 50: break

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    print(f"     -> 최종 정확도: {acc:.2f}%")
    return acc


# ==============================================================================
# 5. 메인 실행부
# ==============================================================================
# ==============================================================================
# 5. 메인 실행부 (시각화 기능 강화)
# ==============================================================================
# ==============================================================================
# 5. 메인 실행부 (시각화 기능 강화)
# ==============================================================================
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    dlg_img, dlg_lbl = next(iter(train_loader))
    dlg_img, dlg_lbl = dlg_img[0:1].to(device), dlg_lbl[0:1].to(device)

    results = {'sparsity': [], 'fl_acc': [], 'mse': [], 'psnr': [], 'ssim': [], 'images': []}

    for ratio in SPARSITY_RATIOS:
        print(f"\n" + "-" * 50)
        print(f"Case: Gradient Retention Ratio {ratio * 100}%")
        results['sparsity'].append(f"{int(ratio * 100)}%")

        # 1. FL 성능 시뮬레이션
        acc = run_fl_simulation(train_loader, test_loader, ratio)
        results['fl_acc'].append(acc)

        # 2. DLG 공격 및 지표 계산
        metrics, recon_img = run_dlg_attack(dlg_img, dlg_lbl, ratio)
        results['mse'].append(metrics[0])
        results['psnr'].append(metrics[1])
        results['ssim'].append(metrics[2])
        results['images'].append(recon_img)

    # --- 1. 종합 지표 시각화 (2x2 Subplots) ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    x_pos = np.arange(len(results['sparsity']))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # (0,0) FL Accuracy
    axs[0, 0].plot(x_pos, results['fl_acc'], marker='o', color=colors[0], linewidth=2)
    axs[0, 0].set_title('Utility: FL Accuracy (%)', fontsize=12, fontweight='bold')
    axs[0, 0].set_ylabel('Accuracy (%)')

    # (0,1) PSNR (Higher is better for reconstruction)
    axs[0, 1].plot(x_pos, results['psnr'], marker='s', color=colors[1], linewidth=2)
    axs[0, 1].set_title('Privacy: DLG PSNR (dB)', fontsize=12, fontweight='bold')
    axs[0, 1].set_ylabel('PSNR (dB)')

    # (1,0) SSIM (Higher is more similar)
    axs[1, 0].plot(x_pos, results['ssim'], marker='^', color=colors[2], linewidth=2)
    axs[1, 0].set_title('Privacy: DLG SSIM', fontsize=12, fontweight='bold')
    axs[1, 0].set_ylabel('SSIM Score')

    # (1,1) MSE (Lower is more similar)
    axs[1, 1].plot(x_pos, results['mse'], marker='v', color=colors[3], linewidth=2)
    axs[1, 1].set_title('Privacy: DLG MSE', fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel('Mean Squared Error')

    for ax in axs.flat:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results['sparsity'])
        ax.set_xlabel('Gradient Retention Ratio')
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.suptitle('Performance Metrics under Gradient Sparsification', fontsize=16, y=1.02)
    plt.show()

    # --- 2. 복원 이미지 시각화 (지표 포함) ---
    plt.figure(figsize=(18, 5))
    for i, img in enumerate(results['images']):
        plt.subplot(1, len(SPARSITY_RATIOS), i + 1)
        disp_img = (img[0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
        plt.imshow(disp_img)

        # 타이틀에 모든 지표 표시
        title_str = (f"Sparsity: {results['sparsity'][i]}\n"
                     f"PSNR: {results['psnr'][i]:.2f}dB\n"
                     f"SSIM: {results['ssim'][i]:.4f}\n"
                     f"MSE: {results['mse'][i]:.5f}")
        plt.title(title_str, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()