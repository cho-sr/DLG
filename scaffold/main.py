# -*- coding: utf-8 -*-
"""
DLG Attack on SCAFFOLD Federated Learning
연합학습 SCAFFOLD 환경에서 DLG 공격을 통한 이미지 복원
"""
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.vision import LeNet, weights_init
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils import label_to_onehot, cross_entropy_for_onehot

# ==================== 설정 ====================
parser = argparse.ArgumentParser(description='DLG Attack on SCAFFOLD')
parser.add_argument('--index', type=int, default=25, help='CIFAR-10 이미지 인덱스')
parser.add_argument('--image', type=str, default="", help='커스텀 이미지 경로')
parser.add_argument('--local_epochs', type=int, default=1, help='로컬 학습 에포크 수')
parser.add_argument('--lr', type=float, default=0.1, help='학습률')
parser.add_argument('--dlg_lr', type=float, default=0.01, help='DLG 최적화 학습률')
parser.add_argument('--dlg_iterations', type=int, default=300, help='DLG 반복 횟수')
args = parser.parse_args()

# 디바이스 설정 (GPU 사용 가능 시 자동으로 사용)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")

print(f"Device: {device}")
print(f"DLG Iterations: {args.dlg_iterations}\n")

# ==================== 데이터셋 로드 ====================
dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# Ground Truth 데이터 준비
img_index = args.index
if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)
else:
    gt_data = tp(dst[img_index][0]).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device).view(1,)
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

print(f"Image index: {img_index}, Label: {gt_label.item()}")

# ==================== 모델 초기화 ====================
net = LeNet().to(device)
torch.manual_seed(1234)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

# ==================== SCAFFOLD 로컬 학습 ====================
# SCAFFOLD의 핵심: Control Variates를 사용하여 클라이언트 드리프트 감소

# 학습 전 가중치 저장 (공격자가 알고 있다고 가정)
initial_weights = [p.detach().clone() for p in net.parameters()]

# Step 1: Control Variate 초기화를 위한 사전 학습
# 실제 SCAFFOLD에서는 이전 라운드의 control variate를 사용하지만,
# 여기서는 몇 번의 사전 학습으로 평균 그래디언트를 계산하여 초기화
print("Initializing SCAFFOLD control variates...")
client_control = []
for param in net.parameters():
    client_control.append(torch.zeros_like(param.data))

# 사전 학습으로 control variate 추정 (3번의 forward-backward)
net.train()
accumulated_grads = [torch.zeros_like(p.data) for p in net.parameters()]
num_warmup = 3

for warmup_iter in range(num_warmup):
    net.zero_grad()
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    loss.backward()
    
    # 그래디언트 누적
    with torch.no_grad():
        for i, param in enumerate(net.parameters()):
            if param.grad is not None:
                accumulated_grads[i].add_(param.grad.data / num_warmup)

# Client control variate = 평균 그래디언트의 일부 (0.3배)
# 너무 크면 학습이 불안정하므로 적절히 스케일링
for i in range(len(client_control)):
    client_control[i] = accumulated_grads[i] * 0.3

# Server control variate는 여러 클라이언트의 평균이지만, 
# 단일 클라이언트 시뮬레이션이므로 client의 0.5배로 설정
server_control = [c * 0.5 for c in client_control]

# Control variate 통계 출력
client_norm = sum((c ** 2).sum().item() for c in client_control) ** 0.5
server_norm = sum((s ** 2).sum().item() for s in server_control) ** 0.5
print(f"Client control norm: {client_norm:.6f}")
print(f"Server control norm: {server_norm:.6f}")

# 모델을 초기 상태로 리셋
with torch.no_grad():
    for param, w_init in zip(net.parameters(), initial_weights):
        param.copy_(w_init)

# Step 2: SCAFFOLD 로컬 학습 수행
optimizer_client = torch.optim.SGD(net.parameters(), lr=args.lr)
net.train()

for epoch in range(args.local_epochs):
    optimizer_client.zero_grad()
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    loss.backward()
    
    # SCAFFOLD의 그래디언트 보정: g_corrected = g + (c_server - c_client)
    # 이 보정이 클라이언트 드리프트를 줄이고 수렴을 개선
    with torch.no_grad():
        correction_norm = 0
        for param, c_server, c_client in zip(net.parameters(), server_control, client_control):
            if param.grad is not None:
                correction = c_server - c_client
                param.grad.data.add_(correction)
                correction_norm += (correction ** 2).sum().item()
        
        if epoch == 0:
            print(f"Gradient correction norm: {correction_norm ** 0.5:.6f}")
    
    optimizer_client.step()

# Step 3: 가중치 업데이트량 계산 (서버로 전송되는 정보)
trained_weights = [p.detach().clone() for p in net.parameters()]
weight_deltas = [w_new - w_old for w_new, w_old in zip(trained_weights, initial_weights)]

# Step 4: Client Control Variate 업데이트 (SCAFFOLD 알고리즘)
# c_i^{new} = c_i^{old} - c + (y - x) / (K * lr)
# 여기서 y=trained_weights, x=initial_weights
option_drift = [-(w_new - w_old) / (args.local_epochs * args.lr) 
                for w_new, w_old in zip(trained_weights, initial_weights)]

new_client_control = []
for c_client, c_server, drift in zip(client_control, server_control, option_drift):
    c_i_new = c_client - c_server + drift
    new_client_control.append(c_i_new)

# 업데이트된 control variate의 변화량 확인
control_delta_norm = sum(((c_new - c_old) ** 2).sum().item() 
                        for c_new, c_old in zip(new_client_control, client_control)) ** 0.5
print(f"Control variate delta norm: {control_delta_norm:.6f}")

# Step 5: Target Gradient 계산 (공격자가 가로챈 정보로부터 역산)
# 실제 SCAFFOLD에서 서버로 전송되는 것은 weight_deltas와 control_variate_delta
target_grad = []
for delta_w in weight_deltas:
    effective_grad = -delta_w / (args.lr * args.local_epochs)
    target_grad.append(effective_grad.detach())

# 모델을 초기 상태로 복원 (공격자는 글로벌 모델을 알고 있음)
with torch.no_grad():
    for param, w_init in zip(net.parameters(), initial_weights):
        param.copy_(w_init)

print("SCAFFOLD local training completed.\n")

# ==================== DLG 공격 시작 ====================
print(f"Starting DLG Attack ({args.dlg_iterations} iterations)...\n")

# Dummy 데이터 초기화 (랜덤 노이즈로 시작)
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = gt_onehot_label.clone().to(device)  # 레이블은 알고 있다고 가정

# LBFGS 옵티마이저 (그래디언트 매칭에 효과적)
optimizer = torch.optim.LBFGS([dummy_data], lr=args.dlg_lr, max_iter=1)

# DLG 최적화 루프
for iters in range(args.dlg_iterations):
    def closure():
        optimizer.zero_grad()
        
        # 픽셀 값을 [0, 1] 범위로 제한
        dummy_data.data.clamp_(0, 1)
        
        # Dummy 데이터로부터 그래디언트 계산
        dummy_pred = net(dummy_data)
        dummy_loss = criterion(dummy_pred, dummy_label)
        dummy_dy_dx = torch.autograd.grad(
            dummy_loss, 
            net.parameters(), 
            create_graph=True
        )
        
        # 그래디언트 매칭 손실: ||dummy_grad - target_grad||^2
        # 목표: Dummy 데이터의 그래디언트가 실제 그래디언트와 같아지도록
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, target_grad):
            grad_diff += ((gx - gy) ** 2).sum()
        
        grad_diff.backward()
        return grad_diff
    
    # 옵티마이저 스텝
    optimizer.step(closure)
    
    # 진행 상황 출력 (50번마다)
    if iters % 50 == 0 or iters == args.dlg_iterations - 1:
        loss_val = closure().item()
        print(f"Iter {iters:4d}/{args.dlg_iterations}: Loss = {loss_val:.6f}")

print("\nDLG Attack completed!\n")

# ==================== 결과 시각화 ====================
# 품질 평가 지표 계산
mse = F.mse_loss(dummy_data[0], gt_data[0]).item()
psnr = 20 * np.log10(1.0) - 10 * np.log10(mse) if mse > 0 else float('inf')
correlation = torch.corrcoef(torch.stack([
    dummy_data[0].flatten(),
    gt_data[0].flatten()
]))[0, 1].item()

print("=" * 60)
print("Results:")
print(f"  MSE: {mse:.6f}")
print(f"  PSNR: {psnr:.2f} dB")
print(f"  Correlation: {correlation:.4f}")
print("=" * 60)

# 원본 vs 복원 이미지 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 원본 이미지
axes[0].imshow(tt(gt_data[0].cpu()))
axes[0].set_title(f"Ground Truth\n(Label: {gt_label.item()})", 
                 fontsize=14, fontweight='bold')
axes[0].axis('off')

# 복원된 이미지
axes[1].imshow(tt(dummy_data[0].cpu()))
axes[1].set_title(f"DLG Reconstructed\n(MSE: {mse:.4f}, PSNR: {psnr:.2f}dB)", 
                 fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.suptitle("DLG Attack on SCAFFOLD Federated Learning", 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dlg_result.png', dpi=150, bbox_inches='tight')
print("\nResult saved as 'dlg_result.png'")
plt.show()

print("\n✓ Done!")
