# -*- coding: utf-8 -*-
"""
FedAvg with Deep Leakage from Gradients (DLG) + Gradient Sparsification

이 스크립트는 FedAvg 환경에서 DLG 공격을 수행합니다:
1. 클라이언트가 로컬 데이터로 모델 학습
2. 그래디언트를 스파시피케이션으로 압축
3. 공격자가 스파시파이된 그래디언트로부터 데이터 복원 시도

Usage:
    python fedavg_dlg.py --sparsity 1.0 --index 25
    python fedavg_dlg.py --sparsity 0.1 --index 25
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import datasets, transforms

print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

# ============================================================================
# Argument Parser
# ============================================================================
parser = argparse.ArgumentParser(description='FedAvg with DLG + Sparsification')
parser.add_argument('--index', type=int, default=25,
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
parser.add_argument('--sparsity', type=float, default=0.5,
                    help='Gradient retention ratio (1.0=100%, 0.1=10%, 0.05=5%, 0.01=1%)')
parser.add_argument('--local_epochs', type=int, default=5,
                    help='Number of local training epochs')
parser.add_argument('--local_lr', type=float, default=0.01,
                    help='Learning rate for local training')
parser.add_argument('--dlg_iters', type=int, default=100,
                    help='Number of DLG optimization iterations')
args = parser.parse_args()

# ============================================================================
# Device Setup
# ============================================================================
device = "mps"
if torch.cuda.is_available():
    device = "cuda"
print(f"Device: {device}")
print(f"Sparsity Ratio: {args.sparsity * 100}% (top {args.sparsity * 100}% gradients retained)")

# ============================================================================
# Data Preparation
# ============================================================================
dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

print(f"\n✓ Ground truth: shape={gt_data.shape}, label={gt_label.item()}")

# ============================================================================
# Model Setup
# ============================================================================
from models.vision import LeNet, weights_init

net = LeNet().to(device)
torch.manual_seed(1234)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

total_params = sum(p.numel() for p in net.parameters())
print(f"✓ LeNet loaded ({total_params:,} parameters)")

# ============================================================================
# Sparsification Function
# ============================================================================
def sparsify_gradients(gradients, ratio):
    """Keep only top-k% gradients by magnitude, set others to zero."""
    flat_grads = torch.cat([g.view(-1) for g in gradients])
    num_params = flat_grads.numel()
    
    if ratio >= 1.0:
        return gradients, {
            'threshold': 0.0,
            'retained': 1.0,
            'nonzero': num_params,
            'total': num_params
        }
    
    k = max(1, int(num_params * ratio))
    top_values, _ = torch.topk(torch.abs(flat_grads), k)
    threshold = top_values[-1]
    
    sparsified = []
    for g in gradients:
        mask = (torch.abs(g) >= threshold).float()
        sparsified.append(g * mask)
    
    nonzero_count = sum((g != 0).sum().item() for g in sparsified)
    retained_ratio = nonzero_count / num_params
    
    return sparsified, {
        'threshold': threshold.item(),
        'retained': retained_ratio,
        'nonzero': nonzero_count,
        'total': num_params
    }

# ============================================================================
# FedAvg Local Training (Client Simulation)
# ============================================================================
print("\n" + "="*60)
print("FedAvg Local Training (Client Simulation)")
print("="*60)

initial_weights = [p.detach().clone() for p in net.parameters()]
optimizer_local = torch.optim.SGD(net.parameters(), lr=args.local_lr)

net.train()
for epoch in range(args.local_epochs):
    optimizer_local.zero_grad()
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    loss.backward()
    optimizer_local.step()
    print(f"  Epoch {epoch}: Loss {loss.item():.4f}")

print("✓ Local training completed")

# ============================================================================
# Compute Target Gradient with Sparsification
# ============================================================================
print("\n" + "="*60)
print("Gradient Computation & Sparsification")
print("="*60)

# Restore model to initial state
with torch.no_grad():
    for param, w_init in zip(net.parameters(), initial_weights):
        param.copy_(w_init)

# Compute gradient
net.zero_grad()
pred = net(gt_data)
loss = criterion(pred, gt_onehot_label)
loss.backward()

# Extract and sparsify gradients
target_gradients = [p.grad.detach().clone() for p in net.parameters()]

print(f"\nBefore Sparsification:")
total_grad_norm = sum(g.norm().item() for g in target_gradients)
total_nonzero = sum((g != 0).sum().item() for g in target_gradients)
print(f"  Total gradient norm: {total_grad_norm:.4f}")
print(f"  Non-zero elements: {total_nonzero:,}")

print(f"\nApplying sparsification (ratio={args.sparsity})...")
target_gradients, sparse_stats = sparsify_gradients(target_gradients, args.sparsity)

print(f"After Sparsification:")
print(f"  Threshold: {sparse_stats['threshold']:.6f}")
print(f"  Retained: {sparse_stats['nonzero']:,} / {sparse_stats['total']:,} ({sparse_stats['retained']*100:.2f}%)")

# Restore model to initial state for DLG
with torch.no_grad():
    for param, w_init in zip(net.parameters(), initial_weights):
        param.copy_(w_init)

net.eval()

# ============================================================================
# DLG Attack
# ============================================================================
print("\n" + "="*60)
print("Deep Leakage from Gradients (DLG) Attack")
print("="*60)

# Initialize dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_data.data.clamp_(0, 1)  # Initialize in valid range
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

# Optimizer
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

history = []
loss_history = []

print(f"\nRunning DLG for {args.dlg_iters} iterations...")
for iters in range(args.dlg_iters):
    def closure():
        optimizer.zero_grad()
        
        # Clamp data to [0, 1]
        dummy_data.data.clamp_(0, 1)
        
        # Forward pass
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        
        # Compute gradients
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        # Gradient matching loss
        grad_diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(dummy_dy_dx, target_gradients):
            grad_diff = grad_diff + ((gx - gy) ** 2).sum()
        
        grad_diff.backward()
        return grad_diff
    
    optimizer.step(closure)
    
    if iters % 10 == 0:
        # Compute loss for logging
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=False)
        
        grad_diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(dummy_dy_dx, target_gradients):
            grad_diff = grad_diff + ((gx - gy) ** 2).sum()
        
        loss_history.append(grad_diff.item())
        print(f"  Iter {iters:3d}: Loss {grad_diff.item():.4f}")
        
        with torch.no_grad():
            history.append(tt(dummy_data[0].cpu()))

# ============================================================================
# Calculate Metrics
# ============================================================================
print("\n" + "="*60)
print("Results & Metrics")
print("="*60)

with torch.no_grad():
    gt_np = gt_data[0].cpu().numpy()
    dummy_np = dummy_data[0].detach().cpu().numpy()
    
    mse = np.mean((gt_np - dummy_np) ** 2)
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10)) if mse > 0 else 100.0
    
    final_pred = net(dummy_data)
    final_pred_probs = F.softmax(final_pred, dim=1)
    pred_label = torch.argmax(final_pred_probs, dim=1).item()

print(f"\nReconstruction Quality:")
print(f"  MSE:  {mse:.6f}")
print(f"  PSNR: {psnr:.2f} dB")
print(f"  True Label: {gt_label.item()}")
print(f"  Predicted Label: {pred_label}")
print(f"  Label Match: {'YES ✓' if pred_label == gt_label.item() else 'NO ✗'}")

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*60)
print("Generating Visualizations")
print("="*60)

sparsity_str = f"{int(args.sparsity*100)}" if args.sparsity >= 0.01 else f"{args.sparsity*100:.1f}"

# 1. Reconstruction Progress
plt.figure(figsize=(15, 6))
for i in range(min(10, len(history))):
    plt.subplot(2, 5, i + 1)
    plt.imshow(history[i])
    plt.title(f"Iter {i * 10}")
    plt.axis('off')
plt.suptitle(f"DLG Reconstruction Progress (Sparsity: {sparsity_str}%)", fontsize=14)
plt.tight_layout()
plt.savefig(f"dlg_progress_sparsity_{sparsity_str}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: dlg_progress_sparsity_{sparsity_str}.png")

# 2. Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(range(0, len(loss_history) * 10, 10), loss_history, 'b-', linewidth=2, marker='o')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Gradient Matching Loss', fontsize=12)
plt.title(f'DLG Convergence (Sparsity: {sparsity_str}%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig(f"dlg_loss_sparsity_{sparsity_str}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: dlg_loss_sparsity_{sparsity_str}.png")

# 3. Final Comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(tt(gt_data[0].cpu()))
axes[0].set_title(f"Ground Truth\nClass: {gt_label.item()}", fontsize=12)
axes[0].axis('off')

axes[1].imshow(tt(dummy_data[0].detach().cpu()))
axes[1].set_title(f"Reconstructed\nClass: {pred_label}\nPSNR: {psnr:.2f}dB", fontsize=12)
axes[1].axis('off')

plt.suptitle(f"DLG Attack Result (Sparsity: {sparsity_str}%)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"dlg_final_sparsity_{sparsity_str}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: dlg_final_sparsity_{sparsity_str}.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)

print(f"\nConfiguration:")
print(f"  Sparsity Ratio: {args.sparsity * 100}%")
print(f"  Local Epochs: {args.local_epochs}")
print(f"  DLG Iterations: {args.dlg_iters}")
print(f"  Model: LeNet ({total_params:,} parameters)")

print(f"\nSparsification:")
print(f"  Retained Parameters: {sparse_stats['nonzero']:,} / {sparse_stats['total']:,}")
print(f"  Retention Ratio: {sparse_stats['retained']*100:.2f}%")

print(f"\nAttack Results:")
print(f"  MSE: {mse:.6f}")
print(f"  PSNR: {psnr:.2f} dB")
print(f"  Label Match: {'YES ✓' if pred_label == gt_label.item() else 'NO ✗'}")

print(f"\nGenerated Files:")
print(f"  - dlg_progress_sparsity_{sparsity_str}.png")
print(f"  - dlg_loss_sparsity_{sparsity_str}.png")
print(f"  - dlg_final_sparsity_{sparsity_str}.png")

print("\n" + "="*60)
print("✓ Experiment completed successfully!")
print("="*60)
