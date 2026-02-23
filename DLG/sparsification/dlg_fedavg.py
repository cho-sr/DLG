# -*- coding: utf-8 -*-
"""
DLG with FedAvg + Gradient Sparsification
Based on original DLG paper implementation

Usage:
    python dlg_fedavg.py --index 25 --sparsity 1.0    # 100% (no sparsification)
    python dlg_fedavg.py --index 25 --sparsity 0.1    # 10% (top 10%)
    python dlg_fedavg.py --index 25 --sparsity 0.01   # 1% (top 1%)
"""

import argparse
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import datasets, transforms

print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import ResNet18, weights_init

# -------------------------
# Argument Parser
# -------------------------
parser = argparse.ArgumentParser(description='DLG with FedAvg + Sparsification')
parser.add_argument('--index', type=int, default=25,
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
parser.add_argument('--sparsity', type=float, default=1.0,
                    help='gradient retention ratio (1.0=100%, 0.1=10%, 0.01=1%)')
parser.add_argument('--local_epochs', type=int, default=1,
                    help='number of local epochs in FedAvg')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate for local training')
parser.add_argument('--dlg_iters', type=int, default=300,
                    help='DLG optimization iterations')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
args = parser.parse_args()

# -------------------------
# Device Setup
# -------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Running on {device}")
print(f"Sparsity Ratio: {args.sparsity * 100}% (retaining top {args.sparsity * 100}% gradients)")

# Seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# -------------------------
# DLG Hyperparameters
# -------------------------
DLG_TV_WEIGHT = 0.001   # Total Variation regularization weight (noise reduction)
DLG_L2_WEIGHT = 0.0001  # L2 regularization weight (stability)

# -------------------------
# Data Preparation
# -------------------------
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
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

plt.figure(figsize=(4, 4))
plt.imshow(tt(gt_data[0].cpu()))
plt.title(f"Ground Truth (Class {gt_label.item()})")
plt.axis('off')
plt.tight_layout()
plt.savefig("ground_truth.png", dpi=150, bbox_inches='tight')
plt.close()
print("Ground truth image saved to ground_truth.png")

# -------------------------
# Model Setup (ResNet18)
# -------------------------
net = ResNet18().to(device)
torch.manual_seed(args.seed)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

print(f"Model: ResNet18 ({sum(p.numel() for p in net.parameters()):,} parameters)")

# -------------------------
# Sparsification Function
# -------------------------
def sparsify_gradients(model, ratio):
    """
    Keep only top-k% gradients by magnitude, set others to zero.
    
    Args:
        model: neural network model
        ratio: retention ratio (1.0 = keep all, 0.1 = keep top 10%)
    """
    if ratio >= 1.0:
        return  # No sparsification
    
    # Collect all gradients
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    
    if not grads:
        return
    
    # Concatenate all gradients
    all_grads = torch.cat(grads)
    num_params = all_grads.numel()
    
    # Find threshold (top-k values)
    k = max(1, int(num_params * ratio))
    top_values, _ = torch.topk(torch.abs(all_grads), k)
    threshold = top_values[-1]
    
    # Apply sparsification mask
    for param in model.parameters():
        if param.grad is not None:
            mask = (torch.abs(param.grad) >= threshold).float()
            param.grad.data *= mask
    
    # Report sparsification stats
    total_params = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
    nonzero_params = sum((p.grad != 0).sum().item() for p in model.parameters() if p.grad is not None)
    actual_ratio = nonzero_params / total_params if total_params > 0 else 0
    print(f"  Sparsification: {nonzero_params:,} / {total_params:,} params retained ({actual_ratio*100:.2f}%)")

# -------------------------
# FedAvg Local Training
# -------------------------
print("\n" + "=" * 50)
print("FedAvg Local Training (Client Simulation)")
print("=" * 50)

# Save initial weights
original_weights = [p.detach().clone() for p in net.parameters()]

# Local optimizer
optimizer_client = torch.optim.SGD(net.parameters(), lr=args.lr)

# Local training
net.train()
for epoch in range(args.local_epochs):
    optimizer_client.zero_grad()
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    loss.backward()
    optimizer_client.step()
    
    print(f"  Local Epoch {epoch}: Loss {loss.item():.4f}")

print("\nLocal training completed!")

# -------------------------
# Compute Target Gradient with Sparsification
# -------------------------
print("\n" + "=" * 50)
print("Computing Target Gradient (What Attacker Intercepts)")
print("=" * 50)

# Restore model to initial state
with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)

# Compute gradient at initial state
net.zero_grad()
pred = net(gt_data)
loss = criterion(pred, gt_onehot_label)
loss.backward()

# Apply sparsification
print(f"\nApplying sparsification (ratio={args.sparsity})...")
sparsify_gradients(net, args.sparsity)

# Save target gradient
original_dy_dx = [p.grad.detach().clone() for p in net.parameters()]

# Compute gradient norm
total_grad_norm = sum(g.norm().item() for g in original_dy_dx)
nonzero_grads = sum((g != 0).sum().item() for g in original_dy_dx)
print(f"Target gradient norm: {total_grad_norm:.4f}")
print(f"Non-zero gradient elements: {nonzero_grads:,}")

# Restore model to initial state for DLG
with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)

net.eval()

# -------------------------
# DLG Attack
# -------------------------
print("\n" + "=" * 50)
print("Starting DLG Reconstruction Attack")
print("=" * 50)

# Initialize dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.figure(figsize=(4, 4))
plt.imshow(tt(dummy_data[0].cpu()))
plt.title("Initial Dummy (Random Noise)")
plt.axis('off')
plt.tight_layout()
plt.savefig("initial_dummy.png", dpi=150, bbox_inches='tight')
plt.close()
print("Initial dummy image saved to initial_dummy.png")

# Total Variation regularization
def total_variation(x):
    """Smoothness penalty to reduce noise"""
    diff_i = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    diff_j = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return diff_i.sum() + diff_j.sum()

# L2 regularization
def l2_penalty(x):
    """L2 penalty to prevent extreme values"""
    return (x ** 2).sum()

# LBFGS optimizer
optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=1.0)

history = []
loss_history = []

for iters in range(args.dlg_iters):
    def closure():
        optimizer.zero_grad()
        
        # Clamp data to [0, 1]
        dummy_data.data.clamp_(0, 1)
        
        # Forward pass
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        
        # Compute dummy gradient
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        # Gradient matching loss (initialize as tensor)
        grad_diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff = grad_diff + ((gx - gy) ** 2).sum()
        
        # Total loss with regularization
        tv_loss = total_variation(dummy_data)
        l2_loss = l2_penalty(dummy_data)
        total_loss = grad_diff + DLG_TV_WEIGHT * tv_loss + DLG_L2_WEIGHT * l2_loss
        
        total_loss.backward()
        return total_loss
    
    optimizer.step(closure)
    
    if iters % 10 == 0:
        # Compute losses for reporting (without no_grad for autograd.grad)
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=False)
        
        # Compute metrics
        grad_diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff = grad_diff + ((gx - gy) ** 2).sum()
        
        tv_loss = total_variation(dummy_data)
        total_loss = grad_diff + DLG_TV_WEIGHT * tv_loss
        loss_history.append(grad_diff.item())
        
        print(f"Iter {iters:3d}: GradLoss {grad_diff.item():.4f} | TV {tv_loss.item():.2f}")
        
        with torch.no_grad():
            history.append(tt(dummy_data[0].cpu()))

# -------------------------
# Calculate Metrics
# -------------------------
print("\n" + "=" * 50)
print("Calculating Metrics")
print("=" * 50)

with torch.no_grad():
    gt_np = gt_data[0].cpu().numpy()
    dummy_np = dummy_data[0].detach().cpu().numpy()
    
    mse = np.mean((gt_np - dummy_np) ** 2)
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10)) if mse > 0 else 100.0
    
    # Predicted label
    final_pred = net(dummy_data)
    pred_label = torch.argmax(final_pred, dim=1).item()
    dummy_label_class = torch.argmax(dummy_label, dim=1).item()

print(f"MSE: {mse:.6f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"True Label: {gt_label.item()}")
print(f"Reconstructed Label: {pred_label}")

# -------------------------
# Visualization
# -------------------------
print("\nGenerating visualizations...")

# 1. Reconstruction Progress
plt.figure(figsize=(15, 6))
for i in range(min(10, len(history))):
    plt.subplot(2, 5, i + 1)
    plt.imshow(history[i])
    plt.title(f"Iter {i * 10}")
    plt.axis('off')
plt.suptitle(f"DLG Reconstruction Progress (Sparsity: {args.sparsity*100}%)", fontsize=14)
plt.tight_layout()
plt.savefig(f"dlg_progress_sparsity_{int(args.sparsity*100)}.png", dpi=150, bbox_inches='tight')
plt.close()

# 2. Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(range(0, len(loss_history) * 10, 10), loss_history, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Gradient Matching Loss', fontsize=12)
plt.title(f'DLG Convergence (Sparsity: {args.sparsity*100}%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig(f"dlg_loss_sparsity_{int(args.sparsity*100)}.png", dpi=150, bbox_inches='tight')
plt.close()

# 3. Final Comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(tt(gt_data[0].cpu()))
axes[0].set_title(f"Ground Truth\nClass: {gt_label.item()}", fontsize=12)
axes[0].axis('off')

axes[1].imshow(tt(dummy_data[0].detach().cpu()))
axes[1].set_title(f"Reconstructed\nClass: {pred_label}\nPSNR: {psnr:.2f}dB", fontsize=12)
axes[1].axis('off')

plt.suptitle(f"DLG Attack Result (Sparsity: {args.sparsity*100}%)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"dlg_final_sparsity_{int(args.sparsity*100)}.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*50}")
print("✓ All results saved successfully!")
print(f"{'='*50}")
print(f"  - ground_truth.png")
print(f"  - initial_dummy.png")
print(f"  - dlg_progress_sparsity_{int(args.sparsity*100)}.png")
print(f"  - dlg_loss_sparsity_{int(args.sparsity*100)}.png")
print(f"  - dlg_final_sparsity_{int(args.sparsity*100)}.png")
