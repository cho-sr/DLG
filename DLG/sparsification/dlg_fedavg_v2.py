# -*- coding: utf-8 -*-
"""
DLG with FedAvg + Gradient Sparsification (Weight Difference Method)

This uses the weight difference method which is more robust for FedAvg scenarios.

Usage:
    python dlg_fedavg_v2.py --index 25 --sparsity 1.0
    python dlg_fedavg_v2.py --index 25 --sparsity 0.1
    python dlg_fedavg_v2.py --index 25 --sparsity 0.01
"""

import argparse
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import ResNet18, weights_init

print(f"PyTorch: {torch.__version__}")

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser(description='DLG with FedAvg + Sparsification (Weight Diff)')
parser.add_argument('--index', type=int, default=25)
parser.add_argument('--sparsity', type=float, default=1.0,
                    help='gradient retention ratio (1.0=100%, 0.1=10%)')
parser.add_argument('--local_epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dlg_iters', type=int, default=300)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
args = parser.parse_args()

# -------------------------
# Setup
# -------------------------
if args.device == 'auto':
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
else:
    device = args.device

print(f"Device: {device}")
print(f"Sparsity: {args.sparsity * 100}%\n")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# -------------------------
# Data
# -------------------------
dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

gt_data = tp(dst[args.index][0]).to(device).view(1, 3, 32, 32)
gt_label = torch.tensor([dst[args.index][1]], dtype=torch.long, device=device)
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

# Save ground truth
plt.figure(figsize=(4, 4))
plt.imshow(tt(gt_data[0].cpu()))
plt.title(f"Ground Truth (Class {gt_label.item()})")
plt.axis('off')
plt.savefig("ground_truth.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Ground truth saved")

# -------------------------
# Model
# -------------------------
net = ResNet18().to(device)
torch.manual_seed(args.seed)
net.apply(weights_init)
criterion = cross_entropy_for_onehot
print(f"✓ ResNet18 loaded ({sum(p.numel() for p in net.parameters()):,} params)\n")

# -------------------------
# Sparsification
# -------------------------
def sparsify_gradients(model, ratio):
    if ratio >= 1.0:
        print("  No sparsification (100%)")
        return
    
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    
    if not grads:
        return
    
    all_grads = torch.cat(grads)
    k = max(1, int(all_grads.numel() * ratio))
    top_values, _ = torch.topk(torch.abs(all_grads), k)
    threshold = top_values[-1]
    
    for param in model.parameters():
        if param.grad is not None:
            mask = (torch.abs(param.grad) >= threshold).float()
            param.grad.data *= mask
    
    nonzero = sum((p.grad != 0).sum().item() for p in model.parameters() if p.grad is not None)
    total = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
    print(f"  Retained: {nonzero:,} / {total:,} ({nonzero/total*100:.2f}%)")

# -------------------------
# FedAvg Local Training
# -------------------------
print("="*50)
print("FedAvg Local Training")
print("="*50)

original_weights = [p.detach().clone() for p in net.parameters()]
optimizer_client = torch.optim.SGD(net.parameters(), lr=args.lr)

net.train()
for epoch in range(args.local_epochs):
    optimizer_client.zero_grad()
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    loss.backward()
    optimizer_client.step()
    print(f"  Epoch {epoch}: Loss {loss.item():.4f}")

print("✓ Local training completed\n")

# -------------------------
# Target Gradient (Weight Difference Method)
# -------------------------
print("="*50)
print("Target Gradient (Weight Difference)")
print("="*50)

# Compute weight difference
target_grad = []
for w_init, w_trained in zip(original_weights, net.parameters()):
    weight_diff = (w_init - w_trained).detach()
    effective_grad = weight_diff / (args.lr * args.local_epochs)
    target_grad.append(effective_grad)

grad_norm = sum(g.norm().item() for g in target_grad)
nonzero = sum((g != 0).sum().item() for g in target_grad)
print(f"✓ Effective gradient norm: {grad_norm:.4f}")
print(f"✓ Non-zero elements: {nonzero:,}\n")

# Apply sparsification to target
print("Applying sparsification to target gradient...")
for i, g in enumerate(target_grad):
    target_grad[i] = g.clone()  # Make a copy

# Create temporary model to use sparsify function
temp_grads = target_grad
if args.sparsity < 1.0:
    all_grads = torch.cat([g.view(-1) for g in temp_grads])
    k = max(1, int(all_grads.numel() * args.sparsity))
    top_values, _ = torch.topk(torch.abs(all_grads), k)
    threshold = top_values[-1]
    
    target_grad = []
    for g in temp_grads:
        mask = (torch.abs(g) >= threshold).float()
        target_grad.append(g * mask)
    
    nonzero_after = sum((g != 0).sum().item() for g in target_grad)
    print(f"  After sparsification: {nonzero_after:,} / {all_grads.numel():,} ({nonzero_after/all_grads.numel()*100:.2f}%)")
else:
    print("  No sparsification (100%)")

grad_norm_after = sum(g.norm().item() for g in target_grad)
print(f"✓ Target gradient norm after sparsification: {grad_norm_after:.4f}\n")

# Restore model
with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)

net.eval()
print("✓ Model restored, using eval mode for DLG")

# -------------------------
# DLG Attack
# -------------------------
print("="*50)
print("DLG Reconstruction Attack")
print("="*50)

dummy_data = torch.randn(gt_data.size(), device=device, requires_grad=True)
dummy_label = torch.randn(gt_onehot_label.size(), device=device, requires_grad=True)

# Save initial
plt.figure(figsize=(4, 4))
plt.imshow(tt(dummy_data[0].detach().cpu()))
plt.title("Initial Dummy")
plt.axis('off')
plt.savefig("initial_dummy.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Initial dummy saved\n")

optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=1.0)

history = []
loss_history = []

for iters in range(args.dlg_iters):
    def closure():
        optimizer.zero_grad()
        dummy_data.data.clamp_(0, 1)
        
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(dummy_dy_dx, target_grad):
            grad_diff = grad_diff + ((gx - gy) ** 2).sum()
        
        grad_diff.backward()
        return grad_diff
    
    optimizer.step(closure)
    
    if iters % 10 == 0:
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=False)
        
        grad_diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(dummy_dy_dx, target_grad):
            grad_diff = grad_diff + ((gx - gy) ** 2).sum()
        
        loss_history.append(grad_diff.item())
        
        data_mean = dummy_data.mean().item()
        data_std = dummy_data.std().item()
        
        print(f"Iter {iters:3d}: GradLoss {grad_diff.item():.4f} | Data(μ={data_mean:.3f}, σ={data_std:.3f})")
        
        with torch.no_grad():
            history.append(tt(dummy_data[0].cpu()))

# -------------------------
# Metrics
# -------------------------
print(f"\n{'='*50}")
print("Results")
print("="*50)

with torch.no_grad():
    mse = F.mse_loss(dummy_data, gt_data).item()
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10)) if mse > 0 else 100.0
    
    final_pred = net(dummy_data)
    pred_label = torch.argmax(final_pred, dim=1).item()
    
    # Dummy label
    dummy_pred_label = torch.argmax(F.softmax(dummy_label, dim=-1), dim=1).item()

print(f"MSE: {mse:.6f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"True Label: {gt_label.item()}")
print(f"Reconstructed Label: {pred_label}")
print(f"Dummy Label (optimized): {dummy_pred_label}\n")

# -------------------------
# Save Results
# -------------------------
# Progress
plt.figure(figsize=(15, 6))
for i in range(min(10, len(history))):
    plt.subplot(2, 5, i + 1)
    plt.imshow(history[i])
    plt.title(f"Iter {i * 10}")
    plt.axis('off')
plt.suptitle(f"Reconstruction Progress (Sparsity: {args.sparsity*100}%)", fontsize=14)
plt.tight_layout()
plt.savefig(f"dlg_progress_{int(args.sparsity*100)}.png", dpi=150, bbox_inches='tight')
plt.close()

# Loss curve
if loss_history:
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(loss_history) * 10, 10), loss_history, 'b-', linewidth=2, marker='o')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Gradient Matching Loss', fontsize=12)
    plt.title(f'DLG Convergence (Sparsity: {args.sparsity*100}%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"dlg_loss_{int(args.sparsity*100)}.png", dpi=150, bbox_inches='tight')
    plt.close()

# Final comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(tt(gt_data[0].cpu()))
axes[0].set_title(f"Ground Truth\nClass: {gt_label.item()}", fontsize=12)
axes[0].axis('off')

axes[1].imshow(tt(dummy_data[0].detach().cpu()))
axes[1].set_title(f"Reconstructed\nClass: {pred_label}\nPSNR: {psnr:.2f}dB", fontsize=12)
axes[1].axis('off')

plt.suptitle(f"DLG Result (Sparsity: {args.sparsity*100}%)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"dlg_final_{int(args.sparsity*100)}.png", dpi=150, bbox_inches='tight')
plt.close()

print("="*50)
print("✓ All results saved:")
print("="*50)
print(f"  - ground_truth.png")
print(f"  - initial_dummy.png")
print(f"  - dlg_progress_{int(args.sparsity*100)}.png")
print(f"  - dlg_loss_{int(args.sparsity*100)}.png")
print(f"  - dlg_final_{int(args.sparsity*100)}.png")
