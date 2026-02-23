# -*- coding: utf-8 -*-
"""
Simplified DLG with FedAvg + Sparsification (No matplotlib GUI)

Usage:
    python dlg_simple.py --index 25 --sparsity 1.0
    python dlg_simple.py --index 25 --sparsity 0.1
    python dlg_simple.py --index 25 --sparsity 0.01
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

print(f"PyTorch version: {torch.__version__}")

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser(description='DLG with FedAvg + Sparsification')
parser.add_argument('--index', type=int, default=25)
parser.add_argument('--sparsity', type=float, default=1.0,
                    help='gradient retention ratio (1.0=100%, 0.1=10%, 0.01=1%)')
parser.add_argument('--local_epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dlg_iters', type=int, default=300)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                    help='device to use (auto=automatic detection)')
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
        print("  No sparsification applied (100%)")
        return
    
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    
    if not grads:
        return
    
    all_grads = torch.cat(grads)
    num_params = all_grads.numel()
    k = max(1, int(num_params * ratio))
    top_values, _ = torch.topk(torch.abs(all_grads), k)
    threshold = top_values[-1]
    
    for param in model.parameters():
        if param.grad is not None:
            mask = (torch.abs(param.grad) >= threshold).float()
            param.grad.data *= mask
    
    nonzero = sum((p.grad != 0).sum().item() for p in model.parameters() if p.grad is not None)
    print(f"  Sparsified: {nonzero:,} / {num_params:,} retained ({nonzero/num_params*100:.2f}%)")

# -------------------------
# FedAvg Local Training
# -------------------------
print("="*50)
print("FedAvg Local Training")
print("="*50)

original_weights = [p.detach().clone() for p in net.parameters()]

# [핵심] BatchNorm running statistics 저장
original_bn_stats = {}
for name, module in net.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        original_bn_stats[name] = {
            'running_mean': module.running_mean.clone(),
            'running_var': module.running_var.clone(),
            'num_batches_tracked': module.num_batches_tracked.clone()
        }

print(f"✓ Saved {len(original_bn_stats)} BatchNorm layers\n")

optimizer_client = torch.optim.SGD(net.parameters(), lr=args.lr)

# Check initial loss
net.eval()
with torch.no_grad():
    pred_init = net(gt_data)
    loss_init = criterion(pred_init, gt_onehot_label)
print(f"  Initial Loss (before training): {loss_init.item():.4f}")

net.train()
for epoch in range(args.local_epochs):
    optimizer_client.zero_grad()
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    loss.backward()
    optimizer_client.step()
    print(f"  Epoch {epoch}: Loss before step {loss.item():.4f}")

# Check final loss
net.eval()
with torch.no_grad():
    pred_final = net(gt_data)
    loss_final = criterion(pred_final, gt_onehot_label)
print(f"  Final Loss (after training): {loss_final.item():.4f}")

print("✓ Local training completed\n")

# -------------------------
# Target Gradient
# -------------------------
print("="*50)
print("Computing Target Gradient")
print("="*50)

# Restore to initial state (weights + BN stats)
with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)
    
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d) and name in original_bn_stats:
            module.running_mean.copy_(original_bn_stats[name]['running_mean'])
            module.running_var.copy_(original_bn_stats[name]['running_var'])
            module.num_batches_tracked.copy_(original_bn_stats[name]['num_batches_tracked'])

print("✓ Model restored to initial state (weights + BN stats)")

# Compute gradient at initial state
net.zero_grad()
pred = net(gt_data)
loss = criterion(pred, gt_onehot_label)
loss.backward()

# Apply sparsification
sparsify_gradients(net, args.sparsity)

# Save target
original_dy_dx = [p.grad.detach().clone() for p in net.parameters()]
grad_norm = sum(g.norm().item() for g in original_dy_dx)
print(f"✓ Target gradient norm: {grad_norm:.4f}\n")

# [핵심] DLG uses EVAL mode (not train!)
# Train mode + batch=1 causes unstable BN stats → false gradient matching
# We restored BN stats, so eval mode will use correct statistics
net.eval()
print("✓ Using eval mode for DLG (stable BN with restored stats)")

# -------------------------
# DLG Attack
# -------------------------
print("="*50)
print("DLG Reconstruction Attack")
print("="*50)

dummy_data = torch.randn(gt_data.size(), device=device, requires_grad=True)
dummy_label = torch.randn(gt_onehot_label.size(), device=device, requires_grad=True)

# Save initial dummy
plt.figure(figsize=(4, 4))
plt.imshow(tt(dummy_data[0].detach().cpu()))
plt.title("Initial Dummy")
plt.axis('off')
plt.savefig("initial_dummy.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Initial dummy saved\n")

optimizer = torch.optim.LBFGS([dummy_data, dummy_label], 
                              lr=1.0,
                              max_iter=20,
                              max_eval=25,
                              tolerance_grad=1e-7,
                              tolerance_change=1e-9,
                              history_size=100,
                              line_search_fn="strong_wolfe")

history = []
loss_history = []

for iters in range(args.dlg_iters):
    def closure():
        optimizer.zero_grad()
        
        # [중요] clamp in no_grad context
        with torch.no_grad():
            dummy_data.clamp_(0, 1)
        
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff = grad_diff + ((gx - gy) ** 2).sum()
        
        grad_diff.backward()
        
        # [Debug] Check if gradient reached dummy_data
        if iters == 0:
            data_grad_inside = dummy_data.grad.norm().item() if dummy_data.grad is not None else 0.0
            print(f"  [Debug Iter 0] DataGrad inside closure: {data_grad_inside:.6f}")
        
        return grad_diff
    
    optimizer.step(closure)
    
    if iters % 10 == 0:
        # Report (check if dummy_data changed)
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=False)
        
        grad_diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff = grad_diff + ((gx - gy) ** 2).sum()
        
        loss_history.append(grad_diff.item())
        
        # Check if dummy_data has gradient
        data_grad_norm = dummy_data.grad.norm().item() if dummy_data.grad is not None else 0.0
        data_mean = dummy_data.mean().item()
        
        print(f"Iter {iters:3d}: GradLoss {grad_diff.item():.4f} | DataGrad {data_grad_norm:.4f} | DataMean {data_mean:.3f}")
        
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

print(f"MSE: {mse:.6f}")
print(f"PSNR: {psnr:.2f} dB")
print(f"True Label: {gt_label.item()}")
print(f"Predicted Label: {pred_label}\n")

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
plt.figure(figsize=(10, 5))
plt.plot(range(0, len(loss_history) * 10, 10), loss_history, 'b-', linewidth=2)
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
print("✓ All images saved:")
print("="*50)
print(f"  - ground_truth.png")
print(f"  - initial_dummy.png")
print(f"  - dlg_progress_{int(args.sparsity*100)}.png")
print(f"  - dlg_loss_{int(args.sparsity*100)}.png")
print(f"  - dlg_final_{int(args.sparsity*100)}.png")
