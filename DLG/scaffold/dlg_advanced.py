# -*- coding: utf-8 -*-
"""
Advanced DLG Attack on SCAFFOLD Federated Learning

Features:
- Total Variation (TV) regularization for smoother reconstructions
- Label inference when labels are unknown
- Multiple initialization strategies
- Enhanced visualization and metrics
- Support for different optimizers
"""
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
import copy
import os
from datetime import datetime

print(f"PyTorch version: {torch.__version__}, TorchVision version: {torchvision.__version__}")

from utils import label_to_onehot, cross_entropy_for_onehot, total_variation

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(description='Advanced DLG Attack on SCAFFOLD')
parser.add_argument('--index', type=int, default=25,
                    help='Image index for CIFAR-10')
parser.add_argument('--image', type=str, default="",
                    help='Path to custom image')
parser.add_argument('--local_epochs', type=int, default=1,
                    help='Number of local epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate for SCAFFOLD training')
parser.add_argument('--dlg_lr', type=float, default=0.1,
                    help='Learning rate for DLG optimization')
parser.add_argument('--dlg_iterations', type=int, default=300,
                    help='Number of DLG iterations')
parser.add_argument('--optimizer', type=str, default='LBFGS',
                    choices=['LBFGS', 'Adam', 'SGD'],
                    help='Optimizer for DLG')
parser.add_argument('--use_tv', action='store_true',
                    help='Use Total Variation regularization')
parser.add_argument('--tv_weight', type=float, default=0.001,
                    help='Weight for TV regularization')
parser.add_argument('--infer_label', action='store_true',
                    help='Infer label instead of using ground truth')
parser.add_argument('--init_strategy', type=str, default='random',
                    choices=['random', 'zeros', 'mean'],
                    help='Initialization strategy for dummy data')
parser.add_argument('--output_dir', type=str, default='./dlg_results',
                    help='Output directory for results')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed')
parser.add_argument('--save_interval', type=int, default=20,
                    help='Save reconstruction every N iterations')
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = os.path.join(args.output_dir, f"exp_{timestamp}")
os.makedirs(exp_dir, exist_ok=True)

print(f"Experiment directory: {exp_dir}")

# ==================== Device Setup ====================
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")

# ==================== Dataset Preparation ====================
print("\n" + "="*70)
print("STAGE 1: Dataset Loading")
print("="*70)

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

print(f"✓ Loaded image index {img_index}, label: {gt_label.item()}")
print(f"✓ Image shape: {gt_data.shape}")

# Save ground truth
plt.figure(figsize=(6, 6))
plt.imshow(tt(gt_data[0].cpu()))
plt.title(f"Ground Truth (Label: {gt_label.item()})", fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, 'ground_truth.png'), dpi=150, bbox_inches='tight')
plt.close()

# ==================== Model Initialization ====================
print("\n" + "="*70)
print("STAGE 2: Model Initialization")
print("="*70)

net = LeNet().to(device)
torch.manual_seed(args.seed)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

num_params = sum(p.numel() for p in net.parameters())
print(f"✓ Model: LeNet")
print(f"✓ Total parameters: {num_params:,}")

# ==================== SCAFFOLD Training ====================
print("\n" + "="*70)
print("STAGE 3: SCAFFOLD Local Training")
print("="*70)
print(f"Configuration:")
print(f"  - Local epochs: {args.local_epochs}")
print(f"  - Learning rate: {args.lr}")
print(f"  - Optimizer: SGD")

# SCAFFOLD control variates
client_control = [torch.zeros_like(p.data) for p in net.parameters()]
server_control = [torch.zeros_like(p.data) for p in net.parameters()]

# Save initial weights
initial_weights = [p.detach().clone() for p in net.parameters()]

# Local training
optimizer_client = torch.optim.SGD(net.parameters(), lr=args.lr)

net.train()
training_losses = []

for epoch in range(args.local_epochs):
    optimizer_client.zero_grad()
    
    # Forward pass
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    training_losses.append(loss.item())
    
    # Backward pass
    loss.backward()
    
    # SCAFFOLD correction
    with torch.no_grad():
        for param, c_server, c_client in zip(net.parameters(), server_control, client_control):
            if param.grad is not None:
                param.grad.data.add_(c_server - c_client)
    
    optimizer_client.step()
    
    print(f"  Epoch {epoch+1}/{args.local_epochs}: Loss = {loss.item():.6f}")

# Compute weight updates and target gradients
trained_weights = [p.detach().clone() for p in net.parameters()]
weight_deltas = [w_new - w_old for w_new, w_old in zip(trained_weights, initial_weights)]

target_grad = []
for delta_w in weight_deltas:
    effective_grad = -delta_w / (args.lr * args.local_epochs)
    target_grad.append(effective_grad.detach())

print(f"✓ Training completed")
print(f"✓ Final loss: {training_losses[-1]:.6f}")

# Reset model
with torch.no_grad():
    for param, w_init in zip(net.parameters(), initial_weights):
        param.copy_(w_init)

# ==================== DLG Attack Setup ====================
print("\n" + "="*70)
print("STAGE 4: DLG Attack Configuration")
print("="*70)
print(f"Configuration:")
print(f"  - Optimizer: {args.optimizer}")
print(f"  - Learning rate: {args.dlg_lr}")
print(f"  - Iterations: {args.dlg_iterations}")
print(f"  - TV regularization: {args.use_tv}")
if args.use_tv:
    print(f"  - TV weight: {args.tv_weight}")
print(f"  - Label inference: {args.infer_label}")
print(f"  - Initialization: {args.init_strategy}")

# Initialize dummy data
if args.init_strategy == 'random':
    dummy_data = torch.randn(gt_data.size()).to(device)
elif args.init_strategy == 'zeros':
    dummy_data = torch.zeros(gt_data.size()).to(device)
elif args.init_strategy == 'mean':
    dummy_data = torch.ones(gt_data.size()).to(device) * 0.5

dummy_data.requires_grad_(True)

# Initialize dummy label
if args.infer_label:
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    optimize_label = True
    print("  - Optimizing label: YES")
else:
    dummy_label = gt_onehot_label.clone().to(device)
    optimize_label = False
    print("  - Using known label: YES")

# Save initial dummy data
plt.figure(figsize=(6, 6))
plt.imshow(tt(dummy_data[0].cpu()))
plt.title(f"Initial Dummy Data ({args.init_strategy})", fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, 'initial_dummy.png'), dpi=150, bbox_inches='tight')
plt.close()

# Setup optimizer
if optimize_label:
    opt_params = [dummy_data, dummy_label]
else:
    opt_params = [dummy_data]

if args.optimizer == 'LBFGS':
    optimizer = torch.optim.LBFGS(opt_params, lr=args.dlg_lr, max_iter=1)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(opt_params, lr=args.dlg_lr)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(opt_params, lr=args.dlg_lr, momentum=0.9)

# ==================== DLG Attack ====================
print("\n" + "="*70)
print("STAGE 5: DLG Attack Execution")
print("="*70)

history = []
loss_history = []
grad_loss_history = []
tv_loss_history = []
label_history = [] if optimize_label else None

print("Iteration | Grad Loss  | TV Loss    | Total Loss | Label Pred")
print("-" * 70)

for iters in range(args.dlg_iterations):
    def closure():
        optimizer.zero_grad()
        
        # Clamp dummy data to valid pixel range
        dummy_data.data.clamp_(0, 1)
        
        # Prepare label
        if optimize_label:
            current_label = F.softmax(dummy_label, dim=-1)
        else:
            current_label = dummy_label
        
        # Forward pass
        dummy_pred = net(dummy_data)
        dummy_loss = criterion(dummy_pred, current_label)
        
        # Compute gradients
        dummy_dy_dx = torch.autograd.grad(
            dummy_loss,
            net.parameters(),
            create_graph=True
        )
        
        # Gradient matching loss
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, target_grad):
            grad_diff += ((gx - gy) ** 2).sum()
        
        # Total Variation regularization
        if args.use_tv:
            tv_loss = total_variation(dummy_data)
            total_loss = grad_diff + args.tv_weight * tv_loss
        else:
            tv_loss = torch.tensor(0.0)
            total_loss = grad_diff
        
        total_loss.backward()
        
        return total_loss, grad_diff, tv_loss
    
    if args.optimizer == 'LBFGS':
        total_loss, grad_loss, tv_loss = optimizer.step(closure)
    else:
        total_loss, grad_loss, tv_loss = closure()
        optimizer.step()
    
    # Record losses
    loss_history.append(total_loss.item())
    grad_loss_history.append(grad_loss.item())
    tv_loss_history.append(tv_loss.item() if args.use_tv else 0)
    
    # Predicted label
    if optimize_label:
        pred_label_idx = F.softmax(dummy_label, dim=-1).argmax().item()
        label_history.append(pred_label_idx)
        label_str = f"{pred_label_idx}"
    else:
        label_str = "Known"
    
    # Log and save progress
    if iters % args.save_interval == 0 or iters == args.dlg_iterations - 1:
        print(f"{iters:9d} | {grad_loss.item():10.6f} | {tv_loss.item():10.6f} | "
              f"{total_loss.item():10.6f} | {label_str:>10s}")
        
        # Save snapshot
        history.append(tt(dummy_data[0].cpu()))
        
        # Save intermediate result
        if iters > 0 and iters % (args.save_interval * 5) == 0:
            plt.figure(figsize=(6, 6))
            plt.imshow(history[-1])
            plt.title(f"Iteration {iters}", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, f'recon_iter_{iters:04d}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()

print("-" * 70)
print("✓ DLG attack completed!")

# ==================== Results Visualization ====================
print("\n" + "="*70)
print("STAGE 6: Results Visualization")
print("="*70)

# Reconstruction progress
num_snapshots = min(len(history), 12)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(num_snapshots):
    idx = int(i * (len(history) - 1) / (num_snapshots - 1)) if num_snapshots > 1 else 0
    axes[i].imshow(history[idx])
    iter_num = idx * args.save_interval
    axes[i].set_title(f"Iteration {iter_num}", fontsize=10)
    axes[i].axis('off')

for i in range(num_snapshots, 12):
    axes[i].axis('off')

plt.suptitle("DLG Reconstruction Progress on SCAFFOLD", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, 'reconstruction_progress.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved reconstruction progress")

# Loss curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total loss
axes[0, 0].plot(loss_history, linewidth=2, color='blue')
axes[0, 0].set_xlabel('Iteration', fontsize=11)
axes[0, 0].set_ylabel('Total Loss', fontsize=11)
axes[0, 0].set_title('Total Loss Curve', fontsize=12, fontweight='bold')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# Gradient loss
axes[0, 1].plot(grad_loss_history, linewidth=2, color='red')
axes[0, 1].set_xlabel('Iteration', fontsize=11)
axes[0, 1].set_ylabel('Gradient Matching Loss', fontsize=11)
axes[0, 1].set_title('Gradient Loss Curve', fontsize=12, fontweight='bold')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# TV loss
if args.use_tv:
    axes[1, 0].plot(tv_loss_history, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Iteration', fontsize=11)
    axes[1, 0].set_ylabel('TV Loss', fontsize=11)
    axes[1, 0].set_title('Total Variation Loss', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'TV Regularization\nNot Used', 
                   ha='center', va='center', fontsize=14, transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')

# Label inference
if optimize_label and label_history:
    axes[1, 1].plot(label_history, linewidth=2, color='purple', marker='o', markersize=3)
    axes[1, 1].axhline(y=gt_label.item(), color='black', linestyle='--', 
                      linewidth=2, label=f'True Label ({gt_label.item()})')
    axes[1, 1].set_xlabel('Iteration', fontsize=11)
    axes[1, 1].set_ylabel('Predicted Label', fontsize=11)
    axes[1, 1].set_title('Label Inference Progress', fontsize=12, fontweight='bold')
    axes[1, 1].set_yticks(range(10))
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
else:
    axes[1, 1].text(0.5, 0.5, 'Label Inference\nNot Used', 
                   ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

plt.suptitle('DLG Attack Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved loss curves")

# Final comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(tt(gt_data[0].cpu()))
axes[0].set_title(f"Ground Truth\n(Label: {gt_label.item()})", 
                 fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(history[-1])
if optimize_label:
    final_pred = label_history[-1] if label_history else "Unknown"
    axes[1].set_title(f"DLG Reconstructed\n(Predicted: {final_pred})", 
                     fontsize=14, fontweight='bold')
else:
    axes[1].set_title("DLG Reconstructed", fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.suptitle("DLG Attack Results on SCAFFOLD", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(exp_dir, 'final_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved final comparison")

# ==================== Quality Metrics ====================
print("\n" + "="*70)
print("STAGE 7: Quality Assessment")
print("="*70)

# MSE
mse = F.mse_loss(dummy_data[0], gt_data[0]).item()
print(f"Mean Squared Error (MSE): {mse:.8f}")

# PSNR
if mse > 0:
    psnr = 20 * np.log10(1.0) - 10 * np.log10(mse)
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.4f} dB")
else:
    psnr = float('inf')
    print("PSNR: Infinity (perfect reconstruction)")

# Correlation
correlation = torch.corrcoef(torch.stack([
    dummy_data[0].flatten(),
    gt_data[0].flatten()
]))[0, 1].item()
print(f"Pixel-wise Correlation: {correlation:.6f}")

# L1 distance
l1_dist = F.l1_loss(dummy_data[0], gt_data[0]).item()
print(f"L1 Distance: {l1_dist:.8f}")

# SSIM (simplified version)
def ssim_simple(img1, img2):
    c1, c2 = 0.01**2, 0.03**2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1)**2).mean()
    sigma2_sq = ((img2 - mu2)**2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim_val = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_val.item()

ssim_val = ssim_simple(dummy_data[0], gt_data[0])
print(f"Structural Similarity (SSIM): {ssim_val:.6f}")

# Label accuracy
if optimize_label and label_history:
    final_label_pred = label_history[-1]
    label_correct = (final_label_pred == gt_label.item())
    print(f"Label Inference: {final_label_pred} (Ground Truth: {gt_label.item()})")
    print(f"Label Accuracy: {'✓ Correct' if label_correct else '✗ Incorrect'}")

# ==================== Summary Report ====================
print("\n" + "="*70)
print("EXPERIMENT SUMMARY")
print("="*70)

summary = f"""
Configuration:
  Dataset: CIFAR-10
  Image Index: {args.index}
  True Label: {gt_label.item()}
  
SCAFFOLD Training:
  Local Epochs: {args.local_epochs}
  Learning Rate: {args.lr}
  Final Training Loss: {training_losses[-1]:.6f}

DLG Attack:
  Optimizer: {args.optimizer}
  Learning Rate: {args.dlg_lr}
  Iterations: {args.dlg_iterations}
  TV Regularization: {args.use_tv}
  TV Weight: {args.tv_weight if args.use_tv else 'N/A'}
  Label Inference: {args.infer_label}
  Initialization: {args.init_strategy}

Quality Metrics:
  MSE: {mse:.8f}
  PSNR: {psnr:.4f} dB
  Correlation: {correlation:.6f}
  L1 Distance: {l1_dist:.8f}
  SSIM: {ssim_val:.6f}
  Final Gradient Loss: {grad_loss_history[-1]:.8f}
"""

if optimize_label and label_history:
    summary += f"  Label Prediction: {label_history[-1]} ({'Correct' if label_correct else 'Incorrect'})\n"

summary += f"""
Output Files:
  Directory: {exp_dir}
  - ground_truth.png
  - initial_dummy.png
  - reconstruction_progress.png
  - loss_curves.png
  - final_comparison.png
  - recon_iter_*.png (intermediate results)
  - summary.txt (this report)
"""

print(summary)

# Save summary
with open(os.path.join(exp_dir, 'summary.txt'), 'w') as f:
    f.write(summary)

print("="*70)
print(f"✓ All results saved to: {exp_dir}")
print("="*70)

