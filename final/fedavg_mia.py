# -*- coding: utf-8 -*-
"""
FedAvg with Model Inversion Attack (MIA) + Gradient Sparsification

이 스크립트는 FedAvg 환경에서 Model Inversion Attack을 수행합니다:
1. 클라이언트가 로컬 데이터로 모델 학습
2. 그래디언트를 스파시피케이션으로 압축
3. 공격자가 스파시파이된 그래디언트로부터 데이터 복원 시도

Usage:
    # 100% 그래디언트 전송 (기준선)
    python fedavg_mia.py --sparsity 1.0 --index 25
    
    # 상위 10% 파라미터만 전송
    python fedavg_mia.py --sparsity 0.1 --index 25
    
    # 상위 5% 파라미터만 전송
    python fedavg_mia.py --sparsity 0.05 --index 25
    
    # 상위 1% 파라미터만 전송
    python fedavg_mia.py --sparsity 0.01 --index 25
"""

import argparse
import numpy as np
from PIL import Image
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

print(f"PyTorch: {torch.__version__}")

from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import LeNet, weights_init

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some metrics will be skipped.")

# ============================================================================
# Argument Parser
# ============================================================================
parser = argparse.ArgumentParser(description='FedAvg with Model Inversion Attack')
parser.add_argument('--index', type=int, default=25,
                    help='Target image index in CIFAR-10')
parser.add_argument('--image', type=str, default='',
                    help='Path to custom image')
parser.add_argument('--sparsity', type=float, default=1.0,
                    help='Gradient retention ratio (1.0=100%, 0.1=10%, 0.05=5%, 0.01=1%)')
parser.add_argument('--local_epochs', type=int, default=5,
                    help='Number of local training epochs')
parser.add_argument('--local_lr', type=float, default=0.01,
                    help='Learning rate for local training')
parser.add_argument('--mia_iters', type=int, default=100,
                    help='Number of MIA optimization iterations')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed')
args = parser.parse_args()

# ============================================================================
# Device Setup
# ============================================================================
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")
print(f"Sparsity Ratio: {args.sparsity * 100}% (top {args.sparsity * 100}% gradients retained)")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ============================================================================
# MIA Hyperparameters
# ============================================================================
MIA_L2_WEIGHT = 0.0001   # L2 regularization for stability

# ============================================================================
# Data Preparation
# ============================================================================
print("\n" + "="*60)
print("1. DATA PREPARATION")
print("="*60)

dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# Ground truth data
img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

if hasattr(args, 'image') and args.image and len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

# Save ground truth
plt.figure(figsize=(4, 4))
plt.imshow(tt(gt_data[0].cpu()))
plt.title(f"Ground Truth Image (Class {gt_label.item()})")
plt.axis('off')
plt.tight_layout()
plt.savefig("mia_ground_truth.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Ground truth image saved: mia_ground_truth.png")
print(f"  Image shape: {gt_data.shape}, Label: {gt_label.item()}")

# ============================================================================
# Model Setup
# ============================================================================
print("\n" + "="*60)
print("2. MODEL SETUP")
print("="*60)

net = LeNet().to(device)
torch.manual_seed(args.seed)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

total_params = sum(p.numel() for p in net.parameters())
print(f"✓ LeNet loaded")
print(f"  Total parameters: {total_params:,}")

# ============================================================================
# Metrics Calculation Functions
# ============================================================================
def calculate_metrics(gt_label, pred_label, mse, psnr):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        gt_label: Ground truth label (scalar)
        pred_label: Predicted label (scalar)
        mse: Mean Squared Error
        psnr: Peak Signal-to-Noise Ratio
    
    Returns:
        metrics dictionary with ACC, F1, Precision, Recall, MSE, PSNR
    """
    metrics = {
        'ACC': 1.0 if gt_label == pred_label else 0.0,
        'MSE': mse,
        'PSNR': psnr,
    }
    
    if SKLEARN_AVAILABLE:
        # For binary classification (correct/incorrect)
        y_true = [1]  # 1 = correct
        y_pred = [1 if gt_label == pred_label else 0]
        
        try:
            metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            print(f"Warning: Could not calculate F1 metrics: {e}")
    
    return metrics
def sparsify_gradients(gradients, ratio):
    """
    Keep only top-k% gradients by magnitude, set others to zero.
    
    Args:
        gradients: list of gradient tensors
        ratio: retention ratio (1.0 = keep all, 0.1 = keep top 10%)
    
    Returns:
        sparsified gradients, sparsification statistics
    """
    # Flatten all gradients
    flat_grads = torch.cat([g.view(-1) for g in gradients])
    num_params = flat_grads.numel()
    
    if ratio >= 1.0:
        return gradients, {
            'threshold': 0.0,
            'retained': 1.0,
            'nonzero': num_params,
            'total': num_params
        }
    
    # Find threshold for top-k%
    k = max(1, int(num_params * ratio))
    top_values, _ = torch.topk(torch.abs(flat_grads), k)
    threshold = top_values[-1]
    
    # Apply sparsification mask
    sparsified = []
    for g in gradients:
        mask = (torch.abs(g) >= threshold).float()
        sparsified.append(g * mask)
    
    # Calculate statistics
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
print("3. FEDAVG LOCAL TRAINING (Client Simulation)")
print("="*60)

# Save initial weights for gradient computation
initial_weights = [p.detach().clone() for p in net.parameters()]

# Local optimizer (client)
optimizer_local = torch.optim.SGD(net.parameters(), lr=args.local_lr)

# Local training phase
net.train()
for epoch in range(args.local_epochs):
    optimizer_local.zero_grad()
    
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    loss.backward()
    
    optimizer_local.step()
    
    with torch.no_grad():
        pred_probs = F.softmax(pred, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1).item()
    
    print(f"  Local Epoch {epoch}: Loss {loss.item():.4f}, Predicted Label: {pred_label}")

print("✓ Local training completed")

# ============================================================================
# Compute Target Gradient (What Attacker Intercepts)
# ============================================================================
print("\n" + "="*60)
print("4. GRADIENT COMPUTATION & SPARSIFICATION")
print("="*60)

# Restore model to initial state
with torch.no_grad():
    for param, w_init in zip(net.parameters(), initial_weights):
        param.copy_(w_init)

# Compute gradient on fresh model
net.zero_grad()
pred = net(gt_data)
loss = criterion(pred, gt_onehot_label)
loss.backward()

# Extract gradients
target_gradients = [p.grad.detach().clone() for p in net.parameters()]

# Before sparsification
total_grad_norm = sum(g.norm().item() for g in target_gradients)
total_nonzero = sum((g != 0).sum().item() for g in target_gradients)

print(f"\nBefore Sparsification:")
print(f"  Total gradient norm: {total_grad_norm:.4f}")
print(f"  Non-zero elements: {total_nonzero:,}")

# Apply sparsification
print(f"\nApplying sparsification (ratio={args.sparsity})...")
target_gradients, sparse_stats = sparsify_gradients(target_gradients, args.sparsity)

print(f"After Sparsification:")
print(f"  Threshold: {sparse_stats['threshold']:.6f}")
print(f"  Retained: {sparse_stats['nonzero']:,} / {sparse_stats['total']:,} ({sparse_stats['retained']*100:.2f}%)")

# Restore model to initial state for MIA
with torch.no_grad():
    for param, w_init in zip(net.parameters(), initial_weights):
        param.copy_(w_init)

net.eval()

# ============================================================================
# Model Inversion Attack (MIA)
# ============================================================================
print("\n" + "="*60)
print("5. MODEL INVERSION ATTACK (MIA)")
print("="*60)

# Initialize dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

# Save initial dummy
plt.figure(figsize=(4, 4))
plt.imshow(tt(dummy_data[0].cpu().detach()))
plt.title("Initial Dummy (Random)")
plt.axis('off')
plt.tight_layout()
plt.savefig("mia_initial_dummy.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Initial dummy image saved: mia_initial_dummy.png")

# L2 regularization
def l2_penalty(x):
    """Prevent extreme values"""
    return (x ** 2).sum()

# LBFGS optimizer for MIA
optimizer_mia = torch.optim.LBFGS([dummy_data, dummy_label], lr=1.0)

history = []
loss_history = []

print(f"\nRunning MIA for {args.mia_iters} iterations...")
for iters in range(args.mia_iters):
    def closure():
        optimizer_mia.zero_grad()
    
        
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
        
        # Total loss with regularization
        total_loss = grad_diff
        total_loss.backward()
        return total_loss
    
    optimizer_mia.step(closure)
    
    if iters % 10 == 0:
        # Compute current losses (without no_grad to enable autograd)
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=False)
        
        grad_diff = torch.tensor(0.0, device=device)
        for gx, gy in zip(dummy_dy_dx, target_gradients):
            grad_diff = grad_diff + ((gx - gy) ** 2).sum()
        
        loss_history.append(grad_diff.item())
        print(f"  Iter {iters:3d}: GradLoss {grad_diff.item():.4f}")
        
        with torch.no_grad():
            history.append(tt(dummy_data[0].cpu()))

# ============================================================================
# Calculate Metrics
# ============================================================================
print("\n" + "="*60)
print("6. RESULTS & METRICS")
print("="*60)

with torch.no_grad():
    gt_np = gt_data[0].cpu().numpy()
    dummy_np = dummy_data[0].detach().cpu().numpy()
    
    # MSE
    mse = np.mean((gt_np - dummy_np) ** 2)
    
    # PSNR
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10)) if mse > 0 else 100.0
    
    # Predicted labels
    final_pred = net(dummy_data)
    final_pred_probs = F.softmax(final_pred, dim=1)
    pred_label = torch.argmax(final_pred_probs, dim=1).item()
    
    dummy_label_softmax = F.softmax(dummy_label, dim=-1)
    dummy_label_class = torch.argmax(dummy_label_softmax, dim=1).item()

# Calculate comprehensive metrics
metrics = calculate_metrics(gt_label.item(), pred_label, mse, psnr)

print(f"\nReconstruction Quality:")
print(f"  MSE:  {mse:.6f}")
print(f"  PSNR: {psnr:.2f} dB")
print(f"  True Label: {gt_label.item()}")
print(f"  Reconstructed Predicted Label: {pred_label}")
print(f"  Dummy Label (optimized): {dummy_label_class}")
print(f"  Label Correctness: {'✓ MATCH' if pred_label == gt_label.item() else '✗ MISMATCH'}")

print(f"\nEvaluation Metrics:")
print(f"  ACC (Accuracy): {metrics['ACC']:.4f}")
if 'F1' in metrics:
    print(f"  F1 Score: {metrics['F1']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
print(f"  MSE: {metrics['MSE']:.6f}")
print(f"  PSNR: {metrics['PSNR']:.2f} dB")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*60)
print("7. GENERATING VISUALIZATIONS")
print("="*60)

sparsity_str = f"{int(args.sparsity*100)}" if args.sparsity >= 0.01 else f"{args.sparsity*100:.1f}"

# 1. Reconstruction Progress
plt.figure(figsize=(15, 6))
for i in range(min(10, len(history))):
    plt.subplot(2, 5, i + 1)
    plt.imshow(history[i])
    plt.title(f"Iter {i * 10}")
    plt.axis('off')
plt.suptitle(f"MIA Reconstruction Progress (Sparsity: {sparsity_str}%)", fontsize=14)
plt.tight_layout()
plt.savefig(f"mia_progress_sparsity_{sparsity_str}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: mia_progress_sparsity_{sparsity_str}.png")

# 2. Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(range(0, len(loss_history) * 10, 10), loss_history, 'b-', linewidth=2, marker='o')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Gradient Matching Loss', fontsize=12)
plt.title(f'MIA Convergence (Sparsity: {sparsity_str}%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig(f"mia_loss_sparsity_{sparsity_str}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: mia_loss_sparsity_{sparsity_str}.png")

# 3. Final Comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(tt(gt_data[0].cpu()))
axes[0].set_title(f"Ground Truth\nClass: {gt_label.item()}", fontsize=12)
axes[0].axis('off')

axes[1].imshow(tt(dummy_data[0].detach().cpu()))
axes[1].set_title(f"Reconstructed\nClass: {pred_label}\nPSNR: {psnr:.2f}dB", fontsize=12)
axes[1].axis('off')

plt.suptitle(f"Model Inversion Attack Result (Sparsity: {sparsity_str}%)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"mia_final_sparsity_{sparsity_str}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: mia_final_sparsity_{sparsity_str}.png")

# 4. Gradient Distribution Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original gradients
all_original_grads = torch.cat([g.view(-1) for g in [p.grad.detach().clone() for p in net.parameters()]])
axes[0].hist(all_original_grads.cpu().numpy(), bins=100, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Gradient Value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Original Gradient Distribution', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Sparsified gradients
all_sparse_grads = torch.cat([g.view(-1) for g in target_gradients])
all_sparse_grads_nonzero = all_sparse_grads[all_sparse_grads != 0]
if len(all_sparse_grads_nonzero) > 0:
    axes[1].hist(all_sparse_grads_nonzero.cpu().numpy(), bins=100, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_xlabel('Gradient Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Sparsified Gradient Distribution ({sparsity_str}%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(f"mia_gradient_dist_sparsity_{sparsity_str}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: mia_gradient_dist_sparsity_{sparsity_str}.png")

# ============================================================================
# Evaluation Metrics Visualization
# ============================================================================
# Create metrics visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Accuracy & Label Match
ax = axes[0, 0]
metrics_names = ['Accuracy', 'Label Match']
metrics_values = [metrics['ACC'], 1.0 if pred_label == gt_label.item() else 0.0]
colors = ['#2ecc71' if v == 1.0 else '#e74c3c' for v in metrics_values]
bars1 = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Score', fontsize=11, fontweight='bold')
ax.set_title('Classification Metrics', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, metrics_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# 2. MSE & PSNR
ax = axes[0, 1]
ax_twin = ax.twinx()

# Create bars with proper positioning
x_pos = [0, 1]
bar1 = ax.bar(x_pos[0], metrics['MSE'], color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=2, width=0.6, label='MSE')
bar2 = ax_twin.bar(x_pos[1], metrics['PSNR'], color='#3498db', alpha=0.7, edgecolor='black', linewidth=2, width=0.6, label='PSNR')

ax.set_ylabel('MSE (Lower is Better)', fontsize=11, fontweight='bold', color='#e74c3c')
ax_twin.set_ylabel('PSNR (Higher is Better)', fontsize=11, fontweight='bold', color='#3498db')
ax.set_title('Reconstruction Quality', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['MSE', 'PSNR'])
ax.tick_params(axis='y', labelcolor='#e74c3c')
ax_twin.tick_params(axis='y', labelcolor='#3498db')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
ax.text(x_pos[0], metrics['MSE'], f'{metrics["MSE"]:.4f}', ha='center', va='bottom', fontweight='bold')
ax_twin.text(x_pos[1], metrics['PSNR'], f'{metrics["PSNR"]:.2f}dB', ha='center', va='bottom', fontweight='bold')

# 3. F1, Precision, Recall (if available)
ax = axes[1, 0]
if 'F1' in metrics:
    metric_names = ['F1', 'Precision', 'Recall']
    metric_values = [metrics['F1'], metrics['Precision'], metrics['Recall']]
    colors = ['#9b59b6', '#3498db', '#2ecc71']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Extended Metrics', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
else:
    ax.text(0.5, 0.5, 'sklearn not available\nfor F1/Precision/Recall', 
            ha='center', va='center', fontsize=11, transform=ax.transAxes)
    ax.axis('off')

# 4. Summary Statistics
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
EXPERIMENT SUMMARY

Sparsity: {args.sparsity * 100:.1f}%
True Label: {gt_label.item()}
Predicted Label: {pred_label}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:   {metrics['ACC']:.4f}
MSE:        {metrics['MSE']:.6f}
PSNR:       {metrics['PSNR']:.2f} dB
"""

if 'F1' in metrics:
    summary_text += f"""F1 Score:   {metrics['F1']:.4f}
Precision:  {metrics['Precision']:.4f}
Recall:     {metrics['Recall']:.4f}
"""

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle(f'Evaluation Metrics Summary (Sparsity: {sparsity_str}%)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f"mia_metrics_sparsity_{sparsity_str}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: mia_metrics_sparsity_{sparsity_str}.png")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"\nExperiment Configuration:")
print(f"  Target Image Index: {args.index}")
print(f"  Sparsity Ratio: {args.sparsity * 100}%")
print(f"  Local Training Epochs: {args.local_epochs}")
print(f"  Local Learning Rate: {args.local_lr}")
print(f"  MIA Iterations: {args.mia_iters}")
print(f"  Model: LeNet ({total_params:,} parameters)")

print(f"\nGradient Sparsification Statistics:")
print(f"  Retained Parameters: {sparse_stats['nonzero']:,} / {sparse_stats['total']:,}")
print(f"  Retention Ratio: {sparse_stats['retained']*100:.2f}%")
print(f"  Threshold: {sparse_stats['threshold']:.6f}")

print(f"\nReconstruction Attack Results:")
print(f"  MSE: {mse:.6f}")
print(f"  PSNR: {psnr:.2f} dB")
print(f"  True Label: {gt_label.item()}")
print(f"  Predicted Label: {pred_label}")
print(f"  Label Match: {'YES ✓' if pred_label == gt_label.item() else 'NO ✗'}")

print(f"\nGenerated Files:")
print(f"  - mia_ground_truth.png")
print(f"  - mia_initial_dummy.png")
print(f"  - mia_progress_sparsity_{sparsity_str}.png")
print(f"  - mia_loss_sparsity_{sparsity_str}.png")
print(f"  - mia_final_sparsity_{sparsity_str}.png")
print(f"  - mia_gradient_dist_sparsity_{sparsity_str}.png")
print(f"  - mia_metrics_sparsity_{sparsity_str}.png (NEW: Evaluation Metrics)")

print("\n" + "="*60)
print("✓ Experiment completed successfully!")
print("="*60)
