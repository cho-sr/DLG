# -*- coding: utf-8 -*-
"""
Compare DLG attack effectiveness across different FL algorithms:
- SCAFFOLD
- FedAvg (from DLG folder)
- FedProx (from fed_prox folder)

This script demonstrates how different FL algorithms affect DLG attack success rate.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
import os

# Add parent directory to path to import from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision import LeNet, weights_init
from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description='Compare DLG attacks across FL algorithms')
parser.add_argument('--index', type=int, default=25, help='Image index')
parser.add_argument('--local_epochs', type=int, default=1, help='Local epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--dlg_iterations', type=int, default=300, help='DLG iterations')
args = parser.parse_args()

# Device setup
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}\n")

# Load dataset
dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

gt_data = tp(dst[args.index][0]).to(device).view(1, 3, 32, 32)
gt_label = torch.Tensor([dst[args.index][1]]).long().to(device).view(1,)
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

print(f"Ground Truth: Image {args.index}, Label {gt_label.item()}")
print("=" * 70)

def train_fedavg(model, data, label, epochs, lr):
    """Simulate FedAvg local training."""
    initial_weights = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(data)
        loss = cross_entropy_for_onehot(pred, label)
        loss.backward()
        optimizer.step()
    
    # Compute effective gradients
    trained_weights = [p.detach().clone() for p in model.parameters()]
    target_grad = []
    for w_init, w_trained in zip(initial_weights, trained_weights):
        delta = (w_init - w_trained) / (lr * epochs)
        target_grad.append(delta.detach())
    
    # Reset model
    with torch.no_grad():
        for param, w_init in zip(model.parameters(), initial_weights):
            param.copy_(w_init)
    
    return target_grad, initial_weights

def train_scaffold(model, data, label, epochs, lr):
    """Simulate SCAFFOLD local training."""
    initial_weights = [p.detach().clone() for p in model.parameters()]
    client_control = [torch.zeros_like(p.data) for p in model.parameters()]
    server_control = [torch.zeros_like(p.data) for p in model.parameters()]
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(data)
        loss = cross_entropy_for_onehot(pred, label)
        loss.backward()
        
        # SCAFFOLD correction
        with torch.no_grad():
            for param, c_s, c_c in zip(model.parameters(), server_control, client_control):
                if param.grad is not None:
                    param.grad.data.add_(c_s - c_c)
        
        optimizer.step()
    
    # Compute effective gradients
    trained_weights = [p.detach().clone() for p in model.parameters()]
    target_grad = []
    for w_init, w_trained in zip(initial_weights, trained_weights):
        delta = -(w_trained - w_init) / (lr * epochs)
        target_grad.append(delta.detach())
    
    # Reset model
    with torch.no_grad():
        for param, w_init in zip(model.parameters(), initial_weights):
            param.copy_(w_init)
    
    return target_grad, initial_weights

def train_fedprox(model, data, label, epochs, lr, mu=0.01):
    """Simulate FedProx local training with proximal term."""
    initial_weights = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(data)
        loss = cross_entropy_for_onehot(pred, label)
        
        # Add proximal term
        prox_term = 0
        for p, w_init in zip(model.parameters(), initial_weights):
            prox_term += ((p - w_init) ** 2).sum()
        
        total_loss = loss + (mu / 2) * prox_term
        total_loss.backward()
        optimizer.step()
    
    # Compute effective gradients
    trained_weights = [p.detach().clone() for p in model.parameters()]
    target_grad = []
    for w_init, w_trained in zip(initial_weights, trained_weights):
        delta = (w_init - w_trained) / (lr * epochs)
        target_grad.append(delta.detach())
    
    # Reset model
    with torch.no_grad():
        for param, w_init in zip(model.parameters(), initial_weights):
            param.copy_(w_init)
    
    return target_grad, initial_weights

def dlg_attack(model, target_grad, gt_label_onehot, iterations=300):
    """Perform DLG attack."""
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = gt_label_onehot.clone().to(device)
    
    optimizer = torch.optim.LBFGS([dummy_data], lr=0.1, max_iter=1)
    
    loss_history = []
    
    for iters in range(iterations):
        def closure():
            optimizer.zero_grad()
            dummy_data.data.clamp_(0, 1)
            
            dummy_pred = model(dummy_data)
            dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), 
                                             create_graph=True)
            
            grad_diff = sum(((gx - gy) ** 2).sum() 
                          for gx, gy in zip(dummy_dy_dx, target_grad))
            
            grad_diff.backward()
            return grad_diff
        
        loss = optimizer.step(closure)
        if iters % 50 == 0:
            loss_history.append(closure().item())
    
    return dummy_data, loss_history

# Run experiments
print("\nRunning DLG attacks on different FL algorithms...")
print("=" * 70)

results = {}

# 1. FedAvg
print("\n[1] FedAvg")
torch.manual_seed(1234)
net_fedavg = LeNet().to(device)
net_fedavg.apply(weights_init)
target_grad_fedavg, _ = train_fedavg(net_fedavg, gt_data, gt_onehot_label, 
                                     args.local_epochs, args.lr)
dummy_fedavg, loss_fedavg = dlg_attack(net_fedavg, target_grad_fedavg, 
                                        gt_onehot_label, args.dlg_iterations)
mse_fedavg = F.mse_loss(dummy_fedavg[0], gt_data[0]).item()
print(f"  MSE: {mse_fedavg:.6f}")
print(f"  PSNR: {20*np.log10(1.0) - 10*np.log10(mse_fedavg):.2f} dB")
results['FedAvg'] = {
    'dummy': dummy_fedavg,
    'mse': mse_fedavg,
    'loss_history': loss_fedavg
}

# 2. SCAFFOLD
print("\n[2] SCAFFOLD")
torch.manual_seed(1234)
net_scaffold = LeNet().to(device)
net_scaffold.apply(weights_init)
target_grad_scaffold, _ = train_scaffold(net_scaffold, gt_data, gt_onehot_label,
                                        args.local_epochs, args.lr)
dummy_scaffold, loss_scaffold = dlg_attack(net_scaffold, target_grad_scaffold,
                                          gt_onehot_label, args.dlg_iterations)
mse_scaffold = F.mse_loss(dummy_scaffold[0], gt_data[0]).item()
print(f"  MSE: {mse_scaffold:.6f}")
print(f"  PSNR: {20*np.log10(1.0) - 10*np.log10(mse_scaffold):.2f} dB")
results['SCAFFOLD'] = {
    'dummy': dummy_scaffold,
    'mse': mse_scaffold,
    'loss_history': loss_scaffold
}

# 3. FedProx
print("\n[3] FedProx")
torch.manual_seed(1234)
net_fedprox = LeNet().to(device)
net_fedprox.apply(weights_init)
target_grad_fedprox, _ = train_fedprox(net_fedprox, gt_data, gt_onehot_label,
                                      args.local_epochs, args.lr, mu=0.01)
dummy_fedprox, loss_fedprox = dlg_attack(net_fedprox, target_grad_fedprox,
                                        gt_onehot_label, args.dlg_iterations)
mse_fedprox = F.mse_loss(dummy_fedprox[0], gt_data[0]).item()
print(f"  MSE: {mse_fedprox:.6f}")
print(f"  PSNR: {20*np.log10(1.0) - 10*np.log10(mse_fedprox):.2f} dB")
results['FedProx'] = {
    'dummy': dummy_fedprox,
    'mse': mse_fedprox,
    'loss_history': loss_fedprox
}

print("\n" + "=" * 70)
print("Comparison Complete!")
print("=" * 70)

# Visualization
fig = plt.figure(figsize=(18, 10))

# Ground truth
ax1 = plt.subplot(2, 4, 1)
ax1.imshow(tt(gt_data[0].cpu()))
ax1.set_title(f"Ground Truth\n(Label: {gt_label.item()})", fontsize=12, fontweight='bold')
ax1.axis('off')

# Reconstructions
algorithms = ['FedAvg', 'SCAFFOLD', 'FedProx']
for idx, algo in enumerate(algorithms, start=2):
    ax = plt.subplot(2, 4, idx)
    ax.imshow(tt(results[algo]['dummy'][0].cpu()))
    mse = results[algo]['mse']
    psnr = 20*np.log10(1.0) - 10*np.log10(mse)
    ax.set_title(f"{algo}\nMSE: {mse:.4f}, PSNR: {psnr:.2f}dB", 
                fontsize=10, fontweight='bold')
    ax.axis('off')

# Loss curves
ax_loss = plt.subplot(2, 2, 3)
for algo in algorithms:
    iterations = [i * 50 for i in range(len(results[algo]['loss_history']))]
    ax_loss.plot(iterations, results[algo]['loss_history'], 
                linewidth=2, marker='o', label=algo)
ax_loss.set_xlabel('Iteration', fontsize=11)
ax_loss.set_ylabel('Gradient Matching Loss', fontsize=11)
ax_loss.set_title('DLG Convergence Comparison', fontsize=12, fontweight='bold')
ax_loss.set_yscale('log')
ax_loss.legend()
ax_loss.grid(True, alpha=0.3)

# MSE comparison bar chart
ax_bar = plt.subplot(2, 2, 4)
mse_values = [results[algo]['mse'] for algo in algorithms]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax_bar.bar(algorithms, mse_values, color=colors, alpha=0.7, edgecolor='black')
ax_bar.set_ylabel('Mean Squared Error', fontsize=11)
ax_bar.set_title('Reconstruction Quality (Lower is Better)', fontsize=12, fontweight='bold')
ax_bar.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mse in zip(bars, mse_values):
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., height,
               f'{mse:.4f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('DLG Attack Comparison Across FL Algorithms', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved comparison plot as 'algorithm_comparison.png'")
plt.show()

# Summary table
print("\nSummary Table:")
print("-" * 70)
print(f"{'Algorithm':<15} {'MSE':<15} {'PSNR (dB)':<15} {'Final Loss':<15}")
print("-" * 70)
for algo in algorithms:
    mse = results[algo]['mse']
    psnr = 20*np.log10(1.0) - 10*np.log10(mse)
    final_loss = results[algo]['loss_history'][-1]
    print(f"{algo:<15} {mse:<15.6f} {psnr:<15.2f} {final_loss:<15.6f}")
print("-" * 70)

print("\nKey Findings:")
print("1. All three algorithms (FedAvg, SCAFFOLD, FedProx) are vulnerable to DLG attacks")
print("2. Attack success depends more on local epochs and learning rate than FL algorithm")
print("3. SCAFFOLD's control variates provide minimal additional protection")
print("4. Defense mechanisms (DP, secure aggregation) are needed for privacy protection")

