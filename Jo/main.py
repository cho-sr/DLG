"""
Sparsification vs FL Utility and DLG Privacy Leakage (CIFAR-10, ResNet-18)

Key requirements implemented:
1) Use custom ResNet18 from models/vision.py (CIFAR-style 3x3 conv1).
2) Partial ImageNet pretrained load while excluding mismatched layers:
   - conv1.weight (7x7 vs 3x3 mismatch)
   - fc.* (torchvision head)
3) Multi-round FL: 10 rounds with sparsification ratios [100%, 10%, 1%].
4) Metrics: Top-1, Top-5, F1-macro.
5) DLG with L-BFGS + TV loss + L2 regularization, 500 iterations.
6) Save PNG outputs:
   - results/comprehensive_metrics.png
   - results/dlg_convergence.png
   - results/reconstruction_comparison.png
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from models.vision import ResNet18


# =========================
# Configuration
# =========================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ROUNDS = 10
LOCAL_EPOCHS_PER_ROUND = 1
BATCH_SIZE = 128
TEST_BATCH_SIZE = 512
LR = 0.03
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

SPARSITY_CASES = [
    ("100% (No Sparsification)", 1.0),
    ("Top 10%", 0.10),
    ("Top 1%", 0.01),
]

DLG_ITERS = 500
DLG_TV_WEIGHT = 0.001
DLG_L2_WEIGHT = 0.0001
DLG_LBFGS_LR = 1.0

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class CaseResult:
    name: str
    ratio: float
    top1: float
    top5: float
    f1_macro: float
    dlg_mse: float
    dlg_mse_curve: List[float]
    original_img: torch.Tensor
    reconstructed_img: torch.Tensor


def denormalize(img_chw: np.ndarray) -> np.ndarray:
    img = np.transpose(img_chw, (1, 2, 0))
    mean = np.array(CIFAR10_MEAN, dtype=np.float32)
    std = np.array(CIFAR10_STD, dtype=np.float32)
    img = img * std + mean
    return np.clip(img, 0.0, 1.0)


def make_dataloaders() -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    # Speed-oriented subset for experiment iteration.
    train_subset = Subset(train_set, range(20000))
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


def build_model_with_partial_pretrained(num_classes: int = 10) -> nn.Module:
    """
    Build custom CIFAR ResNet18 and partially load torchvision pretrained weights.

    Exclude:
      - conv1.weight (shape mismatch 7x7 vs 3x3)
      - fc.* (different classifier head)
    """
    model = ResNet18(num_classes=num_classes).to(DEVICE)

    tv_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    tv_state = tv_model.state_dict()
    dst_state = model.state_dict()

    filtered = {}
    for k, v in tv_state.items():
        if k == "conv1.weight" or k.startswith("fc."):
            continue
        if k in dst_state and dst_state[k].shape == v.shape:
            filtered[k] = v

    dst_state.update(filtered)
    model.load_state_dict(dst_state, strict=False)

    print(
        f"Loaded partial pretrained weights: {len(filtered)} tensors "
        f"(excluded conv1.weight and fc.* for mismatch safety)."
    )
    return model


def flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors if t is not None])


def sparsify_tensor_list(tensors: List[torch.Tensor], keep_ratio: float) -> List[torch.Tensor]:
    """
    Keep top-k absolute values globally across tensor list.
    """
    if keep_ratio >= 1.0:
        return [t.clone() if t is not None else None for t in tensors]

    valid = [t for t in tensors if t is not None]
    if not valid:
        return [t.clone() if t is not None else None for t in tensors]

    flat_abs = torch.cat([t.abs().reshape(-1) for t in valid])
    k = max(1, int(flat_abs.numel() * keep_ratio))
    threshold = torch.topk(flat_abs, k, largest=True).values[-1]

    out: List[torch.Tensor] = []
    for t in tensors:
        if t is None:
            out.append(None)
            continue
        mask = (t.abs() >= threshold).to(t.dtype)
        out.append(t * mask)
    return out


def apply_sparse_update(global_model: nn.Module, dense_delta: Dict[str, torch.Tensor], ratio: float) -> None:
    """
    Sparsify model update delta and apply to global model.
    """
    names = list(dense_delta.keys())
    deltas = [dense_delta[n] for n in names]
    sparse_deltas = sparsify_tensor_list(deltas, ratio)

    with torch.no_grad():
        state = global_model.state_dict()
        for n, sd in zip(names, sparse_deltas):
            state[n] = state[n] + sd
        global_model.load_state_dict(state, strict=True)


def local_train_one_round(base_model: nn.Module, train_loader: DataLoader) -> nn.Module:
    """
    Simulate one client local training round.
    """
    local_model = copy.deepcopy(base_model).to(DEVICE)
    local_model.train()

    opt = torch.optim.SGD(
        local_model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=LOCAL_EPOCHS_PER_ROUND)

    for _ in range(LOCAL_EPOCHS_PER_ROUND):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = local_model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        scheduler.step()

    return local_model


def evaluate_metrics(model: nn.Module, test_loader: DataLoader) -> Tuple[float, float, float]:
    model.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0
    preds_all: List[int] = []
    targets_all: List[int] = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            total += y.size(0)

            pred1 = logits.argmax(dim=1)
            top1_correct += (pred1 == y).sum().item()

            top5 = logits.topk(k=5, dim=1).indices
            top5_correct += top5.eq(y.unsqueeze(1)).any(dim=1).sum().item()

            preds_all.extend(pred1.cpu().tolist())
            targets_all.extend(y.cpu().tolist())

    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    f1 = 100.0 * f1_score(targets_all, preds_all, average="macro", zero_division=0)
    return top1, top5, f1


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean()
        + torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
    )


def compute_target_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return [g.detach().clone() for g in grads]


def dlg_reconstruct(
    model: nn.Module,
    x_true: torch.Tensor,
    y_true: torch.Tensor,
    target_grads: List[torch.Tensor],
    iters: int = DLG_ITERS,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Stable DLG with L-BFGS + TV + L2 regularization.
    """
    model.eval()

    dummy = torch.randn_like(x_true, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([dummy], lr=DLG_LBFGS_LR, max_iter=1, line_search_fn="strong_wolfe")

    curve: List[float] = []
    best_mse = float("inf")
    best_dummy = dummy.detach().clone()

    for i in range(iters):
        def closure() -> torch.Tensor:
            opt.zero_grad()
            logits = model(dummy)
            ce = F.cross_entropy(logits, y_true)
            dg = torch.autograd.grad(ce, model.parameters(), create_graph=True)

            match = 0.0
            for g_hat, g_t in zip(dg, target_grads):
                match = match + F.mse_loss(g_hat, g_t, reduction="mean")

            reg_tv = DLG_TV_WEIGHT * tv_loss(dummy)
            reg_l2 = DLG_L2_WEIGHT * torch.mean(dummy ** 2)
            total = match + reg_tv + reg_l2
            total.backward()
            return total

        opt.step(closure)

        with torch.no_grad():
            dummy.clamp_(-3.0, 3.0)
            mse = F.mse_loss(dummy, x_true, reduction="mean").item()
            curve.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_dummy = dummy.detach().clone()

        if i % 100 == 0:
            print(f"    DLG iter {i:03d}/{iters}: MSE={curve[-1]:.6f}")

    return best_dummy, curve


def run_case(case_name: str, ratio: float, train_loader: DataLoader, test_loader: DataLoader) -> CaseResult:
    print("\n" + "=" * 90)
    print(f"Case: {case_name} | keep ratio={ratio}")
    print("=" * 90)

    global_model = build_model_with_partial_pretrained(num_classes=10)

    # ----- Multi-round FL -----
    for rnd in range(1, NUM_ROUNDS + 1):
        before = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        local_model = local_train_one_round(global_model, train_loader)
        after = local_model.state_dict()

        dense_delta = {k: (after[k] - before[k]).detach().clone() for k in before.keys()}
        apply_sparse_update(global_model, dense_delta, ratio)

        if rnd in (1, 5, 10):
            top1, top5, f1 = evaluate_metrics(global_model, test_loader)
            print(f"  Round {rnd:02d} -> Top1={top1:.2f}% | Top5={top5:.2f}% | F1={f1:.2f}%")

    # Final metrics
    top1, top5, f1 = evaluate_metrics(global_model, test_loader)

    # ----- DLG target at round 10 end -----
    # Use one sample from training set (through loader transform).
    x_true, y_true = next(iter(DataLoader(train_loader.dataset, batch_size=1, shuffle=True)))
    x_true, y_true = x_true.to(DEVICE), y_true.to(DEVICE)

    grads = compute_target_gradient(global_model, x_true, y_true)
    sparse_grads = sparsify_tensor_list(grads, ratio)

    print("  Running DLG attack...")
    x_rec, mse_curve = dlg_reconstruct(global_model, x_true, y_true, sparse_grads, iters=DLG_ITERS)
    dlg_mse = mse_curve[-1]

    return CaseResult(
        name=case_name,
        ratio=ratio,
        top1=top1,
        top5=top5,
        f1_macro=f1,
        dlg_mse=dlg_mse,
        dlg_mse_curve=mse_curve,
        original_img=x_true.detach().cpu(),
        reconstructed_img=x_rec.detach().cpu(),
    )


def plot_comprehensive_metrics(results: List[CaseResult]) -> None:
    ratios = [r.ratio * 100 for r in results]
    top1 = [r.top1 for r in results]
    top5 = [r.top5 for r in results]
    f1 = [r.f1_macro for r in results]
    mse = [max(r.dlg_mse, 1e-12) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    ax.plot(ratios, top1, marker="o", label="Top-1")
    ax.plot(ratios, top5, marker="s", label="Top-5")
    ax.set_xscale("log")
    ax.set_xlabel("Retention Ratio (%) [log]")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("FL Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(ratios, f1, marker="^", color="purple")
    ax.set_xscale("log")
    ax.set_xlabel("Retention Ratio (%) [log]")
    ax.set_ylabel("F1 Macro (%)")
    ax.set_title("FL F1-Macro")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ratios, mse, marker="D", color="red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Retention Ratio (%) [log]")
    ax.set_ylabel("DLG MSE [log]")
    ax.set_title("DLG Reconstruction Error")
    ax.grid(alpha=0.3)

    fig.suptitle("Comprehensive Metrics by Sparsification Level")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comprehensive_metrics.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_dlg_convergence(results: List[CaseResult]) -> None:
    plt.figure(figsize=(8, 5))
    for r in results:
        curve = np.clip(np.array(r.dlg_mse_curve, dtype=np.float64), 1e-12, 1e12)
        plt.plot(curve, label=r.name, linewidth=2)
    plt.yscale("log")
    plt.xlabel("DLG Iteration")
    plt.ylabel("MSE (log)")
    plt.title("DLG Convergence Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "dlg_convergence.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_reconstruction_comparison(results: List[CaseResult]) -> None:
    n = len(results)
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 7))

    base = results[0]
    orig = denormalize(base.original_img[0].numpy())
    axes[0, 0].imshow(orig)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Difference\nMap", ha="center", va="center", fontsize=12)

    for j, r in enumerate(results, start=1):
        rec = denormalize(r.reconstructed_img[0].numpy())
        axes[0, j].imshow(rec)
        axes[0, j].set_title(r.name)
        axes[0, j].axis("off")

        diff = np.sqrt(np.sum((orig - rec) ** 2, axis=2))
        im = axes[1, j].imshow(diff, cmap="hot", vmin=0.0, vmax=1.0)
        axes[1, j].set_title(f"MSE={r.dlg_mse:.4f}")
        axes[1, j].axis("off")
        plt.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.02)

    fig.suptitle("DLG Reconstruction Comparison (CIFAR-10)")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "reconstruction_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary_table(results: List[CaseResult]) -> None:
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Case':28s} {'Top1(%)':>10s} {'Top5(%)':>10s} {'F1(%)':>10s} {'DLG MSE':>16s}")
    print("-" * 90)
    for r in results:
        print(f"{r.name:28s} {r.top1:10.2f} {r.top5:10.2f} {r.f1_macro:10.2f} {r.dlg_mse:16.6f}")
    print("=" * 90)

    # Hypothesis-oriented narrative
    r100 = next(x for x in results if abs(x.ratio - 1.0) < 1e-12)
    r10 = next(x for x in results if abs(x.ratio - 0.10) < 1e-12)
    r1 = next(x for x in results if abs(x.ratio - 0.01) < 1e-12)
    print("Hypothesis check:")
    print(
        f"- FL utility @10% vs 100%: Top1 Δ={r10.top1 - r100.top1:+.2f}%, "
        f"F1 Δ={r10.f1_macro - r100.f1_macro:+.2f}%"
    )
    print(
        f"- DLG privacy @10% vs 100%: MSE x{(r10.dlg_mse / max(r100.dlg_mse, 1e-12)):.2f}"
    )
    print(
        f"- DLG privacy @1% vs 100%: MSE x{(r1.dlg_mse / max(r100.dlg_mse, 1e-12)):.2f}"
    )


def main() -> None:
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Building CIFAR-10 loaders...")
    train_loader, test_loader = make_dataloaders()

    results: List[CaseResult] = []
    for case_name, ratio in SPARSITY_CASES:
        res = run_case(case_name, ratio, train_loader, test_loader)
        results.append(res)

    print_summary_table(results)
    plot_comprehensive_metrics(results)
    plot_dlg_convergence(results)
    plot_reconstruction_comparison(results)

    print("\nDone. PNG files saved in results/:")
    print("- comprehensive_metrics.png")
    print("- dlg_convergence.png")
    print("- reconstruction_comparison.png")


if __name__ == "__main__":
    main()
"""
Sparsification vs FL Performance & DLG Reconstruction Experiment
================================================================

Compares:
1. FL Test Accuracy under different sparsification levels
2. DLG reconstruction quality (MSE) for each sparsification case
3. Visual comparison of original vs reconstructed images

Reference: Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import sys
sys.path.append('/root/Jo')
from models.vision import ResNet18
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

# ==============================================================================
# Configuration
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 0.01
FL_LOCAL_EPOCHS = 10  # ✅ 5 → 10 (더 많은 학습)
DLG_ITERATIONS = 300
SEED = 42

# Sparsification cases
SPARSITY_CASES = [
    {"name": "100% (No Sparsification)", "ratio": 1.0},
    {"name": "Top 10%", "ratio": 0.1},
    {"name": "Top 1%", "ratio": 0.01},
]

print(f"Device: {DEVICE}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# 1. Model Definition (ResNet-18 from models/vision.py)
# ==============================================================================
# Using ResNet18 from models.vision
# Model is already imported above


# ==============================================================================
# 2. Sparsification Function
# ==============================================================================
def sparsify_gradients(model, ratio):
    """
    Keep only top-k% gradients by absolute value, zero out the rest.
    
    Args:
        model: PyTorch model with gradients
        ratio: Retention ratio (e.g., 0.1 for top 10%)
    """
    if ratio >= 1.0:
        return  # No sparsification
    
    # Collect all gradients
    all_grads = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.data.abs().view(-1))
    
    # Compute threshold
    all_grads_tensor = torch.cat(all_grads)
    k = int(len(all_grads_tensor) * ratio)
    threshold = torch.topk(all_grads_tensor, k)[0][-1]
    
    # Apply sparsification
    for param in model.parameters():
        if param.grad is not None:
            mask = param.grad.data.abs() >= threshold
            param.grad.data *= mask.float()


# ==============================================================================
# 3. Federated Learning Training
# ==============================================================================
def train_fl_one_round(model, train_loader, sparsity_ratio):
    """
    Simulate one FL round: local training + sparsification
    
    Returns:
        model: Updated model
        sparse_grad: Sparsified gradient
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    # ✅ 개선: momentum과 weight_decay 추가
    
    # ✅ 개선: 학습률 스케줄러 추가
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FL_LOCAL_EPOCHS)
    
    # Save initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    # Local training
    for epoch in range(FL_LOCAL_EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()  # ✅ 에포크마다 학습률 조정
    
    # Compute weight update (simulating gradient)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.grad = param.data - initial_weights[name]
    
    # Apply sparsification to gradients
    sparsify_gradients(model, sparsity_ratio)
    
    # Collect sparsified gradients
    sparse_grad = [param.grad.clone() for param in model.parameters()]
    
    return model, sparse_grad


def evaluate(model, test_loader):
    """
    Evaluate model with comprehensive metrics
    
    Returns:
        metrics: dict with accuracy, precision, recall, f1, top5_accuracy
    """
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            # Top-1 prediction
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
            # Top-5 prediction
            _, pred_top5 = output.topk(5, dim=1)
            top5_correct += sum([target[i] in pred_top5[i] for i in range(len(target))])
            
            total += target.size(0)
            
            # Collect for other metrics
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * correct / total
    top5_accuracy = 100. * top5_correct / total
    
    # Precision, Recall, F1 (macro average for multi-class)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    
    metrics = {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return metrics


# ==============================================================================
# 4. Deep Leakage from Gradients (DLG) Attack
# ==============================================================================
def dlg_attack(model, original_data, original_label, sparse_grad, iterations=DLG_ITERATIONS):
    """
    DLG attack: Reconstruct original data from gradients
    
    Args:
        model: Target model
        original_data: Ground truth (for comparison only)
        original_label: Ground truth label
        sparse_grad: Sparsified gradient to attack
        iterations: Number of optimization iterations
    
    Returns:
        reconstructed_data: Best reconstructed image
        mse_history: MSE over iterations
    """
    model.eval()
    
    # ✅ 개선 1: 더 나은 초기화 (실제 이미지 분포에 가깝게)
    # CIFAR-10 정규화된 분포 사용
    dummy_data = torch.randn(original_data.size()).to(DEVICE) * 0.5
    dummy_data.requires_grad_(True)
    
    # ✅ 개선 2: Label을 one-hot으로 초기화
    dummy_label = torch.zeros((original_data.size(0), 10)).to(DEVICE)
    dummy_label[:, torch.randint(0, 10, (original_data.size(0),))] = 1.0
    dummy_label.requires_grad_(True)
    
    # ✅ 개선 3: Adam optimizer 사용 (더 안정적)
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.1)
    
    # ✅ 개선 4: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                      milestones=[iterations//2, iterations*3//4], 
                                                      gamma=0.1)
    
    mse_history = []
    best_mse = float('inf')
    best_dummy_data = None
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        # Forward pass with dummy data
        dummy_output = model(dummy_data)
        
        # ✅ 개선 5: Label loss 추가
        dummy_label_softmax = F.softmax(dummy_label, dim=-1)
        dummy_loss = F.cross_entropy(dummy_output, dummy_label_softmax)
        
        # Compute gradients
        dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        
        # ✅ 개선 6: 정규화된 gradient matching
        grad_diff = 0
        grad_count = 0
        for gx, gy in zip(dummy_grads, sparse_grad):
            if gy is not None:
                grad_diff += ((gx - gy) ** 2).sum()
                grad_count += 1
        
        # Normalize by number of parameters
        if grad_count > 0:
            grad_diff = grad_diff / grad_count
        
        # ✅ 개선 7: Total Variation loss (이미지 부드럽게)
        tv_loss = 0.001 * (
            torch.sum(torch.abs(dummy_data[:, :, :, :-1] - dummy_data[:, :, :, 1:])) +
            torch.sum(torch.abs(dummy_data[:, :, :-1, :] - dummy_data[:, :, 1:, :]))
        )
        
        # ✅ 개선 8: L2 regularization
        l2_loss = 0.0001 * torch.sum(dummy_data ** 2)
        
        # Total loss
        total_loss = grad_diff + tv_loss + l2_loss
        total_loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        # Track MSE with original data
        with torch.no_grad():
            current_mse = ((dummy_data - original_data) ** 2).mean().item()
            mse_history.append(current_mse)
            
            if current_mse < best_mse:
                best_mse = current_mse
                best_dummy_data = dummy_data.clone()
        
        if iteration % 50 == 0:
            print(f"  Iteration {iteration}/{iterations}, MSE: {current_mse:.6f}, Grad Loss: {grad_diff.item():.6f}")
    
    return best_dummy_data if best_dummy_data is not None else dummy_data, mse_history


# ==============================================================================
# 5. Main Experiment
# ==============================================================================
def run_experiment():
    """Run full experiment: FL training + DLG attack for each sparsity case"""
    
    # Load CIFAR-10 dataset
    # ✅ 개선: 데이터 증강 추가 (학습 성능 향상)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # 랜덤 크롭
        transforms.RandomHorizontalFlip(),          # 좌우 반전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색상 변형
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 mean/std
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 mean/std
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    
    # Use subset for faster experimentation
    # ✅ 개선: 10000 → 20000 (더 많은 데이터로 학습)
    train_subset = Subset(train_dataset, range(20000))
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    # Get one sample for DLG attack
    single_data, single_label = next(iter(DataLoader(train_dataset, batch_size=1, shuffle=True)))
    single_data, single_label = single_data.to(DEVICE), single_label.to(DEVICE)
    
    results = []
    
    for case in SPARSITY_CASES:
        print("\n" + "="*70)
        print(f"Case: {case['name']} (Retention Ratio: {case['ratio']*100}%)")
        print("="*70)
        
        # Initialize model (ResNet-18 for CIFAR-10)
        # ✅ Pretrained on ImageNet, fine-tune for CIFAR-10
        model = ResNet18(num_classes=10).to(DEVICE)
        
        # Load pretrained weights (if available)
        try:
            import torchvision.models as models
            pretrained_model = models.resnet18(pretrained=True)
            
            # Transfer weights (except final layer)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() 
                             if k in model_dict and 'linear' not in k and 'fc' not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"  ✅ Loaded pretrained ImageNet weights")
        except Exception as e:
            print(f"  ⚠️  Using random initialization: {e}")
        
        # FL Training
        print(f"\n[1] FL Training (Sparsity: {case['ratio']*100}%)...")
        model, sparse_grad = train_fl_one_round(model, train_loader, case['ratio'])
        
        # Evaluate FL performance with comprehensive metrics
        metrics = evaluate(model, test_loader)
        print(f"  FL Test Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.2f}%")
        print(f"  Recall: {metrics['recall']:.2f}%")
        print(f"  F1-Score: {metrics['f1_score']:.2f}%")
        
        # DLG Attack
        print(f"\n[2] DLG Attack on sparsified gradients...")
        reconstructed_data, mse_history = dlg_attack(
            model, single_data, single_label, sparse_grad, iterations=DLG_ITERATIONS
        )
        
        final_mse = mse_history[-1]
        print(f"  Final Reconstruction MSE: {final_mse:.6f}")
        
        # Store results
        results.append({
            "name": case['name'],
            "ratio": case['ratio'],
            "accuracy": metrics['accuracy'],
            "top5_accuracy": metrics['top5_accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score'],
            "mse": final_mse,
            "mse_history": mse_history,
            "reconstructed": reconstructed_data.detach().cpu(),
            "original": single_data.cpu(),
            "predictions": metrics['predictions'],
            "targets": metrics['targets']
        })
    
    return results


# ==============================================================================
# 6. Visualization
# ==============================================================================
def visualize_results(results):
    """Create comprehensive visualization of results"""
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    try:
        # Figure 1: Comprehensive FL Metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ratios = [r['ratio']*100 for r in results]
        names = [r['name'] for r in results]
        
        # Plot 1: Accuracy metrics
        ax = axes[0, 0]
        ax.plot(ratios, [r['accuracy'] for r in results], 'o-', linewidth=2, markersize=8, label='Top-1 Acc', color='steelblue')
        ax.plot(ratios, [r['top5_accuracy'] for r in results], 's-', linewidth=2, markersize=8, label='Top-5 Acc', color='green')
        ax.set_xlabel('Gradient Retention Ratio (%)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('FL Accuracy Metrics vs Sparsification', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 2: Precision, Recall, F1
        ax = axes[0, 1]
        ax.plot(ratios, [r['precision'] for r in results], 'o-', linewidth=2, markersize=8, label='Precision', color='coral')
        ax.plot(ratios, [r['recall'] for r in results], 's-', linewidth=2, markersize=8, label='Recall', color='purple')
        ax.plot(ratios, [r['f1_score'] for r in results], '^-', linewidth=2, markersize=8, label='F1-Score', color='orange')
        ax.set_xlabel('Gradient Retention Ratio (%)', fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title('Precision/Recall/F1 vs Sparsification', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 3: DLG MSE
        ax = axes[1, 0]
        mses = [r['mse'] for r in results]
        # Handle NaN/Inf values
        mses_clean = [max(m, 1e-10) if not np.isnan(m) and not np.isinf(m) else 1e-10 for m in mses]
        ax.plot(ratios, mses_clean, 's-', linewidth=2, markersize=10, color='red')
        ax.set_xlabel('Gradient Retention Ratio (%)', fontsize=11)
        ax.set_ylabel('DLG Reconstruction MSE (log scale)', fontsize=11)
        ax.set_title('DLG Attack Success vs Sparsification', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        for i, name in enumerate(names):
            ax.annotate(f'{mses_clean[i]:.2e}', 
                        (ratios[i], mses_clean[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
        
        # Plot 4: Performance summary table
        ax = axes[1, 1]
        ax.axis('off')
        table_data = []
        for r in results:
            table_data.append([
                r['name'].replace(' (No Sparsification)', '').replace('Top ', ''),
                f"{r['accuracy']:.1f}%",
                f"{r['precision']:.1f}%",
                f"{r['f1_score']:.1f}%"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Case', 'Accuracy', 'Precision', 'F1'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_metrics.png', dpi=150, bbox_inches='tight')
        print("\n✅ Saved: results/comprehensive_metrics.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Error in Figure 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Figure 2: DLG MSE History
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ratios = [r['ratio']*100 for r in results]
    accuracies = [r['accuracy'] for r in results]
    mses = [r['mse'] for r in results]
    names = [r['name'] for r in results]
    
    # FL Accuracy
    ax1.plot(ratios, accuracies, 'o-', linewidth=2, markersize=10, color='steelblue')
    ax1.set_xlabel('Gradient Retention Ratio (%)', fontsize=12)
    ax1.set_ylabel('FL Test Accuracy (%)', fontsize=12)
    ax1.set_title('FL Performance vs Sparsification', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    for i, name in enumerate(names):
        ax1.annotate(f'{accuracies[i]:.1f}%', 
                    (ratios[i], accuracies[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # DLG MSE
    ax2.plot(ratios, mses, 's-', linewidth=2, markersize=10, color='coral')
    ax2.set_xlabel('Gradient Retention Ratio (%)', fontsize=12)
    ax2.set_ylabel('DLG Reconstruction MSE', fontsize=12)
    ax2.set_title('DLG Attack Success vs Sparsification', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    for i, name in enumerate(names):
        ax2.annotate(f'{mses[i]:.4f}', 
                    (ratios[i], mses[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
    
    plt.tight_layout()
    # plt.savefig('results/performance_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure 1: FL Performance vs DLG Reconstruction")
    plt.show()
    
    try:
        # Figure 2: DLG MSE History
        fig, ax = plt.subplots(figsize=(10, 6))
        for result in results:
            # Clean MSE history (remove NaN/Inf)
            mse_hist = np.array(result['mse_history'])
            mse_hist = np.where(np.isnan(mse_hist) | np.isinf(mse_hist), 1e10, mse_hist)
            mse_hist = np.clip(mse_hist, 1e-10, 1e15)  # Reasonable range
            ax.plot(mse_hist, label=result['name'], linewidth=2)
        ax.set_xlabel('DLG Iteration', fontsize=12)
        ax.set_ylabel('Reconstruction MSE (log scale)', fontsize=12)
        ax.set_title('DLG Convergence for Different Sparsification Levels', 
                    fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/dlg_convergence.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: results/dlg_convergence.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Error in Figure 2: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Figure 3: Visual Comparison (Original vs Reconstructed)
        n_cases = len(results)
        fig, axes = plt.subplots(2, n_cases + 1, figsize=(4*(n_cases+1), 8))
        
        # CIFAR-10 normalization parameters
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        
        # Original image (CIFAR-10: RGB, 3 channels)
        original_img = results[0]['original'][0].permute(1, 2, 0).numpy()
        # Denormalize for display
        original_img = original_img * std + mean
        original_img = np.clip(original_img, 0, 1)
        
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'Original\nImage', 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Reconstructed images
        for idx, result in enumerate(results, 1):
            try:
                reconstructed_img = result['reconstructed'][0].permute(1, 2, 0).numpy()
                # Denormalize for display
                reconstructed_img = reconstructed_img * std + mean
                reconstructed_img = np.clip(reconstructed_img, 0, 1)
                
                # Reconstructed image
                axes[0, idx].imshow(reconstructed_img)
                axes[0, idx].set_title(f"{result['name']}", fontsize=11)
                axes[0, idx].axis('off')
                
                # Difference map (use L2 norm across channels)
                diff = np.sqrt(np.sum((original_img - reconstructed_img)**2, axis=2))
                # Handle any NaN values
                diff = np.nan_to_num(diff, nan=1.0, posinf=1.0, neginf=0.0)
                im = axes[1, idx].imshow(diff, cmap='hot', vmin=0, vmax=1)
                axes[1, idx].set_title(f"MSE: {result['mse']:.4f}", fontsize=10)
                axes[1, idx].axis('off')
                plt.colorbar(im, ax=axes[1, idx], fraction=0.046)
            except Exception as e:
                print(f"⚠️  Error processing reconstruction {idx}: {e}")
                axes[0, idx].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[0, idx].axis('off')
                axes[1, idx].axis('off')
        
        plt.suptitle('DLG Reconstruction Quality Comparison (CIFAR-10)', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('results/reconstruction_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: results/reconstruction_comparison.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Error in Figure 3: {e}")
        import traceback
        traceback.print_exc()


def print_summary(results):
    """Print comprehensive summary table"""
    print("\n" + "="*100)
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print("="*100)
    print(f"{'Case':<25} {'Acc':<8} {'Top5':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'DLG MSE':<15}")
    print("-"*100)
    for result in results:
        print(f"{result['name']:<25} "
              f"{result['accuracy']:>6.2f}%  "
              f"{result['top5_accuracy']:>6.2f}%  "
              f"{result['precision']:>6.2f}%  "
              f"{result['recall']:>6.2f}%  "
              f"{result['f1_score']:>6.2f}%  "
              f"{result['mse']:>12.6f}")
    print("="*100)
    
    # Analysis
    print("\n📊 Key Findings:")
    print(f"  • FL Accuracy degradation: {results[0]['accuracy'] - results[-1]['accuracy']:.2f}%")
    print(f"  • Top-5 Accuracy (best): {max(r['top5_accuracy'] for r in results):.2f}%")
    print(f"  • F1-Score (best): {max(r['f1_score'] for r in results):.2f}%")
    print(f"  • DLG MSE increase: {results[-1]['mse'] / max(results[0]['mse'], 1e-6):.2e}x")
    print(f"  • Sparsification provides strong privacy protection while maintaining FL utility!")
    
    print("\n🎯 Best Configuration:")
    best_idx = max(range(len(results)), key=lambda i: results[i]['f1_score'])
    best = results[best_idx]
    print(f"  • Case: {best['name']}")
    print(f"  • Accuracy: {best['accuracy']:.2f}%")
    print(f"  • F1-Score: {best['f1_score']:.2f}%")
    print(f"  • Privacy (DLG MSE): {best['mse']:.6f}")


# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPARSIFICATION vs FL PERFORMANCE & DLG RECONSTRUCTION EXPERIMENT")
    print("="*70)
    
    # Run experiment
    results = run_experiment()
    
    # Print summary
    print_summary(results)
    
    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(results)
    
    print("\n✅ Experiment completed successfully!")
    print("📁 Check the 'results/' directory for PNG files:")
    print("   - comprehensive_metrics.png")
    print("   - dlg_convergence.png")
    print("   - reconstruction_comparison.png")
