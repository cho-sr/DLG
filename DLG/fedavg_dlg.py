import argparse
import copy
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.vision import LeNet, weights_init
from utils import cross_entropy_for_onehot, label_to_onehot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_clients(num_items: int, num_clients: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_items)
    return [x.tolist() for x in np.array_split(perm, num_clients)]


def fedavg(local_states, local_sizes):
    total = float(sum(local_sizes))
    out = OrderedDict()
    for k in local_states[0].keys():
        out[k] = sum(state[k] * (sz / total) for state, sz in zip(local_states, local_sizes))
    return out


def train_local(global_model, loader, device, num_classes, lr, epochs):
    model = copy.deepcopy(global_model).to(device)
    model.train()
    optimizer_local = torch.optim.SGD(model.parameters(), lr=lr)

    loss_sum = 0.0
    step_count = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_onehot = label_to_onehot(y, num_classes=num_classes)

            optimizer_local.zero_grad()
            pred = model(x)
            loss = cross_entropy_for_onehot(pred, y_onehot)
            loss.backward()
            optimizer_local.step()

            loss_sum += float(loss.item())
            step_count += 1

    avg_loss = loss_sum / max(1, step_count)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, avg_loss


def sample_gradient(model, x, y, device, num_classes):
    net = copy.deepcopy(model).to(device)
    net.eval()

    x, y = x.to(device), y.to(device)
    y_onehot = label_to_onehot(y, num_classes=num_classes)

    pred = net(x)
    loss = cross_entropy_for_onehot(pred, y_onehot)
    dy_dx = torch.autograd.grad(loss, net.parameters())
    return [g.detach().clone() for g in dy_dx]


def sparsify_gradients_topk(gradients, sparsity_ratio):
    if sparsity_ratio <= 0.0 or sparsity_ratio > 1.0:
        raise ValueError("sparsity_ratio must be in (0.0, 1.0].")

    flat = torch.cat([g.reshape(-1) for g in gradients])
    total = flat.numel()

    if sparsity_ratio >= 1.0:
        kept = total
        stats = {
            "kept": kept,
            "total": total,
            "retention_ratio": kept / float(total),
        }
        return [g.clone() for g in gradients], stats

    k = max(1, int(total * sparsity_ratio))
    _, topk_idx = torch.topk(flat.abs(), k, largest=True, sorted=False)

    mask = torch.zeros_like(flat)
    mask[topk_idx] = 1.0
    sparse_flat = flat * mask

    sparse_grads = []
    offset = 0
    for g in gradients:
        n = g.numel()
        sparse_grads.append(sparse_flat[offset : offset + n].view_as(g))
        offset += n

    stats = {
        "kept": k,
        "total": total,
        "retention_ratio": k / float(total),
    }
    return sparse_grads, stats


def compute_mse_psnr(gt_tensor, recon_tensor):
    mse = float(torch.mean((gt_tensor - recon_tensor) ** 2).item())
    psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-12)) if mse > 0 else 100.0
    return mse, float(psnr)


def compute_ssim(gt_tensor, recon_tensor):
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except Exception:
        print("Warning: skimage not available, SSIM is skipped.")
        return None

    gt_np = gt_tensor[0].detach().cpu().clamp(0, 1).numpy()
    recon_np = recon_tensor[0].detach().cpu().clamp(0, 1).numpy()
    gt_hwc = np.transpose(gt_np, (1, 2, 0))
    recon_hwc = np.transpose(recon_np, (1, 2, 0))

    try:
        return float(ssim_fn(gt_hwc, recon_hwc, data_range=1.0, channel_axis=2))
    except TypeError:
        return float(ssim_fn(gt_hwc, recon_hwc, data_range=1.0, multichannel=True))


def compute_lpips(gt_tensor, recon_tensor, device):
    try:
        import lpips
    except Exception:
        print("Warning: lpips package not available, LPIPS is skipped.")
        return None

    try:
        loss_fn = lpips.LPIPS(net="alex").to(device)
        gt = gt_tensor.to(device).clamp(0, 1) * 2.0 - 1.0
        recon = recon_tensor.to(device).clamp(0, 1) * 2.0 - 1.0
        with torch.no_grad():
            val = loss_fn(gt, recon)
        return float(val.item())
    except Exception as exc:
        print(f"Warning: LPIPS compute failed, skipped. ({exc})")
        return None


def compute_attack_metrics(attack_net, gt_tensor, recon_tensor, true_label, dummy_class, device):
    recon_on_device = recon_tensor.to(device)
    with torch.no_grad():
        logits = attack_net(recon_on_device)
        probs = F.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        top1_conf = float(torch.max(probs, dim=1).values.item())
        true_conf = float(probs[0, true_label].item())

    mse, psnr = compute_mse_psnr(gt_tensor, recon_tensor)
    l1_mae = float(torch.mean(torch.abs(gt_tensor - recon_tensor)).item())
    ssim = compute_ssim(gt_tensor, recon_tensor)
    lpips_score = compute_lpips(gt_tensor, recon_tensor, device)

    metrics = {
        "label_match": float(pred_class == true_label),
        "mse": mse,
        "psnr": psnr,
        "l1_mae": l1_mae,
        "top1_conf": top1_conf,
        "true_label_conf": true_conf,
        "recon_pred_class": float(pred_class),
        "dummy_class": float(dummy_class),
    }

    if ssim is not None:
        metrics["ssim"] = ssim
    if lpips_score is not None:
        metrics["lpips"] = lpips_score

    return metrics, pred_class


def save_visualizations(progress_snapshots, loss_history, metric_history):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib not available, visualization files are skipped.")
        return []

    saved_files = []

    # Progress grid
    if progress_snapshots:
        num = len(progress_snapshots)
        cols = min(6, num)
        rows = int(np.ceil(num / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)
        for idx, (it, img_tensor) in enumerate(progress_snapshots):
            img = img_tensor[0].detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
            axes[idx].imshow(img)
            axes[idx].set_title(f"iter {it}")
            axes[idx].axis("off")
        for idx in range(num, len(axes)):
            axes[idx].axis("off")
        fig.tight_layout()
        fig.savefig("dlg_progress_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append("dlg_progress_grid.png")

    # Loss curve
    if loss_history:
        xs = [it for it, _ in loss_history]
        ys = [loss for _, loss in loss_history]
        fig = plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o", linewidth=1.8)
        plt.xlabel("Iteration")
        plt.ylabel("Gradient Matching Loss")
        plt.title("DLG Loss Curve")
        plt.grid(alpha=0.3)
        if all(v > 0 for v in ys):
            plt.yscale("log")
        fig.tight_layout()
        fig.savefig("dlg_loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append("dlg_loss_curve.png")

    # MSE/PSNR curve (optional)
    if metric_history:
        xs = [it for it, _, _ in metric_history]
        mses = [m for _, m, _ in metric_history]
        psnrs = [p for _, _, p in metric_history]

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        ax1.plot(xs, mses, color="tab:red", marker="o", label="MSE")
        ax2.plot(xs, psnrs, color="tab:blue", marker="s", label="PSNR")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("MSE", color="tab:red")
        ax2.set_ylabel("PSNR (dB)", color="tab:blue")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("dlg_metrics_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append("dlg_metrics_curve.png")

    return saved_files


def dlg_attack(model, target_gradients, gt_shape, label_shape, iters, device, gt_data=None, log_interval=10):
    net = copy.deepcopy(model).to(device)
    net.eval()
    criterion = cross_entropy_for_onehot

    dummy_data = torch.randn(gt_shape).to(device).requires_grad_(True)
    dummy_label = torch.randn(label_shape).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    progress_snapshots = []
    loss_history = []
    metric_history = []

    for it in range(iters):

        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, target_gradients):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)

        if it % log_interval == 0:
            with torch.enable_grad():
                dummy_pred = net(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=False)
                current_loss = 0
                for gx, gy in zip(dummy_dy_dx, target_gradients):
                    current_loss += ((gx - gy) ** 2).sum()
                current_loss_val = float(current_loss.item())

            print(f"  DLG iter {it:3d} | grad loss {current_loss_val:.6f}")
            loss_history.append((it, current_loss_val))
            progress_snapshots.append((it, dummy_data.detach().cpu().clone()))

            if gt_data is not None:
                mse_it, psnr_it = compute_mse_psnr(gt_data, dummy_data.detach().cpu())
                metric_history.append((it, mse_it, psnr_it))

    with torch.no_grad():
        pred_class = int(torch.argmax(net(dummy_data), dim=1).item())
        dummy_class = int(torch.argmax(F.softmax(dummy_label, dim=-1), dim=1).item())

    return dummy_data.detach().cpu(), pred_class, dummy_class, progress_snapshots, loss_history, metric_history


def parse_args():
    p = argparse.ArgumentParser("FedAvg + DLG (main-style)")
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--local_lr", type=float, default=0.05)
    p.add_argument("--attack_round", type=int, default=0)
    p.add_argument("--victim_client", type=int, default=0)
    p.add_argument("--victim_pos", type=int, default=0)
    p.add_argument("--dlg_iters", type=int, default=300)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument(
        "--sparsity",
        type=float,
        default=1.0,
        help="Gradient retention ratio (1.0=100%, 0.1=10%, 0.01=1%).",
    )
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f"Device: {device}")
    dst = datasets.CIFAR10("~/.torch", train=True, download=True, transform=transforms.ToTensor())
    client_ids = split_clients(len(dst), args.num_clients, args.seed)

    print("\n[FedAvg Config]")
    print(f"  clients={args.num_clients}, rounds={args.rounds}, frac={args.frac}")
    print(f"  local_epochs={args.local_epochs}, batch_size={args.batch_size}, local_lr={args.local_lr}")
    print(f"  sparsity={args.sparsity}")

    net = LeNet()
    torch.manual_seed(args.seed)
    net.apply(weights_init)
    net = net.to(device)
    num_classes = net.fc[-1].out_features

    captured_grad = None
    captured_gt = None
    captured_label = None
    captured_model_state = None

    for rnd in range(args.rounds):
        num_sel = max(1, int(args.frac * args.num_clients))
        selected = random.sample(range(args.num_clients), num_sel)
        print(f"\n[Round {rnd}] selected={selected}")

        local_states, local_sizes, local_losses = [], [], []

        for cid in selected:
            ids = client_ids[cid]
            loader = DataLoader(Subset(dst, ids), batch_size=args.batch_size, shuffle=True)

            if rnd == args.attack_round and cid == args.victim_client and captured_grad is None:
                victim_idx = ids[args.victim_pos]
                x, y = dst[victim_idx]
                x = x.unsqueeze(0)
                y = torch.tensor([y], dtype=torch.long)

                captured_gt = x.detach().cpu().clone()
                captured_label = int(y.item())
                captured_model_state = copy.deepcopy(net.state_dict())
                captured_grad = sample_gradient(net, x, y, device, num_classes)
                captured_grad, sparse_stats = sparsify_gradients_topk(captured_grad, args.sparsity)
                print(f"  [Attack Capture] client={cid}, label={captured_label}, idx={victim_idx}")
                print(
                    "  [Sparsification] kept="
                    f"{sparse_stats['kept']}/{sparse_stats['total']} "
                    f"({sparse_stats['retention_ratio'] * 100:.2f}%)"
                )

            local_state, local_loss = train_local(net, loader, device, num_classes, args.local_lr, args.local_epochs)
            local_states.append(local_state)
            local_sizes.append(len(ids))
            local_losses.append(local_loss)
            print(f"  client={cid:2d} | samples={len(ids):5d} | local_loss={local_loss:.6f}")

        round_loss = sum(l * n for l, n in zip(local_losses, local_sizes)) / float(sum(local_sizes))
        net.load_state_dict(fedavg(local_states, local_sizes))
        print(f"  [Round {rnd} Summary] weighted_local_loss={round_loss:.6f}, total_samples={sum(local_sizes)}")

    if captured_grad is None:
        raise RuntimeError("No gradient captured. Check --attack_round/--victim_client.")

    attack_net = LeNet().to(device)
    attack_net.load_state_dict(captured_model_state)
    attack_net.eval()

    print("\nRunning DLG attack...")
    recon, pred_class, dummy_class, progress_snapshots, loss_history, metric_history = dlg_attack(
        model=attack_net,
        target_gradients=[g.to(device) for g in captured_grad],
        gt_shape=tuple(captured_gt.shape),
        label_shape=(1, num_classes),
        iters=args.dlg_iters,
        device=device,
        gt_data=captured_gt,
        log_interval=max(1, args.log_interval),
    )

    metrics, pred_class_from_metric = compute_attack_metrics(
        attack_net=attack_net,
        gt_tensor=captured_gt,
        recon_tensor=recon,
        true_label=captured_label,
        dummy_class=dummy_class,
        device=device,
    )

    save_image(captured_gt, "fedavg_gt.png")
    save_image(recon.clamp(0, 1), "fedavg_dlg_recon.png")
    viz_files = save_visualizations(progress_snapshots, loss_history, metric_history)

    print("\nAttack Summary")
    print(f"  true label: {captured_label}")
    print(f"  reconstructed pred class: {pred_class}")
    print(f"  reconstructed pred class (metric pass): {pred_class_from_metric}")
    print(f"  optimized dummy class: {dummy_class}")

    print("  metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}")

    print("\nExample command:")
    print(
        "  python fedavg_dlg.py --num_clients 5 --rounds 5 --frac 1.0 "
        "--local_epochs 1 --batch_size 64 --local_lr 0.05 "
        "--attack_round 0 --victim_client 0 --victim_pos 0 --dlg_iters 300 --log_interval 10"
    )

    print("\nGenerated files:")
    generated = ["fedavg_gt.png", "fedavg_dlg_recon.png"] + viz_files
    for f in generated:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
