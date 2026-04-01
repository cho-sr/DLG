# -*- coding: utf-8 -*-
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from main_layerwise import BACK_RATIO, FRONT_RATIO, sparsify_gradients_layerwise_topk
from models.vision import LeNet, weights_init
from utils import cross_entropy_for_onehot, label_to_onehot


PRED_GRID_SAMPLES = 25


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
    return [chunk.tolist() for chunk in np.array_split(perm, num_clients)]


def compute_grad_l2(gradients) -> float:
    total = 0.0
    for grad_tensor in gradients:
        total += float(torch.sum(grad_tensor.detach().float() ** 2).item())
    return math.sqrt(total)


def compute_client_fedsgd_gradient(global_model, loader, device, num_classes):
    model = LeNet().to(device)
    model.load_state_dict(global_model.state_dict(), strict=True)
    model.train()
    model.zero_grad(set_to_none=True)

    total_samples = len(loader.dataset)
    total_loss = 0.0
    total_correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        onehot = label_to_onehot(labels, num_classes=num_classes)
        outputs = model(images)
        loss = cross_entropy_for_onehot(outputs, onehot)

        batch_size = labels.size(0)
        scaled_loss = loss * (batch_size / max(1, total_samples))
        scaled_loss.backward()

        total_loss += float(loss.item()) * batch_size
        total_correct += int((outputs.detach().argmax(dim=1) == labels).sum().item())

    gradients = [param.grad.detach().cpu().clone() for param in model.parameters()]
    metrics = {
        "train_loss": total_loss / max(1, total_samples),
        "train_accuracy": 100.0 * total_correct / max(1, total_samples),
        "grad_l2": compute_grad_l2(gradients),
    }
    return gradients, metrics


def aggregate_gradients(gradient_lists, client_sizes):
    total = float(sum(client_sizes))
    aggregated = []
    for grads in zip(*gradient_lists):
        agg = sum(grad * (size / total) for grad, size in zip(grads, client_sizes))
        aggregated.append(agg)
    return aggregated


def apply_fedsgd_update(model, gradients, lr):
    with torch.no_grad():
        for param, grad_tensor in zip(model.parameters(), gradients):
            param.sub_(lr * grad_tensor.to(param.device))


def evaluate_model(model, loader, device, num_classes, collect_predictions=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            onehot = label_to_onehot(labels, num_classes=num_classes)
            outputs = model(images)
            loss = cross_entropy_for_onehot(outputs, onehot)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size

            if collect_predictions:
                preds.append(outputs.argmax(dim=1).detach().cpu())
                targets.append(labels.detach().cpu())

    result = {
        "loss": total_loss / max(1, total_samples),
        "accuracy": 100.0 * total_correct / max(1, total_samples),
    }
    if collect_predictions:
        result["preds"] = torch.cat(preds, dim=0)
        result["targets"] = torch.cat(targets, dim=0)
    return result


def compute_confusion_matrix(targets, preds, num_classes):
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(targets.tolist(), preds.tolist()):
        conf[int(true_label), int(pred_label)] += 1
    return conf


def compute_classification_metrics(confusion_matrix):
    supports = confusion_matrix.sum(axis=1).astype(np.float64)
    pred_counts = confusion_matrix.sum(axis=0).astype(np.float64)
    tp = np.diag(confusion_matrix).astype(np.float64)

    precision = np.divide(tp, pred_counts, out=np.zeros_like(tp), where=pred_counts > 0)
    recall = np.divide(tp, supports, out=np.zeros_like(tp), where=supports > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    total = max(1.0, supports.sum())
    weights = supports / total

    metrics = {
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
        "per_class_accuracy": recall.astype(float).tolist(),
    }
    return metrics


def save_training_curves(round_history, out_path):
    if not round_history:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib not available, training curve is skipped.")
        return

    rounds = [entry["round"] for entry in round_history]
    train_acc = [entry["train_accuracy"] for entry in round_history]
    test_acc = [entry["test_accuracy"] for entry in round_history]
    train_loss = [entry["train_loss"] for entry in round_history]
    test_loss = [entry["test_loss"] for entry in round_history]
    grad_norm = [entry["aggregated_grad_l2"] for entry in round_history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(rounds, train_acc, marker="o", label="Train Acc")
    axes[0].plot(rounds, test_acc, marker="s", label="Test Acc")
    axes[0].set_title("FedSGD Accuracy")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(rounds, train_loss, marker="o", label="Train Loss")
    axes[1].plot(rounds, test_loss, marker="s", label="Test Loss")
    axes[1].set_title("FedSGD Loss")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(rounds, grad_norm, marker="o", color="tab:purple")
    axes[2].set_title("Aggregated Gradient L2")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("L2 Norm")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_image(confusion_matrix, class_names, out_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib not available, confusion matrix image is skipped.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Final Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    max_value = float(confusion_matrix.max()) if confusion_matrix.size else 0.0
    threshold = max_value / 2.0 if max_value > 0 else 0.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            value = int(confusion_matrix[i, j])
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prediction_grid(model, dataset, device, num_samples, seed, out_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib not available, prediction grid is skipped.")
        return

    model.eval()
    rng = np.random.default_rng(seed)
    sample_count = min(num_samples, len(dataset))
    indices = rng.choice(len(dataset), size=sample_count, replace=False)

    cols = int(math.ceil(math.sqrt(sample_count)))
    rows = int(math.ceil(sample_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    with torch.no_grad():
        for plot_idx, data_idx in enumerate(indices):
            image, label = dataset[int(data_idx)]
            logits = model(image.unsqueeze(0).to(device))
            pred = int(logits.argmax(dim=1).item())

            img_np = image.detach().cpu().numpy().transpose(1, 2, 0)
            axes[plot_idx].imshow(img_np)
            axes[plot_idx].set_title(
                f"GT:{dataset.classes[label]}\nPred:{dataset.classes[pred]}",
                color="green" if pred == label else "red",
                fontsize=9,
            )
            axes[plot_idx].axis("off")

    for plot_idx in range(sample_count, len(axes)):
        axes[plot_idx].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser("FedSGD FL-only evaluation with layer-wise sparsification")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--server_lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "fedsgd_eval_outputs"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(torch.__version__, torchvision.__version__)
    print(f"Running on {device}")
    print(
        "Using layer-wise sparsification from main_layerwise.py | "
        f"front={FRONT_RATIO} back={BACK_RATIO}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dst = datasets.CIFAR10("~/.torch", train=True, download=True, transform=transforms.ToTensor())
    test_dst = datasets.CIFAR10("~/.torch", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dst, batch_size=args.test_batch_size, shuffle=False)
    client_ids = split_clients(len(train_dst), args.num_clients, args.seed)

    global_model = LeNet().to(device)
    torch.manual_seed(args.seed)
    global_model.apply(weights_init)
    num_classes = global_model.fc[-1].out_features
    param_names = [name for name, _ in global_model.named_parameters()]

    initial_metrics = evaluate_model(global_model, test_loader, device, num_classes)
    print(
        f"[Initial Eval] test_loss={initial_metrics['loss']:.6f} "
        f"| test_accuracy={initial_metrics['accuracy']:.2f}%"
    )

    round_history = []

    for rnd in range(args.rounds):
        num_selected = max(1, int(args.frac * args.num_clients))
        selected = random.sample(range(args.num_clients), num_selected)
        print(f"\n[Round {rnd}] selected={selected}")

        client_gradient_lists = []
        client_sizes = []
        local_metric_rows = []
        round_sparsity_stats = None

        for cid in selected:
            ids = client_ids[cid]
            if not ids:
                continue

            loader = DataLoader(Subset(train_dst, ids), batch_size=args.batch_size, shuffle=False)
            client_grads, client_metrics = compute_client_fedsgd_gradient(
                global_model, loader, device, num_classes
            )
            sparse_grads, sparse_stats = sparsify_gradients_layerwise_topk(client_grads, param_names)

            client_gradient_lists.append(sparse_grads)
            client_sizes.append(len(ids))
            local_metric_rows.append(client_metrics)
            if round_sparsity_stats is None:
                round_sparsity_stats = sparse_stats

            print(
                f"  client={cid:2d} | samples={len(ids):5d} | "
                f"train_loss={client_metrics['train_loss']:.6f} | "
                f"train_acc={client_metrics['train_accuracy']:.2f}% | "
                f"grad_l2={client_metrics['grad_l2']:.6f}"
            )

        if not client_gradient_lists:
            raise RuntimeError("No client gradients were computed in this round.")

        aggregated_gradients = aggregate_gradients(client_gradient_lists, client_sizes)
        aggregated_grad_l2 = compute_grad_l2(aggregated_gradients)
        apply_fedsgd_update(global_model, aggregated_gradients, args.server_lr)
        test_metrics = evaluate_model(global_model, test_loader, device, num_classes)

        weighted_train_loss = sum(
            metrics["train_loss"] * size for metrics, size in zip(local_metric_rows, client_sizes)
        ) / float(sum(client_sizes))
        weighted_train_accuracy = sum(
            metrics["train_accuracy"] * size for metrics, size in zip(local_metric_rows, client_sizes)
        ) / float(sum(client_sizes))

        round_entry = {
            "round": rnd,
            "train_loss": weighted_train_loss,
            "train_accuracy": weighted_train_accuracy,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "aggregated_grad_l2": aggregated_grad_l2,
            "upload_sparsity": round_sparsity_stats,
        }
        round_history.append(round_entry)

        print(
            f"  [Round Summary] train_loss={weighted_train_loss:.6f} "
            f"| train_acc={weighted_train_accuracy:.2f}% "
            f"| test_loss={test_metrics['loss']:.6f} "
            f"| test_acc={test_metrics['accuracy']:.2f}% "
            f"| agg_grad_l2={aggregated_grad_l2:.6f}"
        )
        if round_sparsity_stats is not None:
            print(
                "  [Upload Sparsity] kept="
                f"{round_sparsity_stats['kept']}/{round_sparsity_stats['total']} "
                f"({round_sparsity_stats['retention_ratio'] * 100:.2f}%)"
            )

    final_eval = evaluate_model(
        global_model, test_loader, device, num_classes, collect_predictions=True
    )
    confusion_matrix = compute_confusion_matrix(final_eval["targets"], final_eval["preds"], num_classes)
    final_metrics = compute_classification_metrics(confusion_matrix)
    final_metrics["test_loss"] = final_eval["loss"]
    final_metrics["test_accuracy"] = final_eval["accuracy"]

    curves_path = out_dir / "fedsgd_training_curves.png"
    confusion_path = out_dir / "fedsgd_confusion_matrix.png"
    pred_grid_path = out_dir / "fedsgd_prediction_grid.png"
    summary_json = out_dir / "fedsgd_summary.json"
    summary_txt = out_dir / "fedsgd_summary.txt"

    save_training_curves(round_history, curves_path)
    save_confusion_matrix_image(confusion_matrix, test_dst.classes, confusion_path)
    save_prediction_grid(global_model, test_dst, device, PRED_GRID_SAMPLES, args.seed, pred_grid_path)

    summary_payload = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "config": {
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "frac": args.frac,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "server_lr": args.server_lr,
            "seed": args.seed,
            "front_ratio": FRONT_RATIO,
            "back_ratio": BACK_RATIO,
        },
        "initial_test_metrics": initial_metrics,
        "round_history": round_history,
        "final_metrics": final_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": test_dst.classes,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    with summary_txt.open("w", encoding="utf-8") as fh:
        fh.write(f"device: {device}\n")
        fh.write(f"front_ratio: {FRONT_RATIO}\n")
        fh.write(f"back_ratio: {BACK_RATIO}\n")
        fh.write(f"initial_test_loss: {initial_metrics['loss']:.6f}\n")
        fh.write(f"initial_test_accuracy: {initial_metrics['accuracy']:.2f}\n")
        for round_entry in round_history:
            fh.write(
                f"round_{round_entry['round']}: "
                f"train_loss={round_entry['train_loss']:.6f}, "
                f"train_acc={round_entry['train_accuracy']:.2f}, "
                f"test_loss={round_entry['test_loss']:.6f}, "
                f"test_acc={round_entry['test_accuracy']:.2f}, "
                f"agg_grad_l2={round_entry['aggregated_grad_l2']:.6f}, "
                f"upload_retention={round_entry['upload_sparsity']['retention_ratio']:.6f}\n"
            )
        fh.write(f"final_test_loss: {final_metrics['test_loss']:.6f}\n")
        fh.write(f"final_test_accuracy: {final_metrics['test_accuracy']:.2f}\n")
        fh.write(f"macro_precision: {final_metrics['macro_precision']:.6f}\n")
        fh.write(f"macro_recall: {final_metrics['macro_recall']:.6f}\n")
        fh.write(f"macro_f1: {final_metrics['macro_f1']:.6f}\n")
        fh.write(f"weighted_precision: {final_metrics['weighted_precision']:.6f}\n")
        fh.write(f"weighted_recall: {final_metrics['weighted_recall']:.6f}\n")
        fh.write(f"weighted_f1: {final_metrics['weighted_f1']:.6f}\n")
        for class_name, class_acc in zip(test_dst.classes, final_metrics["per_class_accuracy"]):
            fh.write(f"class_accuracy[{class_name}]: {class_acc:.6f}\n")

    print("\nFinal Metrics")
    print(f"  test_loss: {final_metrics['test_loss']:.6f}")
    print(f"  test_accuracy: {final_metrics['test_accuracy']:.2f}%")
    print(f"  macro_precision: {final_metrics['macro_precision']:.6f}")
    print(f"  macro_recall: {final_metrics['macro_recall']:.6f}")
    print(f"  macro_f1: {final_metrics['macro_f1']:.6f}")
    print(f"  weighted_precision: {final_metrics['weighted_precision']:.6f}")
    print(f"  weighted_recall: {final_metrics['weighted_recall']:.6f}")
    print(f"  weighted_f1: {final_metrics['weighted_f1']:.6f}")

    print("\nGenerated files:")
    for output_path in [curves_path, confusion_path, pred_grid_path, summary_json, summary_txt]:
        print(f"  - {output_path}")


if __name__ == "__main__":
    main()
