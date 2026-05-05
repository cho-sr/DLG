# -*- coding: utf-8 -*-
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from models.vision import LeNet, weights_init
from utils import cross_entropy_for_onehot, label_to_onehot


DEFAULT_SAVE_ROUNDS = [10, 20, 30, 40, 50]


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


class ClientSubsetWithIndex(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_index = self.indices[idx]
        image, label = self.dataset[sample_index]
        return image, label, sample_index


def clone_state_dict_to_cpu(state_dict):
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def compute_grad_l2(named_grads) -> float:
    total = 0.0
    for grad_tensor in named_grads.values():
        total += float(torch.sum(grad_tensor.detach().float() ** 2).item())
    return math.sqrt(total)


def average_state_dicts(state_dicts, client_sizes):
    if not state_dicts:
        raise ValueError("state_dicts must not be empty.")

    total_weight = float(sum(client_sizes))
    if total_weight <= 0:
        raise ValueError("Sum of client_sizes must be positive.")

    averaged_state = {}
    for name in state_dicts[0]:
        first_tensor = state_dicts[0][name]
        if torch.is_floating_point(first_tensor):
            averaged = torch.zeros_like(first_tensor, dtype=torch.float32)
            for state_dict, size in zip(state_dicts, client_sizes):
                averaged += state_dict[name].float() * (size / total_weight)
            averaged_state[name] = averaged.to(dtype=first_tensor.dtype)
        else:
            averaged_state[name] = first_tensor.clone()
    return averaged_state


def evaluate_model(model, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": 100.0 * total_correct / max(1, total_samples),
    }


def build_gradient_snapshot(model, round_idx, client_id, sample_indices, labels, param_names, images):
    sample_indices_list = [int(idx) for idx in sample_indices.detach().cpu().tolist()]
    labels_list = [int(label) for label in labels.detach().cpu().tolist()]
    named_grads = {
        name: param.grad.detach().cpu().clone()
        for name, param in model.named_parameters()
    }

    snapshot = {
        "round": int(round_idx),
        "client_id": int(client_id),
        "sample_index": sample_indices_list[0] if len(sample_indices_list) == 1 else sample_indices_list,
        "label": labels_list[0] if len(labels_list) == 1 else labels_list,
        "param_names": list(param_names),
        "named_grads": named_grads,
        "grad_l2": compute_grad_l2(named_grads),
        "model_state_dict_before_step": clone_state_dict_to_cpu(model.state_dict()),
        "input_shape": tuple(int(dim) for dim in images.shape),
    }
    return snapshot


def train_client_fedavg(
    global_state_dict,
    dataset,
    indices,
    client_id,
    round_idx,
    args,
    device,
    num_classes,
    should_capture_snapshot,
):
    model = LeNet().to(device)
    model.load_state_dict(global_state_dict, strict=True)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.client_lr)
    loader = DataLoader(
        ClientSubsetWithIndex(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    param_names = [name for name, _ in model.named_parameters()]

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    snapshot = None
    first_local_step_seen = False

    for _ in range(args.local_epochs):
        for images, labels, sample_indices in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            onehot = label_to_onehot(labels, num_classes=num_classes)
            outputs = model(images)
            loss = cross_entropy_for_onehot(outputs, onehot)
            loss.backward()

            # Capture the raw first-step local gradient before optimizer.step() updates weights.
            if should_capture_snapshot and not first_local_step_seen:
                snapshot = build_gradient_snapshot(
                    model=model,
                    round_idx=round_idx,
                    client_id=client_id,
                    sample_indices=sample_indices,
                    labels=labels,
                    param_names=param_names,
                    images=images,
                )

            optimizer.step()
            first_local_step_seen = True

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((outputs.detach().argmax(dim=1) == labels).sum().item())
            total_samples += batch_size

    metrics = {
        "train_loss": total_loss / max(1, total_samples),
        "train_accuracy": 100.0 * total_correct / max(1, total_samples),
        "num_examples_seen": total_samples,
        "snapshot_saved": snapshot is not None,
    }
    return clone_state_dict_to_cpu(model.state_dict()), metrics, snapshot


def parse_args():
    parser = argparse.ArgumentParser("FedAvg training with per-round DLG gradient capture")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--client_lr", type=float, default=0.01)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--save_rounds",
        nargs="+",
        type=int,
        default=DEFAULT_SAVE_ROUNDS,
        help="1-based round numbers whose first local-step gradients should be saved.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "fedavg_iter_outputs"),
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.num_clients < 1:
        raise ValueError("--num_clients must be at least 1.")
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1.")
    if not (0.0 < args.frac <= 1.0):
        raise ValueError("--frac must satisfy 0 < frac <= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if args.local_epochs < 1:
        raise ValueError("--local_epochs must be at least 1.")
    if args.client_lr <= 0:
        raise ValueError("--client_lr must be positive.")
    if args.test_batch_size < 1:
        raise ValueError("--test_batch_size must be at least 1.")
    if not args.save_rounds:
        raise ValueError("--save_rounds must contain at least one round.")
    if any(round_idx < 1 for round_idx in args.save_rounds):
        raise ValueError("--save_rounds must contain only positive 1-based round indices.")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = get_device()
    save_rounds = sorted(set(args.save_rounds))

    print(torch.__version__, torchvision.__version__)
    print(f"Running on {device}")
    print(
        "FedAvg config | "
        f"clients={args.num_clients} rounds={args.rounds} frac={args.frac} "
        f"batch_size={args.batch_size} local_epochs={args.local_epochs} client_lr={args.client_lr}"
    )
    print(f"Gradient save rounds (1-based): {save_rounds}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dst = datasets.CIFAR10("~/.torch", train=True, download=True, transform=transforms.ToTensor())
    test_dst = datasets.CIFAR10("~/.torch", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dst, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    client_ids = split_clients(len(train_dst), args.num_clients, args.seed)

    global_model = LeNet().to(device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    global_model.apply(weights_init)
    num_classes = global_model.fc[-1].out_features

    initial_metrics = evaluate_model(global_model, test_loader, device, num_classes)
    print(
        f"[Initial Eval] test_loss={initial_metrics['loss']:.6f} "
        f"| test_accuracy={initial_metrics['accuracy']:.2f}%"
    )

    gradient_snapshots = {}
    round_history = []

    for round_idx in range(1, args.rounds + 1):
        num_selected = max(1, int(args.frac * args.num_clients))
        selected_clients = random.sample(range(args.num_clients), num_selected)
        should_capture_this_round = round_idx in save_rounds

        print(f"\n[Round {round_idx}] selected={selected_clients}")

        client_state_dicts = []
        client_sizes = []
        local_metric_rows = []
        snapshot_client_ids = []

        for client_id in selected_clients:
            ids = client_ids[client_id]
            if not ids:
                continue

            local_state_dict, client_metrics, snapshot = train_client_fedavg(
                global_state_dict=global_model.state_dict(),
                dataset=train_dst,
                indices=ids,
                client_id=client_id,
                round_idx=round_idx,
                args=args,
                device=device,
                num_classes=num_classes,
                should_capture_snapshot=should_capture_this_round,
            )

            client_state_dicts.append(local_state_dict)
            client_sizes.append(len(ids))
            local_metric_rows.append(client_metrics)

            print(
                f"  client={client_id:4d} | samples={len(ids):5d} | "
                f"train_loss={client_metrics['train_loss']:.6f} | "
                f"train_acc={client_metrics['train_accuracy']:.2f}%"
            )

            if snapshot is not None:
                gradient_snapshots.setdefault(round_idx, {})[client_id] = snapshot
                snapshot_client_ids.append(client_id)
                print(
                    f"    saved gradient snapshot | sample_index={snapshot['sample_index']} "
                    f"| label={snapshot['label']} | grad_l2={snapshot['grad_l2']:.6f}"
                )

        if not client_state_dicts:
            raise RuntimeError("No client updates were produced in this round.")

        averaged_state_dict = average_state_dicts(client_state_dicts, client_sizes)
        global_model.load_state_dict(averaged_state_dict, strict=True)
        test_metrics = evaluate_model(global_model, test_loader, device, num_classes)

        weighted_train_loss = sum(
            metrics["train_loss"] * size for metrics, size in zip(local_metric_rows, client_sizes)
        ) / float(sum(client_sizes))
        weighted_train_accuracy = sum(
            metrics["train_accuracy"] * size for metrics, size in zip(local_metric_rows, client_sizes)
        ) / float(sum(client_sizes))

        round_entry = {
            "round": round_idx,
            "selected_clients": selected_clients,
            "train_loss": weighted_train_loss,
            "train_accuracy": weighted_train_accuracy,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "snapshot_saved": bool(snapshot_client_ids),
            "snapshot_client_ids": snapshot_client_ids,
        }
        round_history.append(round_entry)

        print(
            f"  [Round Summary] train_loss={weighted_train_loss:.6f} "
            f"| train_acc={weighted_train_accuracy:.2f}% "
            f"| test_loss={test_metrics['loss']:.6f} "
            f"| test_acc={test_metrics['accuracy']:.2f}% "
            f"| snapshots={snapshot_client_ids}"
        )

    gradients_path = out_dir / "fedavg_iter_gradients.pt"
    summary_json_path = out_dir / "fedavg_iter_summary.json"
    summary_txt_path = out_dir / "fedavg_iter_summary.txt"

    torch.save(
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "config": {
                "num_clients": args.num_clients,
                "rounds": args.rounds,
                "frac": args.frac,
                "batch_size": args.batch_size,
                "local_epochs": args.local_epochs,
                "client_lr": args.client_lr,
                "test_batch_size": args.test_batch_size,
                "seed": args.seed,
                "save_rounds": save_rounds,
            },
            "gradient_snapshots": gradient_snapshots,
        },
        gradients_path,
    )

    summary_payload = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "config": {
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "frac": args.frac,
            "batch_size": args.batch_size,
            "local_epochs": args.local_epochs,
            "client_lr": args.client_lr,
            "test_batch_size": args.test_batch_size,
            "seed": args.seed,
            "save_rounds": save_rounds,
        },
        "initial_test_metrics": initial_metrics,
        "round_history": round_history,
        "saved_rounds_completed": sorted(int(round_idx) for round_idx in gradient_snapshots.keys()),
        "gradient_snapshot_counts": {
            str(round_idx): len(round_snapshots)
            for round_idx, round_snapshots in gradient_snapshots.items()
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    with summary_txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"device: {device}\n")
        fh.write(f"num_clients: {args.num_clients}\n")
        fh.write(f"rounds: {args.rounds}\n")
        fh.write(f"frac: {args.frac}\n")
        fh.write(f"batch_size: {args.batch_size}\n")
        fh.write(f"local_epochs: {args.local_epochs}\n")
        fh.write(f"client_lr: {args.client_lr}\n")
        fh.write(f"seed: {args.seed}\n")
        fh.write(f"save_rounds: {save_rounds}\n")
        fh.write(f"initial_test_loss: {initial_metrics['loss']:.6f}\n")
        fh.write(f"initial_test_accuracy: {initial_metrics['accuracy']:.2f}\n")
        for round_entry in round_history:
            fh.write(
                f"round_{round_entry['round']}: "
                f"selected={round_entry['selected_clients']}, "
                f"train_loss={round_entry['train_loss']:.6f}, "
                f"train_acc={round_entry['train_accuracy']:.2f}, "
                f"test_loss={round_entry['test_loss']:.6f}, "
                f"test_acc={round_entry['test_accuracy']:.2f}, "
                f"snapshot_saved={round_entry['snapshot_saved']}, "
                f"snapshot_client_ids={round_entry['snapshot_client_ids']}\n"
            )
        fh.write(
            f"saved_rounds_completed: "
            f"{sorted(int(round_idx) for round_idx in gradient_snapshots.keys())}\n"
        )

    print("\nGenerated files:")
    for output_path in [gradients_path, summary_json_path, summary_txt_path]:
        print(f"  - {output_path}")


if __name__ == "__main__":
    main()
