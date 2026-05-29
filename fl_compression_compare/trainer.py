from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn

from compression import (
    add_delta_to_model,
    aggregate_deltas,
    clone_state_dict,
    compress_delta,
    mean_stats,
    subtract_state_dicts,
)
from data import make_client_loader
from models import get_model


def local_train(
    model: nn.Module,
    loader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            total_seen += batch_size

    return {
        "train_loss": total_loss / max(1, total_seen),
        "train_accuracy": 100.0 * total_correct / max(1, total_seen),
    }


def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += batch_size

    return {
        "test_loss": total_loss / max(1, total_seen),
        "test_accuracy": 100.0 * total_correct / max(1, total_seen),
    }


def make_client_schedule(
    num_clients: int,
    rounds: int,
    frac: float,
    seed: int,
) -> List[List[int]]:
    if frac <= 0.0 or frac > 1.0:
        raise ValueError("frac must satisfy 0 < frac <= 1.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    clients_per_round = max(1, int(num_clients * frac))
    schedule: List[List[int]] = []

    for _ in range(rounds):
        perm = torch.randperm(num_clients, generator=generator).tolist()
        schedule.append(perm[:clients_per_round])

    return schedule


def run_condition(
    *,
    condition_name: str,
    dataset_name: str,
    num_classes: int,
    base_state: Mapping[str, torch.Tensor],
    train_dataset,
    test_loader,
    partitions: Sequence[Sequence[int]],
    schedule: Sequence[Sequence[int]],
    device: torch.device,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    keep_ratio: float,
    quant_bits: int,
    seed: int,
) -> List[Dict[str, float]]:
    global_model = get_model(dataset_name, num_classes=num_classes).to(device)
    global_model.load_state_dict(base_state, strict=True)

    initial_metrics = evaluate(global_model, test_loader, device)
    history: List[Dict[str, float]] = [
        {
            "condition": condition_name,
            "round": 0,
            "keep_ratio": keep_ratio,
            "quant_bits": quant_bits,
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "test_loss": initial_metrics["test_loss"],
            "test_accuracy": initial_metrics["test_accuracy"],
            "round_upload_mbits": 0.0,
            "cumulative_upload_mbits": 0.0,
            "relative_upload": 0.0,
            "compression_mse": 0.0,
            "quantization_mse": 0.0,
        }
    ]
    cumulative_upload_bits = 0.0

    for round_idx in range(1, rounds + 1):
        selected_clients = schedule[round_idx - 1]
        global_state = clone_state_dict(global_model.state_dict())

        client_deltas = []
        client_sizes = []
        local_metrics = []
        compression_rows = []

        for client_id in selected_clients:
            loader_seed = seed + round_idx * 1000 + client_id
            loader = make_client_loader(
                train_dataset,
                partitions[client_id],
                batch_size=batch_size,
                seed=loader_seed,
            )

            local_model = get_model(dataset_name, num_classes=num_classes).to(device)
            local_model.load_state_dict(global_state, strict=True)
            metrics = local_train(
                local_model,
                loader,
                device,
                epochs=local_epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

            local_state = clone_state_dict(local_model.state_dict())
            delta = subtract_state_dicts(local_state, global_state)
            compressed_delta, compression_stats = compress_delta(delta, keep_ratio, quant_bits)

            client_deltas.append(compressed_delta)
            client_sizes.append(len(partitions[client_id]))
            local_metrics.append(metrics)
            compression_rows.append(compression_stats)

        aggregated_delta = aggregate_deltas(client_deltas, client_sizes)
        add_delta_to_model(global_model, aggregated_delta)
        test_metrics = evaluate(global_model, test_loader, device)

        selected_size = float(sum(client_sizes))
        train_loss = sum(
            row["train_loss"] * size for row, size in zip(local_metrics, client_sizes)
        ) / selected_size
        train_accuracy = sum(
            row["train_accuracy"] * size for row, size in zip(local_metrics, client_sizes)
        ) / selected_size
        compression_stats = mean_stats(compression_rows)

        round_upload_bits = sum(row["upload_bits"] for row in compression_rows)
        cumulative_upload_bits += round_upload_bits

        history.append(
            {
                "condition": condition_name,
                "round": round_idx,
                "keep_ratio": keep_ratio,
                "quant_bits": quant_bits,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_metrics["test_loss"],
                "test_accuracy": test_metrics["test_accuracy"],
                "round_upload_mbits": round_upload_bits / 1_000_000.0,
                "cumulative_upload_mbits": cumulative_upload_bits / 1_000_000.0,
                "relative_upload": compression_stats.get("relative_upload", 0.0),
                "compression_mse": compression_stats.get("compression_mse", 0.0),
                "quantization_mse": compression_stats.get("quantization_mse", 0.0),
            }
        )

    return history
