from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from compression import clone_state_dict
from data import (
    dirichlet_partition,
    get_device,
    load_datasets,
    make_test_loader,
    partition_summary,
    set_seed,
)
from models import get_model
from plotting import save_plots, write_history_csv, write_json
from trainer import make_client_schedule, run_condition


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FL convergence under non-IID data, sparsification, and quantization."
    )
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["fake", "mnist", "cifar10"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./fl_compression_compare/outputs")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--frac", type=float, default=0.5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Dirichlet concentration for non-IID split. Smaller values are more non-IID.",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=12000,
        help="Use 0 for the full train set.",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=2000,
        help="Use 0 for the full test set.",
    )
    parser.add_argument(
        "--keep_ratios",
        type=float,
        nargs="+",
        default=[1.0, 0.2, 0.05],
        help="Top-k update retention ratios. Example: 1.0 0.2 0.05",
    )
    parser.add_argument(
        "--quant_bits",
        type=int,
        nargs="+",
        default=[32, 8, 4],
        help="Uniform quantization bits. 32 means no quantization.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def format_condition_name(keep_ratio: float, quant_bits: int) -> str:
    ratio_text = f"{keep_ratio:g}".replace(".", "p")
    return f"keep_{ratio_text}_q{quant_bits}"


def validate_args(args) -> None:
    if args.num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive.")
    if args.local_epochs <= 0:
        raise ValueError("local_epochs must be positive.")
    if args.batch_size <= 0 or args.test_batch_size <= 0:
        raise ValueError("batch sizes must be positive.")
    for keep_ratio in args.keep_ratios:
        if keep_ratio < 0.0 or keep_ratio > 1.0:
            raise ValueError("Each keep_ratio must satisfy 0 <= keep_ratio <= 1.")
    for bits in args.quant_bits:
        if bits < 1:
            raise ValueError("Each quant_bits value must be >= 1.")


def build_base_state(dataset_name: str, num_classes: int, seed: int) -> OrderedDict[str, torch.Tensor]:
    set_seed(seed)
    model = get_model(dataset_name, num_classes=num_classes)
    return clone_state_dict(model.state_dict())


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Non-IID Dirichlet alpha: {args.alpha}")
    print(f"Output directory: {out_dir.resolve()}")

    dataset_bundle = load_datasets(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )
    train_dataset = dataset_bundle["train_dataset"]
    test_dataset = dataset_bundle["test_dataset"]
    train_labels = dataset_bundle["train_labels"]
    num_classes = dataset_bundle["num_classes"]

    partitions = dirichlet_partition(
        train_labels,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        min_size=1,
    )
    partition_rows = partition_summary(partitions, train_labels, num_classes)
    write_json(partition_rows, out_dir / "client_label_distribution.json")

    test_loader = make_test_loader(test_dataset, args.test_batch_size)
    schedule = make_client_schedule(
        num_clients=args.num_clients,
        rounds=args.rounds,
        frac=args.frac,
        seed=args.seed,
    )
    base_state = build_base_state(args.dataset, num_classes, args.seed)

    histories: Dict[str, List[Dict[str, float]]] = {}
    flat_rows: List[Dict[str, float]] = []
    conditions: List[Tuple[float, int]] = [
        (keep_ratio, quant_bits)
        for keep_ratio in args.keep_ratios
        for quant_bits in args.quant_bits
    ]

    for index, (keep_ratio, quant_bits) in enumerate(conditions, start=1):
        condition_name = format_condition_name(keep_ratio, quant_bits)
        print(
            f"\n[{index}/{len(conditions)}] {condition_name}: "
            f"keep_ratio={keep_ratio}, quant_bits={quant_bits}"
        )
        history = run_condition(
            condition_name=condition_name,
            dataset_name=args.dataset,
            num_classes=num_classes,
            base_state=base_state,
            train_dataset=train_dataset,
            test_loader=test_loader,
            partitions=partitions,
            schedule=schedule,
            device=device,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            keep_ratio=keep_ratio,
            quant_bits=quant_bits,
            seed=args.seed,
        )
        histories[condition_name] = history
        flat_rows.extend(history)

        final_row = history[-1]
        print(
            f"  final_test_acc={final_row['test_accuracy']:.2f}% "
            f"final_test_loss={final_row['test_loss']:.4f} "
            f"upload={final_row['cumulative_upload_mbits']:.2f} Mbits"
        )

    history_csv = out_dir / "history.csv"
    summary_json = out_dir / "summary.json"
    write_history_csv(flat_rows, history_csv)
    plot_paths = save_plots(histories, out_dir)

    summary = {
        "config": vars(args),
        "device": str(device),
        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_dataset),
        "conditions": [
            {
                "name": format_condition_name(keep_ratio, quant_bits),
                "keep_ratio": keep_ratio,
                "quant_bits": quant_bits,
                "final_test_accuracy": histories[format_condition_name(keep_ratio, quant_bits)][-1][
                    "test_accuracy"
                ],
                "final_test_loss": histories[format_condition_name(keep_ratio, quant_bits)][-1][
                    "test_loss"
                ],
                "cumulative_upload_mbits": histories[format_condition_name(keep_ratio, quant_bits)][-1][
                    "cumulative_upload_mbits"
                ],
            }
            for keep_ratio, quant_bits in conditions
        ],
        "client_label_distribution_file": str(out_dir / "client_label_distribution.json"),
        "history_csv": str(history_csv),
        "plots": [str(path) for path in plot_paths],
    }
    write_json(summary, summary_json)

    print("\nGenerated files:")
    print(f"  {history_csv}")
    print(f"  {summary_json}")
    print(f"  {out_dir / 'client_label_distribution.json'}")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
