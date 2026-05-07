#!/usr/bin/env python3
"""Visualize saved FedAvg layer-wise DLG attack outputs.

This script reads `fedavg_iter_attack_layerwise_outputs` and creates one PNG
report per `(round, client)` pair without rerunning the attack.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "fedavg_iter_attack_layerwise_outputs"
ROUND_DIR_RE = re.compile(r"^round_(\d+)$")
CLIENT_DIR_RE = re.compile(r"^client_(\d+)$")


@dataclass
class RatioResult:
    label: str
    front_ratio: float
    back_ratio: float
    iterations: list[int]
    losses: list[float]
    final_loss: float
    retention_ratio: float
    recon_path: Path


@dataclass
class LayerwiseRecord:
    round_idx: int
    client_id: int
    client_dir: Path
    gt_path: Path
    comparison_path: Path
    recon_grid_path: Path
    results: list[RatioResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PNG reports from fedavg_iter_attack_layerwise_outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="fedavg_iter_attack_layerwise_outputs directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for report PNG files. Defaults to INPUT/visualizations.",
    )
    parser.add_argument(
        "--round",
        dest="round_idx",
        type=int,
        default=None,
        help="Only visualize this round.",
    )
    parser.add_argument(
        "--client_id",
        type=int,
        default=None,
        help="Only visualize this client id.",
    )
    return parser.parse_args()


def numeric_suffix(path: Path, pattern: re.Pattern[str]) -> int | None:
    matched = pattern.match(path.name)
    if matched is None:
        return None
    return int(matched.group(1))


def set_axis_missing(ax: plt.Axes, label: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10, color="#6b7280")
    ax.set_title(label, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")


def show_image(ax: plt.Axes, path: Path, label: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if path.exists():
        ax.imshow(mpimg.imread(path))
        ax.set_title(label, fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        set_axis_missing(ax, label)


def maybe_use_log_scale(ax: plt.Axes, values: list[float]) -> None:
    if values and all(value > 0 for value in values):
        ax.set_yscale("log")


def load_results_from_summary(summary_path: Path, client_dir: Path) -> list[RatioResult]:
    payload: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_results = payload.get("results", [])
    if not raw_results:
        raise ValueError(f"No ratio results found in {summary_path}")

    results: list[RatioResult] = []
    for row in raw_results:
        front_ratio = float(row["front_ratio"])
        back_ratio = float(row["back_ratio"])
        ratio_dir = client_dir / f"front_{front_ratio:.2f}_back_{back_ratio:.2f}"
        results.append(
            RatioResult(
                label=str(row["label"]),
                front_ratio=front_ratio,
                back_ratio=back_ratio,
                iterations=[int(iteration) for iteration in row.get("iterations", [])],
                losses=[float(loss) for loss in row.get("losses", [])],
                final_loss=float(row["final_loss"]),
                retention_ratio=float(row["sparse_stats"]["retention_ratio"]),
                recon_path=ratio_dir / "dlg_recon.png",
            )
        )
    return results


def discover_records(
    input_dir: Path, requested_round: int | None, requested_client_id: int | None
) -> list[LayerwiseRecord]:
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    records: list[LayerwiseRecord] = []
    round_dirs = [
        (round_idx, path)
        for path in input_dir.iterdir()
        if path.is_dir()
        for round_idx in [numeric_suffix(path, ROUND_DIR_RE)]
        if round_idx is not None
    ]

    for round_idx, round_dir in sorted(round_dirs, key=lambda item: item[0]):
        if requested_round is not None and round_idx != requested_round:
            continue

        client_dirs = [
            (client_id, path)
            for path in round_dir.iterdir()
            if path.is_dir()
            for client_id in [numeric_suffix(path, CLIENT_DIR_RE)]
            if client_id is not None
        ]

        for client_id, client_dir in sorted(client_dirs, key=lambda item: item[0]):
            if requested_client_id is not None and client_id != requested_client_id:
                continue

            summary_path = client_dir / "summary.json"
            if not summary_path.exists():
                continue

            records.append(
                LayerwiseRecord(
                    round_idx=round_idx,
                    client_id=client_id,
                    client_dir=client_dir,
                    gt_path=client_dir / "gt.png",
                    comparison_path=client_dir / "loss_comparison.png",
                    recon_grid_path=client_dir / "final_reconstruction_grid.png",
                    results=load_results_from_summary(summary_path, client_dir),
                )
            )

    if not records:
        raise FileNotFoundError(
            f"No readable layer-wise attack outputs found in {input_dir}"
        )

    return records


def make_report(record: LayerwiseRecord, output_path: Path) -> None:
    num_ratios = max(1, len(record.results))
    fig_height = max(8.5, 5.5 + num_ratios * 1.8)
    fig = plt.figure(figsize=(15, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=num_ratios + 3,
        ncols=4,
        height_ratios=[2.8, 1.7, 0.35] + [1.0] * num_ratios,
    )

    fig.suptitle(
        f"FedAvg Layer-wise DLG Report - Round {record.round_idx}, Client {record.client_id}",
        fontsize=18,
    )

    loss_ax = fig.add_subplot(grid[0, :3])
    all_loss_values: list[float] = []
    cmap = plt.get_cmap("tab10")
    for idx, result in enumerate(record.results):
        all_loss_values.extend(result.losses)
        loss_ax.plot(
            result.iterations,
            result.losses,
            marker="o",
            markersize=2.5,
            linewidth=1.8,
            color=cmap(idx % 10),
            label=result.label,
        )
    maybe_use_log_scale(loss_ax, all_loss_values)
    loss_ax.set_title("DLG Loss by Iteration")
    loss_ax.set_xlabel("Iteration")
    loss_ax.set_ylabel("Gradient loss")
    loss_ax.grid(True, which="both", alpha=0.25)
    loss_ax.legend(fontsize=8)

    final_ax = fig.add_subplot(grid[0, 3])
    ratio_labels = [result.label.replace(", ", "\n") for result in record.results]
    final_losses = [result.final_loss for result in record.results]
    bars = final_ax.bar(range(len(record.results)), final_losses, color=[cmap(i % 10) for i in range(len(record.results))])
    maybe_use_log_scale(final_ax, final_losses)
    final_ax.set_title("Final Loss by Ratio")
    final_ax.set_ylabel("Final grad loss")
    final_ax.set_xticks(range(len(record.results)))
    final_ax.set_xticklabels(ratio_labels, fontsize=8)
    for bar, loss in zip(bars, final_losses):
        final_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{loss:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    show_image(fig.add_subplot(grid[1, 0]), record.gt_path, "gt")
    show_image(fig.add_subplot(grid[1, 1]), record.comparison_path, "loss_comparison")
    show_image(fig.add_subplot(grid[1, 2:4]), record.recon_grid_path, "final_reconstruction_grid")

    headers = ["ratio", "retained", "final_loss", "reconstruction"]
    for col_idx, header in enumerate(headers):
        header_ax = fig.add_subplot(grid[2, col_idx])
        header_ax.axis("off")
        header_ax.text(0.5, 0.5, header, ha="center", va="center", fontsize=12, weight="bold")

    for row_idx, result in enumerate(record.results, start=3):
        text_ax = fig.add_subplot(grid[row_idx, 0])
        text_ax.axis("off")
        text_ax.text(
            0.5,
            0.5,
            result.label.replace(", ", "\n"),
            ha="center",
            va="center",
            fontsize=10,
        )

        retained_ax = fig.add_subplot(grid[row_idx, 1])
        retained_ax.axis("off")
        retained_ax.text(
            0.5,
            0.5,
            f"{result.retention_ratio * 100:.2f}%",
            ha="center",
            va="center",
            fontsize=11,
        )

        loss_text_ax = fig.add_subplot(grid[row_idx, 2])
        loss_text_ax.axis("off")
        loss_text_ax.text(
            0.5,
            0.5,
            f"{result.final_loss:.6f}",
            ha="center",
            va="center",
            fontsize=11,
        )

        show_image(fig.add_subplot(grid[row_idx, 3]), result.recon_path, "dlg_recon")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "visualizations"
    )

    records = discover_records(input_dir, args.round_idx, args.client_id)
    for record in records:
        output_path = output_dir / (
            f"round_{record.round_idx}_client_{record.client_id}_layerwise_report.png"
        )
        make_report(record, output_path)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
