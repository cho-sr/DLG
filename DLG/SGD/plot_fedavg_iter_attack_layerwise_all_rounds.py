#!/usr/bin/env python3
"""Visualize layer-wise FedAvg attack outputs across all rounds.

This script reads `fedavg_iter_attack_layerwise_outputs` and creates one PNG
report per client where all discovered rounds are collected into a single
figure.
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
class RatioSummary:
    label: str
    front_ratio: float
    back_ratio: float
    final_loss: float
    retention_ratio: float


@dataclass
class RoundRecord:
    round_idx: int
    client_id: int
    client_dir: Path
    gt_path: Path
    comparison_path: Path
    recon_grid_path: Path
    ratios: list[RatioSummary]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one PNG report per client with all rounds from fedavg_iter_attack_layerwise_outputs."
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
        help="Directory for report PNG files. Defaults to INPUT/visualizations_all_rounds.",
    )
    parser.add_argument(
        "--client_id",
        type=int,
        default=None,
        help="Only visualize this client id. Defaults to every discovered client.",
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


def load_ratio_summaries(summary_path: Path) -> list[RatioSummary]:
    payload: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_results = payload.get("results", [])
    if not raw_results:
        raise ValueError(f"No ratio summaries found in {summary_path}")

    results: list[RatioSummary] = []
    for row in raw_results:
        results.append(
            RatioSummary(
                label=str(row["label"]),
                front_ratio=float(row["front_ratio"]),
                back_ratio=float(row["back_ratio"]),
                final_loss=float(row["final_loss"]),
                retention_ratio=float(row["sparse_stats"]["retention_ratio"]),
            )
        )
    return results


def discover_records(input_dir: Path, requested_client_id: int | None) -> dict[int, list[RoundRecord]]:
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    records_by_client: dict[int, list[RoundRecord]] = {}

    round_dirs = [
        (round_idx, path)
        for path in input_dir.iterdir()
        if path.is_dir()
        for round_idx in [numeric_suffix(path, ROUND_DIR_RE)]
        if round_idx is not None
    ]

    for round_idx, round_dir in sorted(round_dirs, key=lambda item: item[0]):
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

            record = RoundRecord(
                round_idx=round_idx,
                client_id=client_id,
                client_dir=client_dir,
                gt_path=client_dir / "gt.png",
                comparison_path=client_dir / "loss_comparison.png",
                recon_grid_path=client_dir / "final_reconstruction_grid.png",
                ratios=load_ratio_summaries(summary_path),
            )
            records_by_client.setdefault(client_id, []).append(record)

    if not records_by_client:
        target = f"client_{requested_client_id}" if requested_client_id is not None else "any client"
        raise FileNotFoundError(f"No readable layer-wise attack outputs found for {target} in {input_dir}")

    return {
        client_id: sorted(records, key=lambda record: record.round_idx)
        for client_id, records in sorted(records_by_client.items())
    }


def build_ratio_series(records: list[RoundRecord]) -> dict[str, dict[str, list[float]]]:
    ratio_series: dict[str, dict[str, list[float]]] = {}
    for record in records:
        for ratio in record.ratios:
            entry = ratio_series.setdefault(
                ratio.label,
                {
                    "rounds": [],
                    "final_losses": [],
                    "retention_ratios": [],
                },
            )
            entry["rounds"].append(record.round_idx)
            entry["final_losses"].append(ratio.final_loss)
            entry["retention_ratios"].append(ratio.retention_ratio * 100.0)
    return ratio_series


def round_summary_text(record: RoundRecord) -> str:
    lines = [f"round {record.round_idx}"]
    for ratio in record.ratios:
        lines.append(
            f"{ratio.label}\nloss={ratio.final_loss:.4g}, kept={ratio.retention_ratio * 100:.2f}%"
        )
    return "\n".join(lines)


def make_report(records: list[RoundRecord], output_path: Path) -> None:
    if not records:
        raise ValueError("No round records were provided.")

    client_id = records[0].client_id
    ratio_series = build_ratio_series(records)
    num_rounds = len(records)
    figure_height = max(9.0, 5.8 + num_rounds * 1.6)
    fig = plt.figure(figsize=(16, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=num_rounds + 2,
        ncols=4,
        height_ratios=[3.0, 0.35] + [1.0] * num_rounds,
    )

    fig.suptitle(
        f"FedAvg Layer-wise DLG All-Round Report - Client {client_id}",
        fontsize=18,
    )

    final_loss_ax = fig.add_subplot(grid[0, :2])
    retention_ax = fig.add_subplot(grid[0, 2:])
    cmap = plt.get_cmap("tab10")
    all_final_losses: list[float] = []

    for idx, (label, series) in enumerate(ratio_series.items()):
        color = cmap(idx % 10)
        rounds = [int(value) for value in series["rounds"]]
        final_losses = [float(value) for value in series["final_losses"]]
        retained = [float(value) for value in series["retention_ratios"]]
        all_final_losses.extend(final_losses)

        final_loss_ax.plot(
            rounds,
            final_losses,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        retention_ax.plot(
            rounds,
            retained,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

    maybe_use_log_scale(final_loss_ax, all_final_losses)
    final_loss_ax.set_title("Final Loss by Round")
    final_loss_ax.set_xlabel("Round")
    final_loss_ax.set_ylabel("Final grad loss")
    final_loss_ax.grid(True, which="both", alpha=0.25)
    final_loss_ax.legend(fontsize=8)

    retention_ax.set_title("Retention Ratio by Round")
    retention_ax.set_xlabel("Round")
    retention_ax.set_ylabel("Retained gradients (%)")
    retention_ax.grid(True, alpha=0.25)
    retention_ax.legend(fontsize=8)

    headers = ["round summary", "gt", "loss comparison", "reconstruction grid"]
    for col_idx, header in enumerate(headers):
        header_ax = fig.add_subplot(grid[1, col_idx])
        header_ax.axis("off")
        header_ax.text(0.5, 0.5, header, ha="center", va="center", fontsize=12, weight="bold")

    for row_idx, record in enumerate(records, start=2):
        round_ax = fig.add_subplot(grid[row_idx, 0])
        round_ax.axis("off")
        round_ax.text(
            0.5,
            0.5,
            round_summary_text(record),
            ha="center",
            va="center",
            fontsize=9,
        )

        show_image(fig.add_subplot(grid[row_idx, 1]), record.gt_path, "gt")
        show_image(fig.add_subplot(grid[row_idx, 2]), record.comparison_path, "loss_comparison")
        show_image(fig.add_subplot(grid[row_idx, 3]), record.recon_grid_path, "final_reconstruction_grid")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "visualizations_all_rounds"
    )

    records_by_client = discover_records(input_dir, args.client_id)
    for client_id, records in records_by_client.items():
        output_path = output_dir / f"client_{client_id}_all_rounds_layerwise_report.png"
        make_report(records, output_path)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
