#!/usr/bin/env python3
"""Visualize saved FedAvg iterative DLG attack outputs.

This script reads per-round attack outputs from `fedavg_iter_attack_outputs`
and writes one PNG report per client. It does not rerun the attack.
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


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "fedavg_iter_attack_outputs"
ROUND_DIR_RE = re.compile(r"^round_(\d+)$")
CLIENT_DIR_RE = re.compile(r"^client_(\d+)$")


@dataclass
class AttackRecord:
    round_idx: int
    client_id: int
    client_dir: Path
    loss_history: list[tuple[int, float]]
    image_paths: dict[str, Path]

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PNG reports from fedavg_iter_attack_outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="fedavg_iter_attack_outputs directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for report PNG files. Defaults to INPUT/visualizations.",
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


def load_loss_history(client_dir: Path) -> list[tuple[int, float]]:
    json_path = client_dir / "loss_history.json"
    txt_path = client_dir / "loss_history.txt"

    if json_path.exists():
        payload: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        history = payload.get("loss_history", [])
        rows = [
            (int(row["iter"]), float(row["grad_loss"]))
            for row in history
            if "iter" in row and "grad_loss" in row
        ]
        if rows:
            return rows

    if txt_path.exists():
        rows: list[tuple[int, float]] = []
        for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith("iter"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((int(parts[0]), float(parts[1])))
        if rows:
            return rows

    raise FileNotFoundError(f"No readable loss history found in {client_dir}")


def discover_records(input_dir: Path, requested_client_id: int | None) -> dict[int, list[AttackRecord]]:
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    records_by_client: dict[int, list[AttackRecord]] = {}

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
            loss_history = load_loss_history(client_dir)
            image_paths = {
                "gt": client_dir / "gt.png",
                "dummy_init": client_dir / "dummy_init.png",
                "dlg_recon": client_dir / "dlg_recon.png",
            }
            records_by_client.setdefault(client_id, []).append(
                AttackRecord(
                    round_idx=round_idx,
                    client_id=client_id,
                    client_dir=client_dir,
                    loss_history=loss_history,
                    image_paths=image_paths,
                )
            )

    if not records_by_client:
        target = f"client_{requested_client_id}" if requested_client_id is not None else "any client"
        raise FileNotFoundError(f"No readable attack outputs found for {target} in {input_dir}")

    return {
        client_id: sorted(records, key=lambda record: record.round_idx)
        for client_id, records in sorted(records_by_client.items())
    }


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


def make_report(records: list[AttackRecord], output_path: Path) -> None:
    if not records:
        raise ValueError("No attack records were provided.")

    client_id = records[0].client_id
    num_rounds = len(records)
    figure_height = max(8.0, 4.8 + num_rounds * 1.35)
    fig = plt.figure(figsize=(14, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=num_rounds + 2,
        ncols=4,
        height_ratios=[3.0, 0.35] + [1.0] * num_rounds,
    )

    fig.suptitle(f"FedAvg Iterative DLG Attack Report - Client {client_id}", fontsize=18)

    loss_ax = fig.add_subplot(grid[0, :3])
    all_loss_values: list[float] = []
    cmap = plt.get_cmap("tab20")
    for idx, record in enumerate(records):
        iters = [item[0] for item in record.loss_history]
        losses = [item[1] for item in record.loss_history]
        all_loss_values.extend(losses)
        loss_ax.plot(
            iters,
            losses,
            marker="o",
            markersize=2.5,
            linewidth=1.5,
            color=cmap(idx % 20),
            label=f"round {record.round_idx}",
        )
    maybe_use_log_scale(loss_ax, all_loss_values)
    loss_ax.set_title("DLG Loss by Iteration")
    loss_ax.set_xlabel("Iteration")
    loss_ax.set_ylabel("Gradient loss")
    loss_ax.grid(True, which="both", alpha=0.25)
    loss_ax.legend(fontsize=8, ncol=2)

    final_ax = fig.add_subplot(grid[0, 3])
    rounds = [record.round_idx for record in records]
    final_losses = [record.final_loss for record in records]
    final_ax.plot(rounds, final_losses, marker="o", linewidth=2, color="#dc2626")
    maybe_use_log_scale(final_ax, final_losses)
    final_ax.set_title("Final Loss by Round")
    final_ax.set_xlabel("Round")
    final_ax.set_ylabel("Final grad loss")
    final_ax.grid(True, which="both", alpha=0.25)
    final_ax.set_xticks(rounds)
    final_ax.tick_params(axis="x", labelrotation=45)

    headers = ["round", "gt", "dummy_init", "dlg_recon"]
    for col_idx, header in enumerate(headers):
        header_ax = fig.add_subplot(grid[1, col_idx])
        header_ax.axis("off")
        header_ax.text(0.5, 0.5, header, ha="center", va="center", fontsize=12, weight="bold")

    for row_idx, record in enumerate(records, start=2):
        round_ax = fig.add_subplot(grid[row_idx, 0])
        round_ax.axis("off")
        round_ax.text(
            0.5,
            0.55,
            f"round {record.round_idx}\nfinal loss\n{record.final_loss:.4g}",
            ha="center",
            va="center",
            fontsize=10,
        )

        show_image(fig.add_subplot(grid[row_idx, 1]), record.image_paths["gt"], "gt")
        show_image(fig.add_subplot(grid[row_idx, 2]), record.image_paths["dummy_init"], "dummy_init")
        show_image(fig.add_subplot(grid[row_idx, 3]), record.image_paths["dlg_recon"], "dlg_recon")

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

    records_by_client = discover_records(input_dir, args.client_id)
    for client_id, records in records_by_client.items():
        output_path = output_dir / f"client_{client_id}_attack_report.png"
        make_report(records, output_path)
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
