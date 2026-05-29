from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


CSV_COLUMNS = [
    "condition",
    "round",
    "keep_ratio",
    "quant_bits",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "round_upload_mbits",
    "cumulative_upload_mbits",
    "relative_upload",
    "compression_mse",
    "quantization_mse",
]


def write_history_csv(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def write_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_plots(histories: Mapping[str, List[Dict[str, float]]], out_dir: Path) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib is unavailable; skipping plots. ({exc})")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    def plot_round_metric(metric: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, history in histories.items():
            rounds = [row["round"] for row in history]
            values = [row[metric] for row in history]
            ax.plot(rounds, values, marker="o", linewidth=1.8, markersize=3, label=name)
        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)

    plot_round_metric("test_accuracy", "Test accuracy (%)", "convergence_accuracy.png")
    plot_round_metric("test_loss", "Test loss", "convergence_loss.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, history in histories.items():
        x_values = [row["cumulative_upload_mbits"] for row in history]
        y_values = [row["test_accuracy"] for row in history]
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=3, label=name)
    ax.set_xlabel("Cumulative client upload (Mbits)")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "accuracy_vs_upload.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)

    return generated
