#!/usr/bin/env python3
"""Plot FedAvg iteration outputs without rerunning training.

This script reads `fedavg_iter_summary.json` when available and falls back to
`fedavg_iter_summary.txt`. It generates an SVG report so it works even in
environments where matplotlib is unavailable.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


ROUND_LINE_RE = re.compile(
    r"^round_(\d+): .*?"
    r"train_loss=([0-9eE+\-.]+), "
    r"train_acc=([0-9eE+\-.]+), "
    r"test_loss=([0-9eE+\-.]+), "
    r"test_acc=([0-9eE+\-.]+), "
    r"snapshot_saved=(True|False), "
    r"snapshot_client_ids=(\[.*\])$"
)

CONFIG_CASTS = {
    "num_clients": int,
    "rounds": int,
    "frac": float,
    "batch_size": int,
    "local_epochs": int,
    "client_lr": float,
    "seed": int,
    "test_batch_size": int,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot curves from fedavg_iter_outputs without retraining."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=Path(__file__).resolve().parent / "fedavg_iter_outputs",
        type=Path,
        help="fedavg_iter_outputs directory or a summary json/txt file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output SVG path (default: next to the summary file)",
    )
    return parser.parse_args()


def load_summary(input_path: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    if input_path.is_dir():
        json_path = input_path / "fedavg_iter_summary.json"
        txt_path = input_path / "fedavg_iter_summary.txt"
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
        if txt_path.exists():
            return parse_summary_txt(txt_path)
        raise FileNotFoundError(
            f"No fedavg_iter_summary.json or fedavg_iter_summary.txt found in {input_path}"
        )

    if input_path.suffix.lower() == ".json":
        return json.loads(input_path.read_text(encoding="utf-8"))
    if input_path.suffix.lower() == ".txt":
        return parse_summary_txt(input_path)

    raise ValueError(
        f"Unsupported input: {input_path}. Expected a directory, .json, or .txt file."
    )


def parse_summary_txt(txt_path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    round_history: list[dict[str, Any]] = []
    saved_rounds_completed: list[int] = []
    initial_test_metrics: dict[str, float] = {}
    device = "unknown"

    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("device:"):
            device = line.split(":", 1)[1].strip()
            continue

        if line.startswith("save_rounds:"):
            config["save_rounds"] = ast.literal_eval(line.split(":", 1)[1].strip())
            continue

        if line.startswith("initial_test_loss:"):
            initial_test_metrics["loss"] = float(line.split(":", 1)[1].strip())
            continue

        if line.startswith("initial_test_accuracy:"):
            initial_test_metrics["accuracy"] = float(line.split(":", 1)[1].strip())
            continue

        if line.startswith("saved_rounds_completed:"):
            saved_rounds_completed = ast.literal_eval(line.split(":", 1)[1].strip())
            continue

        matched_round = ROUND_LINE_RE.match(line)
        if matched_round:
            round_idx = int(matched_round.group(1))
            snapshot_client_ids = ast.literal_eval(matched_round.group(7))
            round_history.append(
                {
                    "round": round_idx,
                    "train_loss": float(matched_round.group(2)),
                    "train_accuracy": float(matched_round.group(3)),
                    "test_loss": float(matched_round.group(4)),
                    "test_accuracy": float(matched_round.group(5)),
                    "snapshot_saved": matched_round.group(6) == "True",
                    "snapshot_client_ids": snapshot_client_ids,
                }
            )
            continue

        if ":" not in line:
            continue

        key, value = [part.strip() for part in line.split(":", 1)]
        if key in CONFIG_CASTS:
            config[key] = CONFIG_CASTS[key](value)

    return {
        "device": device,
        "config": config,
        "initial_test_metrics": initial_test_metrics,
        "round_history": round_history,
        "saved_rounds_completed": saved_rounds_completed,
        "gradient_snapshot_counts": {
            str(entry["round"]): len(entry["snapshot_client_ids"])
            for entry in round_history
            if entry["snapshot_client_ids"]
        },
    }


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path.resolve()

    if input_path.is_dir():
        return (input_path / "fedavg_iter_curves.svg").resolve()

    return input_path.with_name(f"{input_path.stem}_curves.svg").resolve()


def nice_step(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def build_ticks(min_value: float, max_value: float, target_count: int = 6) -> list[float]:
    if math.isclose(min_value, max_value):
        min_value -= 0.5
        max_value += 0.5

    step = nice_step((max_value - min_value) / max(target_count - 1, 1))
    tick_start = math.floor(min_value / step) * step
    tick_end = math.ceil(max_value / step) * step

    ticks: list[float] = []
    current = tick_start
    max_ticks = 100
    while current <= tick_end + step * 0.5 and len(ticks) < max_ticks:
        ticks.append(round(current, 10))
        current += step
    return ticks


def format_tick(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def metric_bounds(values: list[float], floor_zero: bool = False, ceiling: float | None = None) -> tuple[float, float]:
    min_value = min(values)
    max_value = max(values)

    if floor_zero:
        min_value = min(min_value, 0.0)

    span = max_value - min_value
    if math.isclose(span, 0.0):
        span = 1.0

    padding = span * 0.08
    lower = min_value - padding
    upper = max_value + padding

    if floor_zero:
        lower = max(0.0, lower)
    if ceiling is not None:
        upper = min(ceiling, upper)

    if math.isclose(lower, upper):
        upper = lower + 1.0

    return lower, upper


def polyline_points(
    points: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    coords = []
    for x_value, y_value in points:
        x_pos = left + (x_value - x_min) / (x_max - x_min) * width
        y_pos = top + height - (y_value - y_min) / (y_max - y_min) * height
        coords.append(f"{x_pos:.2f},{y_pos:.2f}")
    return " ".join(coords)


def svg_text(x: float, y: float, text: str, size: int = 14, anchor: str = "start", fill: str = "#1f2937") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" fill="{fill}">'
        f"{escape(text)}</text>"
    )


def draw_line_panel(
    title: str,
    series: list[dict[str, Any]],
    save_rounds: list[int],
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
    x_max: int,
    y_bounds: tuple[float, float],
) -> str:
    parts: list[str] = []
    parts.append(
        f'<rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" '
        'rx="14" fill="#ffffff" stroke="#d1d5db" stroke-width="1.2"/>'
    )
    parts.append(svg_text(panel_x + 20, panel_y + 28, title, size=18))

    plot_left = panel_x + 70
    plot_top = panel_y + 50
    plot_width = panel_w - 95
    plot_height = panel_h - 90
    y_min, y_max = y_bounds

    for save_round in save_rounds:
        marker_x = plot_left + (save_round / max(x_max, 1)) * plot_width
        parts.append(
            f'<line x1="{marker_x:.2f}" y1="{plot_top:.2f}" x2="{marker_x:.2f}" '
            f'y2="{plot_top + plot_height:.2f}" stroke="#e5e7eb" stroke-width="1" '
            'stroke-dasharray="5 5"/>'
        )

    for x_tick in build_ticks(0.0, float(x_max), target_count=6):
        if x_tick < 0 or x_tick > x_max:
            continue
        tick_x = plot_left + (x_tick / max(x_max, 1)) * plot_width
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{plot_top:.2f}" x2="{tick_x:.2f}" '
            f'y2="{plot_top + plot_height:.2f}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{plot_top + plot_height:.2f}" '
            f'x2="{tick_x:.2f}" y2="{plot_top + plot_height + 6:.2f}" '
            'stroke="#6b7280" stroke-width="1"/>'
        )
        parts.append(svg_text(tick_x, plot_top + plot_height + 24, format_tick(x_tick), size=12, anchor="middle", fill="#4b5563"))

    for y_tick in build_ticks(y_min, y_max, target_count=6):
        if y_tick < y_min - 1e-9 or y_tick > y_max + 1e-9:
            continue
        tick_y = plot_top + plot_height - ((y_tick - y_min) / (y_max - y_min)) * plot_height
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{tick_y:.2f}" x2="{plot_left + plot_width:.2f}" '
            f'y2="{tick_y:.2f}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{plot_left - 6:.2f}" y1="{tick_y:.2f}" x2="{plot_left:.2f}" '
            f'y2="{tick_y:.2f}" stroke="#6b7280" stroke-width="1"/>'
        )
        parts.append(svg_text(plot_left - 10, tick_y + 4, format_tick(y_tick), size=12, anchor="end", fill="#4b5563"))

    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top + plot_height:.2f}" '
        f'x2="{plot_left + plot_width:.2f}" y2="{plot_top + plot_height:.2f}" '
        'stroke="#6b7280" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" x2="{plot_left:.2f}" '
        f'y2="{plot_top + plot_height:.2f}" stroke="#6b7280" stroke-width="1.5"/>'
    )

    legend_x = plot_left + plot_width - 160
    legend_y = panel_y + 24
    for idx, item in enumerate(series):
        legend_item_y = legend_y + idx * 20
        parts.append(
            f'<line x1="{legend_x:.2f}" y1="{legend_item_y:.2f}" '
            f'x2="{legend_x + 18:.2f}" y2="{legend_item_y:.2f}" '
            f'stroke="{item["color"]}" stroke-width="3"/>'
        )
        parts.append(svg_text(legend_x + 26, legend_item_y + 4, item["label"], size=12, fill="#374151"))

    for item in series:
        if len(item["points"]) < 2:
            continue
        line_points = polyline_points(
            item["points"],
            0.0,
            float(max(x_max, 1)),
            y_min,
            y_max,
            plot_left,
            plot_top,
            plot_width,
            plot_height,
        )
        parts.append(
            f'<polyline fill="none" stroke="{item["color"]}" stroke-width="3" '
            f'stroke-linecap="round" stroke-linejoin="round" points="{line_points}"/>'
        )
        last_x, last_y = item["points"][-1]
        circle_x = plot_left + (last_x / max(x_max, 1)) * plot_width
        circle_y = plot_top + plot_height - ((last_y - y_min) / (y_max - y_min)) * plot_height
        parts.append(
            f'<circle cx="{circle_x:.2f}" cy="{circle_y:.2f}" r="4.5" fill="{item["color"]}" stroke="#ffffff" stroke-width="1.5"/>'
        )

    return "".join(parts)


def draw_bar_panel(
    title: str,
    counts_by_round: dict[int, int],
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
    x_max: int,
) -> str:
    parts: list[str] = []
    parts.append(
        f'<rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" '
        'rx="14" fill="#ffffff" stroke="#d1d5db" stroke-width="1.2"/>'
    )
    parts.append(svg_text(panel_x + 20, panel_y + 28, title, size=18))

    plot_left = panel_x + 70
    plot_top = panel_y + 50
    plot_width = panel_w - 95
    plot_height = panel_h - 88
    y_max = max(counts_by_round.values(), default=1)

    for x_tick in build_ticks(0.0, float(x_max), target_count=6):
        if x_tick < 0 or x_tick > x_max:
            continue
        tick_x = plot_left + (x_tick / max(x_max, 1)) * plot_width
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{plot_top:.2f}" x2="{tick_x:.2f}" '
            f'y2="{plot_top + plot_height:.2f}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(svg_text(tick_x, plot_top + plot_height + 22, format_tick(x_tick), size=12, anchor="middle", fill="#4b5563"))

    for y_tick in build_ticks(0.0, float(y_max), target_count=min(y_max + 1, 6)):
        if y_tick < 0 or y_tick > y_max + 1e-9:
            continue
        tick_y = plot_top + plot_height - ((y_tick / max(y_max, 1)) * plot_height)
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{tick_y:.2f}" x2="{plot_left + plot_width:.2f}" '
            f'y2="{tick_y:.2f}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(svg_text(plot_left - 10, tick_y + 4, format_tick(y_tick), size=12, anchor="end", fill="#4b5563"))

    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top + plot_height:.2f}" '
        f'x2="{plot_left + plot_width:.2f}" y2="{plot_top + plot_height:.2f}" '
        'stroke="#6b7280" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" x2="{plot_left:.2f}" '
        f'y2="{plot_top + plot_height:.2f}" stroke="#6b7280" stroke-width="1.5"/>'
    )

    bar_width = max(14.0, min(34.0, plot_width / max(len(counts_by_round) * 2, 8)))
    for round_idx, count in sorted(counts_by_round.items()):
        bar_center_x = plot_left + (round_idx / max(x_max, 1)) * plot_width
        bar_height = (count / max(y_max, 1)) * plot_height
        bar_x = bar_center_x - bar_width / 2
        bar_y = plot_top + plot_height - bar_height
        parts.append(
            f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            'rx="5" fill="#8b5cf6" opacity="0.88"/>'
        )
        parts.append(svg_text(bar_center_x, bar_y - 8, str(count), size=12, anchor="middle", fill="#4b5563"))

    if not counts_by_round:
        parts.append(svg_text(panel_x + panel_w / 2, panel_y + panel_h / 2, "No snapshot counts recorded", size=16, anchor="middle", fill="#6b7280"))

    return "".join(parts)


def build_svg(summary: dict[str, Any]) -> str:
    round_history = summary.get("round_history", [])
    if not round_history:
        raise ValueError("The summary does not contain any round_history entries.")

    rounds = [int(entry["round"]) for entry in round_history]
    max_round = max(rounds)
    save_rounds = [int(value) for value in summary.get("config", {}).get("save_rounds", [])]

    train_loss_points = [(int(entry["round"]), float(entry["train_loss"])) for entry in round_history]
    test_loss_points = [(int(entry["round"]), float(entry["test_loss"])) for entry in round_history]
    train_acc_points = [(int(entry["round"]), float(entry["train_accuracy"])) for entry in round_history]
    test_acc_points = [(int(entry["round"]), float(entry["test_accuracy"])) for entry in round_history]

    initial_test_metrics = summary.get("initial_test_metrics", {})
    if "loss" in initial_test_metrics:
        test_loss_points = [(0, float(initial_test_metrics["loss"]))] + test_loss_points
    if "accuracy" in initial_test_metrics:
        test_acc_points = [(0, float(initial_test_metrics["accuracy"]))] + test_acc_points

    loss_values = [point[1] for point in train_loss_points + test_loss_points]
    acc_values = [point[1] for point in train_acc_points + test_acc_points]
    loss_bounds = metric_bounds(loss_values)
    acc_bounds = metric_bounds(acc_values, floor_zero=True, ceiling=100.0)

    counts_by_round = {
        int(round_idx): int(count)
        for round_idx, count in summary.get("gradient_snapshot_counts", {}).items()
    }

    width = 1280
    height = 940
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    parts.append('<rect width="100%" height="100%" fill="#f8fafc"/>')

    title = "FedAvg Iteration Curves"
    config = summary.get("config", {})
    device = summary.get("device", "unknown")
    subtitle = (
        f"device={device} | clients={config.get('num_clients', '?')} | "
        f"rounds={config.get('rounds', max_round)} | frac={config.get('frac', '?')} | "
        f"batch={config.get('batch_size', '?')} | local_epochs={config.get('local_epochs', '?')} | "
        f"lr={config.get('client_lr', '?')} | seed={config.get('seed', '?')}"
    )

    final_test = round_history[-1]
    best_test = max(round_history, key=lambda item: float(item["test_accuracy"]))
    summary_line = (
        f"final_test_acc={float(final_test['test_accuracy']):.2f}% | "
        f"final_test_loss={float(final_test['test_loss']):.4f} | "
        f"best_test_acc={float(best_test['test_accuracy']):.2f}% @ round {int(best_test['round'])}"
    )

    parts.append(svg_text(36, 48, title, size=28, fill="#111827"))
    parts.append(svg_text(36, 78, subtitle, size=15, fill="#475569"))
    parts.append(svg_text(36, 104, summary_line, size=15, fill="#475569"))

    panel_gap = 26
    panel_width = width - 72
    loss_panel_y = 132
    loss_panel_h = 250
    acc_panel_y = loss_panel_y + loss_panel_h + panel_gap
    acc_panel_h = 250
    bars_panel_y = acc_panel_y + acc_panel_h + panel_gap
    bars_panel_h = 220

    parts.append(
        draw_line_panel(
            title="Loss by Round",
            series=[
                {"label": "Train loss", "color": "#2563eb", "points": train_loss_points},
                {"label": "Test loss", "color": "#dc2626", "points": test_loss_points},
            ],
            save_rounds=save_rounds,
            panel_x=36,
            panel_y=loss_panel_y,
            panel_w=panel_width,
            panel_h=loss_panel_h,
            x_max=max_round,
            y_bounds=loss_bounds,
        )
    )
    parts.append(
        draw_line_panel(
            title="Accuracy by Round",
            series=[
                {"label": "Train accuracy", "color": "#059669", "points": train_acc_points},
                {"label": "Test accuracy", "color": "#f59e0b", "points": test_acc_points},
            ],
            save_rounds=save_rounds,
            panel_x=36,
            panel_y=acc_panel_y,
            panel_w=panel_width,
            panel_h=acc_panel_h,
            x_max=max_round,
            y_bounds=acc_bounds,
        )
    )
    parts.append(
        draw_bar_panel(
            title="Saved Gradient Snapshots per Round",
            counts_by_round=counts_by_round,
            panel_x=36,
            panel_y=bars_panel_y,
            panel_w=panel_width,
            panel_h=bars_panel_h,
            x_max=max_round,
        )
    )

    if save_rounds:
        marker_text = "Saved rounds: " + ", ".join(str(round_idx) for round_idx in save_rounds)
        parts.append(svg_text(width - 36, height - 28, marker_text, size=13, anchor="end", fill="#64748b"))

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.input)
    output_path = resolve_output_path(args.input, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg = build_svg(summary)
    output_path.write_text(svg, encoding="utf-8")
    print(f"Saved graph to: {output_path}")


if __name__ == "__main__":
    main()
