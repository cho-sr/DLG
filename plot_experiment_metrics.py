#!/usr/bin/env python3
"""Create metric comparison plots for Cocktail Party Attack experiments."""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent / "experiment_plots"

EXPERIMENTS = [
    {
        "label": "Baseline",
        "psnr": 26.444,
        "ssim": 0.794,
        "lpips": 0.243,
        "color": "#7f7f7f",
    },
    {
        "label": "Quant 8-bit",
        "psnr": 26.238,
        "ssim": 0.779,
        "lpips": 0.257,
        "color": "#1f77b4",
    },
    {
        "label": "Quant 4-bit",
        "psnr": 21.815,
        "ssim": 0.164,
        "lpips": 0.702,
        "color": "#6baed6",
    },
    {
        "label": "Global 80%",
        "psnr": 25.560,
        "ssim": 0.714,
        "lpips": 0.313,
        "color": "#ff7f0e",
    },
    {
        "label": "Global 50%",
        "psnr": 22.915,
        "ssim": 0.334,
        "lpips": 0.602,
        "color": "#ffbb78",
    },
    {
        "label": "Layerwise 80%",
        "psnr": 25.658,
        "ssim": 0.729,
        "lpips": 0.297,
        "color": "#2ca02c",
    },
    {
        "label": "Layerwise 50%",
        "psnr": 22.998,
        "ssim": 0.350,
        "lpips": 0.590,
        "color": "#98df8a",
    },
]

METRICS = {
    "psnr": {
        "title": "PSNR Comparison",
        "ylabel": "PSNR (higher is better)",
        "filename": "psnr_comparison.png",
        "ylim_padding": 0.12,
    },
    "ssim": {
        "title": "SSIM Comparison",
        "ylabel": "SSIM (higher is better)",
        "filename": "ssim_comparison.png",
        "ylim_padding": 0.18,
    },
    "lpips": {
        "title": "LPIPS Comparison",
        "ylabel": "LPIPS (lower is better)",
        "filename": "lpips_comparison.png",
        "ylim_padding": 0.18,
    },
}


def add_value_labels(ax, bars, values):
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_metric(metric_name, config):
    labels = [experiment["label"] for experiment in EXPERIMENTS]
    values = [experiment[metric_name] for experiment in EXPERIMENTS]
    colors = [experiment["color"] for experiment in EXPERIMENTS]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_title(config["title"], fontsize=15, pad=14)
    ax.set_ylabel(config["ylabel"], fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=35)

    max_value = max(values)
    ax.set_ylim(0, max_value * (1 + config["ylim_padding"]))
    add_value_labels(ax, bars, values)

    fig.tight_layout()
    output_path = OUTPUT_DIR / config["filename"]
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric_name, config in METRICS.items():
        output_path = plot_metric(metric_name, config)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
