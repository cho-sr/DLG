#!/usr/bin/env python3
"""Create per-iteration metric plots for Cocktail Party Attack experiments."""

from pathlib import Path
import re

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "experiment_plots"

EXPERIMENTS = [
    {
        "label": "Baseline",
        "log": ROOT
        / "cocktail_party_attack"
        / "exp/tiny_imagenet/fc2/attack/cp_gm/nodef/decor_1.47_t_12.4_tv_3.1_nv_0_l1_0/128.log",
        "color": "#7f7f7f",
        "linestyle": "-",
    },
    {
        "label": "Quant 8-bit",
        "log": ROOT
        / "cocktail_party_attack_quantization"
        / "exp/tiny_imagenet/fc2/attack/cp_gm/quant/q_bits_8_decor_1.47_t_12.4_tv_3.1_nv_0_l1_0/128.log",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    {
        "label": "Quant 4-bit",
        "log": ROOT
        / "cocktail_party_attack_quantization"
        / "exp/tiny_imagenet/fc2/attack/cp_gm/quant/q_bits_4_decor_1.47_t_12.4_tv_3.1_nv_0_l1_0/128.log",
        "color": "#6baed6",
        "linestyle": "--",
    },
    {
        "label": "Global 80%",
        "log": ROOT
        / "cocktail_party_attack_sparsification"
        / "exp/tiny_imagenet/fc2/attack/cp_gm/sparse/global_topk_sparsity_0.8_decor_1.47_t_12.4_tv_3.1_nv_0_l1_0/128.log",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    {
        "label": "Global 50%",
        "log": ROOT
        / "cocktail_party_attack_sparsification"
        / "exp/tiny_imagenet/fc2/attack/cp_gm/sparse/global_topk_sparsity_0.5_decor_1.47_t_12.4_tv_3.1_nv_0_l1_0/128.log",
        "color": "#ffbb78",
        "linestyle": "--",
    },
    {
        "label": "Layerwise 80%",
        "log": ROOT
        / "cocktail_party_attack_sparsification"
        / "exp/tiny_imagenet/fc2/attack/cp_gm/sparse/layerwise_topk_sparsity_0.8_decor_1.47_t_12.4_tv_3.1_nv_0_l1_0/128.log",
        "color": "#2ca02c",
        "linestyle": "-",
    },
    {
        "label": "Layerwise 50%",
        "log": ROOT
        / "cocktail_party_attack_sparsification"
        / "exp/tiny_imagenet/fc2/attack/cp_gm/sparse/layerwise_topk_sparsity_0.5_decor_1.47_t_12.4_tv_3.1_nv_0_l1_0/128.log",
        "color": "#98df8a",
        "linestyle": "--",
    },
]

METRICS = {
    "psnr": {
        "title": "PSNR by Iteration",
        "ylabel": "PSNR (higher is better)",
        "filename": "psnr_by_iter.png",
    },
    "ssim": {
        "title": "SSIM by Iteration",
        "ylabel": "SSIM (higher is better)",
        "filename": "ssim_by_iter.png",
    },
    "lpips": {
        "title": "LPIPS by Iteration",
        "ylabel": "LPIPS (lower is better)",
        "filename": "lpips_by_iter.png",
    },
}

ITER_LINE_RE = re.compile(
    r"iter:\s+(?P<iter>\d+).*?"
    r"psnr:\s+(?P<psnr>[-+]?\d*\.?\d+).*?"
    r"ssim:\s+(?P<ssim>[-+]?\d*\.?\d+).*?"
    r"lpips:\s+(?P<lpips>[-+]?\d*\.?\d+)"
)


def read_metric_history(log_path):
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = ITER_LINE_RE.search(line)
            if match is None:
                continue
            rows.append(
                {
                    "iter": int(match.group("iter")),
                    "psnr": float(match.group("psnr")),
                    "ssim": float(match.group("ssim")),
                    "lpips": float(match.group("lpips")),
                }
            )
    if not rows:
        raise ValueError(f"No iteration metric rows found in {log_path}")
    return rows


def plot_metric(metric_name, config, histories):
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    for experiment in EXPERIMENTS:
        label = experiment["label"]
        history = histories[label]
        xs = [row["iter"] for row in history]
        ys = [row[metric_name] for row in history]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=2,
            label=label,
            color=experiment["color"],
            linestyle=experiment["linestyle"],
        )

    ax.set_title(config["title"], fontsize=15, pad=14)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(config["ylabel"], fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, ncol=2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path = OUTPUT_DIR / config["filename"]
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = {}

    for experiment in EXPERIMENTS:
        histories[experiment["label"]] = read_metric_history(experiment["log"])

    for metric_name, config in METRICS.items():
        output_path = plot_metric(metric_name, config, histories)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
