# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import LeNet, weights_init


DEFAULT_RATIO_PAIRS = ["0.95:0.95", "0.99:0.91", "0.91:0.99"]
LAYER_ORDER = ["body.0", "body.2", "body.4", "fc.0"]
FRONT_LAYERS = {"body.0", "body.2"}
BACK_LAYERS = {"body.4", "fc.0"}
LINE_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


def parse_ratio_pair(raw_pair):
    parts = raw_pair.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid ratio pair '{raw_pair}'. Expected format 'front:back' like '0.95:0.95'."
        )

    try:
        front_ratio = float(parts[0])
        back_ratio = float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Invalid ratio pair '{raw_pair}'. Both front and back must be floats."
        ) from exc

    for value, name in [(front_ratio, "front"), (back_ratio, "back")]:
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"Invalid {name} ratio '{value}' in '{raw_pair}'. Ratio must satisfy 0 < ratio <= 1."
            )

    return front_ratio, back_ratio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DLG reconstruction across multiple layer-wise sparsification ratios."
    )
    parser.add_argument("--index", type=int, default=25, help="the index for leaking images on CIFAR.")
    parser.add_argument("--image", type=str, default="", help="the path to customized image.")
    parser.add_argument(
        "--ratio_pairs",
        nargs="+",
        default=DEFAULT_RATIO_PAIRS,
        help="space-separated ratio pairs in 'front:back' format, e.g. 0.95:0.95 0.99:0.91",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "layerwise_compare_outputs"),
        help="directory to save comparison figures and summaries",
    )
    args = parser.parse_args()

    try:
        args.ratio_pairs = [parse_ratio_pair(raw_pair) for raw_pair in args.ratio_pairs]
    except ValueError as exc:
        parser.error(str(exc))

    return args


def get_device():
    device = "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def layer_name_from_param_name(param_name):
    parts = param_name.split(".")
    if len(parts) < 2:
        raise ValueError(f"Unsupported parameter name: {param_name}")

    layer_name = ".".join(parts[:2])
    if layer_name not in FRONT_LAYERS | BACK_LAYERS:
        raise ValueError(f"Unexpected LeNet layer: {param_name}")
    return layer_name


def retention_ratio_for_layer(layer_name, front_ratio, back_ratio):
    if layer_name in FRONT_LAYERS:
        return front_ratio
    if layer_name in BACK_LAYERS:
        return back_ratio
    raise ValueError(f"Unknown layer name: {layer_name}")


def sparsify_gradients_layerwise_topk(gradients, param_names, front_ratio, back_ratio):
    if len(gradients) != len(param_names):
        raise ValueError("gradients and param_names must have the same length.")

    grouped = {layer_name: [] for layer_name in LAYER_ORDER}
    for idx, (param_name, grad_tensor) in enumerate(zip(param_names, gradients)):
        grouped[layer_name_from_param_name(param_name)].append((idx, grad_tensor))

    sparse_grads = [None] * len(gradients)
    layer_stats = []
    total_kept = 0
    total_params = 0

    for layer_name in LAYER_ORDER:
        entries = grouped[layer_name]
        flat = torch.cat([grad_tensor.reshape(-1) for _, grad_tensor in entries])
        total = flat.numel()
        ratio = retention_ratio_for_layer(layer_name, front_ratio, back_ratio)
        kept = total if ratio >= 1.0 else max(1, int(total * ratio))

        if kept >= total:
            sparse_flat = flat.clone()
        else:
            _, topk_idx = torch.topk(flat.abs(), kept, largest=True, sorted=False)
            mask = torch.zeros_like(flat)
            mask[topk_idx] = 1.0
            sparse_flat = flat * mask

        offset = 0
        for idx, grad_tensor in entries:
            n = grad_tensor.numel()
            sparse_grads[idx] = sparse_flat[offset : offset + n].view_as(grad_tensor)
            offset += n

        total_kept += kept
        total_params += total
        layer_stats.append(
            {
                "layer": layer_name,
                "kept": kept,
                "total": total,
                "retention_ratio": kept / float(total),
            }
        )

    stats = {
        "kept": total_kept,
        "total": total_params,
        "retention_ratio": total_kept / float(total_params),
        "layer_stats": layer_stats,
    }
    return sparse_grads, stats


def tensor_to_image_array(image_tensor):
    image_tensor = image_tensor.detach().clone().float().cpu()
    image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    image_tensor = image_tensor.clamp(0.0, 1.0)
    return image_tensor.permute(1, 2, 0).numpy()


def prepare_ground_truth(index, image_path, device):
    dst = datasets.CIFAR10("~/.torch", download=True)
    tp = transforms.ToTensor()

    gt_data = tp(dst[index][0]).to(device)
    if image_path:
        gt_data = Image.open(image_path)
        gt_data = tp(gt_data).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.tensor([dst[index][1]], dtype=torch.long, device=device).view(1,)
    gt_onehot_label = label_to_onehot(gt_label, num_classes=10)
    return dst, gt_data, gt_label, gt_onehot_label


def run_single_ratio(gt_data, gt_onehot_label, front_ratio, back_ratio, color, device):
    label = f"front={front_ratio:.2f}, back={back_ratio:.2f}"
    print(f"\n[Run] {label}")

    net = LeNet().to(device)
    param_names = [name for name, _ in net.named_parameters()]

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    net.apply(weights_init)
    criterion = cross_entropy_for_onehot

    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = [grad_tensor.detach().clone() for grad_tensor in dy_dx]
    original_dy_dx, sparse_stats = sparsify_gradients_layerwise_topk(
        original_dy_dx,
        param_names,
        front_ratio,
        back_ratio,
    )

    for layer_stat in sparse_stats["layer_stats"]:
        print(
            f"  {layer_stat['layer']} kept "
            f"{layer_stat['kept']}/{layer_stat['total']} "
            f"({layer_stat['retention_ratio'] * 100:.2f}%)"
        )
    print(
        "  total kept "
        f"{sparse_stats['kept']}/{sparse_stats['total']} "
        f"({sparse_stats['retention_ratio'] * 100:.2f}%)"
    )

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    dummy_data = torch.randn(gt_data.size(), device=device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size(), device=device).requires_grad_(True)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    iterations = []
    losses = []

    for iters in range(400):

        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            current_loss_value = float(current_loss.item())
            iterations.append(iters)
            losses.append(current_loss_value)
            print(f"  iter={iters:3d} | loss={current_loss_value:.4f}")

    return {
        "label": label,
        "front_ratio": front_ratio,
        "back_ratio": back_ratio,
        "color": color,
        "iterations": iterations,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "sparse_stats": sparse_stats,
        "final_reconstruction": tensor_to_image_array(dummy_data[0]),
    }


def save_loss_comparison(results, out_path):
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(
            result["iterations"],
            result["losses"],
            label=result["label"],
            color=result["color"],
            linewidth=2,
        )

    plt.title("DLG Loss Comparison by Layer-wise Sparsification Ratio")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Matching Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_final_reconstruction_grid(gt_data, results, out_path):
    cols = len(results) + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]

    axes[0].imshow(tensor_to_image_array(gt_data[0]))
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    for idx, result in enumerate(results, start=1):
        axes[idx].imshow(result["final_reconstruction"])
        axes[idx].set_title(
            f"{result['label']}\nloss={result['final_loss']:.4f}",
            color=result["color"],
            fontsize=10,
        )
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_files(results, out_dir, image_index, image_path):
    summary_json_path = out_dir / "summary.json"
    summary_txt_path = out_dir / "summary.txt"

    serializable_results = []
    for result in results:
        serializable_results.append(
            {
                "label": result["label"],
                "front_ratio": result["front_ratio"],
                "back_ratio": result["back_ratio"],
                "color": result["color"],
                "iterations": result["iterations"],
                "losses": result["losses"],
                "final_loss": result["final_loss"],
                "sparse_stats": result["sparse_stats"],
            }
        )

    payload = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "image_index": image_index,
        "custom_image": image_path,
        "results": serializable_results,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"torch: {torch.__version__}",
        f"torchvision: {torchvision.__version__}",
        f"image_index: {image_index}",
        f"custom_image: {image_path}",
        "",
    ]
    for result in serializable_results:
        lines.append(result["label"])
        lines.append(f"  final_loss: {result['final_loss']:.6f}")
        lines.append(
            "  total_kept: "
            f"{result['sparse_stats']['kept']}/{result['sparse_stats']['total']} "
            f"({result['sparse_stats']['retention_ratio'] * 100:.2f}%)"
        )
        for layer_stat in result["sparse_stats"]["layer_stats"]:
            lines.append(
                f"  {layer_stat['layer']}: "
                f"{layer_stat['kept']}/{layer_stat['total']} "
                f"({layer_stat['retention_ratio'] * 100:.2f}%)"
            )
        lines.append("")

    summary_txt_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_json_path, summary_txt_path


def main():
    args = parse_args()
    device = get_device()

    print(torch.__version__, torchvision.__version__)
    print(f"Running on {device}")
    print(f"Ratio pairs: {args.ratio_pairs}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, gt_data, _, gt_onehot_label = prepare_ground_truth(args.index, args.image, device)

    results = []
    for idx, (front_ratio, back_ratio) in enumerate(args.ratio_pairs):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        result = run_single_ratio(
            gt_data=gt_data,
            gt_onehot_label=gt_onehot_label,
            front_ratio=front_ratio,
            back_ratio=back_ratio,
            color=color,
            device=device,
        )
        results.append(result)

    loss_path = out_dir / "loss_comparison.png"
    recon_grid_path = out_dir / "final_reconstruction_grid.png"
    summary_json_path, summary_txt_path = write_summary_files(
        results,
        out_dir,
        args.index,
        args.image,
    )

    save_loss_comparison(results, loss_path)
    save_final_reconstruction_grid(gt_data, results, recon_grid_path)

    print("\nSaved files:")
    for output_path in [loss_path, recon_grid_path, summary_json_path, summary_txt_path]:
        print(f"  - {output_path}")


if __name__ == "__main__":
    main()
