# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from utils import label_to_onehot, cross_entropy_for_onehot
from models.vision import LeNet


DEFAULT_PT_FILE = Path(__file__).resolve().parent / "fedavg_iter_outputs" / "fedavg_iter_gradients.pt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "fedavg_iter_attack_layerwise_outputs"
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
        description="Run DLG on a FedAvg snapshot with multiple layer-wise top-k sparsification ratios."
    )
    parser.add_argument(
        "--pt_file",
        type=str,
        default=str(DEFAULT_PT_FILE),
        help="the path to the saved fedavg_iter_gradients.pt file.",
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="the FedAvg round whose saved gradient snapshot will be attacked.",
    )
    parser.add_argument(
        "--client_id",
        type=int,
        required=True,
        help="the client id whose saved gradient snapshot will be attacked.",
    )
    parser.add_argument(
        "--ratio_pairs",
        nargs="+",
        default=DEFAULT_RATIO_PAIRS,
        help="space-separated ratio pairs in 'front:back' format, e.g. 0.95:0.95 0.99:0.91",
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


def load_snapshot(args):
    pt_path = Path(args.pt_file).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"Gradient artifact not found: {pt_path}")

    try:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(pt_path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError("Loaded artifact is not a dictionary.")
    if "gradient_snapshots" not in payload or not isinstance(payload["gradient_snapshots"], dict):
        raise KeyError("Artifact does not contain a valid 'gradient_snapshots' dictionary.")

    available_rounds = sorted(int(round_key) for round_key in payload["gradient_snapshots"].keys())
    round_snapshots = payload["gradient_snapshots"].get(args.round)
    if round_snapshots is None:
        round_snapshots = payload["gradient_snapshots"].get(str(args.round))
    if not isinstance(round_snapshots, dict):
        raise ValueError(f"round={args.round} was not found. Available rounds: {available_rounds}")

    available_clients = sorted(int(client_key) for client_key in round_snapshots.keys())
    snapshot = round_snapshots.get(args.client_id)
    if snapshot is None:
        snapshot = round_snapshots.get(str(args.client_id))
    if not isinstance(snapshot, dict):
        raise ValueError(
            f"client_id={args.client_id} was not found in round={args.round}. "
            f"Available clients for round {args.round}: {available_clients}"
        )

    return pt_path, snapshot


def prepare_ground_truth(snapshot, device):
    if isinstance(snapshot.get("sample_index"), (list, tuple)):
        raise ValueError("This script only supports single-sample snapshots, but sample_index is not scalar.")
    if isinstance(snapshot.get("label"), (list, tuple)):
        raise ValueError("This script only supports single-sample snapshots, but label is not scalar.")

    input_shape = tuple(int(dim) for dim in snapshot.get("input_shape", (1, 3, 32, 32)))
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError(
            f"This script only supports single-sample snapshots, but input_shape={input_shape}."
        )

    dst = datasets.CIFAR10("~/.torch", download=True)
    tp = transforms.ToTensor()

    img_index = int(snapshot["sample_index"])
    artifact_label = int(snapshot["label"])
    gt_data = tp(dst[img_index][0]).to(device)
    gt_label = torch.tensor([dst[img_index][1]], dtype=torch.long, device=device)
    if int(gt_label.item()) != artifact_label:
        raise ValueError(
            f"Snapshot label mismatch: artifact label={artifact_label}, dataset label={int(gt_label.item())}"
        )

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = gt_label.view(1,)
    gt_onehot_label = label_to_onehot(gt_label, num_classes=10)
    return gt_data, gt_label, gt_onehot_label, img_index, artifact_label, input_shape


def load_model_and_gradients(snapshot, device):
    net = LeNet().to(device)
    model_state_dict = snapshot.get("model_state_dict_before_step")
    if not isinstance(model_state_dict, dict):
        raise KeyError("Snapshot does not contain 'model_state_dict_before_step'.")
    net.load_state_dict(model_state_dict, strict=True)

    if "named_grads" not in snapshot or not isinstance(snapshot["named_grads"], dict):
        raise KeyError("Snapshot does not contain a valid 'named_grads' dictionary.")

    param_names = []
    original_dy_dx = []
    missing_names = []
    for name, _ in net.named_parameters():
        param_names.append(name)
        if name not in snapshot["named_grads"]:
            missing_names.append(name)
            continue
        original_dy_dx.append(snapshot["named_grads"][name].detach().clone().to(device))

    if missing_names:
        raise KeyError(f"Snapshot gradients are missing model parameters: {missing_names}")

    return net, param_names, original_dy_dx


def run_single_ratio(net, gt_data, gt_onehot_label, base_gradients, param_names, front_ratio, back_ratio, ratio_dir, device):
    criterion = cross_entropy_for_onehot
    tp_to_pil = transforms.ToPILImage()
    label = f"front={front_ratio:.2f}, back={back_ratio:.2f}"
    print(f"\n[Run] {label}")

    target_gradients, sparse_stats = sparsify_gradients_layerwise_topk(
        [grad.detach().clone() for grad in base_gradients],
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
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    dummy_data = torch.randn(gt_data.size(), device=device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size(), device=device).requires_grad_(True)
    tp_to_pil(dummy_data[0].detach().cpu()).save(ratio_dir / "dummy_init.png")

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    history = []
    loss_history = []
    num_iters = 400

    for iters in range(num_iters):

        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, target_gradients):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            current_loss_value = float(current_loss.item())
            print(f"  iter={iters:3d} | loss={current_loss_value:.4f}")
            loss_history.append({"iter": int(iters), "grad_loss": current_loss_value})
            history.append(tensor_to_image_array(dummy_data[0]))

    tp_to_pil(dummy_data[0].detach().cpu()).save(ratio_dir / "dlg_recon.png")

    with torch.no_grad():
        pred_class = int(net(gt_data).argmax(dim=1).item())
        dummy_class = int(F.softmax(dummy_label, dim=-1).argmax(dim=1).item())

    with open(ratio_dir / "loss_history.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ratio_label": label,
                "front_ratio": front_ratio,
                "back_ratio": back_ratio,
                "loss_history": loss_history,
                "sparse_stats": sparse_stats,
            },
            fh,
            indent=2,
        )

    with open(ratio_dir / "loss_history.txt", "w", encoding="utf-8") as fh:
        fh.write("iter\tgrad_loss\n")
        for row in loss_history:
            fh.write(f"{row['iter']}\t{row['grad_loss']:.6f}\n")

    with open(ratio_dir / "dlg_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"ratio_label: {label}\n")
        fh.write(f"front_ratio: {front_ratio}\n")
        fh.write(f"back_ratio: {back_ratio}\n")
        fh.write(f"pred_class: {pred_class}\n")
        fh.write(f"dummy_class: {dummy_class}\n")
        fh.write(f"iters: {num_iters}\n")
        if loss_history:
            fh.write(f"final_grad_loss: {loss_history[-1]['grad_loss']:.6f}\n")
        fh.write(
            "total_kept: "
            f"{sparse_stats['kept']}/{sparse_stats['total']} "
            f"({sparse_stats['retention_ratio'] * 100:.2f}%)\n"
        )
        for layer_stat in sparse_stats["layer_stats"]:
            fh.write(
                f"{layer_stat['layer']}: "
                f"{layer_stat['kept']}/{layer_stat['total']} "
                f"({layer_stat['retention_ratio'] * 100:.2f}%)\n"
            )

    plt.figure(figsize=(12, 8))
    for idx, image_array in enumerate(history):
        plt.subplot(4, 10, idx + 1)
        plt.imshow(image_array)
        plt.title(f"iter={idx * 10}")
        plt.axis("off")
    plt.savefig(ratio_dir / "dlg_progress_grid.png", bbox_inches="tight")
    plt.close()

    return {
        "label": label,
        "front_ratio": front_ratio,
        "back_ratio": back_ratio,
        "iterations": [entry["iter"] for entry in loss_history],
        "losses": [entry["grad_loss"] for entry in loss_history],
        "final_loss": loss_history[-1]["grad_loss"] if loss_history else None,
        "sparse_stats": sparse_stats,
        "final_reconstruction": tensor_to_image_array(dummy_data[0]),
    }


def save_loss_comparison(results, out_path):
    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(results):
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        plt.plot(
            result["iterations"],
            result["losses"],
            label=result["label"],
            color=color,
            linewidth=2,
        )

    plt.title("FedAvg DLG Loss Comparison by Layer-wise Sparsification Ratio")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Matching Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_loss_comparison_txt(results, out_path):
    all_iterations = sorted(
        {int(iteration) for result in results for iteration in result["iterations"]}
    )
    loss_by_label = {}
    for result in results:
        loss_by_label[result["label"]] = {
            int(iteration): float(loss)
            for iteration, loss in zip(result["iterations"], result["losses"])
        }

    with out_path.open("w", encoding="utf-8") as fh:
        header = ["iter"] + [result["label"] for result in results]
        fh.write("\t".join(header) + "\n")
        for iteration in all_iterations:
            row = [str(iteration)]
            for result in results:
                loss_value = loss_by_label[result["label"]].get(iteration)
                row.append("" if loss_value is None else f"{loss_value:.10f}")
            fh.write("\t".join(row) + "\n")


def save_final_reconstruction_grid(gt_data, results, out_path):
    cols = len(results) + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]

    axes[0].imshow(tensor_to_image_array(gt_data[0]))
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    for idx, result in enumerate(results, start=1):
        color = LINE_COLORS[(idx - 1) % len(LINE_COLORS)]
        axes[idx].imshow(result["final_reconstruction"])
        axes[idx].set_title(
            f"{result['label']}\nloss={result['final_loss']:.4f}",
            color=color,
            fontsize=10,
        )
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_files(results, output_dir, pt_path, args, img_index, artifact_label, input_shape):
    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"

    payload = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pt_file": str(pt_path),
        "round": int(args.round),
        "client_id": int(args.client_id),
        "sample_index": img_index,
        "label": artifact_label,
        "input_shape": list(input_shape),
        "results": [
            {
                "label": result["label"],
                "front_ratio": result["front_ratio"],
                "back_ratio": result["back_ratio"],
                "iterations": result["iterations"],
                "losses": result["losses"],
                "final_loss": result["final_loss"],
                "sparse_stats": result["sparse_stats"],
            }
            for result in results
        ],
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"torch: {torch.__version__}",
        f"torchvision: {torchvision.__version__}",
        f"pt_file: {pt_path}",
        f"round: {args.round}",
        f"client_id: {args.client_id}",
        f"sample_index: {img_index}",
        f"artifact_label: {artifact_label}",
        f"input_shape: {input_shape}",
        "",
    ]
    for result in payload["results"]:
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

    pt_path, snapshot = load_snapshot(args)
    gt_data, gt_label, gt_onehot_label, img_index, artifact_label, input_shape = prepare_ground_truth(
        snapshot, device
    )
    net, param_names, original_dy_dx = load_model_and_gradients(snapshot, device)

    output_dir = DEFAULT_OUTPUT_DIR / f"round_{args.round}" / f"client_{args.client_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms.ToPILImage()(gt_data[0].detach().cpu()).save(output_dir / "gt.png")

    results = []
    for front_ratio, back_ratio in args.ratio_pairs:
        ratio_dir = output_dir / f"front_{front_ratio:.2f}_back_{back_ratio:.2f}"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        result = run_single_ratio(
            net=net,
            gt_data=gt_data,
            gt_onehot_label=gt_onehot_label,
            base_gradients=original_dy_dx,
            param_names=param_names,
            front_ratio=front_ratio,
            back_ratio=back_ratio,
            ratio_dir=ratio_dir,
            device=device,
        )
        results.append(result)

    loss_path = output_dir / "loss_comparison.png"
    loss_txt_path = output_dir / "loss_comparison.txt"
    recon_grid_path = output_dir / "final_reconstruction_grid.png"
    summary_json_path, summary_txt_path = write_summary_files(
        results,
        output_dir,
        pt_path,
        args,
        img_index,
        artifact_label,
        input_shape,
    )

    save_loss_comparison(results, loss_path)
    write_loss_comparison_txt(results, loss_txt_path)
    save_final_reconstruction_grid(gt_data, results, recon_grid_path)

    with open(output_dir / "run_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"pt_file: {pt_path}\n")
        fh.write(f"round: {args.round}\n")
        fh.write(f"client_id: {args.client_id}\n")
        fh.write(f"sample_index: {img_index}\n")
        fh.write(f"artifact_label: {artifact_label}\n")
        fh.write(f"gt_label: {int(gt_label.item())}\n")
        fh.write(f"input_shape: {input_shape}\n")
        fh.write(f"ratio_pairs: {args.ratio_pairs}\n")
        fh.write(f"summary_json: {summary_json_path}\n")
        fh.write(f"summary_txt: {summary_txt_path}\n")
        fh.write(f"loss_comparison: {loss_path}\n")
        fh.write(f"loss_comparison_txt: {loss_txt_path}\n")
        fh.write(f"final_reconstruction_grid: {recon_grid_path}\n")

    print("\nSaved files:")
    for output_path in [loss_path, loss_txt_path, recon_grid_path, summary_json_path, summary_txt_path]:
        print(f"  - {output_path}")


if __name__ == "__main__":
    main()
