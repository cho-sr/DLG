# -*- coding: utf-8 -*-
import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct an image from the .pt gradient artifact saved by 0326.py."
    )
    parser.add_argument(
        "--gradient-path",
        type=str,
        required=True,
        help="Path to the .pt file saved by 0326.py with --save-dlg.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of outer optimization steps for the DLG attack.",
    )
    parser.add_argument(
        "--history-every",
        type=int,
        default=10,
        help="Save and log one reconstruction snapshot every N steps.",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.0,
        help="Optional total-variation regularization weight.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for dummy image and dummy label initialization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device used for the attack.",
    )
    parser.add_argument(
        "--optimize-label",
        action="store_true",
        help="Optimize a dummy label instead of using the label stored in the .pt artifact.",
    )
    parser.add_argument(
        "--model-init-seed",
        type=int,
        default=None,
        help=(
            "Fallback seed used only when the .pt artifact does not contain model_state_dict. "
            "Useful mainly for old round-1 artifacts."
        ),
    )
    parser.add_argument(
        "--save-figure",
        type=str,
        default="",
        help="Optional output path for the reconstruction progress figure.",
    )
    parser.add_argument(
        "--save-final-image",
        type=str,
        default="",
        help="Optional output path for the final reconstructed image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a matplotlib window. Useful in headless environments.",
    )
    return parser


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.history_every < 1:
        raise ValueError("--history-every must be at least 1.")
    if args.tv_weight < 0:
        raise ValueError("--tv-weight must be non-negative.")


def load_0326_module(script_dir: Path):
    module_path = script_dir / "0326.py"
    spec = importlib.util.spec_from_file_location("fedsgd_0326_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_gradient_artifact(path: Path) -> Dict[str, object]:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise ValueError("Loaded artifact is not a dictionary.")
    if "named_grads" not in artifact:
        raise ValueError("Artifact does not contain 'named_grads'.")
    if "label" not in artifact:
        raise ValueError("Artifact does not contain 'label'.")
    return artifact


def build_model(fedsgd_module, artifact: Dict[str, object], device: torch.device, model_init_seed: int | None):
    model_name = artifact.get("model_name")
    if model_name is not None and model_name != "SimpleCifarCNN":
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            "This attack script currently supports only SimpleCifarCNN artifacts from 0326.py."
        )

    model = fedsgd_module.SimpleCifarCNN().to(device)

    model_state_dict = artifact.get("model_state_dict")
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    else:
        if model_init_seed is None:
            raise ValueError(
                "This .pt artifact does not contain model_state_dict. "
                "Please regenerate it with the updated 0326.py, or pass --model-init-seed "
                "only if this was captured before any server update."
            )
        torch.manual_seed(model_init_seed)
        model = fedsgd_module.SimpleCifarCNN().to(device)

    model.eval()
    return model


def get_normalization(artifact: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
    normalization = artifact.get("normalization")
    if isinstance(normalization, dict):
        mean = normalization.get("mean", (0.4914, 0.4822, 0.4465))
        std = normalization.get("std", (0.2023, 0.1994, 0.2010))
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return mean_tensor, std_tensor


def get_input_shape(artifact: Dict[str, object]) -> Tuple[int, int, int, int]:
    sample_shape = artifact.get("sample_shape")
    if isinstance(sample_shape, tuple) and len(sample_shape) == 4:
        return sample_shape
    if isinstance(sample_shape, tuple) and len(sample_shape) == 3:
        return (1, *sample_shape)
    return (1, 3, 32, 32)


def cross_entropy_for_soft_labels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), dim=1))


def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def denormalize_image(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu()
    mean = mean.detach().cpu()
    std = std.detach().cpu()
    return (image * std + mean).clamp(0.0, 1.0)


def reconstruct(
    model: nn.Module,
    artifact: Dict[str, object],
    device: torch.device,
    steps: int,
    history_every: int,
    tv_weight: float,
    seed: int,
    optimize_label: bool,
) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]], List[Tuple[int, float]], int | None]:
    torch.manual_seed(seed)

    mean, std = get_normalization(artifact)
    mean_device = mean.to(device)
    std_device = std.to(device)
    lower_bound = (torch.zeros_like(mean_device) - mean_device) / std_device
    upper_bound = (torch.ones_like(mean_device) - mean_device) / std_device

    input_shape = get_input_shape(artifact)
    target_grads = {
        name: grad_tensor.to(device)
        for name, grad_tensor in artifact["named_grads"].items()
    }
    target_label = torch.tensor([int(artifact["label"])], dtype=torch.long, device=device)

    expected_names = [name for name, _ in model.named_parameters()]
    missing_names = [name for name in expected_names if name not in target_grads]
    if missing_names:
        raise ValueError(f"Artifact gradients are missing model parameters: {missing_names}")

    dummy_data = torch.randn(input_shape, device=device, requires_grad=True)
    parameters_to_optimize = [dummy_data]
    dummy_label_logits = None

    if optimize_label:
        dummy_label_logits = torch.randn((1, 10), device=device, requires_grad=True)
        parameters_to_optimize.append(dummy_label_logits)

    optimizer = torch.optim.LBFGS(parameters_to_optimize, lr=1.0, max_iter=1)
    history: List[Tuple[int, torch.Tensor]] = []
    loss_history: List[Tuple[int, float]] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad()

        with torch.no_grad():
            dummy_data.clamp_(lower_bound, upper_bound)

        dummy_pred = model(dummy_data)
        if optimize_label:
            assert dummy_label_logits is not None
            dummy_label = F.softmax(dummy_label_logits, dim=-1)
            dummy_loss = cross_entropy_for_soft_labels(dummy_pred, dummy_label)
        else:
            dummy_loss = nn.CrossEntropyLoss()(dummy_pred, target_label)

        dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        grad_diff = torch.zeros((), device=device)
        for (name, _), dummy_grad in zip(model.named_parameters(), dummy_grads):
            grad_diff = grad_diff + ((dummy_grad - target_grads[name]) ** 2).sum()

        if tv_weight > 0:
            grad_diff = grad_diff + tv_weight * total_variation(dummy_data)

        grad_diff.backward()
        return grad_diff

    for step_idx in range(steps):
        optimizer.step(closure)

        should_log = (step_idx % history_every == 0) or (step_idx == steps - 1)
        if should_log:
            current_loss = closure().item()
            current_image = denormalize_image(dummy_data, mean, std).squeeze(0)
            history.append((step_idx, current_image))
            loss_history.append((step_idx, current_loss))
            print(f"step={step_idx:04d} grad_loss={current_loss:.6f}")

    inferred_label = None
    if optimize_label:
        assert dummy_label_logits is not None
        inferred_label = int(torch.argmax(F.softmax(dummy_label_logits, dim=-1), dim=1).item())

    final_image = denormalize_image(dummy_data, mean, std)
    return final_image, history, loss_history, inferred_label


def plot_history(history: List[Tuple[int, torch.Tensor]]):
    num_panels = len(history)
    if num_panels == 0:
        raise ValueError("No history snapshots were collected for plotting.")
    cols = min(5, num_panels)
    rows = math.ceil(num_panels / cols)
    figure = plt.figure(figsize=(3 * cols, 3 * rows))

    for index, (step_idx, image) in enumerate(history, start=1):
        axis = figure.add_subplot(rows, cols, index)
        axis.imshow(image.permute(1, 2, 0))
        axis.set_title(f"step={step_idx}")
        axis.axis("off")

    figure.tight_layout()
    return figure


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gradient_path = Path(args.gradient_path).expanduser().resolve()
    if not gradient_path.exists():
        raise FileNotFoundError(f"Gradient artifact not found: {gradient_path}")

    device = get_device(args.device)
    print(f"Running on {device}")
    print(f"Loading artifact from {gradient_path}")

    artifact = load_gradient_artifact(gradient_path)
    script_dir = Path(__file__).resolve().parent
    fedsgd_module = load_0326_module(script_dir)
    model = build_model(fedsgd_module, artifact, device, args.model_init_seed)

    final_image, history, _, inferred_label = reconstruct(
        model=model,
        artifact=artifact,
        device=device,
        steps=args.steps,
        history_every=args.history_every,
        tv_weight=args.tv_weight,
        seed=args.seed,
        optimize_label=args.optimize_label,
    )

    print(f"artifact_label={artifact['label']}")
    if inferred_label is not None:
        print(f"inferred_label={inferred_label}")

    figure = plot_history(history)
    if args.save_figure:
        figure_path = Path(args.save_figure).expanduser().resolve()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, bbox_inches="tight")
        print(f"Saved progress figure to {figure_path}")

    if args.save_final_image:
        final_image_path = Path(args.save_final_image).expanduser().resolve()
        final_image_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(final_image_path, final_image.squeeze(0).permute(1, 2, 0).numpy())
        print(f"Saved final reconstructed image to {final_image_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
