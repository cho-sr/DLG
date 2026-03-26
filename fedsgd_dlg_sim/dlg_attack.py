import copy
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


GradientList = List[torch.Tensor]
GradientMatchingLoss = Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], torch.Tensor]


def cross_entropy_for_onehot(predictions: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that accepts soft one-hot-like label distributions."""
    log_probs = F.log_softmax(predictions, dim=-1)
    return torch.mean(torch.sum(-soft_targets * log_probs, dim=-1))


def gradient_l2_loss(
    dummy_gradients: Sequence[torch.Tensor],
    target_gradients: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Pure L2 gradient matching loss kept separate for easy future swaps."""
    if len(dummy_gradients) != len(target_gradients):
        raise ValueError("Gradient lists must have the same length.")

    loss = torch.tensor(0.0, device=dummy_gradients[0].device)
    for dummy_grad, target_grad in zip(dummy_gradients, target_gradients):
        loss = loss + torch.sum((dummy_grad - target_grad) ** 2)
    return loss


def _compute_gradient_matching_objective(
    model: nn.Module,
    dummy_image: torch.Tensor,
    dummy_label_logits: torch.Tensor,
    target_gradients: Sequence[torch.Tensor],
    matching_loss_fn: GradientMatchingLoss,
    create_graph: bool,
) -> torch.Tensor:
    predictions = model(dummy_image)
    soft_labels = F.softmax(dummy_label_logits, dim=-1)
    classification_loss = cross_entropy_for_onehot(predictions, soft_labels)
    dummy_gradients = torch.autograd.grad(
        classification_loss,
        tuple(model.parameters()),
        create_graph=create_graph,
    )
    return matching_loss_fn(dummy_gradients, target_gradients)


def run_dlg(
    model: nn.Module,
    target_gradients: Sequence[torch.Tensor],
    device: torch.device,
    image_shape: Tuple[int, int, int, int],
    num_classes: int = 10,
    num_iterations: int = 300,
    lbfgs_lr: float = 1.0,
    matching_loss_fn: GradientMatchingLoss = gradient_l2_loss,
) -> Dict[str, object]:
    """
    Reconstruct a private example from leaked gradients using the original DLG
    setup: soft labels, L-BFGS, and pure L2 gradient matching.
    """
    reference_model = copy.deepcopy(model).to(device)
    reference_model.eval()

    leaked_gradients = [grad.detach().clone().to(device) for grad in target_gradients]

    dummy_image = torch.randn(image_shape, device=device, requires_grad=True)
    dummy_label_logits = torch.randn((image_shape[0], num_classes), device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [dummy_image, dummy_label_logits],
        lr=lbfgs_lr,
        max_iter=1,
        line_search_fn="strong_wolfe",
    )

    loss_history: List[float] = []

    for _ in range(num_iterations):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            matching_loss = _compute_gradient_matching_objective(
                model=reference_model,
                dummy_image=dummy_image,
                dummy_label_logits=dummy_label_logits,
                target_gradients=leaked_gradients,
                matching_loss_fn=matching_loss_fn,
                create_graph=True,
            )
            matching_loss.backward()
            return matching_loss

        optimizer.step(closure)

        with torch.no_grad():
            dummy_image.clamp_(0.0, 1.0)

        current_loss = _compute_gradient_matching_objective(
            model=reference_model,
            dummy_image=dummy_image,
            dummy_label_logits=dummy_label_logits,
            target_gradients=leaked_gradients,
            matching_loss_fn=matching_loss_fn,
            create_graph=False,
        )
        loss_history.append(float(current_loss.item()))

    reconstructed_labels = F.softmax(dummy_label_logits.detach(), dim=-1)

    return {
        "reconstructed_image": dummy_image.detach().cpu(),
        "reconstructed_label_distribution": reconstructed_labels.detach().cpu(),
        "loss_history": loss_history,
    }
