import copy
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


GradientList = List[torch.Tensor]
Batch = Tuple[torch.Tensor, torch.Tensor]


def client_compute_gradient(
    global_model: nn.Module,
    batch: Batch,
    device: torch.device,
) -> Tuple[GradientList, float]:
    """
    Simulate a client receiving the global model, computing one local gradient,
    and returning the raw gradients without any real communication layer.
    """
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    criterion = nn.CrossEntropyLoss()
    logits = local_model(images)
    loss = criterion(logits, labels)

    # These are the exact gradients that would leak from the victim client.
    gradients = torch.autograd.grad(loss, local_model.parameters())
    detached_gradients = [grad.detach().clone() for grad in gradients]

    return detached_gradients, float(loss.item())


def aggregate_gradients(client_gradients: Sequence[GradientList]) -> GradientList:
    """Average raw client gradients parameter-wise for FedSGD."""
    if not client_gradients:
        raise ValueError("No client gradients were provided for aggregation.")

    num_clients = len(client_gradients)
    aggregated: GradientList = []

    for grads_for_param in zip(*client_gradients):
        mean_grad = torch.stack(grads_for_param, dim=0).mean(dim=0)
        aggregated.append(mean_grad)

    if num_clients < 2:
        raise ValueError("FedSGD aggregation expects gradients from at least 2 clients.")

    return aggregated


def server_update_model(
    global_model: nn.Module,
    averaged_gradients: GradientList,
    server_lr: float,
) -> None:
    """Apply a plain SGD-style update with no momentum to the global model."""
    with torch.no_grad():
        for parameter, grad in zip(global_model.parameters(), averaged_gradients):
            parameter.add_(grad, alpha=-server_lr)


def fedsgd_round(
    global_model: nn.Module,
    client_batches: Sequence[Batch],
    server_lr: float,
    device: torch.device,
    victim_client_id: int = 0,
) -> Dict[str, object]:
    """
    Run one synchronous FedSGD round:
    1. each client gets the same global model state
    2. each client computes local gradients on its own batch
    3. the server averages returned gradients
    4. the server applies one global SGD update
    """
    if len(client_batches) < 2:
        raise ValueError("Run FedSGD with at least 2 simulated clients.")
    if not 0 <= victim_client_id < len(client_batches):
        raise IndexError("victim_client_id is out of range.")

    all_client_gradients: List[GradientList] = []
    client_losses: List[float] = []
    victim_gradients: GradientList | None = None

    for client_id, batch in enumerate(client_batches):
        client_gradients, client_loss = client_compute_gradient(global_model, batch, device)
        all_client_gradients.append(client_gradients)
        client_losses.append(client_loss)

        if client_id == victim_client_id:
            victim_gradients = [grad.clone() for grad in client_gradients]

    if victim_gradients is None:
        raise RuntimeError("Victim gradients were not captured during the FedSGD round.")

    averaged_gradients = aggregate_gradients(all_client_gradients)
    server_update_model(global_model, averaged_gradients, server_lr)

    return {
        "client_gradients": all_client_gradients,
        "client_losses": client_losses,
        "averaged_gradients": averaged_gradients,
        "victim_gradients": victim_gradients,
    }
