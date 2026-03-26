import argparse
import copy
import json
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from data import get_cifar10_datasets, partition_dataset_among_clients
from dlg_attack import run_dlg
from fedsgd_sim import fedsgd_round
from model import get_lenet


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_one_batch_per_client(
    client_subsets: List[Subset],
    victim_client_id: int,
    default_batch_size: int,
    loader_seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Draw one local batch per client for a single FedSGD round."""
    client_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for client_id, client_subset in enumerate(client_subsets):
        batch_size = 1 if client_id == victim_client_id else default_batch_size
        generator = torch.Generator().manual_seed(loader_seed + client_id)
        loader = DataLoader(
            client_subset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        images, labels = next(iter(loader))
        client_batches.append((images, labels))

    return client_batches


def _save_outputs(
    output_dir: str,
    ground_truth_image: torch.Tensor,
    reconstructed_image: torch.Tensor,
    loss_history: List[float],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    ground_truth_path = os.path.join(output_dir, "ground_truth.png")
    reconstruction_path = os.path.join(output_dir, "reconstructed.png")
    comparison_path = os.path.join(output_dir, "comparison.png")
    history_json_path = os.path.join(output_dir, "dlg_loss_history.json")
    history_pt_path = os.path.join(output_dir, "dlg_loss_history.pt")

    save_image(ground_truth_image, ground_truth_path)
    save_image(reconstructed_image, reconstruction_path)
    save_image(torch.cat([ground_truth_image, reconstructed_image], dim=0), comparison_path, nrow=2)

    with open(history_json_path, "w", encoding="utf-8") as handle:
        json.dump(loss_history, handle, indent=2)

    torch.save(torch.tensor(loss_history, dtype=torch.float32), history_pt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-process FedSGD + DLG simulation on CIFAR-10.")
    parser.add_argument("--data-root", type=str, default="./fedsgd_dlg_sim/data", help="Where CIFAR-10 is stored.")
    parser.add_argument("--output-dir", type=str, default="./fedsgd_dlg_sim/outputs", help="Where to save outputs.")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of simulated clients.")
    parser.add_argument("--victim-client-id", type=int, default=0, help="Client used for DLG reconstruction.")
    parser.add_argument(
        "--client-batch-size",
        type=int,
        default=1,
        help="Batch size for non-victim clients. The victim is forced to batch size 1.",
    )
    parser.add_argument("--server-lr", type=float, default=0.1, help="FedSGD server learning rate.")
    parser.add_argument("--dlg-iters", type=int, default=300, help="Number of L-BFGS outer iterations for DLG.")
    parser.add_argument("--dlg-lr", type=float, default=1.0, help="L-BFGS learning rate for DLG.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not already present.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, _ = get_cifar10_datasets(data_root=args.data_root, download=args.download)
    client_subsets = partition_dataset_among_clients(
        dataset=train_dataset,
        num_clients=args.num_clients,
        seed=args.seed,
    )

    client_batches = _select_one_batch_per_client(
        client_subsets=client_subsets,
        victim_client_id=args.victim_client_id,
        default_batch_size=args.client_batch_size,
        loader_seed=args.seed,
    )

    victim_images, victim_labels = client_batches[args.victim_client_id]

    global_model = get_lenet().to(device)

    # This frozen copy is the exact model state seen by every client in the round.
    attack_model = copy.deepcopy(global_model).to(device)

    round_result = fedsgd_round(
        global_model=global_model,
        client_batches=client_batches,
        server_lr=args.server_lr,
        device=device,
        victim_client_id=args.victim_client_id,
    )

    dlg_result = run_dlg(
        model=attack_model,
        target_gradients=round_result["victim_gradients"],
        device=device,
        image_shape=tuple(victim_images.shape),
        num_classes=10,
        num_iterations=args.dlg_iters,
        lbfgs_lr=args.dlg_lr,
    )

    reconstructed_image = dlg_result["reconstructed_image"]
    loss_history = dlg_result["loss_history"]
    reconstructed_label_distribution = dlg_result["reconstructed_label_distribution"]

    _save_outputs(
        output_dir=args.output_dir,
        ground_truth_image=victim_images.cpu(),
        reconstructed_image=reconstructed_image,
        loss_history=loss_history,
    )

    print(f"Client losses: {round_result['client_losses']}")
    print(f"Victim label: {int(victim_labels.item())}")
    print(f"Reconstructed label guess: {int(torch.argmax(reconstructed_label_distribution, dim=-1).item())}")
    print(f"Initial DLG loss: {loss_history[0]:.6f}")
    print(f"Final DLG loss: {loss_history[-1]:.6f}")
    print(f"Saved outputs to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
