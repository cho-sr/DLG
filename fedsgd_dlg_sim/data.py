from typing import List, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


def get_cifar10_datasets(
    data_root: str = "./fedsgd_dlg_sim/data",
    download: bool = True,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 with a minimal transform so images stay in [0, 1]."""
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
        transform=transform,
    )
    return train_dataset, test_dataset


def partition_dataset_among_clients(
    dataset: Dataset,
    num_clients: int,
    seed: int = 0,
) -> List[Subset]:
    """Split the dataset into deterministic client subsets with near-equal sizes."""
    if num_clients < 2:
        raise ValueError("FedSGD simulation requires at least 2 clients.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()

    partitions: List[Subset] = []
    base_size = len(dataset) // num_clients
    remainder = len(dataset) % num_clients
    start = 0

    for client_id in range(num_clients):
        subset_size = base_size + (1 if client_id < remainder else 0)
        subset_indices = shuffled_indices[start : start + subset_size]
        partitions.append(Subset(dataset, subset_indices))
        start += subset_size

    return partitions
