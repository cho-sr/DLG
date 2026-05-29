from __future__ import annotations

import random
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class SyntheticImageDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        channels: int,
        image_size: int,
        seed: int,
        prototypes: torch.Tensor | None = None,
    ):
        generator = torch.Generator()
        generator.manual_seed(seed)

        if prototypes is None:
            prototypes = torch.randn(
                num_classes,
                channels,
                image_size,
                image_size,
                generator=generator,
            )
        labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
        noise = 0.35 * torch.randn(
            num_samples,
            channels,
            image_size,
            image_size,
            generator=generator,
        )

        images = prototypes[labels] + noise
        self.data = (images - images.mean()) / images.std().clamp_min(1e-6)
        self.targets = labels.tolist()
        self.classes = [str(i) for i in range(num_classes)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.data[index], int(self.targets[index])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_subset(dataset, labels: np.ndarray, limit: int, seed: int):
    if limit <= 0 or limit >= len(labels):
        return dataset, labels

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(labels), size=limit, replace=False)
    indices = np.sort(indices)
    return Subset(dataset, indices.tolist()), labels[indices]


def load_datasets(
    dataset_name: str,
    data_dir: str,
    train_samples: int,
    test_samples: int,
    seed: int,
):
    dataset_name = dataset_name.lower()

    if dataset_name in {"fake", "synthetic"}:
        num_classes = 10
        train_count = train_samples if train_samples > 0 else 10000
        test_count = test_samples if test_samples > 0 else 2000
        generator = torch.Generator()
        generator.manual_seed(seed)
        prototypes = torch.randn(num_classes, 1, 28, 28, generator=generator)
        train_dataset = SyntheticImageDataset(
            train_count,
            num_classes,
            1,
            28,
            seed,
            prototypes=prototypes,
        )
        test_dataset = SyntheticImageDataset(
            test_count,
            num_classes,
            1,
            28,
            seed + 1,
            prototypes=prototypes,
        )
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "train_labels": np.asarray(train_dataset.targets, dtype=np.int64),
            "num_classes": num_classes,
            "input_channels": 1,
            "image_size": 28,
            "class_names": train_dataset.classes,
        }

    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        train_labels = np.asarray(train_dataset.targets, dtype=np.int64)
        test_labels = np.asarray(test_dataset.targets, dtype=np.int64)
        train_dataset, train_labels = _maybe_subset(train_dataset, train_labels, train_samples, seed)
        test_dataset, _ = _maybe_subset(test_dataset, test_labels, test_samples, seed + 1)
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "train_labels": train_labels,
            "num_classes": 10,
            "input_channels": 1,
            "image_size": 28,
            "class_names": [str(i) for i in range(10)],
        }

    if dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_labels = np.asarray(train_dataset.targets, dtype=np.int64)
        test_labels = np.asarray(test_dataset.targets, dtype=np.int64)
        train_dataset, train_labels = _maybe_subset(train_dataset, train_labels, train_samples, seed)
        test_dataset, _ = _maybe_subset(test_dataset, test_labels, test_samples, seed + 1)
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "train_labels": train_labels,
            "num_classes": 10,
            "input_channels": 3,
            "image_size": 32,
            "class_names": (
                train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes
            ),
        }

    raise ValueError("dataset_name must be one of: fake, mnist, cifar10.")


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 1,
) -> List[List[int]]:
    if alpha <= 0:
        raise ValueError("alpha must be positive. Smaller alpha means stronger non-IID.")

    labels = np.asarray(labels, dtype=np.int64)
    num_classes = int(labels.max()) + 1
    min_size = min(min_size, max(1, len(labels) // max(1, num_clients)))

    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

            for client_id, split in enumerate(np.split(class_indices, split_points)):
                client_indices[client_id].extend(split.tolist())

        lengths = [len(indices) for indices in client_indices]
        if min(lengths) >= min_size:
            for indices in client_indices:
                rng.shuffle(indices)
            return client_indices

    raise RuntimeError(
        "Could not create a non-empty Dirichlet partition. "
        "Try a larger alpha, more training samples, or fewer clients."
    )


def partition_summary(
    partitions: Sequence[Sequence[int]],
    labels: Sequence[int],
    num_classes: int,
) -> List[Dict[str, object]]:
    labels = np.asarray(labels, dtype=np.int64)
    rows: List[Dict[str, object]] = []

    for client_id, indices in enumerate(partitions):
        hist = np.bincount(labels[list(indices)], minlength=num_classes)
        rows.append(
            {
                "client_id": client_id,
                "num_samples": int(len(indices)),
                "label_histogram": hist.astype(int).tolist(),
            }
        )

    return rows


def make_client_loader(
    train_dataset,
    indices: Sequence[int],
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    subset = Subset(train_dataset, list(indices))
    return DataLoader(subset, batch_size=batch_size, shuffle=True, generator=generator)


def make_test_loader(test_dataset, batch_size: int) -> DataLoader:
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
