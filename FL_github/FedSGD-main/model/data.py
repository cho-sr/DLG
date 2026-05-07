from pathlib import Path

import torch

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


class loader(object):
    def __init__(self, cmd='cifar10', batch_size=64):
        self.cmd = cmd
        self.batch_size = batch_size
        self.dataset_root = Path(__file__).resolve().parent.parent / 'dataset'
        self.__load_dataset()
        self.__get_index()

    def __load_dataset(self):
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        if self.cmd == 'cifar10':
            cifar_root = self.dataset_root / 'cifar10_data'
            self.train_dataset = datasets.CIFAR10(
                str(cifar_root),
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            )
            self.test_dataset = datasets.CIFAR10(
                str(cifar_root),
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            )
        else:
            mnist_root = self.dataset_root / 'mnist_data'
            self.train_dataset = datasets.MNIST(
                str(mnist_root),
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )
            self.test_dataset = datasets.MNIST(
                str(mnist_root),
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )

    def __get_index(self):
        self.indices = [[], [], [], [], [], [], [], [], [], []]
        for index, data in enumerate(self.train_dataset):
            self.indices[data[1]].append(index)

    def get_loader(self, rank):
        dataset_indices = []
        difference = list(set(range(10)).difference(set(rank)))
        for i in difference:
            dataset_indices.extend(self.indices[i])

        dataset = torch.utils.data.Subset(self.train_dataset, dataset_indices)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader
