from pathlib import Path

import torch

from matplotlib import pyplot as plt


def plot():
    cache_dir = Path(__file__).resolve().parent.parent / 'cache'
    accuracy_path = cache_dir / 'accuracy.pkl'
    accuracy = torch.load(accuracy_path, map_location='cpu')
    plt.plot([e for e in range(1, len(accuracy) + 1)], accuracy, label='FedSGD')

    plt.title("Test Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.ylim(0, 1)
    plt.xlim(1, len(accuracy))
    plt.legend(loc=4)

    output_path = cache_dir / 'accuracy.png'
    plt.savefig(output_path)
    print('Saved accuracy plot to {}'.format(output_path))
