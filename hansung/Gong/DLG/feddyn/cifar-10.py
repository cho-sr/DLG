from torchvision import datasets, transforms

tfm = transforms.ToTensor()
datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
print("done")
