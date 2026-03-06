# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str,default="",
                    help='the path to customized image.')
parser.add_argument('--pt_file', type=str, default="",
                    help='path to .pt gradient record (uses payload["gradients"]).')
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Running on %s" % device)

dst = datasets.CIFAR10("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)


gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
num_classes = len(dst.classes)
gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)

plt.imshow(tt(gt_data[0].cpu()))

class Network1(nn.Module):
    def __init__(self, num_classes=10):
        super(Network1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

from models.vision import weights_init
net = Network1(num_classes=num_classes).to(device)


torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute target gradient
if len(args.pt_file) > 0:
    payload = torch.load(args.pt_file, map_location="cpu", weights_only=False)
    if "gradients" in payload:
        source_grads = payload["gradients"]
    elif "grads" in payload:
        source_grads = list(payload["grads"].values())
    elif "attack" in payload and isinstance(payload["attack"], dict) and "gradients" in payload["attack"]:
        source_grads = payload["attack"]["gradients"]
    else:
        raise KeyError("No gradients found in .pt (checked: gradients, grads, attack.gradients)")

    original_dy_dx = [g.detach().clone().to(device) for g in source_grads]
    if "model_state" in payload:
        net.load_state_dict(payload["model_state"], strict=False)
    if "input_shape" in payload:
        gt_data = torch.randn(tuple(payload["input_shape"]), device=device)
    if "num_classes" in payload:
        num_classes = int(payload["num_classes"])
        gt_onehot_label = torch.randn((gt_data.size(0), num_classes), device=device)
else:
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


history = []
for iters in range(300):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    if iters % 10 == 0: 
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
