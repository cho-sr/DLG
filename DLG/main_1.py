# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from models.vision import LeNet, weights_init
from utils import label_to_onehot, cross_entropy_for_onehot

# 1. 환경 설정 및 데이터 준비
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dst = datasets.CIFAR10("~/.torch", download=True)
tp, tt = transforms.ToTensor(), transforms.ToPILImage()

img_index = 25
gt_data = tp(dst[img_index][0]).to(device).view(1, 3, 32, 32)
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device).view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

# 2. 모델 및 FedAvg 시뮬레이션
net = LeNet().to(device)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

original_weights = [p.detach().clone() for p in net.parameters()]

# FedAvg 로컬 학습
local_epochs, lr = 1, 0.01
optimizer_client = torch.optim.SGD(net.parameters(), lr=lr)
for _ in range(local_epochs):
    optimizer_client.zero_grad()
    loss = criterion(net(gt_data), gt_onehot_label)
    loss.backward()
    optimizer_client.step()

target_grad = []
for w_init, w_trained in zip(original_weights, net.parameters()):
    target_grad.append((w_init - w_trained).detach() / (lr * local_epochs))

with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)

# 3. DLG 공격 설정
torch.manual_seed(1234)
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
initial_dummy_img = tt(dummy_data[0].detach().clone().clamp(0, 1).cpu())

optimizer = torch.optim.LBFGS([dummy_data], lr=0.1)
history = []

# 4. 공격 루프 (Gradient Matching)
print("Starting DLG Attack...")

for iters in range(101):
    def closure():
        optimizer.zero_grad()
        dummy_data.data.clamp_(0, 1)
        dummy_dy_dx = torch.autograd.grad(criterion(net(dummy_data), gt_onehot_label),
                                          net.parameters(), create_graph=True)
        grad_diff = sum(((gx - gy) ** 2).sum() for gx, gy in zip(dummy_dy_dx, target_grad))
        grad_diff.backward()
        return grad_diff


    optimizer.step(closure)

    # [차이점] 여기서는 closure()를 다시 호출하지 않고 시각화 데이터만 저장합니다.
    if iters % 10 == 0:
        history.append(tt(dummy_data[0].detach().clone().clamp(0, 1).cpu()))

# 5. 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(tt(gt_data[0].cpu()));
axes[0].set_title("1. Ground Truth");
axes[0].axis('off')
axes[1].imshow(initial_dummy_img);
axes[1].set_title("2. Initial Dummy");
axes[1].axis('off')
axes[2].imshow(history[-1]);
axes[2].set_title("3. Reconstructed Result");
axes[2].axis('off')
plt.show()# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from models.vision import LeNet, weights_init
from utils import label_to_onehot, cross_entropy_for_onehot

# 1. 환경 설정 및 데이터 준비
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dst = datasets.CIFAR10("~/.torch", download=True)
tp, tt = transforms.ToTensor(), transforms.ToPILImage()

img_index = 25
gt_data = tp(dst[img_index][0]).to(device).view(1, 3, 32, 32)
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device).view(1,)
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

# 2. 모델 및 FedAvg 시뮬레이션
net = LeNet().to(device)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

original_weights = [p.detach().clone() for p in net.parameters()]

# FedAvg 로컬 학습
local_epochs, lr = 1, 0.01
optimizer_client = torch.optim.SGD(net.parameters(), lr=lr)
for _ in range(local_epochs):
    optimizer_client.zero_grad()
    loss = criterion(net(gt_data), gt_onehot_label)
    loss.backward()
    optimizer_client.step()

target_grad = []
for w_init, w_trained in zip(original_weights, net.parameters()):
    target_grad.append((w_init - w_trained).detach() / (lr * local_epochs))

with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)

# 3. DLG 공격 설정
torch.manual_seed(1234)
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
initial_dummy_img = tt(dummy_data[0].detach().clone().clamp(0, 1).cpu())

optimizer = torch.optim.LBFGS([dummy_data], lr=0.1)
history = []

# 4. 공격 루프 (Gradient Matching)
print("Starting DLG Attack...")

for iters in range(101):
    def closure():
        optimizer.zero_grad()
        dummy_data.data.clamp_(0, 1)
        dummy_dy_dx = torch.autograd.grad(criterion(net(dummy_data), gt_onehot_label),
                                          net.parameters(), create_graph=True)
        grad_diff = sum(((gx - gy) ** 2).sum() for gx, gy in zip(dummy_dy_dx, target_grad))
        grad_diff.backward()
        return grad_diff

    optimizer.step(closure)

    # [차이점] 여기서는 closure()를 다시 호출하지 않고 시각화 데이터만 저장합니다.
    if iters % 10 == 0:
        history.append(tt(dummy_data[0].detach().clone().clamp(0, 1).cpu()))

# 5. 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(tt(gt_data[0].cpu())); axes[0].set_title("1. Ground Truth"); axes[0].axis('off')
axes[1].imshow(initial_dummy_img); axes[1].set_title("2. Initial Dummy"); axes[1].axis('off')
axes[2].imshow(history[-1]); axes[2].set_title("3. Reconstructed Result"); axes[2].axis('off')
plt.show()