# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
from models.vision import LeNet, weights_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms

print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot
import matplotlib
try:
    # Mac/Window/Linux 호환성이 가장 좋은 팝업창 모드
    matplotlib.use('TkAgg')
except:
    # Mac 전용 네이티브 모드 (TkAgg가 없을 경우 대비)
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
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
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

plt.imshow(tt(gt_data[0].cpu()))
plt.title("Ground Truth")
plt.show()

net = LeNet().to(device)

torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot

# --- 수정 전
# compute original gradient 
# pred = net(gt_data)
# y = criterion(pred, gt_onehot_label)
# dy_dx = torch.autograd.grad(y, net.parameters())
#
# original_dy_dx = list((_.detach().clone() for _ in dy_dx))
# --- [수정 후] FedAvg 방식 (Local Epoch 수행) ---
# 1. 학습 전 초기 가중치 저장
original_weights = [p.detach().clone() for p in net.parameters()]

# 2. 클라이언트 로컬 학습 설정 (예: Local Epochs = 5)
local_epochs = 1
lr = 0.01
optimizer_client = torch.optim.SGD(net.parameters(), lr=lr)

# 3. 로컬 학습 수행 (FedAvg 시뮬레이션)
net.train()
for _ in range(local_epochs):
    optimizer_client.zero_grad()
    pred = net(gt_data)
    loss = criterion(pred, gt_onehot_label)
    loss.backward()
    optimizer_client.step()

# 4. 가중치 업데이트값(Weight Difference) 계산
# 공격자가 훔쳐보게 될 Target 값입니다.
target_grad = []
for w_init, w_trained in zip(original_weights, net.parameters()):
    # W_diff = W_init - W_trained (혹은 그 반대)
    # 이것을 "누적된 그래디언트"처럼 취급합니다.
    weight_diff = (w_init - w_trained).detach()
    effective_grad = weight_diff / (lr * local_epochs)
    target_grad.append(effective_grad)

# 모델을 다시 초기 상태로 되돌림 (공격자는 초기 상태 W_init을 알고 있다고 가정)
# FedAvg에서도 Global Model 파라미터는 공유되므로 공격자가 알고 있습니다.
with torch.no_grad():
    for param, w_init in zip(net.parameters(), original_weights):
        param.copy_(w_init)

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
# dummy_label = gt_onehot_label.clone().to(device)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

history = []

print("Starting DLG Attack...")
current_loss_value = [0.0]  # Loss 값을 저장할 리스트 (Closure 내부 접근용)

for iters in range(101):
    def closure():
        optimizer.zero_grad()
        dummy_data.data.clamp_(0, 1)
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, target_grad):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()

        # # [핵심] 계산된 Loss를 변수에 저장 (재계산 X)
        # current_loss_value[0] = grad_diff.item()
        return grad_diff


    optimizer.step(closure)
    # if iters % 10 == 0:
    #     # [수정] closure() 호출 대신 저장된 값 사용
    #     print(f"Iter {iters:3d}: Loss = {current_loss_value[0]:.6f}")
    #     history.append(tt(dummy_data[0].detach().clone().clamp(0, 1).cpu()))
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()