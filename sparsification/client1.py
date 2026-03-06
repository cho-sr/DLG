import socket
import pickle
from tqdm import tqdm
import time
import torch
import random
import numpy as np
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import struct
from collections import OrderedDict
import warnings
import select
import os
from pathlib import Path
from torchvision import models
from torchvision import datasets
import torchvision.transforms.v2 as v2
from utils import label_to_onehot, cross_entropy_for_onehot

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


############################################## 수정 금지 1 ##############################################
IMG_SIZE = 32
NUM_CLASSES = 10
DATASET_ROOT = "./dataset"
NORM_MEAN = [0.4914, 0.4822, 0.4465]
NORM_STD = [0.2023, 0.1994, 0.2010]
######################################################################################################


############################################# 수정 가능 #############################################
local_epochs = 1
lr = 0.01
batch_size = 1
local_steps = 10
host_ip = "127.0.0.1"
port = 8081
client_id = 1
STANDALONE_MODE = True
standalone_rounds = 30
GRAD_RECORD_DIR = Path(__file__).resolve().parent / "client_grad_records"


################# 전처리 코드 수정 가능하나 꼭 IMG_SIZE로 resize한 뒤 정규화 해야 함#################
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    # v2.RandomHorizontalFlip(0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


scaler = torch.amp.GradScaler('cuda')


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
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),
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
        x = x.view(x.size(0), -1)
        return x


def extract_attack_gradient(model, x, y):
    model.eval()
    model.zero_grad(set_to_none=True)

    x = x.to(device)
    y = y.to(device)

    pred = model(x)
    loss = F.cross_entropy(pred, y)

    dy_dx = torch.autograd.grad(loss, model.parameters())

    grads_dict = {}
    for (name, param), grad in zip(model.named_parameters(), dy_dx):
        grads_dict[name] = grad.detach().cpu().clone()
        
    return grads_dict

def train(model, criterion, optimizer, train_loader):
    model.to(device)
    use_amp = device == "cuda"

    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for (images, labels) in tqdm(train_loader, desc="Train"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_corrects.float() / total
        print(f"Epoch [{epoch + 1}/{local_epochs}] => Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy * 100:.2f}%")

    return model

##############################################################################################################################

def compute_fedsgd_gradient(model, criterion, train_loader, train_iter, local_steps=1):
    model.train()
    model.to(device)
    model.zero_grad(set_to_none=True)
    steps = max(1, int(local_steps))

    running_loss = 0.0
    running_corrects = 0
    total = 0

    for _ in range(steps):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        running_loss += batch_loss.item()
        running_corrects += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        # Average over local mini-batch steps before sending to server.
        loss = batch_loss / steps
        loss.backward()

    epoch_loss = running_loss / steps
    epoch_accuracy = (100.0 * running_corrects / total) if total > 0 else 0.0
    print(
        f"FedSGD Local ({steps} step) => Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%"
    )

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grads[name] = param.grad.detach().cpu().clone()
    return grads, train_iter



####################################################### 수정 가능 ##############################################################


class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.images = [item["tensor"] for item in blob["items"]]
        self.labels = [int(item["label"]) for item in blob["items"]]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        x = self.images[idx].float() / 255.0
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def main():
    train_dataset = datasets.CIFAR10(
        root=DATASET_ROOT,
        train=True,
        download=True,
        transform=train_transform,
    )
    indices = np.arange(len(train_dataset))
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    model = Network1(num_classes=NUM_CLASSES)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
##############################################################################################################################





########################################################### 수정 금지 2 ##############################################################
    round_idx = 0
    train_iter = iter(train_loader)

    if STANDALONE_MODE:
        GRAD_RECORD_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Standalone mode enabled. Saving records to: {GRAD_RECORD_DIR}")

        for _ in range(standalone_rounds):
            round_idx += 1
# 1. 공격 타겟 데이터 선정 (정규화된 상태 확인)
            sample_idx = indices[round_idx % len(indices)] # 셔플된 인덱스 활용
            attack_x, attack_y = train_dataset[sample_idx]
            attack_x = attack_x.unsqueeze(0).to(device)
            attack_y = torch.tensor([int(attack_y)], dtype=torch.long).to(device)

            # 2. 핵심: 그래디언트 추출 당시의 모델 가중치(W) 저장
            # 이 W를 기준으로 공격자가 더미 그래디언트를 계산하게 됨
            model_state_at_grad = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # 3. 깨끗한 1개 샘플 그래디언트 추출
            attack_grad = extract_attack_gradient(model, attack_x, attack_y)

            # 4. 실제 학습 진행 (FedSGD 누적 그래디언트)
            # 이 과정은 model 가중치를 직접 바꾸지 않고 param.grad만 채움
            fedsgd_grads, train_iter = compute_fedsgd_gradient(
                model, criterion, train_loader, train_iter, local_steps
            )

            payload = {
                "type": "client_grad",
                "client_id": client_id,
                "round": round_idx,
                "num_samples": 1, 
                "fedsgd_grads": fedsgd_grads, # 실제 전송되는 정보
                "attack": {
                    "num_classes": NUM_CLASSES,
                    "input_shape": tuple(attack_x.shape),
                    "norm_mean": list(NORM_MEAN),
                    "norm_std": list(NORM_STD),
                    "gradients": attack_grad,
                    "model_state": model_state_at_grad,
                    "gt_data": attack_x.detach().cpu().clone(),
                    "gt_label": int(attack_y.item()),
                },
            }
            out_path = GRAD_RECORD_DIR / f"round_{round_idx:03d}_client_{client_id}.pt"
            torch.save(payload, out_path)
            print(f"Saved local gradient record: {out_path.name}")
        return

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host_ip, port))

    while True:
        data_size = struct.unpack('>I', client.recv(4))[0]
        rec_payload = b""
        remaining_payload = data_size
        while remaining_payload != 0:
            rec_payload += client.recv(remaining_payload)
            remaining_payload = data_size - len(rec_payload)
        dict_weight = pickle.loads(rec_payload)
        weight = OrderedDict(dict_weight)
        print("\nReceived updated global model from server")

        model.load_state_dict(weight, strict=True)
       
        read_sockets, _, _ = select.select([client], [], [], 0)
        if read_sockets:
            print("Federated Learning finished")
            break

        # Capture one-sample true gradient before local training for DLG.
        attack_x, attack_y = train_dataset[round_idx % len(train_dataset)]
        attack_x = attack_x.unsqueeze(0)
        attack_y = torch.tensor([int(attack_y)], dtype=torch.long)
        attack_grad = extract_attack_gradient(model, attack_x, attack_y)
        model_state_at_grad = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # model = train(model, criterion, optimizer, train_loader)
        fedsgd_grads, train_iter = compute_fedsgd_gradient(
            model, criterion, train_loader, train_iter, local_steps
        )

        round_idx += 1
        # payload = {
        #     "type": "client_update",
        #     "client_id": client_id,
        #     "round": round_idx,
        #     "model_state": dict(model.state_dict().items()),
        #     "attack": {
        #         "num_classes": NUM_CLASSES,
        #         "input_shape": tuple(attack_x.shape),
        #         "norm_mean": list(NORM_MEAN),
        #         "norm_std": list(NORM_STD),
        #         "gradients": attack_grad,
        #         "model_state": model_state_at_grad,
        #         "gt_data": attack_x.detach().cpu().clone(),
        #         "gt_label": int(attack_y.item()),
        #     },
        # }
        payload = {
            "type": "client_grad",
            "client_id": client_id,
            "round": round_idx,
            "num_samples": len(train_dataset),
            "grads": fedsgd_grads,
            "attack": {
                "num_classes": NUM_CLASSES,
                "input_shape": tuple(attack_x.shape),
                "norm_mean": list(NORM_MEAN),
                "norm_std": list(NORM_STD),
                "gradients": attack_grad,
                "model_state": model_state_at_grad,
                "gt_data": attack_x.detach().cpu().clone(),
                "gt_label": int(attack_y.item()),
            },
        }
        model_data = pickle.dumps(payload)
        client.sendall(struct.pack('>I', len(model_data)))
        client.sendall(model_data)

        print("Sent updated local model to server.")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("\nThe model will be running on", device, "device")

    time.sleep(1)
    main()

######################################################################################################################
