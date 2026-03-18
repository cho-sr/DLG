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
from torchvision import models
from torchvision import datasets
import torchvision.transforms.v2 as v2
from models.vision import LeNet
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
# =========================================================
# [추가할 부분] DLG 페이로드 구성을 위한 정규화 상수 추가
# =========================================================
NORM_MEAN = [0.4914, 0.4822, 0.4465]
NORM_STD = [0.2023, 0.1994, 0.2010]
######################################################################################################


############################################# 수정 가능 #############################################
local_epochs = 1
lr = 0.001
batch_size = 64
client_id = 2
host_ip = "127.0.0.1"
port = 8081


################# 전처리 코드 수정 가능하나 꼭 IMG_SIZE로 resize한 뒤 정규화 해야 함 #################
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomHorizontalFlip(0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010]),
])

attack_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

def build_model():
    model = LeNet()
    if NUM_CLASSES != 10:
        model.fc[0] = nn.Linear(model.fc[0].in_features, NUM_CLASSES)
    return model


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    model.to(device)
    
    # CUDA일 때만 AMP를 사용하도록 분기 처리 (MPS에서 GradScaler 오류 방지)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(local_epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = running_corrects.float() / total
        print(f"Epoch [{epoch + 1}/{local_epochs}] => Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy * 100:.2f}%")

    return model
##############################################################################################################################

# client1.py의 compute_fedsgd_gradient 내부

def compute_fedsgd_gradient(
    model,
    criterion,
    train_loader,
    attack_criterion,
    attack_x,
    attack_y,
    attack_global_index,
):
    model.train()
    model.to(device)
    model.zero_grad(set_to_none=True)

    # -------------------------------------------------------------
    # 1. 정상적인 글로벌 모델 업데이트를 위한 "전체 배치 평균 그래디언트"
    # -------------------------------------------------------------
    # 1. 정상 글로벌 모델 업데이트용 그래디언트 축적
    num_batches = len(train_loader)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_correct = torch.zeros((), device=device, dtype=torch.int64)
    total_samples = 0
    for images, labels in tqdm(train_loader, desc="FedSGD Grad"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # 기존: loss = criterion() / len(train_loader) -> backward()가 작아져서 학습 불가
        # 수정: CrossEntropyLoss(reduction='mean')을 그대로 사용해 batch 평균을 구하고, 
        # 이를 누적한 뒤 마지막에 배치 수로 나누어 진정한 전역 평균을 구함.
        loss = criterion(outputs, labels)
        loss.backward()
        batch_count = labels.size(0)
        with torch.no_grad():
            total_loss += loss.detach() * batch_count
            total_correct += (outputs.detach().argmax(1) == labels).sum()
        total_samples += batch_count

    # 정상 통신용 그래디언트 저장
    fedsgd_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 전체 배치 수로 나누어 FedSGD 기준에 맞춤
            fedsgd_grads[name] = (param.grad.detach() / num_batches).cpu().clone()
            
    fedsgd_buffers = {name: buf.detach().cpu().clone() for name, buf in model.named_buffers()}
    train_metrics = {
        "train_loss": (total_loss / max(1, total_samples)).item(),
        "train_accuracy": (100.0 * total_correct.float() / max(1, total_samples)).item(),
    }

    # -------------------------------------------------------------
    # 2. [몰래 끼워넣기] DLG 공격을 당할 불운한 "1장의 타겟 이미지" 연산
    # -------------------------------------------------------------
    model.zero_grad(set_to_none=True)
    attack_x, attack_y = attack_x.to(device), attack_y.to(device)
    model_state_at_grad = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    attack_outputs = model(attack_x)
    attack_loss = attack_criterion(attack_outputs, attack_y)
    attack_loss.backward()

    attack_payload = {
        "global_index": int(attack_global_index),
        "num_classes": NUM_CLASSES,
        "input_shape": tuple(attack_x.shape),
        "norm_mean": list(NORM_MEAN),
        "norm_std": list(NORM_STD),
        "gradients": [param.grad.detach().cpu().clone() for param in model.parameters()],
        "model_state": model_state_at_grad,
        "gt_data": attack_x.detach().cpu().clone(),
        "gt_label": int(attack_y.item()),
    }

    return fedsgd_grads, fedsgd_buffers, attack_payload, train_metrics


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

    half = len(indices) // 2
    client_idx = indices[:half]
    train_dataset = Subset(train_dataset, client_idx)
    attack_dataset = datasets.CIFAR10(
        root=DATASET_ROOT,
        train=True,
        download=True,
        transform=attack_transform,
    )
    attack_global_index = int(np.min(client_idx))
    attack_x, attack_y = attack_dataset[attack_global_index]
    attack_x = attack_x.unsqueeze(0)
    attack_y = torch.tensor([int(attack_y)], dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    model = build_model().to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    attack_criterion = torch.nn.CrossEntropyLoss()
##############################################################################################################################





########################################################### 수정 금지 2 ##############################################################
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

# =================================================================
        # [수정 전] 에러가 나던 부분
        # grads, buffers, actual_num_samples = compute_fedsgd_gradient(...)
        # =================================================================
        
        fedsgd_grads, fedsgd_buffers, attack_payload, train_metrics = compute_fedsgd_gradient(
            model,
            criterion,
            train_loader,
            attack_criterion,
            attack_x,
            attack_y,
            attack_global_index,
        )
        print(
            f"Client {client_id} local loss: {train_metrics['train_loss']:.4f} | "
            f"local accuracy: {train_metrics['train_accuracy']:.2f}%"
        )

        # 그 후 페이로드를 구성할 때 받아온 변수들을 그대로 사용합니다.
        payload = {
            "type": "client_grad",
            "client_id": client_id,
            "num_samples": len(train_dataset), # 정상 학습을 위해 전체 데이터 수 전송
            "grads": fedsgd_grads,             # 정상적인 평균 그래디언트
            "buffers": fedsgd_buffers,         # 정상적인 BatchNorm 통계량
            "attack": attack_payload,
            "train_metrics": train_metrics,
        }
        
        model_data = pickle.dumps(payload)
        client.sendall(struct.pack('>I', len(model_data)))
        client.sendall(model_data)
        print("Sent local gradients to server.")


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
