import threading
import socket
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np

import struct
from tqdm import tqdm
import copy
import warnings
import random
import os
from pathlib import Path
from torchvision import models
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
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
test_loader = None

############################################## 수정 불가 1 ##############################################
IMG_SIZE = 32
NUM_CLASSES = 50
DATASET_ROOT = "./dataset"
######################################################################################################

####################################################### 수정 가능 #######################################################
target_accuracy = 90.0  # 사용자 편의에 맞게 조정 (70~80 범위)
global_round = 10   # 사용자 편의에 맞게 조정
batch_size = 64  # 사용자 편의에 맞게 조정
num_samples = 1280   # 사용자 편의에 맞게 조정
server_lr = 0.01
host = '127.0.0.1' # loop back으로 연합학습 수행 시 사용될 ip
port = 8081 # 1024번 ~ 65535번
GRAD_RECORD_DIR = Path(__file__).resolve().parent / "gradient_records"
GRAD_SAVE_ROUNDS = {1, 5}
NORM_MEAN = [0.4914, 0.4822, 0.4465]
NORM_STD = [0.2023, 0.1994, 0.2010]


test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=NORM_MEAN,
                 std=NORM_STD),
])

def build_model():
    model = LeNet()
    if NUM_CLASSES != 10:
        model.fc[0] = nn.Linear(model.fc[0].in_features, NUM_CLASSES)
    return model


class CustomDataset(Dataset):
    def __init__(self, pt_path: str, is_train: bool = False, transform=None):
        print(pt_path)
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

def measure_accuracy(global_model, test_loader):
    model = build_model().to(device)
    model.load_state_dict(global_model)
    model.eval()

    accuracy = 0.0
    total = 0.0
    correct = 0

    inference_start = time.time()
    with torch.no_grad():
        print("\n")
        for inputs, labels in tqdm(test_loader, desc="Test"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        accuracy = (100 * correct / total)

    inference_end = time.time()
    inference_time = inference_end - inference_start

    return accuracy, model, inference_time

def build_weighted_average(client_payloads):
    total_samples = sum(max(1, int(item.get("num_samples", 1))) for item in client_payloads)
    if total_samples <= 0:
        return {}, {}

    avg_grads = {}
    avg_buffers = {}
    for item in client_payloads:
        weight = max(1, int(item.get("num_samples", 1))) / float(total_samples)

        for name, grad in item["grads"].items():
            grad_cpu = grad.detach().cpu()
            if name not in avg_grads:
                avg_grads[name] = grad_cpu * weight
            else:
                avg_grads[name] += grad_cpu * weight

        for name, buf in item.get("buffers", {}).items():
            buf_cpu = buf.detach().cpu().float()
            if name not in avg_buffers:
                avg_buffers[name] = buf_cpu * weight
            else:
                avg_buffers[name] += buf_cpu * weight

    return avg_grads, avg_buffers

# =====================================================================
# [수정 1] server.py 내부 apply_fedsgd_update 함수 완전 교체
# 이유: 수동 연산(param.sub_) 대신 옵티마이저의 step()을 활용하기 위함
# =====================================================================
def apply_fedsgd_update(model, optimizer, client_payloads):
    avg_grads, avg_buffers = build_weighted_average(client_payloads)
    if not avg_grads and not avg_buffers:
        return

    optimizer.zero_grad()
    
    # 1. 수신한 평균 그래디언트를 모델 파라미터의 .grad에 직접 주입
    for name, param in model.named_parameters():
        if name in avg_grads:
            param.grad = avg_grads[name].to(param.device, dtype=param.dtype)

    # 2. BatchNorm 통계량 업데이트
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name not in avg_buffers:
                continue
            averaged = avg_buffers[name].to(buf.device)
            if torch.is_floating_point(buf):
                buf.copy_(averaged.to(buf.dtype))
            else:
                buf.copy_(averaged.round().to(buf.dtype))

    # 3. 옵티마이저 업데이트 실행 (Momentum 적용됨)
    optimizer.step()

def clone_attack_payload(attack):
    if not isinstance(attack, dict):
        return None

    cloned = {
        "global_index": int(attack.get("global_index", -1)),
        "gt_label": int(attack.get("gt_label", -1)),
        "num_classes": int(attack.get("num_classes", NUM_CLASSES)),
        "input_shape": tuple(attack.get("input_shape", (1, 3, IMG_SIZE, IMG_SIZE))),
        "norm_mean": list(attack.get("norm_mean", NORM_MEAN)),
        "norm_std": list(attack.get("norm_std", NORM_STD)),
    }

    if "gradients" in attack:
        cloned["gradients"] = [grad.detach().cpu().clone() for grad in attack["gradients"]]
    if "model_state" in attack:
        cloned["model_state"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in attack["model_state"].items()
        }
    if "gt_data" in attack:
        cloned["gt_data"] = attack["gt_data"].detach().cpu().clone()

    return cloned

def save_round_gradients(round_idx, client_payloads, model):
    if round_idx not in GRAD_SAVE_ROUNDS:
        return

    avg_grads, avg_buffers = build_weighted_average(client_payloads)
    GRAD_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GRAD_RECORD_DIR / f"round_{int(round_idx):03d}.pt"

    payload = {
        "round": int(round_idx),
        "num_classes": NUM_CLASSES,
        "input_shape": (1, 3, IMG_SIZE, IMG_SIZE),
        "norm_mean": list(NORM_MEAN),
        "norm_std": list(NORM_STD),
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "clients": [],
        "avg_grads": {k: v.detach().cpu().clone() for k, v in avg_grads.items()},
        "avg_buffers": {k: v.detach().cpu().clone() for k, v in avg_buffers.items()},
    }
    for item in client_payloads:
        payload["clients"].append(
            {
                "client_id": int(item.get("client_id", -1)),
                "num_samples": int(item.get("num_samples", 1)),
                "grads": {k: v.detach().cpu().clone() for k, v in item["grads"].items()},
                "buffers": {k: v.detach().cpu().clone() for k, v in item.get("buffers", {}).items()},
                "attack": clone_attack_payload(item.get("attack")),
            }
        )

    torch.save(payload, out_path)
    print(f"Saved gradient record: {out_path.name}")
##############################################################################################################################






####################################################### 수정 금지 ##############################################################
cnt = []
grad_list = []  # 수신받은 gradient 저장할 리스트
semaphore = threading.Semaphore(0)

global_model = None
global_optimizer = None
global_model_size = 0
global_accuracy = 0.0
current_round = 0
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
def handle_client(conn, addr, model, test_loader):
    global grad_list, global_model, global_accuracy, global_model_size, current_round, cnt, global_optimizer
    print(f"Connected by {addr}")

    while True:
        if len(cnt) < 2:
            cnt.append(1)
            weight = pickle.dumps(dict(model.state_dict().items()))
            # print(weight)
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)

        data_size = struct.unpack('>I', conn.recv(4))[0]
        received_payload = b""
        remaining_payload_size = data_size
        while remaining_payload_size != 0:
            received_payload += conn.recv(remaining_payload_size)
            remaining_payload_size = data_size - len(received_payload)
        received = pickle.loads(received_payload)
        if not isinstance(received, dict) or "grads" not in received:
            raise ValueError("Expected FedSGD payload with 'grads'.")

        grad_list.append(
            {
                "client_id": int(received.get("client_id", -1)),
                "num_samples": int(received.get("num_samples", 1)),
                "grads": received["grads"],
                "buffers": received.get("buffers", {}),
                "attack": received.get("attack"),
            }
        )

        if len(grad_list) == 2:
            current_round += 1
            save_round_gradients(current_round, grad_list, global_model)
            apply_fedsgd_update(global_model, global_optimizer,grad_list)
            global_accuracy, global_model, _ = measure_accuracy(
                dict(global_model.state_dict().items()),
                test_loader,
            )
            print(f"Global round [{current_round} / {global_round}] Accuracy : {global_accuracy}%")
            global_model_size = get_model_size(global_model)
            grad_list = []
            semaphore.release()
        else:
            semaphore.acquire()

        if (current_round == global_round) or (global_accuracy >= target_accuracy):
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)
            conn.close()
            break
        else:
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)

# def apply_fedsgd_update(model, client_payloads, lr):
#     avg_grads, avg_buffers = build_weighted_average(client_payloads)
#     if not avg_grads and not avg_buffers:
#         return

#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if name not in avg_grads:
#                 continue
#             param.sub_(lr * avg_grads[name].to(param.device, dtype=param.dtype))

#         for name, buf in model.named_buffers():
#             if name not in avg_buffers:
#                 continue
#             averaged = avg_buffers[name].to(buf.device)
#             if torch.is_floating_point(buf):
#                 buf.copy_(averaged.to(buf.dtype))
#             else:
#                 buf.copy_(averaged.round().to(buf.dtype))

def get_model_size(global_model):
    model_size = len(pickle.dumps(dict(global_model.state_dict().items())))
    model_size = model_size / (1024 ** 2)

    return model_size


def get_random_subset(dataset, num_samples):
    if num_samples > len(dataset):
        raise ValueError(f"num_samples should not exceed {len(dataset)} (total number of samples in test dataset).")

    indices = random.sample(range(len(dataset)), num_samples)
    subset = Subset(dataset, indices)

    return subset

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()
    connection = []
    address = []

    ############################ 수정 가능 ############################
    train_dataset = datasets.CIFAR10(
        root=DATASET_ROOT,
        train=False,
        download=True,
        transform=test_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    global global_model,global_optimizer
    model = build_model().to(device)
    global_model = model
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=server_lr)
    ####################################################################

    print(f"Server is listening on {host}:{port}")

    while len(address) < 2 and len(connection) < 2:
        conn, addr = server.accept()
        connection.append(conn)
        address.append(addr)

    training_start = time.time()

    connection1 = threading.Thread(target=handle_client, args=(connection[0], address[0], model, test_loader))
    connection2 = threading.Thread(target=handle_client, args=(connection[1], address[1], model, test_loader))

    connection1.start();connection2.start()
    connection1.join();connection2.join()

    training_end = time.time()
    total_time = training_end - training_start

    # 평가지표 1
    print(f"\n학습 성능 : {global_accuracy} %")
    # 평가지표 2
    print(f"\n학습 소요 시간: {int(total_time // 3600)} 시간 {int((total_time % 3600) // 60)} 분 {(total_time % 60):.2f} 초")

    # 평가지표 3
    print(f"\n최종 모델 크기: {global_model_size:.4f} MB")

    final_model = dict(global_model.state_dict().items())
    _, _, inference_time = measure_accuracy(final_model, test_loader)
    # 평가지표 4
    print(f"\n예측 소요 시간 : {(inference_time):.2f} 초")

    print("연합학습 종료")


if __name__ == "__main__":
    main()
##############################################################################################################################
