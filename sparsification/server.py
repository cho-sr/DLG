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
NUM_CLASSES = 10
DATASET_ROOT = "./dataset"
######################################################################################################

####################################################### 수정 가능 #######################################################
target_accuracy = 90.0  # 사용자 편의에 맞게 조정 (70~80 범위)
global_round = 5   # 사용자 편의에 맞게 조정
batch_size = 64  # 사용자 편의에 맞게 조정
num_samples = 1280   # 사용자 편의에 맞게 조정
host = '127.0.0.1' # loop back으로 연합학습 수행 시 사용될 ip
port = 8081 # 1024번 ~ 65535번


test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010])
])





class Network1(nn.Module):
    def __init__(self, num_classes=4):
        super(Network1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),

            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            nn.Conv2d(256, num_classes,kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)

        return x

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
##############################################################################################################################






####################################################### 수정 금지 ##############################################################
cnt = []
model_list = []  # 수신받은 model 저장할 리스트
semaphore = threading.Semaphore(0)

global_model = None
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
    global model_list, global_model, global_accuracy, global_model_size, current_round, cnt
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
        model = pickle.loads(received_payload)

        model_list.append(model)
        # print(models)
        if len(model_list) == 2:
            current_round += 1
            global_model = average_models(model_list)
            global_accuracy, global_model, _ = measure_accuracy(global_model, test_loader)
            print(f"Global round [{current_round} / {global_round}] Accuracy : {global_accuracy}%")
            global_model_size = get_model_size(global_model)
            model_list = []
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

def average_models(models):
    weight_avg = copy.deepcopy(models[0])

    for key in weight_avg.keys():
        for i in range(1, len(models)):
            weight_avg[key] += models[i][key]
        weight_avg[key] = torch.div(weight_avg[key], len(models))

    return weight_avg


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

    model = build_model().to(device)
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
