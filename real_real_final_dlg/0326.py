import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# 전체 실험에서 재현 가능성을 높이기 위해 Python/PyTorch 난수를 고정한다.
# CUDA를 사용할 경우 GPU 난수 시드도 함께 설정한다.
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# 사용 가능한 연산 장치를 선택한다.
# Apple Silicon(MPS) -> CUDA -> CPU 순으로 우선 사용한다.
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# CIFAR-10 분류를 위한 간단한 CNN 모델이다.
# 합성곱 블록으로 특징을 추출한 뒤, 선형 계층으로 10개 클래스를 분류한다.
class SimpleCifarCNN(nn.Module):
    # 특징 추출기와 분류기를 초기화한다.
    # 32x32 입력이 세 번의 pooling 이후 4x4가 된다는 점을 전제로 한다.
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    # 입력 이미지를 특징 추출기와 분류기에 통과시켜 클래스 로짓을 반환한다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# CIFAR-10용 학습/평가 전처리를 구성한다.
# 현재는 ToTensor와 정규화만 적용하며, 데이터 증강은 사용하지 않는다.
def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform


# 전체 학습 데이터를 무작위로 섞은 뒤 client들에게 균등하게 나누는 IID 분할 함수다.
def partition_dataset_iid(dataset: datasets.CIFAR10, num_clients: int, seed: int) -> List[List[int]]:
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()
    client_splits = [[] for _ in range(num_clients)]

    for idx, sample_index in enumerate(shuffled_indices):
        client_splits[idx % num_clients].append(sample_index)

    return client_splits


# 각 client가 제한된 수의 클래스를 중심으로 데이터를 갖도록 non-IID 분할을 수행한다.
# 남은 샘플은 마지막에 다시 전체 client에 분배하므로 완전히 분리된 클래스 구조는 아닐 수 있다.
def partition_dataset_noniid(
    dataset: datasets.CIFAR10,
    num_clients: int,
    classes_per_client: int,
    seed: int,
) -> List[List[int]]:
    if classes_per_client < 1 or classes_per_client > 10:
        raise ValueError("classes_per_client must be between 1 and 10.")

    label_to_indices: Dict[int, List[int]] = {label: [] for label in range(10)}
    for index, label in enumerate(dataset.targets):
        label_to_indices[int(label)].append(index)

    rng = random.Random(seed)
    for indices in label_to_indices.values():
        rng.shuffle(indices)

    client_splits = [[] for _ in range(num_clients)]
    for client_id in range(num_clients):
        labels = rng.sample(range(10), classes_per_client)
        for label in labels:
            take_count = max(1, len(label_to_indices[label]) // num_clients)
            selected = label_to_indices[label][:take_count]
            label_to_indices[label] = label_to_indices[label][take_count:]
            client_splits[client_id].extend(selected)

    leftovers = []
    for indices in label_to_indices.values():
        leftovers.extend(indices)
    rng.shuffle(leftovers)
    for idx, sample_index in enumerate(leftovers):
        client_splits[idx % num_clients].append(sample_index)

    return client_splits


# CIFAR-10 학습/테스트 데이터를 내려받고 지정한 transform을 적용해 반환한다.
def build_datasets(data_dir: Path) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    train_transform, test_transform = build_transforms()
    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


# client별 인덱스 목록을 받아 각 client 전용 DataLoader를 생성한다.
# 실제 데이터는 Subset으로 원본 train_dataset을 참조한다.
def build_client_loaders(
    train_dataset: datasets.CIFAR10,
    client_indices: List[List[int]],
    batch_size: int,
) -> List[DataLoader]:
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        client_loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
        )
    return client_loaders


# 전체 테스트셋 평가용 DataLoader를 생성한다.
# 평가 단계이므로 shuffle은 사용하지 않는다.
def build_test_loader(test_dataset: datasets.CIFAR10, batch_size: int) -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# 한 client가 한 번의 배치 계산으로 만든 결과를 저장하는 구조체다.
# gradient, loss, 정답 수, 배치 크기를 함께 보관한다.
@dataclass
class ClientBatchResult:
    grads: Dict[str, torch.Tensor]
    loss: float
    correct: int
    batch_size: int


class FedSGDClient:
    # 각 client의 식별자, 로컬 DataLoader, 연산 장치를 저장하고 배치 iterator를 준비한다.
    def __init__(self, client_id: int, loader: DataLoader, device: torch.device) -> None:
        self.client_id = client_id
        self.loader = loader
        self.device = device
        self.iterator: Iterator = iter(loader)

    # 로컬 DataLoader에서 다음 배치를 가져온다.
    # iterator가 끝났다면 다시 처음부터 순환하도록 재생성한다.
    def _next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            images, labels = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            images, labels = next(self.iterator)
        return images.to(self.device), labels.to(self.device)

    # 현재 글로벌 모델을 복사한 뒤, 로컬 미니배치 하나에 대한 gradient를 계산한다.
    # FedSGD에서 서버로 전송할 배치 기반 gradient와 통계값을 함께 만든다.
    def compute_batch_gradient(self, global_model: nn.Module) -> ClientBatchResult:
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        local_model.zero_grad(set_to_none=True)

        images, labels = self._next_batch()
        logits = local_model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in local_model.named_parameters()
        }
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        return ClientBatchResult(
            grads=grads,
            loss=loss.item(),
            correct=correct,
            batch_size=labels.size(0),
        )

    # MODIFIED: add DLG-specific single-sample gradient extraction from a fixed local sample.
    # 로컬 데이터 중 특정 샘플 하나만 사용해 gradient를 계산한다.
    # DLG 같은 gradient inversion 실험에서 사용할 수 있도록 CPU 텐서 형태로 반환한다.
    def compute_single_sample_gradient(self, global_model: nn.Module, sample_index: int) -> Dict[str, object]:
        if sample_index < 0 or sample_index >= len(self.loader.dataset):
            raise IndexError(
                f"dlg_sample_index={sample_index} is out of range for client {self.client_id} "
                f"with local dataset size {len(self.loader.dataset)}."
            )

        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        local_model.zero_grad(set_to_none=True)

        image, label = self.loader.dataset[sample_index]
        image = image.unsqueeze(0).to(self.device)
        label_tensor = torch.tensor([label], device=self.device)

        logits = local_model(image)
        loss = nn.CrossEntropyLoss()(logits, label_tensor)
        loss.backward()

        grads = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in local_model.named_parameters()
        }
        return {
            "client_id": self.client_id,
            "sample_index": sample_index,
            "label": int(label),
            "loss": float(loss.item()),
            "named_grads": grads,
        }


class FedSGDServer:
    # 글로벌 모델과 이를 갱신할 SGD optimizer를 초기화한다.
    def __init__(self, model: nn.Module, lr: float, device: torch.device) -> None:
        self.model = model.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.0)
        self.device = device

    # 여러 client가 보낸 gradient를 배치 크기 비율로 가중 평균해 글로벌 모델을 업데이트한다.
    def aggregate(self, client_results: List[ClientBatchResult]) -> None:
        total_weight = sum(result.batch_size for result in client_results)
        aggregated_grads: Dict[str, torch.Tensor] = {}

        for result in client_results:
            weight = result.batch_size / total_weight
            for name, grad in result.grads.items():
                if name not in aggregated_grads:
                    aggregated_grads[name] = grad * weight
                else:
                    aggregated_grads[name] += grad * weight

        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            parameter.grad = aggregated_grads[name].to(self.device)
        self.optimizer.step()

    @torch.no_grad()
    # 현재 글로벌 모델을 주어진 DataLoader에서 평가한다.
    # 평균 loss와 accuracy를 반환하며, 로컬 데이터와 테스트셋 평가에 공통으로 사용된다.
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        # MODIFIED: this evaluator is now used for both client-local datasets and the global test set.
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = nn.CrossEntropyLoss()(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += logits.argmax(dim=1).eq(labels).sum().item()
            total_samples += batch_size

        return total_loss / total_samples, total_correct / total_samples


# 전체 FedSGD 실험을 실행하는 메인 루프다.
# 데이터 준비, 분할, client/server 생성, 라운드별 학습, DLG gradient 추출, 성능 평가를 담당한다.
def run_fedsgd(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_dataset, test_dataset = build_datasets(data_dir)

    if args.partition == "iid":
        client_indices = partition_dataset_iid(train_dataset, args.num_clients, args.seed)
    else:
        client_indices = partition_dataset_noniid(
            train_dataset,
            args.num_clients,
            args.classes_per_client,
            args.seed,
        )

    client_loaders = build_client_loaders(train_dataset, client_indices, args.batch_size)
    test_loader = build_test_loader(test_dataset, args.test_batch_size)

    clients = [FedSGDClient(client_id=i, loader=loader, device=device) for i, loader in enumerate(client_loaders)]
    server = FedSGDServer(SimpleCifarCNN(), lr=args.lr, device=device)
    # MODIFIED: prepare DLG settings once so extraction only happens on selected rounds.
    dlg_rounds = set(args.dlg_rounds)
    dlg_client = clients[args.dlg_client_id]
    if args.save_dlg:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for round_idx in range(1, args.rounds + 1):
        selected_clients = random.sample(clients, args.clients_per_round)
        client_results = [client.compute_batch_gradient(server.model) for client in selected_clients]

        # MODIFIED: extract the single-sample DLG gradient before aggregation and before server update.
        if round_idx in dlg_rounds:
            dlg_gradient = dlg_client.compute_single_sample_gradient(
                server.model,
                args.dlg_sample_index,
            )
            if args.save_dlg:
                save_path = Path(args.save_dir) / (
                    f"dlg_round_{round_idx:03d}_client_{args.dlg_client_id}_"
                    f"sample_{args.dlg_sample_index}.pt"
                )
                torch.save(dlg_gradient, save_path)
                print(f"[DLG] Saved single-sample gradient to {save_path}")
            else:
                print(
                    f"[DLG] Extracted single-sample gradient for round={round_idx:03d} "
                    f"client={args.dlg_client_id} sample={args.dlg_sample_index}"
                )

        server.aggregate(client_results)

        # MODIFIED: rename the previous batch-based train accuracy to local_batch_acc.
        avg_train_loss = sum(result.loss for result in client_results) / len(client_results)
        total_correct = sum(result.correct for result in client_results)
        total_samples = sum(result.batch_size for result in client_results)
        local_batch_acc = total_correct / total_samples

        # MODIFIED: local_acc means the global model evaluated on each client's full local dataset.
        local_accs = []
        for client in clients:
            _, local_acc = server.evaluate(client.loader)
            local_accs.append(local_acc)
        mean_local_acc = sum(local_accs) / len(local_accs)

        # MODIFIED: global_acc means the global model evaluated on the full test set.
        _, global_acc = server.evaluate(test_loader)
        print(
            f"[Round {round_idx:03d}] "
            f"train_loss={avg_train_loss:.4f} "
            f"local_batch_acc={local_batch_acc:.4f} "
            f"mean_local_acc={mean_local_acc:.4f} "
            f"global_acc={global_acc:.4f}"
        )


# 실행에 사용할 CLI 인자를 정의한다.
# 학습 설정과 DLG gradient 추출 관련 옵션을 함께 등록한다.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FedSGD on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to store CIFAR-10")
    parser.add_argument("--rounds", type=int, default=100, help="Number of communication rounds")
    parser.add_argument("--num-clients", type=int, default=10, help="Total number of clients")
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=5,
        help="Number of sampled clients at each round",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Client mini-batch size")
    parser.add_argument("--test-batch-size", type=int, default=256, help="Test batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Server learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "noniid"],
        help="How to split training data across clients",
    )
    parser.add_argument(
        "--classes-per-client",
        type=int,
        default=2,
        help="Used only for noniid partition",
    )
    # MODIFIED: add DLG extraction controls.
    parser.add_argument(
        "--dlg-rounds",
        type=int,
        nargs="*",
        default=[],
        help="Rounds at which to extract a single-sample DLG gradient",
    )
    parser.add_argument(
        "--dlg-client-id",
        type=int,
        default=0,
        help="Client id used for DLG single-sample gradient extraction",
    )
    parser.add_argument(
        "--dlg-sample-index",
        type=int,
        default=0,
        help="Fixed local sample index used for DLG gradient extraction",
    )
    parser.add_argument(
        "--save-dlg",
        action="store_true",
        help="Save extracted DLG gradients to disk",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./dlg_outputs",
        help="Directory used to save extracted DLG gradients",
    )
    return parser


# 전달된 실행 인자가 유효한지 검사한다.
# client 수, round 수, DLG 관련 인덱스와 라운드 범위를 확인한다.
def validate_args(args: argparse.Namespace) -> None:
    if args.clients_per_round > args.num_clients:
        raise ValueError("clients_per_round must be less than or equal to num_clients.")
    if args.num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if args.rounds < 1:
        raise ValueError("rounds must be at least 1.")
    # MODIFIED: validate DLG-related arguments.
    if args.dlg_client_id < 0 or args.dlg_client_id >= args.num_clients:
        raise ValueError("dlg_client_id must be between 0 and num_clients - 1.")
    if args.dlg_sample_index < 0:
        raise ValueError("dlg_sample_index must be non-negative.")
    for dlg_round in args.dlg_rounds:
        if dlg_round < 1 or dlg_round > args.rounds:
            raise ValueError("Each dlg round must be between 1 and rounds.")


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args()
    validate_args(parsed_args)
    run_fedsgd(parsed_args)
