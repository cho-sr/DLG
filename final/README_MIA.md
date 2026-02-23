# FedAvg Model Inversion Attack with Gradient Sparsification

이 프로젝트는 **Federated Learning (FedAvg)** 환경에서 **Model Inversion Attack (MIA)**을 수행하면서 **Gradient Sparsification**을 적용하여 파라미터 비율을 조절할 수 있는 시스템입니다.

## 📋 개요

### 주요 특징

1. **FedAvg 환경**: 클라이언트가 로컬 데이터로 모델을 학습한 후 그래디언트를 서버로 전송
2. **Model Inversion Attack (MIA)**: 스파시파이된 그래디언트로부터 원본 데이터 복원 시도
3. **Gradient Sparsification**: 상위 N% 파라미터만 전송하여 통신 비용 감소 및 프라이버시 향상
4. **유연한 비율 제어**: 0.01(1%)부터 1.0(100%)까지 자유롭게 조절 가능

### 파라미터 비율 설정 예시

```
--sparsity 1.0   → 상위 100% (전체 파라미터 전송 - 기준선)
--sparsity 0.5   → 상위 50% (절반만 전송)
--sparsity 0.1   → 상위 10% (10%만 전송)
--sparsity 0.05  → 상위 5% (상위 5%만 전송)
--sparsity 0.01  → 상위 1% (1%만 전송 - 강력한 압축)
```

## 🚀 빠른 시작

### 기본 사용법

```bash
# 상위 5% 파라미터만 전송하는 MIA 공격
python fedavg_mia.py --sparsity 0.05 --index 25

# 상위 10% 파라미터 전송 (더 적은 압축)
python fedavg_mia.py --sparsity 0.1 --index 25

# 전체 파라미터 전송 (기준선/baseline)
python fedavg_mia.py --sparsity 1.0 --index 25
```

### 여러 비율 한 번에 실행

```bash
# 기본 설정 (1.0, 0.5, 0.1, 0.05, 0.01)
python batch_mia_experiments.py --index 25

# 커스텀 비율 설정
python batch_mia_experiments.py --index 25 --sparsities 1.0 0.2 0.05 0.01
```

### 결과 비교 및 분석

```bash
# 비교 분석 및 시각화 생성
python compare_mia_results.py --sparsities 1.0 0.1 0.05 0.01

# 커스텀 출력 이름
python compare_mia_results.py --sparsities 1.0 0.5 0.1 0.05 0.01 --output results/mia_analysis
```

## 📊 상세 명령어

### `fedavg_mia.py` - 메인 공격 스크립트

**기본 인자:**

```bash
python fedavg_mia.py [options]
```

**주요 옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--index` | 25 | CIFAR-10 데이터셋의 대상 이미지 인덱스 |
| `--sparsity` | 1.0 | 그래디언트 유지 비율 (0.01~1.0) |
| `--local_epochs` | 1 | 로컬 학습 에포크 수 |
| `--local_lr` | 0.01 | 로컬 학습률 |
| `--mia_iters` | 300 | MIA 최적화 반복 횟수 |
| `--seed` | 1234 | 난수 시드 |

**실행 예시:**

```bash
# 기본 설정
python fedavg_mia.py

# 커스텀 설정: 상위 5%, 500 반복
python fedavg_mia.py --sparsity 0.05 --mia_iters 500

# 다른 이미지, 더 많은 로컬 에포크
python fedavg_mia.py --index 42 --sparsity 0.1 --local_epochs 5

# 고정된 난수 시드로 재현 가능한 실험
python fedavg_mia.py --sparsity 0.01 --seed 42
```

### `batch_mia_experiments.py` - 배치 실험

**기본 인자:**

```bash
python batch_mia_experiments.py [options]
```

**주요 옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--index` | 25 | 대상 이미지 인덱스 |
| `--sparsities` | [1.0, 0.5, 0.1, 0.05, 0.01] | 테스트할 sparsity 비율 목록 |
| `--mia_iters` | 300 | 각 실험의 MIA 반복 횟수 |
| `--local_epochs` | 1 | 로컬 학습 에포크 |
| `--seed` | 1234 | 난수 시드 |

**실행 예시:**

```bash
# 기본 설정 (5가지 sparsity 비율)
python batch_mia_experiments.py

# 커스텀 비율: 100%, 10%, 1%만 테스트
python batch_mia_experiments.py --sparsities 1.0 0.1 0.01

# 더 자세한 분석: 더 많은 반복
python batch_mia_experiments.py --mia_iters 500 --local_epochs 2
```

### `compare_mia_results.py` - 결과 비교 분석

**기본 인자:**

```bash
python compare_mia_results.py [options]
```

**주요 옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--sparsities` | [1.0, 0.5, 0.1, 0.05, 0.01] | 비교할 sparsity 비율 목록 |
| `--output` | mia_comparison | 출력 파일 프리픽스 |

**실행 예시:**

```bash
# 기본 설정으로 비교 분석
python compare_mia_results.py

# 특정 비율만 비교
python compare_mia_results.py --sparsities 1.0 0.05 0.01

# 커스텀 출력 파일명
python compare_mia_results.py --output results/privacy_analysis
```

## 📈 출력 파일 설명

### 메인 실행 결과 (`fedavg_mia.py`)

각 실험마다 다음 파일들이 생성됩니다:

```
mia_ground_truth.png              # 원본 이미지
mia_initial_dummy.png             # 초기 임의 노이즈
mia_progress_sparsity_5.png       # 복원 과정 (10단계)
mia_loss_sparsity_5.png           # 손실 함수 곡선 (로그 스케일)
mia_final_sparsity_5.png          # 최종 비교 (원본 vs 복원)
mia_gradient_dist_sparsity_5.png  # 그래디언트 분포 히스토그램
```

### 비교 분석 결과 (`compare_mia_results.py`)

```
mia_comparison_tradeoff.png       # 프라이버시-유틸리티 트레이드오프
mia_comparison_compression.png    # 압축 효율성
mia_comparison_difficulty.png     # 공격 난이도 분석
mia_comparison_summary_table.png  # 요약 테이블
mia_comparison_report.txt         # 상세 텍스트 리포트
```

## 🔍 기술 상세

### Sparsification 메커니즘

```python
def sparsify_gradients(gradients, ratio):
    """
    상위 k% 그래디언트만 유지, 나머지는 0으로 설정
    
    1. 모든 그래디언트를 절댓값 기준으로 정렬
    2. 상위 k개만 선택 (k = 전체 * ratio)
    3. 임계값 이상인 값만 유지, 나머지는 0
    """
```

**예시:**
- `ratio=0.05`: 100,000 파라미터 중 5,000개만 유지
- `ratio=0.1`: 100,000 파라미터 중 10,000개 유지
- `ratio=1.0`: 전체 100,000개 모두 유지

### Model Inversion Attack 단계

1. **FedAvg 로컬 학습**
   - 클라이언트가 원본 이미지로 모델 학습
   - 손실 함수 기울기 계산

2. **그래디언트 스파시피케이션**
   - 상위 N% 파라미터만 유지
   - 공격자가 이 스파시파이된 그래디언트 수신

3. **Model Inversion (공격)**
   - 임의 노이즈로 시작
   - 스파시파이된 그래디언트와 일치하도록 최적화
   - LBFGS 옵티마이저 사용

4. **메트릭 계산**
   - MSE (Mean Squared Error)
   - PSNR (Peak Signal-to-Noise Ratio)
   - 레이블 복원 성공 여부

## 📊 실험 시나리오

### 시나리오 1: 프라이버시 vs 유틸리티 트레이드오프

```bash
# 5가지 sparsity 수준 비교
python batch_mia_experiments.py --sparsities 1.0 0.5 0.1 0.05 0.01
python compare_mia_results.py --sparsities 1.0 0.5 0.1 0.05 0.01
```

**예상 결과:**
- Sparsity 1.0 (100%): 공격 성공률 높음, 프라이버시 낮음
- Sparsity 0.01 (1%): 공격 실패 가능성 높음, 프라이버시 높음

### 시나리오 2: 최적 압축 비율 찾기

```bash
# 더 촘촘한 범위 테스트
python batch_mia_experiments.py \
  --sparsities 1.0 0.5 0.2 0.1 0.05 0.02 0.01
```

### 시나리오 3: 다양한 이미지로 강건성 테스트

```bash
for idx in 10 25 42 99 123; do
  python fedavg_mia.py --index $idx --sparsity 0.05
done
```

## 🎯 성능 지표

### 복원 품질 지표

- **MSE**: 값이 작을수록 좋음 (범위: 0~1)
- **PSNR**: 값이 클수록 좋음 (일반적으로: 20~40 dB)
- **레이블 정확도**: 원본 레이블과의 일치 여부

### 프라이버시 지표

- **Sparsity Ratio**: 낮을수록 더 많은 프라이버시 보호
- **압축률**: 네트워크 대역폭 절약 정도
- **공격 난이도**: 높을수록 데이터 복원이 어려움

## 🔧 커스터마이제이션

### 모델 변경

`models/vision.py`에서 다른 모델 사용 가능:

```python
from models.vision import ResNet18, ResNet50, LeNet

# 다른 모델 테스트
net = ResNet50().to(device)  # ResNet18 대신 ResNet50
```

### 데이터셋 변경

```python
# CIFAR-10 대신 CIFAR-100
dst = datasets.CIFAR100("~/.torch", download=True)

# 또는 커스텀 데이터셋
from torchvision.datasets import ImageNet
dst = ImageNet("./data", split='train')
```

### 정규화 파라미터 조정

```python
# Total Variation 가중치 (큰 값 = 더 부드러운 이미지)
MIA_TV_WEIGHT = 0.01  # 기본: 0.001

# L2 정규화 (큰 값 = 더 작은 값으로 제약)
MIA_L2_WEIGHT = 0.001  # 기본: 0.0001
```

## 📝 코드 구조

```
DLG/sparsification/
├── fedavg_mia.py                 # 메인: FedAvg + MIA + Sparsification
├── batch_mia_experiments.py      # 배치 실험 자동화
├── compare_mia_results.py        # 결과 비교 및 시각화
├── utils.py                      # 유틸리티 함수 (레이블 변환 등)
├── models/
│   └── vision.py                 # ResNet18, LeNet 등 모델 정의
└── README.md                     # 이 파일

```

## 🐛 트러블슈팅

### 메모리 부족 에러

```bash
# MIA 반복 횟수 감소
python fedavg_mia.py --sparsity 0.05 --mia_iters 100

# 또는 배치 크기 감소 (코드 내부 수정)
```

### 느린 실행 속도

```bash
# GPU 사용 확인
python -c "import torch; print(torch.cuda.is_available())"

# LBFGS 반복 횟수 감소
python fedavg_mia.py --mia_iters 200
```

### 복원 실패 (손실 값이 감소하지 않음)

```bash
# 정규화 파라미터 조정
# fedavg_mia.py 내부에서:
MIA_TV_WEIGHT = 0.01  # 더 큰 값
MIA_L2_WEIGHT = 0.001

# 또는 학습률 조정
# optimizer = torch.optim.LBFGS([...], lr=0.5)  # 기본: 1.0
```

## 📚 참고 자료

### 관련 논문

- **Deep Leakage from Gradients** (DLG)
  - Zhu et al., NeurIPS 2019
  - 그래디언트로부터 개인정보 추출

- **Federated Learning**
  - McMahan et al., AISTATS 2017
  - 분산 학습 및 프라이버시

- **Gradient Sparsification**
  - 통신 효율성 및 프라이버시 향상
  - 상위-k 선택 메커니즘

## 💡 주요 인사이트

1. **Sparsity가 낮을수록 프라이버시 보호 증가**
   - 1%만 전송 → 공격 난이도 매우 높음

2. **프라이버시-유틸리티 트레이드오프**
   - 과도한 압축 → 모델 성능 저하
   - 최적점 찾기 필요

3. **상위-k 선택 메커니즘의 효과**
   - 절댓값이 큰 그래디언트만 유지
   - 작은 값(노이즈) 제거

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## 👤 작성자

Hansung University - AI & ML Research Lab

---

**마지막 업데이트**: 2026년 2월

**버전**: 1.0
