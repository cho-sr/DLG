# DLG with FedAvg + Gradient Sparsification

ResNet18 기반 DLG 공격 with FedAvg 환경 및 Gradient Sparsification

## 주요 개선사항

✅ **모델**: LeNet → ResNet18  
✅ **병목 제거**: Layer2/3/4 stride=2 (메모리/연산량 64배 감소)  
✅ **FedAvg**: Local training simulation  
✅ **Sparsification**: Top-k% gradient retention  
✅ **조절 가능**: argparse로 sparsity ratio 변경  

## 빠른 시작

### 1. 기본 실행 (100%, sparsification 없음)

```bash
cd /Users/joseoglae/hansung/Gong/DLG/sparsification
python3 dlg_fedavg.py --index 25 --sparsity 1.0
```

### 2. Sparsity 조절

```bash
# 10% gradient retention (top 10%)
python3 dlg_fedavg.py --index 25 --sparsity 0.1

# 1% gradient retention (top 1%)
python3 dlg_fedavg.py --index 25 --sparsity 0.01

# 50% gradient retention
python3 dlg_fedavg.py --index 25 --sparsity 0.5
```

### 3. 다양한 이미지 테스트

```bash
# 첫 번째 이미지
python3 dlg_fedavg.py --index 0 --sparsity 1.0

# 100번째 이미지
python3 dlg_fedavg.py --index 100 --sparsity 0.1
```

## 전체 파라미터

```bash
python3 dlg_fedavg.py \
    --index 25 \              # CIFAR-10 이미지 인덱스 (0-49999)
    --sparsity 1.0 \          # Gradient retention ratio (0.01-1.0)
    --local_epochs 1 \        # FedAvg local training epochs
    --lr 0.01 \               # Local learning rate
    --dlg_iters 300 \         # DLG optimization iterations
    --seed 1234               # Random seed
```

## 출력 파일

실행 후 자동 생성:
- `dlg_progress_sparsity_100.png` - Reconstruction 진행 과정
- `dlg_loss_sparsity_100.png` - Loss convergence curve
- `dlg_final_sparsity_100.png` - Ground truth vs Reconstructed

## 예상 결과

### Sparsity 100% (No sparsification)
- **Gradient Loss**: 500 → 0.05 (수렴)
- **PSNR**: ~25-30 dB
- **품질**: 거의 완벽한 reconstruction

### Sparsity 10%
- **Gradient Loss**: 500 → 1.0 (부분 수렴)
- **PSNR**: ~15-20 dB
- **품질**: 대략적인 형태 복원

### Sparsity 1%
- **Gradient Loss**: 500 → 50 (수렴 어려움)
- **PSNR**: ~10 dB
- **품질**: 매우 흐릿함, 프라이버시 보호 효과

## 병목 해결

### Before (stride=1 everywhere)
```python
self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)  # 32×32×128
self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)  # 32×32×256
self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)  # 32×32×512
```
- Feature map: 32×32 고정
- 메모리: ~8GB (batch_size=64)
- 연산량: 매우 큼

### After (proper downsampling)
```python
self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16×16×128
self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 8×8×256
self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4×4×512
```
- Feature map: 32×32 → 16×16 → 8×8 → 4×4
- 메모리: ~500MB ✅
- 연산량: 64배 감소 ✅

## Troubleshooting

### Loss가 감소하지 않음
```bash
# Iteration 증가
python3 dlg_fedavg.py --index 25 --sparsity 1.0 --dlg_iters 500

# Learning rate 조절
# dlg_fedavg.py line 247: lr=1.0 → lr=0.5
```

### 노이즈가 심함
```bash
# TV weight 증가 (dlg_fedavg.py line 60)
DLG_TV_WEIGHT = 0.001 → 0.01

# 또는 L2 weight 증가
DLG_L2_WEIGHT = 0.0001 → 0.001
```

### Sparsity가 너무 높음
```bash
# 더 작은 sparsity ratio 시도
python3 dlg_fedavg.py --index 25 --sparsity 0.001  # 0.1%
```

## 실험 예시

```bash
# 1. Baseline (100%)
python3 dlg_fedavg.py --index 25 --sparsity 1.0 --dlg_iters 300

# 2. Medium sparsity (10%)
python3 dlg_fedavg.py --index 25 --sparsity 0.1 --dlg_iters 400

# 3. High sparsity (1%)
python3 dlg_fedavg.py --index 25 --sparsity 0.01 --dlg_iters 500

# 결과 비교:
# - dlg_final_sparsity_100.png
# - dlg_final_sparsity_10.png
# - dlg_final_sparsity_1.png
```
