# FedAvg Model Inversion Attack with Sparsification - 완성 보고서

## 🎉 프로젝트 완성

사용자님의 요청사항이 모두 완성되었습니다!

### ✅ 구현 완료 항목

1. **✅ FedAvg 환경에서 Model Inversion Attack (MIA)**
   - 클라이언트 로컬 학습 시뮬레이션
   - 스파시파이된 그래디언트로부터 데이터 복원 공격
   - 강력한 공격 알고리즘 (LBFGS 옵티마이저)

2. **✅ Sparsification 비율 제어**
   - 상위 N% 파라미터만 전송 가능
   - 유연한 비율 설정 (0.01 ~ 1.0)
   - 예: `--sparsity 0.05` = 상위 5%만 전송

3. **✅ 기존 코드베이스 유지**
   - 기존 `dlg_fedavg.py`, `dlg_fedavg_v2.py` 등 호환성 유지
   - 기존 `models/vision.py`, `utils.py` 재사용
   - 새로운 기능이 추가된 형태

## 📦 제공 파일 목록

### 🔴 메인 실행 스크립트 (필수)
```
✓ fedavg_mia.py                 [새로 작성] - 핵심 스크립트
```

### 🟡 자동화 & 배치 처리
```
✓ batch_mia_experiments.py      [새로 작성] - 여러 sparsity 자동 실행
✓ quick_test_mia.py             [새로 작성] - 빠른 검증용
```

### 🟢 결과 분석
```
✓ compare_mia_results.py        [새로 작성] - 결과 비교 및 시각화
```

### 📘 문서 (상세 설명)
```
✓ README_MIA.md                 [새로 작성] - 상세 사용 설명서
✓ EXAMPLES.md                   [새로 작성] - 다양한 사용 예제
✓ INTEGRATION_GUIDE.md          [새로 작성] - 기존 코드와의 통합 가이드
✓ mia_config.ini                [새로 작성] - 실험 설정 파일
```

### 🔵 기존 파일 (변경 없음)
```
✓ models/vision.py              [기존] - ResNet18, LeNet 등
✓ utils.py                      [기존] - 유틸리티 함수
✓ dlg_fedavg.py                 [기존] - 기존 DLG 코드
✓ dlg_fedavg_v2.py              [기존] - 기존 DLG 코드
```

## 🚀 빠른 시작

### 1️⃣ 기본 사용법 (가장 간단)
```bash
# 상위 5% 파라미터만 사용하여 MIA 공격
cd /Users/joseoglae/hansung/Gong/DLG/sparsification
python fedavg_mia.py --sparsity 0.05 --index 25
```

### 2️⃣ 여러 sparsity 비율 비교
```bash
# 5가지 sparsity (100%, 50%, 10%, 5%, 1%) 자동 실행
python batch_mia_experiments.py --index 25

# 결과 비교 & 시각화
python compare_mia_results.py --sparsities 1.0 0.5 0.1 0.05 0.01
```

### 3️⃣ 빠른 테스트 (시스템 확인)
```bash
# 50번 반복만 하는 빠른 테스트
python quick_test_mia.py
```

## 📊 Sparsity 비율 설명

| Sparsity | 설명 | 장점 | 단점 |
|----------|------|------|------|
| **1.0** (100%) | 모든 파라미터 전송 | 공격 성공률 높음 | 프라이버시 없음 |
| **0.5** (50%) | 절반의 파라미터 | 균형잡힘 | - |
| **0.1** (10%) | 상위 10% 만 | 네트워크 효율 | 복원 어려움 |
| **0.05** (5%) | 상위 5% 만 | 강한 압축 | 매우 어려운 복원 |
| **0.01** (1%) | 상위 1% 만 | 최강 프라이버시 | 거의 불가능한 복원 |

## 💡 핵심 개선 사항

### 이전 (기존 코드)
```python
# 고정된 sparsity로만 테스트 가능
python dlg_fedavg.py --sparsity 0.1
python dlg_fedavg.py --sparsity 0.01
# 수동으로 결과 비교...
```

### 현재 (신규 코드)
```python
# 유연한 비율 조절 + 자동 배치 처리
python batch_mia_experiments.py \
    --sparsities 1.0 0.5 0.1 0.05 0.01

# 자동 비교 분석
python compare_mia_results.py
```

## 📈 생성되는 파일 (예시)

### 단일 실험 결과
```
mia_ground_truth.png              # 원본 이미지
mia_progress_sparsity_5.png       # 복원 과정 시각화
mia_loss_sparsity_5.png           # 손실 함수 수렴 곡선
mia_final_sparsity_5.png          # 최종 결과 비교
mia_gradient_dist_sparsity_5.png  # 그래디언트 분포
```

### 비교 분석 결과
```
mia_comparison_tradeoff.png       # 프라이버시-유틸리티 곡선
mia_comparison_compression.png    # 압축 효율성
mia_comparison_difficulty.png     # 공격 난이도 분석
mia_comparison_report.txt         # 상세 분석 리포트
```

## 🔍 기술 특징

### 1. Sparsification 메커니즘
```python
# 상위 k% 그래디언트만 유지
ratio = 0.05  # 상위 5%
k = int(total_params * ratio)
# 절댓값이 큰 순서대로 선택
top_values = torch.topk(abs(gradients), k)
```

### 2. FedAvg 시뮬레이션
```python
# 1. 클라이언트 로컬 학습
net.train()
pred = net(gt_data)
loss.backward()

# 2. 그래디언트 스파시피케이션
sparsify_gradients(net, ratio)

# 3. 공격자가 스파시파이된 그래디언트 수신
target_gradients = [g.clone() for g in net.parameters()]
```

### 3. Model Inversion Attack
```python
# 임의 노이즈로 시작
dummy_data = torch.randn(...)

# 스파시파이된 그래디언트와 일치하도록 최적화
for iter in range(mia_iters):
    grad_diff = compute_gradient_difference()
    grad_diff.backward()  # 역전파로 dummy_data 업데이트
```

## 📋 주요 파라미터

### 공격 강도 조절
```bash
# 약한 공격 (빠름)
python fedavg_mia.py --mia_iters 100 --sparsity 1.0

# 중간 공격 (기본)
python fedavg_mia.py --mia_iters 300 --sparsity 0.5

# 강한 공격 (느림)
python fedavg_mia.py --mia_iters 1000 --sparsity 0.1
```

### 로컬 학습 조절
```bash
# 적은 학습
python fedavg_mia.py --local_epochs 1

# 많은 학습
python fedavg_mia.py --local_epochs 10
```

### 재현성 확보
```bash
# 고정 시드로 재현 가능
python fedavg_mia.py --seed 42
```

## 🎯 추천 사용 시나리오

### 📌 시나리오 1: 프라이버시 효과 분석
```bash
# 다양한 sparsity에서의 프라이버시 효과 비교
python batch_mia_experiments.py \
    --sparsities 1.0 0.5 0.2 0.1 0.05 0.02 0.01
python compare_mia_results.py
```

### 📌 시나리오 2: 최적 sparsity 찾기
```bash
# 정확도와 프라이버시의 균형점 찾기
python batch_mia_experiments.py
```

### 📌 시나리오 3: 새로운 이미지로 강건성 검증
```bash
# 다양한 이미지로 모델의 일반성 확인
for idx in 10 25 50 100; do
    python fedavg_mia.py --index $idx --sparsity 0.05
done
```

## ✅ 확인 체크리스트

- [x] FedAvg 환경 구현
- [x] Model Inversion Attack 구현
- [x] Sparsification 메커니즘 구현
- [x] 유연한 비율 제어 (0.01~1.0)
- [x] 배치 자동화 처리
- [x] 결과 비교 기능
- [x] 시각화 생성
- [x] 상세 문서 작성
- [x] 기존 코드 호환성 유지

## 📚 제공 문서

| 파일 | 용도 |
|------|------|
| `README_MIA.md` | 전체 기능 설명서 |
| `EXAMPLES.md` | 사용 사례 및 예제 |
| `INTEGRATION_GUIDE.md` | 기존 코드와의 통합 |
| `mia_config.ini` | 실험 설정 파일 |

## 🔧 설치 및 실행

### 1. 폴더 이동
```bash
cd /Users/joseoglae/hansung/Gong/DLG/sparsification
```

### 2. 필요 패키지 확인
```bash
python -c "import torch, torchvision, numpy, matplotlib; print('OK')"
```

### 3. 빠른 테스트 (선택)
```bash
python quick_test_mia.py
```

### 4. 메인 실행
```bash
python fedavg_mia.py --sparsity 0.05 --index 25
```

## 💬 주요 기능 요약

| 기능 | 설명 | 파일 |
|-----|------|------|
| **단일 MIA** | 한 번에 하나의 sparsity로 공격 | `fedavg_mia.py` |
| **배치 처리** | 여러 sparsity 자동 순차 실행 | `batch_mia_experiments.py` |
| **결과 비교** | 모든 sparsity 결과 시각화 | `compare_mia_results.py` |
| **빠른 검증** | 시스템 동작 확인 (50반복) | `quick_test_mia.py` |

## 🎓 학습 결과

이 구현을 통해 다음을 이해할 수 있습니다:

1. **Federated Learning의 프라이버시 위험**
   - 그래디언트로부터 데이터 복원 가능

2. **Gradient Sparsification의 효과**
   - 적은 파라미터로도 상당한 프라이버시 보호

3. **프라이버시-성능 트레이드오프**
   - 높은 압축 = 강한 프라이버시 BUT 모델 성능 저하

4. **공격의 난이도**
   - 극강 압축(1%)에서 거의 불가능한 복원

## 🚀 다음 단계 (선택사항)

1. **다른 데이터셋 적용**
   - CIFAR-100, ImageNet 등

2. **다른 모델 테스트**
   - ResNet50, VGG, EfficientNet 등

3. **다른 공격 방법 적용**
   - Analytical attack, Bayesian attack 등

4. **방어 메커니즘 추가**
   - Differential privacy, DP-SGD 등

## 📞 지원

- 📖 **상세 설명**: `README_MIA.md` 참조
- 💡 **사용 예제**: `EXAMPLES.md` 참조
- 🔄 **통합 가이드**: `INTEGRATION_GUIDE.md` 참조

---

## 🎉 완성!

모든 요청사항이 구현되었습니다:

✅ FedAvg 환경에서 Model Inversion Attack  
✅ Sparsification으로 파라미터 비율 조절  
✅ 상위 N% 선택 가능 (예: 0.05 = 5%)  
✅ 기존 코드베이스 유지  
✅ 자동 배치 처리  
✅ 결과 비교 분석  
✅ 상세 문서 제공  

**Happy Researching! 🎓**

---

**Version**: 1.0  
**Date**: 2026-02-22  
**Status**: ✅ COMPLETE
