# FedAvg MIA Sparsification - 기존 코드와의 통합 가이드

## 📦 새로 추가된 파일

### 메인 스크립트
- **`fedavg_mia.py`** - FedAvg + Model Inversion Attack + Gradient Sparsification
  - 단일 sparsity 비율로 한 번에 한 실험 수행
  - 가장 기본이 되는 메인 스크립트

### 자동화 및 배치 처리
- **`batch_mia_experiments.py`** - 여러 sparsity 비율을 순차적으로 실행
  - 여러 실험을 자동으로 관리
  - 각 실험의 타임아웃 처리

### 결과 분석
- **`compare_mia_results.py`** - 실험 결과 비교 및 시각화
  - 프라이버시-유틸리티 트레이드오프 곡선
  - 압축 효율성 분석
  - 공격 난이도 비교

### 테스트 및 예제
- **`quick_test_mia.py`** - 시스템 동작 확인용 빠른 테스트
  - 적은 반복으로 빠르게 확인
  - 3가지 기본 sparsity 테스트

### 문서
- **`README_MIA.md`** - 상세 사용 설명서
- **`EXAMPLES.md`** - 다양한 사용 사례 예제
- **`mia_config.ini`** - 실험 설정 파일

### 설정 및 상수
- **`mia_config.ini`** - 모든 실험 파라미터 설정

## 🔄 기존 코드와의 관계

### 기존 파일들 (변경 없음)
```
models/vision.py          ✓ 기존 사용 (ResNet18, LeNet 등)
utils.py                  ✓ 기존 사용 (label_to_onehot, cross_entropy_for_onehot)
dlg_fedavg.py            - 기존 코드 (호환성 유지)
dlg_fedavg_v2.py         - 기존 코드 (호환성 유지)
FL.py                    - 기존 코드 (호환성 유지)
```

### 개선 사항
1. **더 명확한 구조**: FedAvg → Sparsification → MIA 단계별 명확화
2. **더 유연한 sparsity 제어**: 비율 기반 조절 (기존: 고정값)
3. **자동 배치 처리**: 여러 실험을 한 번에 실행
4. **결과 비교 기능**: 다양한 sparsity의 효과 비교
5. **더 나은 시각화**: 프라이버시-유틸리티 트레이드오프 표시

## 💡 기존 코드 대비 개선점

### `dlg_fedavg.py` → `fedavg_mia.py`

| 항목 | 기존 | 신규 |
|-----|-----|-----|
| Sparsity 설정 | 고정값 테스트 | 유연한 비율 (0.01~1.0) |
| 단계 명확성 | 섞여있음 | 1-7단계로 명확히 구분 |
| 출력 정보 | 기본 정보 | 상세한 통계 & 리포트 |
| 배치 처리 | 수동 | 자동화된 배치 스크립트 |
| 결과 비교 | 수동 비교 | 자동 비교 & 시각화 |
| 에러 처리 | 기본 | 개선된 에러 처리 |

## 🔄 마이그레이션 가이드

### 기존 코드를 새 코드로 전환

**Before (기존 방식):**
```bash
# 개별 실행, 비율을 코드에 하드코딩
python dlg_fedavg.py --sparsity 0.1  # 10% 테스트
python dlg_fedavg.py --sparsity 0.05 # 5% 테스트
# 수동으로 결과 비교...
```

**After (신규 방식):**
```bash
# 한 번에 모든 비율 테스트
python batch_mia_experiments.py --sparsities 1.0 0.5 0.1 0.05 0.01

# 결과 자동 비교 & 시각화
python compare_mia_results.py --sparsities 1.0 0.5 0.1 0.05 0.01
```

## 📊 호환성 체크리스트

- ✅ `models/vision.py` - 동일한 모델 사용
- ✅ `utils.py` - 동일한 유틸리티 함수
- ✅ CIFAR-10 데이터셋 - 동일하게 사용
- ✅ ResNet18 아키텍처 - 동일하게 사용
- ✅ DLG/MIA 공격 원리 - 개선된 구현

## 🚀 추천 사용 순서

### 1단계: 시스템 검증
```bash
# 빠른 테스트로 시스템 동작 확인
python quick_test_mia.py
```

### 2단계: 단일 실험
```bash
# 특정 sparsity로 한 실험 수행
python fedavg_mia.py --sparsity 0.05 --index 25
```

### 3단계: 배치 실험
```bash
# 여러 sparsity 자동 실행
python batch_mia_experiments.py --index 25
```

### 4단계: 결과 분석
```bash
# 결과 비교 및 시각화
python compare_mia_results.py
```

## 📝 주요 파라미터 설명

### Sparsity Ratio (가장 중요!)
```
--sparsity 1.0   # ✅ 모든 파라미터 전송 (기준선)
--sparsity 0.5   # 50% 전송 (약간의 압축)
--sparsity 0.1   # 10% 전송 (중간 수준 압축)
--sparsity 0.05  # 5% 전송 (강한 압축)
--sparsity 0.01  # 1% 전송 (극강 압축)
```

### 실험 품질 파라미터
```
--mia_iters 300      # 기본 (빠름)
--mia_iters 500      # 더 정교한 복원
--mia_iters 1000     # 최고 정확도 (느림)

--local_epochs 1     # 기본
--local_epochs 5     # 더 많은 로컬 학습

--seed 1234          # 재현성을 위한 고정 시드
```

## 🎯 전형적인 실험 시나리오

### 시나리오 1: 빠른 검증
```bash
python quick_test_mia.py
```
- 소요 시간: ~5-10분
- 3가지 sparsity 테스트

### 시나리오 2: 표준 실험
```bash
python batch_mia_experiments.py --mia_iters 300
python compare_mia_results.py
```
- 소요 시간: ~30-60분
- 5가지 sparsity 테스트
- 완전한 비교 분석

### 시나리오 3: 정밀 실험
```bash
python batch_mia_experiments.py \
  --sparsities 1.0 0.7 0.5 0.3 0.2 0.1 0.05 0.02 0.01 \
  --mia_iters 500

python compare_mia_results.py \
  --sparsities 1.0 0.7 0.5 0.3 0.2 0.1 0.05 0.02 0.01
```
- 소요 시간: ~2-3시간
- 9가지 sparsity 세밀 비교
- 최적 sparsity 찾기

## 📊 생성되는 파일들

### 개별 실험 파일 (각 sparsity마다)
```
mia_ground_truth.png              # 원본 이미지
mia_initial_dummy.png             # 초기 노이즈
mia_progress_sparsity_5.png       # 복원 진행 과정 (10단계)
mia_loss_sparsity_5.png           # 손실 함수 수렴 곡선
mia_final_sparsity_5.png          # 최종 결과 비교
mia_gradient_dist_sparsity_5.png  # 그래디언트 분포 분석
```

### 비교 분석 파일
```
mia_comparison_tradeoff.png       # 프라이버시-유틸리티 곡선
mia_comparison_compression.png    # 압축 효율성
mia_comparison_difficulty.png     # 공격 난이도
mia_comparison_summary_table.png  # 요약 테이블
mia_comparison_report.txt         # 상세 텍스트 리포트
```

## 🔧 트러블슈팅

### 메모리 부족
```bash
# MIA 반복 감소
python fedavg_mia.py --mia_iters 100

# 또는 배치 실험에서 한 번에 하나만 실행
python fedavg_mia.py --sparsity 0.05
```

### 느린 실행
```bash
# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"

# 빠른 테스트 먼저
python quick_test_mia.py
```

### 복원 실패
```bash
# fedavg_mia.py 내부에서 정규화 파라미터 조정
MIA_TV_WEIGHT = 0.01  # 기본값: 0.001
MIA_L2_WEIGHT = 0.001  # 기본값: 0.0001
```

## 📚 추가 참고

### 기존 코드 유지
- `dlg_fedavg.py`, `dlg_fedavg_v2.py`, `FL.py`는 그대로 유지
- 새 코드와 독립적으로 사용 가능
- 필요시 병렬로 실행 가능

### 성능 최적화
- GPU 사용 권장 (실행 시간 10배 이상 단축)
- 배치 처리로 자동화 가능
- 결과 비교 기능으로 의사결정 시간 단축

## ✅ 체크리스트

### 설치 확인
- [ ] PyTorch 설치 확인
- [ ] torchvision 설치 확인
- [ ] matplotlib 설치 확인
- [ ] numpy 설치 확인

### 첫 실행
- [ ] `quick_test_mia.py` 실행 성공
- [ ] 출력 파일 생성 확인
- [ ] GPU 사용 여부 확인

### 완전 실험
- [ ] `batch_mia_experiments.py` 완료
- [ ] `compare_mia_results.py` 시각화 생성
- [ ] 결과 분석 및 해석

---

**Version**: 1.0  
**Last Updated**: 2026-02-22  
**Compatible with**: 기존 DLG 코드베이스
