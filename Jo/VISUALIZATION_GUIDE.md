# 📊 시각화 가이드

## 🎨 생성되는 PNG 이미지

실험 실행 시 `results/` 폴더에 다음 3개의 PNG 파일이 생성됩니다:

### 1. comprehensive_metrics.png
**종합 성능 메트릭 (4-subplot)**

#### 구성:
- **좌상 (Top-left)**: Top-1 vs Top-5 Accuracy
  - X축: Gradient retention ratio (log scale)
  - Y축: Accuracy (%)
  - 두 선 비교: Top-1 (파란색), Top-5 (녹색)

- **우상 (Top-right)**: Precision, Recall, F1-Score
  - X축: Gradient retention ratio (log scale)
  - Y축: Score (%)
  - 세 선 비교: Precision (주황), Recall (보라), F1 (노랑)

- **좌하 (Bottom-left)**: DLG MSE (Privacy)
  - X축: Gradient retention ratio (log scale)
  - Y축: MSE (log scale)
  - Sparsification이 privacy 보호에 미치는 영향

- **우하 (Bottom-right)**: Performance Summary Table
  - 각 케이스별 Accuracy, Precision, F1 요약
  - 색상 코딩으로 가독성 향상

#### 해석:
- 모든 메트릭을 한눈에 비교
- Privacy-Utility trade-off 확인
- 최적의 sparsification ratio 선택

---

### 2. dlg_convergence.png
**DLG 공격 수렴 곡선**

#### 구성:
- X축: DLG iteration (0-300)
- Y축: Reconstruction MSE (log scale)
- 3개 곡선: 100%, 10%, 1% sparsification

#### 해석:
- **수렴하는 곡선**: DLG 공격 성공 (낮은 MSE)
- **발산하는 곡선**: DLG 공격 실패 (높은 MSE)
- **평평한 곡선**: 최적화 정체

**예상 패턴:**
```
100% sparsity: 낮은 MSE로 수렴 (취약)
10% sparsity: 중간 MSE (적당한 보호)
1% sparsity: 높은 MSE로 발산 (강력한 보호)
```

---

### 3. reconstruction_comparison.png
**이미지 복원 품질 비교**

#### 구성:
- **상단 행**: 원본 이미지 vs 복원된 이미지들
  - 왼쪽: Original image (CIFAR-10)
  - 중간: 100% sparsification 복원
  - 오른쪽: 10%, 1% sparsification 복원

- **하단 행**: 차이 맵 (Difference maps)
  - 원본과 복원 이미지 간 L2 distance
  - 색상: 빨강(높은 차이) ~ 노랑(낮은 차이)
  - MSE 값 표시

#### 해석:
- **원본과 유사**: Privacy 취약 (DLG 성공)
- **완전히 다름**: Privacy 보호 (DLG 실패)
- **차이 맵이 빨강**: 높은 MSE, 강력한 보호

---

## 📁 파일 구조

```
/root/Jo/
├── main.py                              # 실험 메인 스크립트
├── view_results.py                      # 이미지 뷰어
└── results/                             # 결과 폴더
    ├── comprehensive_metrics.png        # 종합 메트릭
    ├── dlg_convergence.png             # DLG 수렴
    └── reconstruction_comparison.png    # 복원 비교
```

---

## 🚀 사용 방법

### 1. 실험 실행 (PNG 생성)
```bash
cd /root/Jo
python main.py
```

**출력:**
```
✅ Saved: results/comprehensive_metrics.png
✅ Saved: results/dlg_convergence.png
✅ Saved: results/reconstruction_comparison.png
```

### 2. 이미지 확인

#### 방법 A: 파일 탐색기
```bash
cd /root/Jo/results
ls -lh *.png
```

#### 방법 B: Python 뷰어
```bash
python view_results.py
```

#### 방법 C: 직접 열기
```bash
# Linux
xdg-open results/comprehensive_metrics.png

# Mac
open results/comprehensive_metrics.png

# Windows
start results/comprehensive_metrics.png
```

---

## 🎨 이미지 사양

### 기본 설정
```python
plt.savefig(
    'results/filename.png',
    dpi=150,              # 해상도
    bbox_inches='tight'   # 여백 최소화
)
```

### 파일 크기
- comprehensive_metrics.png: ~100-200 KB
- dlg_convergence.png: ~50-100 KB
- reconstruction_comparison.png: ~150-300 KB

### 이미지 크기
- Figure 1: 14×10 inches (2100×1500 px @ 150 DPI)
- Figure 2: 10×6 inches (1500×900 px @ 150 DPI)
- Figure 3: 가변 (케이스 수에 따라)

---

## 🔧 커스터마이징

### DPI 변경
```python
# main.py에서
plt.savefig('results/filename.png', dpi=300)  # 고해상도
plt.savefig('results/filename.png', dpi=100)  # 저해상도
```

### 파일 형식 변경
```python
plt.savefig('results/filename.pdf')  # PDF (벡터)
plt.savefig('results/filename.svg')  # SVG (벡터)
plt.savefig('results/filename.jpg')  # JPEG (압축)
```

### 투명 배경
```python
plt.savefig('results/filename.png', transparent=True)
```

---

## 📊 시각화 팁

### 1. 색상 의미
- **파란색/녹색**: FL 성능 (높을수록 좋음)
- **빨간색**: Privacy 위협 (낮을수록 좋음)
- **주황/보라**: 균형 지표

### 2. 패턴 해석
```
좋은 결과:
✅ High accuracy (85-95%)
✅ High top-5 (95-99%)
✅ High DLG MSE (1.0+)
✅ Balanced precision/recall

나쁜 결과:
❌ Low accuracy (<70%)
❌ Low DLG MSE (<0.1)
❌ Unbalanced metrics
```

### 3. 비교 포인트
- **100% vs 10%**: Utility 변화 (작을수록 좋음)
- **100% vs 1%**: Privacy 변화 (클수록 좋음)
- **Precision vs Recall**: 균형 확인

---

## 🎓 발표/논문용

### Figure 선택
1. **개요 설명**: comprehensive_metrics.png
2. **Privacy 분석**: dlg_convergence.png
3. **시각적 증거**: reconstruction_comparison.png

### 캡션 예시

**Figure 1:**
> "Comprehensive evaluation of Federated Learning performance under gradient sparsification. (a) Top-1 and Top-5 accuracy remain high even with 10% gradient retention. (b) Precision, Recall, and F1-score show balanced performance. (c) DLG reconstruction error increases dramatically with sparsification, indicating strong privacy protection. (d) Performance summary across all metrics."

**Figure 2:**
> "DLG attack convergence for different sparsification levels. The attack succeeds with full gradients (100%) but fails with sparse gradients (10%, 1%), demonstrating the privacy-preserving effect of gradient sparsification."

**Figure 3:**
> "Visual comparison of DLG reconstruction quality. Top row shows original and reconstructed images. Bottom row shows pixel-wise difference maps with MSE values. Sparse gradients lead to poor reconstruction, preserving privacy."

---

## 🐛 문제 해결

### 이미지가 생성되지 않음
```bash
# 1. 디렉토리 확인
mkdir -p results

# 2. 권한 확인
chmod 755 results

# 3. 재실행
python main.py
```

### 이미지가 잘림
```python
# main.py에서
plt.tight_layout()  # 추가
plt.savefig(..., bbox_inches='tight')  # bbox_inches 확인
```

### 해상도가 낮음
```python
# DPI 증가
plt.savefig('results/filename.png', dpi=300)
```

### 메모리 부족
```python
# 이미지 저장 후 메모리 해제
plt.savefig(...)
plt.close()  # ← 중요!
```

---

## 📚 추가 자료

### Matplotlib 문서
- [savefig 옵션](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
- [Figure 크기 조정](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html)

### 논문 작성 팁
- DPI: 300-600 (출판용)
- 형식: PDF or EPS (벡터)
- 폰트 크기: 8-12pt (가독성)

---

## ✅ 체크리스트

실험 후 확인:
- [ ] 3개 PNG 파일 생성됨
- [ ] 파일 크기 합리적 (< 1MB)
- [ ] 이미지 품질 양호
- [ ] 텍스트 가독성 확인
- [ ] 색상 구분 명확
- [ ] MSE 값 표시 정확

발표/논문용:
- [ ] DPI 300+ 설정
- [ ] 폰트 크기 적절
- [ ] 범례 위치 확인
- [ ] 축 레이블 명확
- [ ] 제목 설명적
- [ ] 색상 접근성 고려

---

## 🎯 요약

**생성되는 파일:**
1. ✅ comprehensive_metrics.png (종합 분석)
2. ✅ dlg_convergence.png (Privacy 보호)
3. ✅ reconstruction_comparison.png (시각적 증거)

**확인 방법:**
```bash
python main.py          # 실험 실행
python view_results.py  # 결과 확인
ls -lh results/*.png    # 파일 목록
```

**활용:**
- 논문/발표 자료
- 보고서 첨부
- 실험 기록
- 성능 분석

모든 이미지가 PNG로 저장되어 쉽게 공유하고 사용할 수 있습니다! 🎨
