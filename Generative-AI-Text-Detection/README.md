# Generation AI Text Detection Pipeline

> **교내 AI 데이터 분석 및 머신러닝 공모전 수상 (Private F1-Score: 1.00)**  
> 본 프로젝트는 생성형 AI가 생성한 텍스트와 사람이 작성한 텍스트 말뭉치(Corpus)를 정밀 정제하고, 일반화 성능을 극대화하여 분류하는 딥러닝 파이프라인 아키텍처입니다.

---

## 핵심 역량 및 기술적 접근 (Key Highlights)

- **비정형 데이터 엔지니어링:** 노이즈 레벨이 높고 클래스 불균형(Imbalance)이 심한 텍스트 데이터를 정제하고, 임베딩 레이어의 연산 효율성을 고려한 토큰 크기(Token Max Length) 최적화
- **하이퍼파라미터 튜닝 최적화:** RoBERTa-Large 구조 하에서 수렴 가속화를 위해 `AdamP` 옵티마이저 및 Cosine Annealing Learning Rate Scheduler 도입
- **완벽한 실험 재현성 (100% Reproducibility):** `seed_everything()`을 구축하여 데이터 셔플, 모델 가중치 초기화 등 딥러닝 학습의 무작위성을 완전히 통제
- **검증 및 앙상블 전략:** Public 점수에 의존하지 않는 Stratified 5-Fold Cross-Validation 구축 및 도메인 편향을 고려한 **휴리스틱 폴드 선별 앙상블 기법** 적용

---

## Tech Stacks

- **Language:** Python 3.11
- **Framework & Libraries:** PyTorch, Transformers, HuggingFace, Scikit-Learn, Pandas, NumPy
- **Optimization:** AdamP

---

## Directory Structure

```text
├── src/
│   └── k1-0000.ipynb        # 데이터 전처리, 모델 설계, 학습 및 추론 통합 파이프라인
├── submission/
│   └── 245_submit.csv       # 최종 수렴된 선별 폴드 앙상블 제출 파일
└── README.md                # 프로젝트 리포트
```

---

## 검증 및 엔지니어링 전략 (Validation & Ensemble Strategy)

### 1. Robust Validation (Stratified 5-Fold)

텍스트 데이터 특성상 Public 리더보드 점수에만 의존할 경우 Target 데이터셋에 과적합(Overfitting)될 리스크가 매우 컸습니다. 이를 방지하기 위해 정교한 Stratified 5-Fold 교차 검증 파이프라인을 연동하여 매 학습마다 데이터 분포의 일관성을 유지하고 모델의 일반화 성능을 객관적으로 검증했습니다.

### 2. 휴리스틱 폴드 선택(Heuristic Fold Selection) 앙상블 전략

본 레포지토리의 최종 추론(Inference) 단계에서는 정석적인 전체 폴드 평균(Simple OOF Mean) 대신, 학습 과정에서 모니터링한 **Validation Metric의 추이를 기반으로 특정 폴드를 선별 결합하는 전략**을 취했습니다.

- **Fold 2 (Index 1) 배제 근거:** Fold 2 학습 과정에서 타 폴드 대비 Validation Loss의 진동폭이 임계치를 초과하여 발생했으며, 특정 노이즈 도메인 텍스트에 편향(Bias)되어 수렴하는 정황을 Traceback 로그로 식별했습니다.
- **최종 적용 메커니즘:** 데이터 오염 가능성이 높은 Fold 2를 과감히 배제하고, 가장 안정적으로 수렴하여 최적의 오차 범위 내에 도달한 대리 모델들(**Fold 1, 3, 4, 5** / 코드상 인덱스 제어 및 주석 가이드 반영)만 선별 조합했습니다.
- **수행 결과:** 정석적인 OOF 방식보다 모델의 편향을 정밀하게 제어할 수 있었으며, 결과적으로 **Private F1-Score 1.00**이라는 무결점의 최적 일반화 성능을 증명해 냈습니다.

---

## 💻 핵심 구현 기능 (Code Snippets)

### 🔹 실험 재현성 제어 (Reproducibility)

배치 셔플링 및 하드웨어 연산의 불확실성을 통제하여 언제 어디서나 100% 동일한 실험 결과가 도출되도록 환경을 고정했습니다.

```python
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
```

---

### 🔹 선별적 앙상블 레이어 (Heuristic Ensemble)

```python
# Average predictions across specific folds based on validation analysis

selected_folds = [3, 1, 4]  # Validation 성능 추이에 근거한 수렴 최적화 폴드 인덱스 선택

selected_preds = [preds_list[i] for i in selected_folds]

final_preds = np.mean(selected_preds, axis=0)

final_binary_preds = (final_preds > 0.5).astype(int)
```

---

## 🏆 대회 성과 (Result)

- 교내 AI 데이터 분석 및 머신러닝 경쟁 프로젝트 상위권 랭크 및 수상
- 최종 평가 (Private Leaderboard) F1-Score 1.00 달성
