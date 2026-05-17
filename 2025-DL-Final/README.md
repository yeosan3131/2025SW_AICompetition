# ? 2025 Deep Learning Final Challenge
> **교내 딥러닝이론및응용 과정 최종 학기말 AI 모델링 컴페티션**
> 본 프로젝트는 딥러닝 아키텍처의 핵심 요소를 직접 커스텀 설계하고, 다양한 하이퍼파라미터 비교 실험 및 규제(Regularization) 기법을 적용하여 최적의 모델 수렴 상태를 달성한 프로젝트입니다.

---

## ? 핵심 엔지니어링 포인트 (Key Highlights)
- **커스텀 레이어 설계 역량:** 주어진 기본 베이스라인에 의존하지 않고, MLP 및 CNN 기반의 커스텀 딥러닝 레이어 구조 변경 실험을 주도하여 구조에 따른 표현력(Capacity) 변화 실증
- **철저한 과적합 방지(Regularization):** 학습 데이터가 소수이거나 특정 패턴에 치우쳐 과적합되는 현상을 방지하기 위해 Dropout, Weight Decay 레이어 옵션을 다각도로 튜닝
- **데이터 기반 Parameter 수렴 실험:** `Adam`, `SGD` 등 옵티마이저의 모멘텀 계수 및 Learning Rate 변동에 따른 오차 역전파(Backpropagation) 수렴 속도를 정밀 비교 분석

---

## ? Tech Stacks
- **Framework:** PyTorch
- **Libraries:** NumPy, SciPy, Scikit-Learn, Matplotlib (Loss 추이 시각화)

---

## ? 학습 제어 전략 (Training Strategy)
- **Metric 추적 기반 모델 세이빙:** 무조건 마지막 에폭의 가중치를 저장하는 것이 아니라, Validation Score가 가장 높았던 최적의 시점(Best Checkpoint)의 가중치를 추적 및 유실 없이 세이빙하는 파이프라인 가동
- 모델 구조 변경에 따른 성능 편차를 단순히 '감'이 아닌 **실험 메트릭 로그 데이터**에 기반하여 정량적으로 분석 및 리포트 작성