# ? Hallym NLP Competition 2025
> **교내 자연어 처리(NLP) 전공 과정 연계 캐글 모델 경쟁 프로젝트 (상위권 기록)**
> 본 프로젝트는 텍스트 말뭉치(Corpus) 데이터의 특성을 분석하고, 최적의 토큰화(Tokenization) 및 NLP 모델 설계 실험을 통해 텍스트 분류 성능을 극대화한 레포지토리입니다.

---

## ? 핵심 엔지니어링 포인트 (Key Highlights)
- **자연어 전처리 파이프라인 최적화:** 텍스트 데이터 내 포함된 불필요한 특수문자, HTML 태그, 공백 등 복잡한 노이즈를 정규표현식(Regex)을 활용해 정밀 정제
- **토큰화 배치 가이드 구축:** 모델 구조 및 하드웨어(GPU VRAM) 리소스를 고려하여 패딩(Padding) 및 절단(Truncation) 임계치(`max_length`)를 유기적으로 튜닝, 임베딩 레이어의 연산 효율성 확보
- **비교 실험을 통한 아키텍처 검증:** 활성화 함수(Activation Function), 옵티마이저 파라미터 변경에 따른 성능 추이를 모니터링하여 최적의 자연어 분류 성능 도출

---

## ? Tech Stacks
- **Language:** Python
- **Libraries:** Transformers, HuggingFace, Scikit-Learn, Pandas, NumPy

---

## ? 실험 및 검증 결과 (Experiment & Evaluation)
- **Train / Validation Split:** 수집된 데이터셋을 학습 및 검증용으로 철저히 분리하여 에폭(Epoch)별 Overfitting 유무를 실시간 모니터링
- 학습률 스케줄러(Learning Rate Scheduler)의 감쇄 주기를 미세 조정하여 Local Minima에 빠지지 않고 안정적으로 글로벌 손실값(Loss)이 수렴하도록 엔지니어링 수행