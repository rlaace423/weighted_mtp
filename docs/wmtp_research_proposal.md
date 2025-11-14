# Weighted Multi-Token Prediction (WMTP) 학술 연구제안서
## — "Not All Tokens Are What You Need"의 이론·실증 통합 —

---

## 1. 초록 (Abstract)

대규모 언어모델(LLM)의 Multi-Token Prediction(MTP)은 동일 연산 예산에서 성능과 효율을 개선하는 패러다임으로 주목받고 있으나, 
표준 MTP는 예측하는 모든 미래 토큰에 균등 가중을 부여하여 비핵심 토큰에도 학습 자원을 낭비할 수 있다. 
본 제안은 "모든 토큰이 동등하게 중요하지 않다(Not All Tokens Are What You Need)"는 통찰을 바탕으로, 토큰별 중요도를 동적으로 반영하는 Weighted MTP(WMTP)를 정식화하고 실증한다.

**본 연구는 3가지 핵심 가중화 방식의 체계적 비교에 집중한다:**

1. **Baseline MTP**: 균등 가중치 (비교 기준선)
2. **Verifiable Critic WMTP**: 데이터셋 레이블 기반 가치함수 가중화 (RM 불필요)
3. **Rho-1 WMTP**: 참조 모델 비교 기반 연속 가중화

코드 생성·추론·일반 언어 이해 벤치마크에서 각 방식의 성능·효율·안정성 이득과 규모에 따른 스케일링 특성을 체계적으로 검증하고, 실용적 구현과 이론적 정당성을 논증한다.

---

## 2. 배경과 문제 정의

- **배경**: NTP(Next-Token Prediction) 대비 MTP는 한 시점에서 다수의 미래 토큰을 병렬 예측해 수렴 가속과 다운스트림 성능 향상을 보고하였다. 또한 병렬 생성·자기추측 디코딩과의 결합으로 추론 속도 이점도 가능하다.

- **문제**: 표준 MTP는 모든 미래 토큰을 균등 가중으로 다루어, 쉬운/비핵심 토큰에도 동일한 학습 자원을 배분한다. 이는 고비용 데이터 구간·장기 의존·결정적 토큰에서 비효율·불안정성을 유발할 수 있다.

- **핵심 가설**: 중요 토큰에 계산을 집중하는 WMTP는 동일 FLOPs에서 더 높은 성능과 안정적 수렴을 달성한다.

- **연구 범위**: 본 연구는 실용적으로 검증 가능한 3가지 가중화 방식(균등, 참조 비교, 검증 가능 가치함수)에 집중하며, 이론적·실증적 기반을 확립한다.

---

## 3. 관련 연구

- **MTP 계열**: 표준 MTP는 병렬 예측을 통해 효율을 높였고, 도약(leap) 예측 등은 장거리 의존·병렬성을 확장했다. (Glöckle et al. 2024, Meta MTP)

- **선택적/가중 학습**:
  - **Rho-1 (Lin et al. NeurIPS 2024)**: 참조 모델 기반 선택적 학습(Selective LM)은 어려운 토큰에 집중해 효율적 성능 향상을 달성. [OpenReview](https://openreview.net/forum?id=0NMzBwqaAJ), [NeurIPS 포스터](https://neurips.cc/virtual/2024/poster/96931), [HF 모델](https://huggingface.co/microsoft/rho-math-7b-v0.1)
  - **Verifiable Rewards**: 코드 실행 결과, 수학 정답 등 객관적 검증 신호를 활용한 학습 (CodeContests, MATH)

- **선호 최적화·정책경사**:
  - Policy Gradient Theorem (Sutton et al. 1999, [NIPS](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf))
  - GAE (Schulman et al. 2015, [arXiv](https://arxiv.org/abs/1506.02438))
  - Off-policy Actor-Critic (Degris et al. 2012, [arXiv](https://arxiv.org/abs/1205.4839))

**본 연구의 위치**: WMTP는 MTP의 병렬 예측 이점을 토큰-수준 중요도 가중과 결합하며, 실용적으로 구현 가능한 3가지 핵심 방식을 체계적으로 비교한다. Verifiable Critic은 RM 의존성을 제거하여 메모리 효율성과 객관성을 강화하고, Rho-1은 참조 모델 비교로 간결한 가중화를 제공한다.

---

## 4. 제안 방법: Weighted Multi-Token Prediction

### 4.1 목적함수 정식화

입력 x, 시점 t, 예측 범위 H에 대해 WMTP 손실은 다음과 같다.

```
L_WMTP = E_x [ Σ_t Σ_{k=1..H} w_{t,k} · CE( P_θ(x_{t+k} | x_{<t}), x_{t+k} ) ]
```

- **제약**: w_{t,k} ≥ 0, Σ_k w_{t,k} = 1
- **전제**: 동일 토큰 공간(토크나이저 일치), 동일 태스크의 시점 정렬(MTP k-헤드 ↔ t+k 예측)

### 4.2 본 연구의 3가지 가중화 방식

#### **방식 1: Baseline MTP (균등 가중)**

```
w_{t,k} = 1.0 (모든 k)
```

- **목적**: 비교 기준선 (Control Group)
- **데이터**: 정답만 학습 (표준 SFT)
- **특징**:
  - 구현 단순성
  - 모든 토큰 균등 학습
  - WMTP 방식들의 성능 이득 측정 기준

#### **방식 2: Verifiable Critic WMTP**

데이터셋의 검증 가능한 레이블(is_correct)을 reward signal로 사용하여 TD error 기반 토큰 가중치를 산출한다.

```
토큰 x_k의 가중치: w_k = exp(δ_k / β)  (β=0.9)

TD error (표준 Temporal Difference):
  # Intermediate tokens (k < T): Bootstrapping
  δ_k = r_k + γV(s_k) - V(s_{k-1})
      = 0 + γV(s_k) - V(s_{k-1})        # r_k = 0 (중간 토큰은 보상 없음)
      = γV(s_k) - V(s_{k-1})            ✅

  # Terminal token (k = T): Direct reward
  δ_T = r_T + γV(terminal) - V(s_{T-1})
      = R + 0 - V(s_{T-1})              # V(terminal) = 0
      = R - V(s_{T-1})                  ✅
```

**TD Error 공식 설명 (표준 RL)**:
- **MTP 시나리오**: 시점 t에서 미래 토큰 x_{t+1}, ..., x_{t+H} 예측
- **표준 TD Learning 적용**:
  - 토큰 x_k의 상태: `s_{k-1} = (x_0, ..., x_{k-1})` (토큰 x_k 생성 **전**)
  - 생성 후 상태: `s_k = (x_0, ..., x_k)` (토큰 x_k 생성 **후**)
- **Intermediate tokens (k < T)**:
  - `δ_k = γV(s_k) - V(s_{k-1})` (Bootstrapping)
  - 다음 상태의 가치 함수로 현재 토큰의 기여도 추정
  - 분산 감소 효과 (GAE의 기반)
- **Terminal token (k = T)**:
  - `δ_T = R - V(s_{T-1})` (Direct reward)
  - 실제 보상 R을 직접 사용 (bootstrapping 불필요)
- **R (reward)**: Terminal 누적 보상 (LLM에서 = 최종 보상, binary: 0 or 1)
- **γ (gamma)**: 할인 계수 (일반적으로 0.99, LLM에서는 1.0 가능)
- **직관**: δ_k는 "토큰 x_k를 선택한 것의 marginal value"
- **이론적 근거**:
  - Sutton & Barto "RL: An Introduction" - 표준 TD(0) 공식
  - TDRM (2024): Intermediate bootstrapping + Terminal direct reward
  - Policy Gradient Theorem (Sutton et al. 1999)

**핵심 특징**:
- **Reward 소스**: 데이터셋 레이블 (코드 실행 결과, 수학 정답)
- **RM 불필요**: ~28GB VRAM 절약
- **객관적 보상**: Ground truth 기반 신뢰성
- **데이터**: 정답+오답 모두 학습 (negative signal 활용)
- **학습 구조**: 2단계 (Pretrain 0.5 epoch + Main 2.5 epoch)

**이론적 정당성**:
- Policy Gradient Theorem: 이득(Advantage)으로 로그우도 가중 시 분산 감소와 수렴 보장 (Sutton et al. 1999)
- GAE: Temporal Difference와 Monte Carlo의 균형으로 분산 저감 (Schulman et al. 2015)

**적용 데이터셋**:
- CodeContests (507K solutions, 57.3% correct)
- MATH, MBPP with incorrect solutions

**RM Critic과의 비교**:

| 비교 항목 | RM Critic | Verifiable Critic |
|----------|-----------|-------------------|
| Reward 소스 | RM 모델 inference | 데이터셋 레이블 (is_correct) |
| RM 모델 필요 | 필수 (7B 모델) | 불필요 |
| 메모리 사용 | 높음 (Base + RM) | 낮음 (Base만, ~28GB 절약) |
| 학습 속도 | 느림 (RM inference) | 빠름 (RM overhead 제거) |
| Reward 신뢰성 | RM 품질에 의존 | Objective (실행 결과/정답) |
| 본 연구 포함 | ❌ 향후 확장 | ✅ 핵심 비교 대상 |

#### **방식 3: Rho-1 WMTP Weighted**

참조 모델과 Base 모델의 Cross-Entropy 차이를 계산하여 연속적 가중치를 부여한다.

```
excess_loss_{t,k} = |CE_ref(x_{t+k}) - CE_mtp(x_{t+k})|
w_{t,k} = softmax(excess_loss_{t,k} / T)
```

**핵심 특징**:
- **가중화**: Reference CE 차이를 softmax로 변환한 연속 가중치
- **온도 T = 1.0**: 표준 softmax (원논문 설정)
- **데이터**: 정답만 학습 (Reference CE 차이만 사용)
- **특징**: 모든 토큰에 연속적 가중치 부여 (부드러운 학습)

**이론적 정당성**:
- **정보 이론**: 토큰별 정보량이 상이하며, 정보 이득이 큰 토큰에 계산 집중으로 샘플 효율 개선
- **중요도 샘플링**: w는 중요도 가중 역할 수행, 적절한 정규화 하에 분산 감소와 안정 추정 기여

**원논문**: Lin et al. "Rho-1: Not All Tokens Are What You Need" (NeurIPS 2024 Oral)

### 4.3 3가지 방식의 체계적 비교

| 특성 | Baseline MTP | Verifiable Critic | Rho-1 Weighted |
|------|-------------|-------------------|----------------|
| **가중치 산출** | 상수 (1.0) | TD error (동적 학습) | Reference CE 차이 (정적 비교) |
| **외부 모델** | 불필요 | 불필요 (RM 제거) | 필요 (Reference) |
| **메모리 효율** | 표준 | 높음 | 표준 |
| **데이터 요구** | 정답만 | 정답+오답 | 정답만 |
| **학습 안정성** | 높음 | 중간 (Value drift 위험) | 높음 |
| **구현 복잡도** | 낮음 | 높음 (2단계 학습) | 중간 (Reference 관리) |
| **이론적 기반** | 표준 SFT | Policy Gradient | 정보 이론 |

### 4.4 실무적 고려(원칙)

- **토크나이저 일치**: 어휘·특수 토큰·분절 정책 불일치는 확률 비교를 왜곡한다. 최소한의 샘플링 검증으로 일치성을 담보.
- **시점 정렬**: 참조/선호/보상 신호는 동일 예측 태스크로 정렬되어야 하며, 헤드 k와 t+k 예측이 정확히 대응되도록 설계.
- **수치 안정성**: 온도 T, 가중치 엔트로피 범위, 가중치 클리핑, outlier 억제, 최소 엔트로피 제약 등을 적용.

---

## 5. 실험 설계

### 5.1 데이터셋·벤치마크

**학습 데이터**:
- CodeContests Train (100K samples, 3 epochs)
- 정답 vs 정답+오답 비교 (Baseline/Rho-1 vs Verifiable Critic)

**평가 벤치마크**:
- **코드 생성**: MBPP, HumanEval
- **수학/추론**: MATH, GSM8K (필요시)
- **일반 이해**: HellaSwag (필요시)
- **In-domain**: CodeContests test

### 5.2 모델·스케일

- **아키텍처**: MTP(H=4) 기반
- **규모**: ≈1B·≈7B 두 스케일로 비교해 스케일링 법칙 분석 (리소스에 따라 조정 가능)

### 5.3 핵심 비교 실험

#### **실험 1: 3가지 방식의 성능 비교**
- Baseline MTP vs Verifiable Critic vs Rho-1 Weighted
- 동일 학습 예산 (3 epochs, 100K samples)
- 지표: Pass@K, Exact Match, 과제별 정답률

#### **실험 2: 메모리 효율성 분석**
- Verifiable Critic의 RM 제거 효과 (~28GB 절약)
- 학습 속도 비교 (RM inference overhead)

#### **실험 3: 데이터 활용 효율성**
- 정답만 (Baseline, Rho-1) vs 정답+오답 (Verifiable Critic)
- Negative signal의 학습 기여도

#### **실험 4: 안정성 및 수렴 분석**
- 수렴 분산, 그래디언트 놈, 가중치 엔트로피 분포
- Value drift (Verifiable Critic), Reference alignment (Rho-1)

#### **실험 5: Ablation Studies**
- Verifiable Critic: 온도 T, GAE λ, Pretrain duration 스윕
- Rho-1: 온도 T, min_ce_diff 스윕
- 가중치 스파르시티(톱-p/톱-k) 영향

### 5.4 지표·통계·재현성

**성능 지표**:
- Exact Match, Pass@K (K=1,10,100)
- 과제별 정답률

**효율 지표**:
- 스텝당 FLOPs, 벽시계 시간, 수렴 스텝 수
- 메모리 사용량 (VRAM)

**안정성 지표**:
- 수렴 분산, 그래디언트 놈, 가중치 엔트로피 분포

**분석 지표**:
- 가중치 분포/엔트로피
- 토큰 난이도별 오류율
- 헤드별 성능 분석

**통계**:
- 비모수 검정(윌콕슨 등), 95% CI 부트스트랩
- 시드≥5 반복

**재현성**:
- 설정·로그·체크포인트 공개
- 시드·데이터 스플릿 고정

### 5.5 핵심 설계 제약·권고

- **토크나이저 강제 일치**: 학습·참조·정책(및 보상평가)이 모두 동일 토크나이저·특수토큰 체계를 사용하도록 강제. 최소 샘플 토큰 세트에 대한 ID 일치 자동 점검을 루프에 포함.

- **시점 정렬 유효성**: MTP 헤드 k ↔ t+k 예측이 항상 일치하도록 라벨 시프트·마스킹을 자동 검증.

- **Verifiable Critic 안정화**:
  - GAE(λ), 보상/이득 정규화, auxiliary loss (히든 상태 앵커, 가치-헤드 정규화)
  - KL/Trust-region으로 Value drift 완화
  - 2단계 학습 (Pretrain 0.5 epoch + Main 2.5 epoch)

- **Rho-1 안정화**:
  - min_ce_diff로 노이즈 필터링
  - 온도 T 조정으로 집중도 제어
  - 동일 토크나이저 강제

---

## 6. 리스크와 완화

**공통 리스크**:
- **토큰화/정렬 불일치**: w 오산정·손실 왜곡 → 동일 토크나이저 강제, 정렬 자동 검증, 실패 시 배치 제외

**Verifiable Critic 특유 리스크**:
- **Value drift**: 가치 표류로 가중 왜곡 → GAE(λ), KL/Trust-region, auxiliary 표현 고정화, 2단계 학습
- **Binary sparsity**: 0/1 보상의 sparsity → Z-score 정규화, 충분한 positive/negative 샘플 확보

**Rho-1 특유 리스크**:
- **참조 비용**: Reference 과도 사용 시 비용 급증 → 샘플링/주기화, 캐시·오프라인 스코어링
- **토큰화 불일치**: 참조 모델 비교 왜곡 → 동일 토크나이저 강제, 자동 검증

---

## 7. 윤리·안전·사회적 영향

- **가중치 편향**: 드문/민감 토큰의 과소·과대 가중을 모니터링하고, 그룹 공정성·유해성 지표를 병행 보고한다.

- **능력 증대와 오용**: 강화된 코드/추론 능력의 오용 가능성에 대비해 책임 있는 공개·제한적 배포·안전 가드레일을 명시한다.

- **데이터·프라이버시**: 참조/선호 신호 추출 시 데이터 정책과 프라이버시를 준수한다.

---

## 8. 기대 효과·기여

**이론적 기여**:
- MTP에 대한 정보 이론·중요도 샘플링·Policy Gradient 관점의 가중화 프레임워크 확립
- 3가지 실용적 가중화 방식의 체계적 비교 및 이론적 정당성 제공

**방법론적 기여**:
- Verifiable Critic: RM 의존성 제거로 메모리 효율성과 객관성 확보
- Rho-1 Weighted: 참조 비교 기반 간결한 가중화 구현
- 안정화 기법(온도·정규화·2단계 학습·검증) 제시

**실증적 기여**:
- 코드·추론·일반 과제에서 3가지 방식의 성능·효율·안정성 동시 비교
- 메모리 효율성, 데이터 활용, 학습 안정성 측면의 실증적 증거 제공

**실용적 기여**:
- 구성 주도형 설계를 통한 확장성·재현성·비용 효율성
- Verifiable rewards를 통한 RM 의존성 제거로 실용성 향상
- 3가지 방식의 trade-off 분석으로 실무 선택 가이드 제공

---

## 9. 향후 확장 방향

본 연구에서 직접 다루지 않는 추가 가중화 방식들:

- **RM Critic 기반**: Reward Model을 활용한 가치함수 학습. Verifiable Critic과 비교하여 RM 품질·비용·안정성 trade-off 분석 가능. 메모리 비용과 RM 의존성으로 인해 본 연구에서는 제외.

- **GRPO 기반**: 그룹 상대 보상 최적화(Group Relative Policy Optimization)를 MTP에 적용하여 critic-free 학습 가능성 검증. 최근 이론적 정식화가 제시되었으나 MTP 적용은 미검증. (DeepSeek 계열, ["What is the Alignment Objective of GRPO?"](https://arxiv.org/abs/2502.18548), ["GRPO's Effective Loss…"](https://arxiv.org/html/2503.06639v1))

- **그래디언트 기반**: 최종 목적에 대한 토큰 표현·로짓의 기여도(∥∇·∥)를 중요도로 사용하는 방법론 탐구. (Goal-Gradient Importance, [Zhuang et al. 2025](https://arxiv.org/abs/2505.08392))

- **하이브리드 접근**: Verifiable Critic + Rho-1 결합 등 복수 신호 융합

---

## 10. 참고문헌

- Glöckle et al. (2024). Better & Faster LLM via Multi-Token Prediction. Meta MTP.

- Lin et al. (2024). Rho-1: Not All Tokens Are What You Need. (NeurIPS 2024 Oral) — [OpenReview](https://openreview.net/forum?id=0NMzBwqaAJ), [NeurIPS 포스터](https://neurips.cc/virtual/2024/poster/96931), [HF 모델 카드](https://huggingface.co/microsoft/rho-math-7b-v0.1)

- Sutton et al. (1999). Policy Gradient Methods for Reinforcement Learning with Function Approximation. — [NIPS 논문](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

- Schulman et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. — [arXiv](https://arxiv.org/abs/1506.02438)

- Degris et al. (2012). Off-Policy Actor-Critic. — [arXiv](https://arxiv.org/abs/1205.4839)

- Vojnovic & Yun (2025). What is the Alignment Objective of GRPO? — [arXiv](https://arxiv.org/abs/2502.18548)

- Mroueh (2025). GRPO's Effective Loss, Dynamics, and Data Efficiency. — [arXiv](https://arxiv.org/html/2503.06639v1)

- Zhuang et al. (2025). Not All Thoughts Are Equal: Thought-Level Reinforcement Learning with Goal-Gradient Importance. — [arXiv](https://arxiv.org/abs/2505.08392)
