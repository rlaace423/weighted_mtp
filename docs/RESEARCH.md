# 연구 배경 및 이론

Weighted Multi-Token Prediction (WMTP)의 이론적 배경 및 실험 설계.

---

## 핵심 가설

**"Not All Tokens Are What You Need"**

표준 MTP는 모든 미래 토큰을 균등 가중으로 다루어, 쉬운/비핵심 토큰에도 동일한 학습 자원을 배분한다. 중요 토큰에 계산을 집중하는 WMTP는 동일 FLOPs에서 더 높은 성능과 안정적 수렴을 달성한다.

---

## 3가지 가중화 방식

### 방식 1: Baseline MTP (Control Group)

```
w_{t,k} = 1.0 (모든 k)
```

- **데이터**: 정답만 학습
- **목적**: 비교 기준선
- **특징**: 표준 SFT, 구현 단순

### 방식 2: Verifiable Critic WMTP

데이터셋의 검증 가능한 레이블(is_correct)을 reward signal로 사용.

**Stage 1: Probabilistic Value Learning (MSE Loss)**:
```python
# Value Head 학습: V(s_t) → E[R | s_t] = P(Success | s_t)
# Target: 모든 토큰에 동일한 reward (R_terminal) 부여
value_targets = rewards.unsqueeze(1).expand(batch_size, seq_len, 1)
value_loss = F.mse_loss(value_logits, value_targets)

# 학습 원리:
# - Correct 샘플 (R=1.0): V → 1.0
# - Incorrect 샘플 (R=0.0): V → 0.0
# - 동일 prefix가 correct/incorrect 모두에 나타나면 V → 확률값
# → MSE 최적해 = E[R | s] = P(Success | s)
```

**Stage 2: TD Error-based Weighting**:
```python
# 1. Value Head로 상태 가치 추정 (Stage 1에서 학습된 weights)
value_logits = model.value_head(hidden_states)

# 2. TD error 계산 (Weight 계산용)
# Intermediate tokens: Bootstrapping
δ_k = γV(s_k) - V(s_{k-1})
# Terminal token: Direct reward
δ_T = R - V(s_{T-1})

# 3. Exponential weighting (IQL/AWR 방식)
weight_k = exp(td_error_k / β)  # β = 0.9
weight_k = clamp(weight_k, min=0.1, max=5.0)

# 4. Weighted policy loss + Value continual learning
weighted_ce_loss = (ce_loss * weights).mean()
value_loss = F.mse_loss(value_logits, value_targets)
total_loss = weighted_ce_loss + value_coef * value_loss
```

**특징**:
- **Value 학습**: MSE loss로 P(Success | s) 학습 (TD error 아님)
- **TD error**: Weight 계산에만 사용
- **데이터**: 정답+오답 모두 학습 (3.7M samples)
- **RM 불필요**: ~28GB VRAM 절약
- **객관적 보상**: 실행 결과/정답 기반
- **2단계 학습**: Stage 1 (Value Head Pretrain 0.5 epoch) + Stage 2 (Weighted Training 2.5 epoch)

**이론적 근거**:
- Probabilistic Value Learning: MSE 최적해 = E[R | s] = P(Success | s)
- TD error weighting: IQL/AWR exponential weighting (`exp(advantage/β)`)
- Critic Continual Learning: PPO best practice (value_coef=0.5)

### 방식 3: Rho-1 WMTP

참조 모델과 Base 모델의 loss 차이 기반.

**Top-k Binary Selection**:
```python
# Signed difference
excess_loss = policy_loss - reference_loss

# Per-head binary weights
weights[:, :, 0] = 1.0  # Head 0 always
for head_idx in range(1, n_future_tokens):
    threshold = torch.quantile(excess_loss[:, :, head_idx], 1 - k_percent)
    weights[:, :, head_idx] = (excess_loss[:, :, head_idx] <= threshold).float()
```

**특징**:
- **데이터**: 정답만 학습
- **Reference 모델**: Sheared LLaMA 2.7B
- **k_percent=0.6**: Top 60% 선택
- **Head 0 always**: MTP 일관성 유지

**원논문**: Lin et al. "Rho-1: Not All Tokens Are What You Need" (NeurIPS 2024 Oral)

---

## 비교 테이블

| 특성 | Baseline | Verifiable Critic | Rho-1 Weighted |
|------|----------|-------------------|----------------|
| **가중치 산출** | 상수 (1.0) | TD error → exp weighting | Reference loss 차이 |
| **Value 학습** | 없음 | MSE loss (Probabilistic) | 없음 |
| **TD error 용도** | 없음 | Weight 계산용 | 없음 |
| **외부 모델** | 불필요 | 불필요 (RM 제거) | 필요 (Reference) |
| **메모리 효율** | 표준 | 높음 (~28GB 절약) | 표준 |
| **데이터 요구** | 정답만 | 정답+오답 (3.7M) | 정답만 |
| **학습 안정성** | 높음 | 중간 (Continual Learning) | 높음 |
| **구현 복잡도** | 낮음 | 높음 (2단계) | 중간 |
| **이론적 기반** | 표준 SFT | MSE + TD weighting | 정보 이론 |
| **Negative signal** | 미사용 | 활용 (weight < 1) | 미사용 |

---

## 실험 설계

### 학습 데이터

- **CodeContests Train**: 3.7M samples (correct: 1.75M, incorrect: 1.94M)
- **샘플 크기**:
  - Baseline: 150K (정답만)
  - Critic: 50K (50:50 균형)
  - Verifiable: 150K (Curriculum Learning, 50:50 균형)
  - Rho-1: 150K (정답만)
- **Epochs**: 3 epochs (동일 학습 예산)

### 평가 벤치마크

- **코드 생성**: MBPP, HumanEval
- **In-domain**: CodeContests test
- **수학/추론**: MATH, GSM8K (선택)

### 핵심 비교 실험

1. **성능 비교**: 3가지 방식의 Pass@K, Exact Match
2. **메모리 효율**: Verifiable Critic의 RM 제거 효과
3. **데이터 활용**: 정답만 vs 정답+오답
4. **안정성**: 수렴 분산, 가중치 엔트로피
5. **Ablation**: 온도 T, GAE λ, Pretrain duration

### 지표

**성능**: Exact Match, Pass@K (K=1,10,100)

**효율**: FLOPs, 벽시계 시간, 메모리 사용량

**안정성**: 수렴 분산, 그래디언트 놈, 가중치 분포

**분석**: 토큰 난이도별 오류율, 헤드별 성능

---

## 안정화 메커니즘

### Verifiable Critic

**Critic Continual Learning**:
- Stage2에서 Value Loss를 Auxiliary Loss로 추가
- Value Coefficient: 0.5 (Stable Baselines3)
- Value Loss Clipping: clip_range=0.2

**Monitoring**:
- Value explained variance
- TD error 분포 (bounded 특성 검증)
- Critic drift (KL/cosine, 선택)

**Gradient Clipping**:
- Global norm clipping (max_grad_norm=0.5~1.0)

### Rho-1

**Noise Filtering**:
- min_ce_diff로 노이즈 필터링
- 온도 T 조정으로 집중도 제어

**정렬 보장**:
- 동일 토크나이저 강제
- 시점 정렬 자동 검증

---

## 향후 확장

본 연구에서 직접 다루지 않는 방식들:

- **RM Critic**: Reward Model 기반 가치함수 (메모리 비용, RM 의존성)
- **GRPO**: Critic-free 학습 (MTP 적용 미검증)
- **Gradient 기반**: Goal-Gradient Importance (∥∇·∥)
- **하이브리드**: Verifiable + Rho-1 결합

---

## 참고문헌

- **Meta MTP**: Glöckle et al. (2024). Better & Faster LLM via Multi-Token Prediction.

- **Rho-1**: Lin et al. (2024). Not All Tokens Are What You Need. NeurIPS 2024 Oral. [OpenReview](https://openreview.net/forum?id=0NMzBwqaAJ)

- **Policy Gradient**: Sutton et al. (1999). Policy Gradient Methods for RL with Function Approximation. [NIPS](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

- **GAE**: Schulman et al. (2015). High-Dimensional Continuous Control Using GAE. [arXiv](https://arxiv.org/abs/1506.02438)

- **IQL**: Kostrikov et al. (2021). Offline RL with Implicit Q-Learning. [arXiv](https://arxiv.org/abs/2110.06169)

- **Sutton & Barto**: Reinforcement Learning: An Introduction (2nd Ed). [PDF](http://incompleteideas.net/book/RLbook2020.pdf)
