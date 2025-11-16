# Phase 5: Value Training 파이프라인 구현 가이드

## 문서 개요

본 문서는 **Phase 5: Value Training 파이프라인 구현**을 위한 실행 가이드입니다. Value weighting 모듈과 학습 파이프라인 구현에 집중하여, WMTP 3가지 실험(Baseline, Verifiable Critic, Rho-1 Weighted)을 실행 가능하게 만듭니다.

**버전**: v1.0 (2025-11-16)
**선행 조건**: Phase 3 (데이터 파이프라인), Phase 4 (Meta Adapter) 완료
**목표**: TD error 기반 가중치 계산 → Stage별 학습 파이프라인 → MLflow 로깅

---

## Part 1: 개요 및 맥락

### 1.1 Phase 5의 위치와 목적

Phase 5는 **모델 → 가중치 → 학습** 연결의 핵심 구간입니다.

```
Phase 4 (model)  →  [Phase 5 (value_weighting + pipeline)]  →  Phase 6 (config/CLI)
   Adapter 구현         TD error 계산 + 학습 파이프라인              사용자 진입점
```

**핵심 질문**: 어떻게 TD error를 계산하고, 이를 학습 파이프라인에 통합할 것인가?

### 1.2 핵심 혁신: 표준 TD Learning + Critic Continual Learning

**연구제안서의 핵심 방법론**:
1. **표준 TD Error 계산** (Verifiable Critic):
   - Intermediate tokens: `γV(s_k) - V(s_{k-1})` (Bootstrapping)
   - Terminal token: `R - V(s_{T-1})` (Direct reward)
2. **Exponential Weighting** (IQL/AWR 방식):
   - `weight = exp(td_error / β)` (β=0.9)
   - Conservative clipping: min=0.1, max=5.0
3. **Critic Continual Learning** (PPO Best Practice):
   - Stage 2에서 Value Loss를 Auxiliary Loss로 추가
   - `total_loss = weighted_ce_loss + value_coef * value_loss`

**아키텍처**:
```
Value Weighting 모듈
    ├── td_error.py: 표준 TD error 계산
    │   ├── compute_td_errors(): Intermediate + Terminal
    │   └── 입력: value_logits, rewards, gamma=1.0
    ├── weight_builder.py: TD error 기반 가중치 산출
    │   ├── build_weights(): Exponential weighting
    │   └── 입력: td_errors, beta=0.9, clipping
    └── metrics.py: TD error/weight 모니터링
        └── compute_weight_stats(), compute_td_stats()

학습 파이프라인
    ├── Stage 0: 분산학습 환경 초기화 (Phase 3에서 구현됨)
    ├── Stage 1: Value Head Pretrain (trunk_forward)
    │   └── Value loss만 계산 (MSE or Huber)
    └── Stage 2: Weighted Training (full_forward)
        ├── TD error 계산 → Weight 산출
        ├── Weighted CE loss 계산
        ├── Value loss (Auxiliary) 계산
        └── total_loss = weighted_ce + value_coef * value_loss
```

### 1.3 기대 효과

1. **이론적 정당성**: 표준 TD Learning (Sutton & Barto) 기반
2. **안정적 학습**: Bootstrapping으로 분산 감소, Continual Learning으로 drift 방지
3. **메모리 효율**: RM 불필요 (~28GB VRAM 절약)
4. **재현성**: Binary reward 환경에서 TD error 자연 bounded

---

## Part 2: Value Weighting 모듈 설계

### 2.1 TD Error 계산 (표준 Temporal Difference)

#### 이론적 배경

**표준 TD(0) 공식** (Sutton & Barto):
```python
# Intermediate tokens (k < T): Bootstrapping
δ_k = r_k + γV(s_k) - V(s_{k-1})
    = γV(s_k) - V(s_{k-1})  # r_k = 0 (중간 토큰 보상 없음)

# Terminal token (k = T): Direct reward
δ_T = R - V(s_{T-1})  # V(terminal) = 0 가정
```

**MTP 시나리오 적용**:
- 시점 t에서 H개 미래 토큰 예측: `x_{t+1}, x_{t+2}, ..., x_{t+H}`
- 상태 표기: `s_k` = 토큰 x_k 생성 후 prefix
- Value head 입력: `s_k` (norm 적용 후 hidden states)

#### 구현 요구사항

**compute_td_errors() 함수**:
```python
def compute_td_errors(
    value_logits: torch.Tensor,  # [batch, seq, 1]
    rewards: torch.Tensor,        # [batch] - Binary: 0 or 1
    gamma: float = 1.0,           # 할인율 (undiscounted, Stage 1과 일관성)
    terminal_indices: Optional[torch.Tensor] = None,  # [batch]
) -> torch.Tensor:
    """표준 TD error 계산

    TD error는 δ_t = V(s_{t+1}) - V(s_t) (γ=1.0)로 계산되며,
    "이 토큰이 성공 확률을 얼마나 변화시켰는가 (ΔP)"를 의미합니다.

    Args:
        value_logits: Value head 출력 [batch, seq, 1]
        rewards: Binary reward [batch] (0: incorrect, 1: correct)
        gamma: 할인율 (기본 1.0, undiscounted)
        terminal_indices: 각 시퀀스의 마지막 토큰 인덱스 [batch]

    Returns:
        td_errors: [batch, seq] TD error (Intermediate + Terminal)
    """
```

**책임**:
1. Value logits squeeze: `[batch, seq, 1]` → `[batch, seq]`
2. Intermediate TD error: `gamma * V(s_k) - V(s_{k-1})`
3. Terminal TD error: `reward - V(s_{T-1})`
4. Masking: padding 토큰 제외

**핵심 로직**:
```python
# Value squeeze
values = value_logits.squeeze(-1)  # [batch, seq]

# Bootstrapping: V(s_k) - V(s_{k-1})
value_next = values[:, 1:]  # [batch, seq-1]
value_current = values[:, :-1]  # [batch, seq-1]
td_errors_intermediate = gamma * value_next - value_current

# Terminal: R - V(s_{T-1})
# terminal_indices로 마지막 value 추출
values_terminal = values[range(batch_size), terminal_indices - 1]
td_errors_terminal = rewards - values_terminal

# Combine: [batch, seq-1] + [batch, 1] → [batch, seq]
# (구체적 구현은 terminal_indices 기반 scatter)
```

#### 검증 기준

**기능 검증**:
- [ ] Intermediate TD error 계산 정확: `gamma * V_next - V_current`
- [ ] Terminal TD error 계산 정확: `reward - V_terminal`
- [ ] Binary reward [0, 1] 환경에서 TD error bounded [-1, 1]
- [ ] Padding mask 정상 동작

**수치 검증**:
```python
# 예시 입력
value_logits = torch.tensor([[[0.5], [0.7], [0.9]]])  # [1, 3, 1]
rewards = torch.tensor([1.0])  # Correct
terminal_indices = torch.tensor([2])  # 마지막 토큰 인덱스 2

# 예상 출력
# Intermediate (0~1): gamma * 0.7 - 0.5 = 0.2
# Intermediate (1~2): gamma * 0.9 - 0.7 = 0.2
# Terminal (2): 1.0 - 0.9 = 0.1
td_errors = torch.tensor([[0.2, 0.2, 0.1]])  # [1, 3]
```

### 2.2 Weight Builder (Exponential Weighting)

#### 이론적 배경

**IQL/AWR Exponential Weighting**:
```python
weight = exp(advantage / β)
```

**WMTP 적용** (TD error 기반):
```python
weight = exp(td_error / β)  # β=0.9 (temperature)
weight = clamp(weight, min=0.1, max=5.0)  # Conservative clipping
```

**직관**:
- Positive TD error (td > 0): weight > 1 → 중요 토큰 강화
- Negative TD error (td < 0): weight < 1 → 비중요 토큰 down-weight
- Incorrect 샘플: reward=0, value>0 → td<0 → weight<1 (자동 필터링)

#### 구현 요구사항

**build_weights() 함수**:
```python
def build_weights(
    td_errors: torch.Tensor,  # [batch, seq]
    beta: float = 0.9,         # Temperature parameter
    min_weight: float = 0.1,   # Conservative minimum
    max_weight: float = 5.0,   # Conservative maximum
) -> torch.Tensor:
    """TD error 기반 exponential weighting

    Args:
        td_errors: TD error [batch, seq]
        beta: Temperature (낮을수록 집중도 높음)
        min_weight: 최소 가중치 (안정성)
        max_weight: 최대 가중치 (극단 방지)

    Returns:
        weights: [batch, seq] Token-level weights
    """
    weights = torch.exp(td_errors / beta)
    weights = torch.clamp(weights, min=min_weight, max=max_weight)
    return weights
```

**책임**:
1. Exponential transformation: `exp(td_error / beta)`
2. Conservative clipping: `[0.1, 5.0]` 범위로 제한
3. Gradient-friendly: autograd 호환

#### 검증 기준

**기능 검증**:
- [ ] Exponential weighting 정확: `exp(td / beta)`
- [ ] Clipping 정상: min=0.1, max=5.0 준수
- [ ] Gradient 계산 가능 (autograd 호환)

**수치 검증**:
```python
# 예시 입력
td_errors = torch.tensor([[0.2, -0.5, 0.1]])  # [1, 3]
beta = 0.9

# 예상 출력
# exp(0.2 / 0.9) = exp(0.222) ≈ 1.25
# exp(-0.5 / 0.9) = exp(-0.556) ≈ 0.57
# exp(0.1 / 0.9) = exp(0.111) ≈ 1.12
weights = torch.tensor([[1.25, 0.57, 1.12]])  # [1, 3]
```

### 2.3 Metrics 모듈

#### 구현 요구사항

**compute_weight_stats() 함수**:
```python
def compute_weight_stats(weights: torch.Tensor) -> Dict[str, float]:
    """Weight 분포 통계

    Args:
        weights: [batch, seq] Token weights

    Returns:
        {
            "weight_mean": float,
            "weight_std": float,
            "weight_min": float,
            "weight_max": float,
            "weight_entropy": float,  # -sum(p * log(p))
        }
    """
```

**compute_td_stats() 함수**:
```python
def compute_td_stats(td_errors: torch.Tensor) -> Dict[str, float]:
    """TD error 분포 통계

    Args:
        td_errors: [batch, seq] TD errors

    Returns:
        {
            "td_mean": float,
            "td_std": float,
            "td_min": float,
            "td_max": float,
        }
    """
```

#### 검증 기준

**기능 검증**:
- [ ] Weight entropy 계산 정확 (정규화 후 `-sum(p * log(p))`)
- [ ] 통계값 범위 검증 (mean, std, min, max)
- [ ] NaN/Inf 처리 (zero division 방지)

---

## Part 3: 학습 파이프라인 설계

### 3.1 Stage 구조 개요

**전체 학습 파이프라인**:
```
Stage 0: 분산학습 환경 초기화 (Phase 3에서 구현됨)
    ├── torch.distributed.init_process_group()
    ├── DistributedSampler 생성
    ├── FSDP wrapping
    └── Rank별 seed, device 설정

Stage 1: Value Head Pretrain (0.5 epoch)
    ├── trunk_forward() 사용
    ├── Value loss만 계산 (MSE or Huber)
    ├── Value head 파라미터만 업데이트
    └── 목적: V(s_t) = P(Success | s_t) 학습 (확률적 가치 학습)

Stage 2: Weighted Training (2.5 epoch)
    ├── full_forward() 사용
    ├── TD error 계산 → Weight 산출
    ├── Weighted CE loss 계산
    ├── Value loss (Auxiliary) 계산
    ├── total_loss = weighted_ce + value_coef * value_loss
    └── 목적: Weighted MTP 학습 + Critic continual learning

Stage 3: 평가 및 로깅 (Rank 0만)
    ├── Checkpoint 저장 (FSDP state_dict)
    ├── MLflow 로깅 (metrics, artifacts)
    └── 평가 리포트 생성 (선택적)
```

### 3.2 Stage 1: Value Head Pretrain

#### 목표
Value head가 **V(s_t) = P(Success | s_t)**를 학습하도록 사전 학습합니다.

**핵심 원리 (Probabilistic Value Learning)**:
- 모든 토큰에 R_terminal (0 or 1)을 target으로 부여 (γ=1.0, undiscounted)
- 배치 학습을 통해 V(s_t) → P(Success | s_t) 자동 수렴
- 동일 prefix가 다른 샘플에서 다른 R을 가지면 확률 추정

**예시**:
```
Sample 1 (R=1): [...A, B, C, D] → V(s_C) target = 1.0
Sample 2 (R=0): [...A, B, C, X] → V(s_C) target = 0.0
배치 학습 → V(s_C) ≈ 0.5 = P(Success | s_C)
```

이후 Stage 2에서 TD error δ_t = V(s_{t+1}) - V(s_t)로
토큰별 성공 확률 변화량(ΔP)을 측정하여 가중치 산출합니다.

#### 구현 요구사항

**train_stage1() 함수**:
```python
def train_stage1(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Dict,
    device: torch.device,
) -> Dict[str, float]:
    """Stage 1: Value Head Pretrain (Probabilistic Value Learning)

    Args:
        adapter: MetaLlamaMTPAdapter (Value head 포함)
        dataloader: Stage 1 DataLoader (correct/incorrect 균형)
        optimizer: Value head 전용 optimizer
        config: {"n_epochs": 0.5, "loss_type": "mse", ...}
        device: torch.device

    Returns:
        metrics: {"stage1_loss": float, "value_explained_variance": float}
    """
```

**핵심 로직**:
```python
for batch in dataloader:
    # 1. trunk_forward (MTP heads 사용 안 함)
    outputs = adapter.trunk_forward(input_ids, attention_mask)
    value_logits = outputs["value_logits"]  # [batch, seq, 1]

    # 2. Value target 생성 (Undiscounted Monte Carlo)
    # 모든 토큰에 동일한 R_terminal 부여 (γ=1.0)
    value_targets = rewards.unsqueeze(1).unsqueeze(2).expand(batch_size, seq_len, 1)

    # Mask padded tokens
    loss_mask = attention_mask.unsqueeze(-1).float()

    # 3. Value loss 계산 (패딩 토큰 제외)
    if loss_type == "mse":
        loss_per_token = F.mse_loss(value_logits, value_targets, reduction="none")
    elif loss_type == "huber":
        loss_per_token = F.smooth_l1_loss(value_logits, value_targets, reduction="none")

    masked_loss = loss_per_token * loss_mask
    value_loss = masked_loss.sum() / loss_mask.sum()

    # 4. Backward & update
    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()
```

**데이터 요구사항** (Phase 3에서 구현됨):
- **Stage 1 샘플링**: `is_correct` 균형 (50% correct, 50% incorrect)
- **샘플 수**: 10,000~50,000 (전체의 0.3~1.4%)
- **Difficulty 무관**: 모든 난이도 균등 샘플링
- **메모리 효율**: 메타데이터 기반 로딩 (~417MB)

#### 검증 기준

**기능 검증**:
- [ ] trunk_forward() 정상 실행
- [ ] Value loss 계산 정확 (MSE or Huber)
- [ ] Gradient 업데이트 정상 (value_head.parameters())
- [ ] Value explained variance 계산 정확

**성능 검증**:
- [ ] 0.5 epoch 학습 완료 시간: <30분 (micro 모델, M3 Mac)
- [ ] Value loss 수렴 확인 (감소 추세)
- [ ] Value explained variance > 0.5 (이상적: 1.0에 가까움)

### 3.3 Stage 2: Weighted Training

#### 목표
TD error 기반 가중치로 MTP를 학습하며, Critic을 continual learning으로 유지합니다.

#### MTP 가중치 시점 정렬 (핵심 원리)

**TD Error의 의미**:
```
δ_{t-1} = γV(s_t) - V(s_{t-1})
```
- `s_{t-1}`: 토큰 x_t 생성 **직전** 상태
- `s_t`: 토큰 x_t 생성 **직후** 상태
- `δ_{t-1}`: 토큰 x_t를 선택했을 때 **가치 변화량** (marginal value)

**핵심 통찰**: δ_{t-1}은 "토큰 x_t의 중요도"를 나타내므로, x_t를 학습할 때 가중치로 사용해야 합니다.

**MTP 적용** (H=4 예시):

시점 t에서 H개 미래 토큰 예측: x_{t+1}, x_{t+2}, x_{t+3}, x_{t+4}

```python
# MTP logits: [batch, seq, n_future_tokens, vocab]
# logits[t, k-1, :]: 시점 t에서 예측한 x_{t+k}의 logits

# 가중치 할당 (TD error 기반):
logits[t, 0, :] (x_{t+1} 예측) → weight = td_errors[t]   (x_{t+1}의 가치 증가량)
logits[t, 1, :] (x_{t+2} 예측) → weight = td_errors[t+1] (x_{t+2}의 가치 증가량)
logits[t, 2, :] (x_{t+3} 예측) → weight = td_errors[t+2] (x_{t+3}의 가치 증가량)
logits[t, 3, :] (x_{t+4} 예측) → weight = td_errors[t+3] (x_{t+4}의 가치 증가량)
```

**일반화**:
```
시점 t에서 예측하는 x_{t+k}의 가중치 = td_errors[t+k-1]
```

이는 Meta MTP 2024의 실제 구현과 일치합니다 (연구제안서 섹션 4.1).

#### 구현 요구사항

**train_stage2() 함수**:
```python
def train_stage2(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Dict,
    device: torch.device,
) -> Dict[str, float]:
    """Stage 2: Weighted Training with Critic Continual Learning

    Args:
        adapter: MetaLlamaMTPAdapter (Value head 포함)
        dataloader: Stage 2 DataLoader (curriculum learning)
        optimizer: 전체 파라미터 optimizer
        config: {
            "n_epochs": 2.5,
            "beta": 0.9,
            "value_coef": 0.5,
            "clip_range": 0.2,
            "max_grad_norm": 0.5,
        }
        device: torch.device

    Returns:
        metrics: {
            "stage2_weighted_ce_loss": float,
            "stage2_value_loss": float,
            "stage2_total_loss": float,
            "td_mean": float,
            "weight_mean": float,
            "value_explained_variance": float,
        }
    """
```

**핵심 로직**:
```python
for batch in dataloader:
    # 1. full_forward (MTP + Value)
    outputs = adapter.full_forward(input_ids)
    logits = outputs["logits"]  # [batch, seq, n_future_tokens, vocab]
    value_logits = outputs["value_logits"]  # [batch, seq, 1]

    # 2. TD error 계산
    td_errors = compute_td_errors(
        value_logits=value_logits,
        rewards=batch["is_correct"].float(),
        attention_mask=attention_mask,
        gamma=1.0,
    )

    # 3. Weight 산출 (TD error → Exponential weighting)
    weights = build_weights(
        td_errors=td_errors,
        beta=config["beta"],
        min_weight=0.1,
        max_weight=5.0,
    )

    # 4. Weighted CE loss 계산 (모든 H개 미래 토큰 평균)
    # logits: [batch, seq, n_future_tokens, vocab]
    # labels: [batch, seq] (next tokens)
    # weights: [batch, seq] (TD errors)

    batch_size, seq_len, n_future, vocab_size = logits.shape
    total_weighted_ce_loss = 0.0

    # H개 미래 토큰 각각에 대한 weighted loss 계산
    for k in range(1, n_future + 1):
        # 시점 t에서 예측한 x_{t+k}
        logits_k = logits[:, :-k, k-1, :]  # [batch, seq-k, vocab]
        labels_k = labels[:, k:]  # [batch, seq-k]
        weights_k = weights[:, k-1:seq_len-1]  # [batch, seq-k]

        # Attention mask 적용
        mask_k = attention_mask[:, k:]  # [batch, seq-k]

        # CE loss (per-token)
        ce_loss_k = F.cross_entropy(
            logits_k.reshape(-1, vocab_size),
            labels_k.reshape(-1),
            reduction="none",
        )  # [batch*(seq-k)]

        # Weighted loss (masking 적용)
        weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.reshape(-1)
        total_weighted_ce_loss += weighted_ce_k.sum() / mask_k.sum()

    # H개 토큰 평균
    weighted_ce_loss = total_weighted_ce_loss / n_future

    # 5. Value loss (Auxiliary, Continual Learning)
    value_targets = batch["rewards"].unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)
    value_loss = F.mse_loss(value_logits, value_targets)

    # Value loss clipping (PPO-style)
    if config.get("clip_range"):
        # Clipping logic (optional)
        pass

    # 6. Total loss
    total_loss = weighted_ce_loss + config["value_coef"] * value_loss

    # 7. Backward & update
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping
    if config.get("max_grad_norm"):
        torch.nn.utils.clip_grad_norm_(
            adapter.parameters(),
            config["max_grad_norm"],
        )

    optimizer.step()
```

**데이터 요구사항** (Phase 3에서 구현됨):
- **Stage 2 샘플링**: Curriculum Learning (difficulty 기반)
  - 초반 epoch (0~30%): low (1-3) 70%, medium (4-7) 30%, high (8-11) 0%
  - 중반 epoch (30~70%): low 30%, medium 60%, high 10%
  - 후반 epoch (70~100%): low 10%, medium 50%, high 40%
- **샘플 수**: 100,000~500,000 (전체의 2.7~13.5%)
- **메모리 효율**: 메타데이터 기반 로딩 (~1GB)

#### Critic Continual Learning 구현

**핵심 아이디어** (PPO Best Practice):
1. **Auxiliary Loss**: Value loss를 policy loss와 함께 최적화
2. **Loss Coefficient**: value_coef=0.5 (Stable Baselines3 표준)
3. **Value Loss Clipping**: clip_range=0.2 (drift 방지)
4. **Gradient Clipping**: max_grad_norm=0.5~1.0 (안정성)

**모니터링 지표**:
```python
# Value explained variance
value_var = value_logits.var()
target_var = value_targets.var()
explained_var = 1 - (value_loss / target_var)  # 1.0에 가까울수록 이상적

# TD error 분포
td_stats = compute_td_stats(td_errors)
# {"td_mean": float, "td_std": float, "td_min": float, "td_max": float}

# Weight 분포
weight_stats = compute_weight_stats(weights)
# {"weight_mean": float, "weight_entropy": float, ...}
```

#### 검증 기준

**기능 검증**:
- [ ] full_forward() 정상 실행
- [ ] TD error 계산 정확
- [ ] Weight 산출 정확
- [ ] **MTP Weight 시점 정렬 정확**: logits[t, k-1, :]의 가중치 = td_errors[t+k-1]
- [ ] **H개 미래 토큰 loss 평균 계산 정확** (Meta MTP 2024 방식)
- [ ] Weighted CE loss 계산 정확 (masking 적용)
- [ ] Value loss (auxiliary) 계산 정확
- [ ] Total loss 업데이트 정상

**안정성 검증**:
- [ ] TD error bounded [-1, 1] (binary reward 환경)
- [ ] Weight clipping 정상 [0.1, 5.0]
- [ ] Gradient clipping 정상 (max_grad_norm)
- [ ] Value loss diverge 방지 (continual learning)

**성능 검증**:
- [ ] 2.5 epoch 학습 완료 시간: <2시간 (micro 모델, M3 Mac)
- [ ] Weighted CE loss 수렴 확인
- [ ] Value explained variance > 0.5 유지

### 3.4 Stage 3: 평가 및 로깅

#### 구현 요구사항

**save_checkpoint() 함수**:
```python
def save_checkpoint(
    adapter: MetaLlamaMTPAdapter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: Path,
) -> None:
    """Checkpoint 저장 (Rank 0만)

    Args:
        adapter: MetaLlamaMTPAdapter
        optimizer: Optimizer state
        epoch: 현재 epoch
        metrics: 학습 metrics
        save_dir: 저장 디렉터리
    """
    if not is_main_process():
        return

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch}.pt")
```

**log_to_mlflow() 함수**:
```python
def log_to_mlflow(
    metrics: Dict[str, float],
    step: int,
    experiment_name: str,
) -> None:
    """MLflow 로깅 (Rank 0만)

    Args:
        metrics: {"loss": float, "td_mean": float, ...}
        step: Global step
        experiment_name: MLflow experiment name
    """
    if not is_main_process():
        return

    mlflow.log_metrics(metrics, step=step)
```

#### 검증 기준

**기능 검증**:
- [ ] Checkpoint 저장 성공 (Rank 0만)
- [ ] MLflow 로깅 성공 (Rank 0만)
- [ ] Metrics 기록 정확

---

## Part 4: Step별 구현 가이드

### 4.1 Step 1: value_weighting/td_error.py 구현

#### 목표
표준 TD error 계산 로직을 구현합니다.

#### 구현 파일
- `src/weighted_mtp/value_weighting/td_error.py`

#### 핵심 함수
1. `compute_td_errors()`: TD error 계산 (Intermediate + Terminal)
2. `_compute_intermediate_td_errors()`: Bootstrapping
3. `_compute_terminal_td_errors()`: Direct reward

#### 검증 기준
- [ ] Unit test 작성 (`tests/unit/test_td_error.py`)
- [ ] 수치 검증 통과 (예시 입력/출력)
- [ ] Edge case 처리 (zero reward, all correct/incorrect)

### 4.2 Step 2: value_weighting/weight_builder.py 구현

#### 목표
TD error 기반 exponential weighting을 구현합니다.

#### 구현 파일
- `src/weighted_mtp/value_weighting/weight_builder.py`

#### 핵심 함수
1. `build_weights()`: Exponential weighting + clipping

#### 검증 기준
- [ ] Unit test 작성 (`tests/unit/test_weight_builder.py`)
- [ ] Gradient 계산 가능 확인
- [ ] Clipping 범위 검증

### 4.3 Step 3: value_weighting/metrics.py 구현

#### 목표
TD error/weight 통계 계산을 구현합니다.

#### 구현 파일
- `src/weighted_mtp/value_weighting/metrics.py`

#### 핵심 함수
1. `compute_weight_stats()`: Weight 분포 통계
2. `compute_td_stats()`: TD error 분포 통계

#### 검증 기준
- [ ] Unit test 작성 (`tests/unit/test_metrics.py`)
- [ ] NaN/Inf 처리 확인

### 4.4 Step 4: pipelines/training.py 구현

#### 목표
Stage 1/2 학습 파이프라인을 구현합니다.

#### 구현 파일
- `src/weighted_mtp/pipelines/training.py`

#### 핵심 함수
1. `train_stage1()`: Value Head Pretrain
2. `train_stage2()`: Weighted Training
3. `run_training_pipeline()`: 전체 오케스트레이션

#### 검증 기준
- [ ] Integration test 작성 (`tests/integration/test_training_pipeline.py`)
- [ ] Micro 모델 + small 데이터로 end-to-end 테스트
- [ ] MLflow 로깅 확인

### 4.5 Step 5: 통합 테스트 및 검증

#### 목표
Micro 모델로 전체 파이프라인을 검증합니다.

#### 검증 항목
1. **기능 검증**:
   - [ ] Stage 1 학습 완료
   - [ ] Stage 2 학습 완료
   - [ ] Checkpoint 저장 성공
   - [ ] MLflow 로깅 성공

2. **성능 검증** (micro 모델 기준):
   - [ ] Stage 1 (0.5 epoch): <30분
   - [ ] Stage 2 (2.5 epoch): <2시간
   - [ ] 메모리 사용: <1GB

3. **품질 검증**:
   - [ ] Value loss 수렴
   - [ ] Weighted CE loss 수렴
   - [ ] Value explained variance > 0.5

---

## Part 5: 검증 및 위험 관리

### 5.1 3-Tier 검증 체계

#### Tier 1: 기능 검증 (Functional Validation)

**Value Weighting 모듈**:
- [ ] TD error 계산 정확 (Intermediate + Terminal)
- [ ] Weight 산출 정확 (Exponential + Clipping)
- [ ] Metrics 계산 정확 (통계값 검증)

**학습 파이프라인**:
- [ ] Stage 1 학습 정상 (trunk_forward)
- [ ] Stage 2 학습 정상 (full_forward)
- [ ] Gradient 업데이트 정상
- [ ] Checkpoint 저장/로드 정상

#### Tier 2: 품질 검증 (Quality Validation)

**성능 목표** (micro 모델 기준):

| 항목 | 목표 | 측정 방법 |
|------|------|-----------|
| Stage 1 시간 | <30분 | Wall clock |
| Stage 2 시간 | <2시간 | Wall clock |
| 메모리 사용 | <1GB | torch.cuda.memory_allocated() |

**코드 품질**:
- [ ] Ruff linting 통과
- [ ] Black formatting 통과
- [ ] Docstring 100% (Args, Returns)

**테스트 커버리지**:
- [ ] value_weighting/: >80%
- [ ] pipelines/: >70%

#### Tier 3: 통합 검증 (Integration Validation)

**Micro 모델 End-to-End**:
```bash
pytest tests/integration/test_training_pipeline.py -v
```
- Stage 1 → Stage 2 → Checkpoint 저장 → MLflow 로깅
- 성능 테스트 통과 (<2.5시간 total)

### 5.2 위험 관리 매트릭스

#### 고위험 (High Impact, High Probability)

**Risk 1: Value loss diverge (Critic drift)**
- **영향**: Stage 2 학습 실패
- **확률**: Medium
- **완화 전략**:
  - Critic Continual Learning (value loss auxiliary)
  - Value loss clipping (clip_range=0.2)
  - Gradient clipping (max_grad_norm=0.5)
  - Value explained variance 모니터링
- **대비책**: Value loss diverge 시 learning rate 감소

**Risk 2: TD error unbounded (수치 폭주)**
- **영향**: Weight 폭주, 학습 불안정
- **확률**: Low (binary reward bounded)
- **완화 전략**:
  - Binary reward [0, 1] 환경 → TD error 자연 bounded
  - Weight clipping [0.1, 5.0]
  - TD error 모니터링 (평균, 표준편차)
- **대비책**: TD error 이상 시 beta 조정

#### 중위험 (Medium Impact, Medium Probability)

**Risk 3: Terminal token indexing 오류**
- **영향**: Terminal TD error 계산 잘못됨
- **확률**: Medium
- **완화 전략**:
  - terminal_indices 명시적 전달
  - Unit test로 검증
- **대비책**: Attention mask 기반 자동 탐지

**Risk 4: Gradient accumulation 버그**
- **영향**: Effective batch size 불일치
- **확률**: Low
- **완화 전략**:
  - Phase 3에서 구현된 분산학습 모듈 활용
  - Integration test로 검증
- **대비책**: 단일 GPU 모드로 디버깅

---

## Part 6: 완료 기준 및 다음 단계

### 6.1 Phase 5 완료 체크리스트

#### 코드 완성
- [ ] `src/weighted_mtp/value_weighting/td_error.py` 구현
  - compute_td_errors() 함수
  - Intermediate + Terminal TD error
- [ ] `src/weighted_mtp/value_weighting/weight_builder.py` 구현
  - build_weights() 함수
  - Exponential weighting + clipping
- [ ] `src/weighted_mtp/value_weighting/metrics.py` 구현
  - compute_weight_stats() 함수
  - compute_td_stats() 함수
- [ ] `src/weighted_mtp/pipelines/training.py` 구현
  - train_stage1() 함수
  - train_stage2() 함수
  - run_training_pipeline() 함수
- [ ] `src/weighted_mtp/value_weighting/__init__.py` 업데이트
  - Public API export

#### 테스트 완성
- [ ] `tests/unit/test_td_error.py`
  - test_compute_td_errors()
  - test_intermediate_td_errors()
  - test_terminal_td_errors()
- [ ] `tests/unit/test_weight_builder.py`
  - test_build_weights()
  - test_weight_clipping()
- [ ] `tests/unit/test_metrics.py`
  - test_compute_weight_stats()
  - test_compute_td_stats()
- [ ] `tests/integration/test_training_pipeline.py`
  - test_stage1_training()
  - test_stage2_training()
  - test_end_to_end_pipeline()

#### 검증 완료
- [ ] Tier 1 (기능): 모든 체크리스트 통과
- [ ] Tier 2 (품질): 성능 목표 달성 (<2.5시간 total)
- [ ] Tier 3 (통합): End-to-end 테스트 통과

#### 문서화
- [ ] Docstring 100% (Args, Returns)
- [ ] `src/weighted_mtp/value_weighting/__init__.py` public API export
- [ ] Phase 5 완료 보고서 작성 (본 문서)

### 6.2 Phase 6 착수 조건

Phase 5 완료 후, 다음 조건을 만족해야 Phase 6 (Config & CLI)로 진행:

**필수 조건**:
1. TD error 계산 구현 완료
2. Weight builder 구현 완료
3. Stage 1/2 학습 파이프라인 구현 완료
4. Unit tests 통과 (coverage >80%)
5. Integration test 통과 (end-to-end)
6. Micro 모델로 성능 검증 완료

**권장 조건**:
1. Production 모델 (7B) 로딩 검증 (VESSL에서) - Phase 6에서 진행
2. MLflow 로깅 확인
3. Code quality 기준 충족 (linting, formatting)

### 6.3 예상 소요 시간

| 작업 | 예상 시간 | 비고 |
|------|-----------|------|
| td_error.py 구현 | 4-6시간 | Intermediate + Terminal 분리 |
| weight_builder.py 구현 | 2-3시간 | Exponential weighting |
| metrics.py 구현 | 2-3시간 | 통계 계산 |
| training.py 구현 | 6-8시간 | Stage 1/2 파이프라인 |
| Unit tests 작성 | 4-6시간 | 3개 모듈 테스트 |
| Integration test 작성 | 3-4시간 | End-to-end 테스트 |
| 통합 테스트 및 디버깅 | 4-6시간 | 성능 최적화 |
| 문서화 | 2-3시간 | 본 문서 작성 |
| **합계** | **27-39시간** | 약 3.5-5일 |

### 6.4 Phase 6 Preview

**Phase 6: Config & CLI 체계** (다음 단계)

핵심 구현:
1. `core/config.py`: Pydantic 모델 정의 (Config, Recipe)
2. `cli/train.py`: argparse, preset 옵션
3. `configs/defaults.yaml`: 환경 고정값
4. `configs/recipe.*.yaml`: 3개 실험 레시피
   - recipe.baseline.yaml
   - recipe.verifiable.yaml
   - recipe.rho1_weighted.yaml

**Phase 5와의 연계**:
- Phase 5 `run_training_pipeline()` → Phase 6 CLI 진입점
- Phase 5 config Dict → Phase 6 Pydantic 모델
- Phase 5 3개 실험 로직 → Phase 6 3개 recipe 설정

---

## 부록

### A. 용어 정리

| 용어 | 정의 |
|------|------|
| **TD Error** | Temporal Difference Error (시간차 오차) |
| **Bootstrapping** | 다음 상태 Value로 현재 토큰 기여도 추정 (분산 감소) |
| **Direct Reward** | 실제 보상 직접 사용 (편향 없음) |
| **Exponential Weighting** | IQL/AWR 방식 `exp(advantage / β)` |
| **Critic Continual Learning** | Policy 학습 중 Critic도 지속 학습 (PPO Best Practice) |
| **Value Explained Variance** | Critic 품질 지표 (1.0에 가까울수록 이상적) |
| **Curriculum Learning** | Difficulty 기반 점진적 난이도 증가 |

### B. 참고 자료

**내부 문서**:
- `docs/00_ideal_structure.md`: 전체 아키텍처
- `docs/02_implementation_plan.md`: Phase 5 요구사항
- `docs/06_phase4_detailed_plan.md`: Phase 4 완료 상태
- `docs/wmtp_research_proposal.md`: WMTP 연구 의도, TD error 공식

**외부 레퍼런스**:
- [Sutton & Barto RL Book](http://incompleteideas.net/book/RLbook2020.pdf): 표준 TD(0) 공식
- [IQL Paper](https://arxiv.org/abs/2110.06169): Exponential weighting 방식
- [PPO Paper](https://arxiv.org/abs/1707.06347): Value loss clipping

### C. TD Error 공식 및 MTP 가중치 시점 정렬 요약

**TD Error 계산 (표준 Temporal Difference)**:

**Intermediate tokens (k < T)**:
```
δ_k = r_k + γV(s_k) - V(s_{k-1})
    = γV(s_k) - V(s_{k-1})  # r_k = 0
```

**Terminal token (k = T)**:
```
δ_T = R - V(s_{T-1})  # V(terminal) = 0
```

**TD Error의 의미**:
```
δ_{t-1} = γV(s_t) - V(s_{t-1})
```
- `s_{t-1}`: 토큰 x_t 생성 직전 상태
- `s_t`: 토큰 x_t 생성 직후 상태
- `δ_{t-1}`: 토큰 x_t를 선택했을 때 가치 변화량 (marginal value)
- **핵심**: δ_{t-1}은 토큰 x_t의 중요도를 나타내므로, x_t 학습 시 가중치로 사용

**MTP 가중치 시점 정렬** (Meta MTP 2024 방식):

시점 t에서 H개 미래 토큰 예측: x_{t+1}, x_{t+2}, ..., x_{t+H}

```python
# 가중치 할당:
logits[t, 0, :] (x_{t+1} 예측) → weight = td_errors[t]     # δ_t
logits[t, 1, :] (x_{t+2} 예측) → weight = td_errors[t+1]   # δ_{t+1}
logits[t, k-1, :] (x_{t+k} 예측) → weight = td_errors[t+k-1]  # δ_{t+k-1}
```

**일반화**:
```
시점 t에서 예측하는 x_{t+k}의 가중치 = td_errors[t+k-1]
```

**Exponential Weighting** (IQL/AWR 방식):
```
weight_k = exp(td_error_k / β)  # β=0.9
weight_k = clamp(weight_k, min=0.1, max=5.0)
```

### D. 개발원칙 준수 체크리스트

**[원칙 1] 앞/뒤 흐름 분석**:
- [ ] Phase 4 Adapter 출력 형식 확인 (trunk/full forward)
- [ ] Phase 3 데이터 파이프라인 출력 확인 (rewards, terminal_indices)
- [ ] Phase 6 Config 입력 요구사항 확인

**[원칙 2] 기존 구조 존중**:
- [ ] 표준 TD Learning 공식 **정확히 구현** (Sutton & Barto)
- [ ] IQL/AWR exponential weighting 방식 차용
- [ ] PPO Critic Continual Learning 베스트 프랙티스 적용

**[원칙 3] 전격적 변경 승인**:
- [ ] 새로운 접근 시 사용자 승인 획득
- [ ] 기존 계획과 차이 발생 시 문서화

**[원칙 4] 하위 호환성 고려 없음**:
- [ ] 주석: 한글, 이모지 없음, 코드 동작 핵심만
- [ ] 로깅: 한글, 이모지 없음
- [ ] 변수명: 통일성 있게 네이밍

**[원칙 5] 계획서와 비교**:
- [ ] Phase 5 완료 후 본 문서 소급 업데이트
- [ ] 차이점 객관적 기술
- [ ] 성과 과장 없음

**[원칙 6] 패키지 의존성 도구 활용**:
- [ ] uv로 의존성 관리
- [ ] pytest 실행 시 `uv run pytest` 사용

---

**문서 종료**

본 문서는 Phase 5 **상세 계획**을 정리한 초안입니다. 구현 과정에서 실제 상태를 반영하여 소급 업데이트할 예정입니다.

**핵심 목표 요약**:
1. 표준 TD error 계산 구현 (Intermediate + Terminal)
2. Exponential weighting 구현 (IQL/AWR 방식)
3. Stage 1/2 학습 파이프라인 구현
4. Critic Continual Learning 적용 (PPO Best Practice)
5. Micro 모델로 성능 검증 (<2.5시간 total)
