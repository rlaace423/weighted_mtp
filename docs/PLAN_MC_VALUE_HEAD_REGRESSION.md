# Value Head 구조 회귀 및 MC Loss 추가 계획서

## 배경

### 문제 인식의 변화
1. 과거: PPO 방식의 value head가 token value 계산에 부적절하다고 판단 → pairwise MLP 방식으로 변경
2. 현재: head 구조나 loss 문제가 아니라 **CodeContests 데이터 분포 문제** (특정 problem의 solutions 과다)
3. 해결: `max_pairs_per_problem` 제한으로 데이터 분포 문제 해결 완료

### 회귀 목표
과거 ARCHITECTURE.md/RESEARCH.md의 설계 의도대로:
- **Terminal reward (correct: 1, incorrect: 0)를 MC 방식으로 학습**
- **4096 linear single head + 2개의 unfreeze trunk** 구조
- **현재 pairwise loss와 별도로 tokenwise MSE loss 추가**

---

## 현재 구현 분석

### Value Head 구조 (`value_head.py`)
```python
# 현재: MLPValueHead (DIAL 스타일 bottleneck)
class MLPValueHead(nn.Module):
    # 4096 → 512 → 256 → 1 (3-layer MLP)
    # dropout 적용
```

### Trunk Gradient 흐름 (`adapter.py`)
```python
# 현재: value_head만 학습, trunk gradient 완전 차단
value_logits = self.value_head(hidden_states.detach())
```

### Value Loss (`run_verifiable.py`)
```python
# 현재: Pairwise Ranking Loss만 사용
value_loss = pairwise_ranking_loss(pos_value_logits, neg_value_logits, pos_mask, neg_mask)
```

### TD Target 계산 (`td_weighting.py`)
```python
# 현재: MC 방식 지원 (lam=1.0, gamma=1.0)
# 이미 구현되어 있으나 pairwise 방식에서는 미사용
if lam == 1.0 and gamma == 1.0:
    td_targets = rewards.view(-1, 1).expand(-1, seq_len)
```

---

## 수정 계획

### Phase 1: Value Head 구조 변경

**목표**: MLPValueHead → LinearValueHead (4096 → 1)

**수정 파일**:
- `configs/verifiable/verifiable_pairwise.yaml`

**변경 사항**:
```yaml
# 추가
value_head:
  type: linear  # "mlp" → "linear"
  dropout: 0.0  # linear에서는 미사용
```

**검증**:
- LinearValueHead가 이미 `value_head.py`에 구현되어 있음
- `adapter.py`의 `from_pretrained()`에서 `value_head_type` 파라미터 지원 확인됨

---

### Phase 2: Trunk Unfreeze 지원

**목표**: Value head 학습 시 trunk 최상위 N개 layer gradient 흐름 허용

**수정 파일**:
- `src/weighted_mtp/models/meta_mtp/adapter.py`
- `configs/verifiable/verifiable_pairwise.yaml`

**변경 사항**:

1. **Config 추가**:
```yaml
value_head:
  type: linear
  num_unfrozen_layers: 2  # 0: value_head만, 2: 상위 2개 trunk layer 포함
```

2. **adapter.py forward() 수정**:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_value_logits: bool = False,
    return_hidden_states: bool = False,
    num_unfrozen_layers: int = 0,  # 새 파라미터
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    num_unfrozen_layers: Value head 학습 시 unfreeze할 trunk layer 수
        - 0: value_head만 학습 (현재 동작)
        - 2: 상위 2개 layer (layers[-2:]) gradient 허용
    """
```

3. **Gradient 흐름 제어 로직**:
```python
# Stage 2: MTP + Value 동시 학습
if return_value_logits and return_hidden_states:
    logits, hidden_states = self.transformer(...)

    if num_unfrozen_layers > 0:
        # trunk 일부 unfreeze: hidden_states gradient 유지
        value_logits = self.value_head(hidden_states)
    else:
        # trunk 완전 freeze: 기존 동작
        value_logits = self.value_head(hidden_states.detach())
```

**주의사항**:
- FSDP wrapping 후에는 layer 단위 freeze가 복잡하므로, `hidden_states.detach()` 여부로 gradient 흐름 제어
- trunk layer 자체를 freeze하지 않고, value_loss gradient가 trunk로 흐르는지 여부만 제어

---

### Phase 3: Tokenwise MSE Loss 추가

**목표**: Pairwise loss + MC 방식 Tokenwise MSE loss 결합

**수정 파일**:
- `src/weighted_mtp/utils/pairwise_utils.py` (또는 새 파일 `mc_loss_utils.py`)
- `src/weighted_mtp/pipelines/run_verifiable.py`
- `configs/verifiable/verifiable_pairwise.yaml`

**새 함수 추가** (`pairwise_utils.py` 또는 별도 파일):
```python
def compute_mc_value_loss(
    value_logits: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Monte Carlo Value Loss (Tokenwise MSE)

    Terminal reward를 모든 토큰에 전파하여 MSE loss 계산
    V(s_t) → R (correct: 1.0, incorrect: 0.0)

    Args:
        value_logits: [batch, seq, 1] Value head 출력
        rewards: [batch] Binary reward (0.0: incorrect, 1.0: correct)
        attention_mask: [batch, seq] 유효 토큰 마스크
        response_mask: [batch, seq] Response 토큰 마스크 (labels != -100)

    Returns:
        mc_loss: scalar tensor
    """
    batch_size, seq_len, _ = value_logits.shape
    dtype = value_logits.dtype

    # MC targets: 모든 토큰에 동일한 terminal reward 할당
    mc_targets = rewards.view(-1, 1).expand(-1, seq_len).to(dtype)  # [batch, seq]

    # Response 토큰만으로 MSE 계산 (Instruction 제외)
    combined_mask = attention_mask * response_mask

    # Tokenwise MSE
    values = value_logits.squeeze(-1)  # [batch, seq]
    mse = (values - mc_targets) ** 2

    # Masked mean
    masked_mse = (mse * combined_mask).sum() / (combined_mask.sum() + 1e-8)

    return masked_mse
```

**Config 추가**:
```yaml
training:
  # Value loss 구성
  value_loss:
    use_pairwise: true       # Pairwise ranking loss 사용
    use_mc_mse: true         # MC tokenwise MSE loss 사용
    pairwise_coef: 1.0       # Pairwise loss 계수
    mc_mse_coef: 0.5         # MC MSE loss 계수
```

**run_verifiable.py 수정**:
```python
# Value Loss 계산 (hybrid: pairwise + MC MSE)
value_loss = torch.tensor(0.0, device=device, dtype=model_dtype)

if config.training.value_loss.use_pairwise:
    pairwise_loss = pairwise_ranking_loss(pos_value_for_ranking, neg_value_logits, pos_mask, neg_mask)
    value_loss = value_loss + config.training.value_loss.pairwise_coef * pairwise_loss

if config.training.value_loss.use_mc_mse:
    # Positive sample에 대한 MC MSE (reward=1)
    pos_mc_loss = compute_mc_value_loss(
        pos_value_logits, pos_rewards, pos_attention_mask, pos_response_mask
    )
    # Negative sample에 대한 MC MSE (reward=0)
    neg_rewards = torch.zeros(neg_input_ids.size(0), device=device, dtype=model_dtype)
    neg_mc_loss = compute_mc_value_loss(
        neg_value_logits, neg_rewards, neg_attention_mask, neg_response_mask
    )
    mc_mse_loss = (pos_mc_loss + neg_mc_loss) / 2
    value_loss = value_loss + config.training.value_loss.mc_mse_coef * mc_mse_loss
```

---

### Phase 4: Config 및 Pipeline 통합

**새 Config 파일**: `configs/verifiable/verifiable_mc_hybrid.yaml`

```yaml
# Verifiable WMTP + MC Hybrid Loss
# Linear Value Head + Trunk Unfreeze + Pairwise + MC MSE

project:
  name: weighted-mtp
  version: "2.1.0"

experiment:
  name: verifiable-mc-hybrid
  description: "WMTP with Linear Value Head, MC MSE + Pairwise Loss"
  stage: verifiable
  tags:
    - verifiable
    - mc-mse
    - pairwise
    - linear-value-head
    - trunk-unfreeze

# 모델 설정
models:
  policy:
    name: meta-llama-mtp
    path: storage/checkpoints/critic/critic-pretrain-pairwise/checkpoint_epoch_2.00.pt
    tokenizer_path: storage/models/meta-llama-mtp/tokenizer
    params:
      dim: 4096
      n_layers: 32
      n_heads: 32
      n_future_tokens: 4
    dtype: bfloat16

# Value Head 설정 (회귀)
value_head:
  type: linear                # 4096 → 1 single linear
  num_unfrozen_layers: 2      # 상위 2개 trunk layer gradient 허용

# 학습 설정
training:
  n_epochs: 2.0
  batch_size: 32
  gradient_accumulation_steps: 1
  trunk_learning_rate: 1.0e-5
  value_head_learning_rate: 1.0e-4
  max_grad_norm: 0.5

  # TD error weighting (정답 샘플만)
  beta: 1.0
  weight_clip_min: 0.1
  weight_clip_max: 2.0
  gamma: 1.0
  lam: 1.0  # MC

  # Value loss 구성 (hybrid)
  value_loss:
    use_pairwise: true
    use_mc_mse: true
    pairwise_coef: 1.0
    mc_mse_coef: 0.5

  # CE loss 계수
  value_loss_coef: 0.05

# 데이터 샘플링 (문제 해결됨: max_pairs_per_problem)
data_sampling:
  seed: 42
  val_n_samples: 500
  use_pairwise: true
  n_samples: 100000
  max_pairs_per_problem: 25

  difficulty_bins:
    zero: [0, 0]
    diff_7: [7, 7]
    else: [8, 25]
  difficulty_weights:
    zero: 0.4
    diff_7: 0.3
    else: 0.3
```

---

## 수정 순서 및 의존성

```
Phase 1: Value Head 구조 변경
    └── Config 파라미터 추가 (value_head.type)
    └── adapter.py에서 value_head_type 전달 확인

Phase 2: Trunk Unfreeze 지원
    └── Phase 1 완료 필요
    └── Config 파라미터 추가 (value_head.num_unfrozen_layers)
    └── adapter.py forward() 수정

Phase 3: MC MSE Loss 추가
    └── Phase 1, 2 독립 (병렬 진행 가능)
    └── compute_mc_value_loss() 함수 추가
    └── Config 파라미터 추가 (training.value_loss.*)

Phase 4: Pipeline 통합
    └── Phase 1, 2, 3 완료 필요
    └── run_verifiable.py 수정
    └── 새 config 파일 생성
    └── validation 함수 수정
```

---

## 검증 항목

### Phase 1 검증
- [ ] LinearValueHead 로드 확인
- [ ] Checkpoint에서 value_head_type 자동 감지 확인

### Phase 2 검증
- [ ] `num_unfrozen_layers=0`: trunk gradient 차단 확인
- [ ] `num_unfrozen_layers=2`: trunk gradient 흐름 확인
- [ ] FSDP 환경에서 정상 동작 확인

### Phase 3 검증
- [ ] MC MSE loss 계산 정확성 (correct: target=1, incorrect: target=0)
- [ ] Pairwise + MC MSE hybrid loss 합산 확인
- [ ] Response 마스크 적용 확인

### Phase 4 검증
- [ ] 새 config 로드 및 파라미터 적용 확인
- [ ] 전체 training loop 정상 동작
- [ ] MLflow 메트릭 로깅 확인 (pairwise_loss, mc_mse_loss 분리)

---

## 예상 효과

1. **Linear Value Head**: 파라미터 수 감소 (MLP 대비), 단순한 구조로 해석 용이
2. **Trunk Unfreeze**: Value function 표현력 증가, 더 정확한 token-level 가치 추정
3. **MC MSE Loss**:
   - Tokenwise 학습 신호 제공 (pairwise는 시퀀스 평균만)
   - Terminal reward 직접 학습으로 수렴 안정성
4. **Hybrid Loss**: Pairwise의 상대 비교 + MC의 절대 값 학습 시너지

---

## 롤백 계획

문제 발생 시 각 Phase별 독립 롤백 가능:
- Phase 1: `value_head.type: mlp`로 복원
- Phase 2: `num_unfrozen_layers: 0`으로 복원
- Phase 3: `use_mc_mse: false`로 비활성화
