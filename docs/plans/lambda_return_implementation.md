# λ-Return Value Loss 구현 계획서

## 배경 및 목적

### 현재 문제
- **MC MSE Loss**: 모든 토큰에 동일한 terminal reward(1/0) 할당
- **결과**: Train acc 70%+ vs Val acc 59% (일반화 실패)
- **원인**: 토큰별 차별화 없이 problem-level 패턴만 학습

### 해결 방안
InstructGPT의 GAE(γ=1.0, λ=0.95) 개념을 Offline Supervised 환경에 적용
- **Fitted λ-Return**: V 예측값과 terminal reward의 가중평균으로 타겟 생성
- **효과**: 후반 토큰은 R에 가깝고, 전반 토큰은 V에 의존 → 자연스러운 credit 분배

### 안정화 전략
학습 초기 불안정성 방지를 위한 3단계 접근:
1. **Bias 초기화**: Value head bias를 0.5로 설정 (기대값 일치)
2. **EOS Warmup**: 초기 N step은 마지막 토큰만 학습 (anchor 형성)
3. **Full λ-Return**: Warmup 후 전체 토큰에 λ-return 적용

---

## Phase 0: 학습 안정화 기반 구축

### 0.1 Value Head Bias 초기화

**수정 대상**: `src/weighted_mtp/models/value_model.py`

**목적**:
- 초기 예측값을 기대값(0.5)에 맞춤
- 학습 초기 gradient 폭발/소실 방지

**구현**:
```python
class ValueModel(nn.Module):
    def __init__(self, ...):
        ...
        # Value head 마지막 레이어 bias를 0.5로 초기화
        if hasattr(self.value_head, '__getitem__'):
            # Sequential인 경우
            self.value_head[-1].bias.data.fill_(0.5)
        else:
            self.value_head.bias.data.fill_(0.5)
```

### 0.2 EOS-only Warmup 로직 ✅ 구현 완료

**수정 대상**: `src/weighted_mtp/utils/pairwise_utils.py`

**목적**:
- 학습 초기에 terminal position(EOS)만 학습
- Value model이 "정답=높음, 오답=낮음" 기본 패턴 학습
- Warmup 후 valid acc 0.6~0.7 도달 시 본 학습 시작

**구현** (Instruction 마스킹 고려):
```python
def create_eos_only_mask(loss_mask: torch.Tensor) -> torch.Tensor:
    """마지막 유효 토큰(Output 끝)만 1인 마스크 생성

    Warmup 학습에서 terminal position만 학습하기 위한 마스크.
    loss_mask가 [0,0,1,1,1,0,0] 형태(Instruction 마스킹)일 때
    마지막 1의 위치를 정확히 찾음.

    Args:
        loss_mask: [batch, seq] Output 토큰 마스크 (labels != -100)

    Returns:
        eos_mask: [batch, seq] 마지막 유효 토큰만 1
    """
    batch_size, seq_len = loss_mask.shape
    device = loss_mask.device

    # 인덱스 [1, 2, ..., seq_len] 생성 (0이 아닌 1부터 시작하여 0값과 구분)
    seq_indices = torch.arange(1, seq_len + 1, device=device).unsqueeze(0)

    # 유효 위치만 인덱스 값 유지, 나머지는 0
    valid_indices = seq_indices * loss_mask.float()

    # 각 배치의 최대 인덱스 = 마지막 유효 토큰 위치
    last_valid_pos = valid_indices.argmax(dim=1)

    # EOS 마스크 생성
    eos_mask = torch.zeros_like(loss_mask, dtype=torch.float32)
    batch_indices = torch.arange(batch_size, device=device)

    # 유효 토큰이 있는 배치만 마킹
    has_valid_tokens = loss_mask.sum(dim=1) > 0
    eos_mask[batch_indices[has_valid_tokens], last_valid_pos[has_valid_tokens]] = 1.0

    return eos_mask
```

**핵심 수정 사항** (기존 계획 대비):
- 기존: `loss_mask.sum(dim=1) - 1` → Instruction 마스킹 미고려로 잘못된 위치 반환
- 수정: `(seq_indices * loss_mask).argmax(dim=1)` → Instruction 마스킹 환경에서 정확한 마지막 Output 토큰 위치 반환

**Training loop 수정**:
```python
# Warmup 설정
warmup_steps = config.training.value_loss.lambda_return.get("warmup_steps", 0)
use_eos_warmup = warmup_steps > 0

# Training loop 내
if use_eos_warmup and global_step < warmup_steps:
    # Warmup: EOS 토큰만 학습
    pos_effective_mask = create_eos_only_mask(pos_loss_mask)
    neg_effective_mask = create_eos_only_mask(neg_loss_mask)
else:
    # 본 학습: 전체 output 토큰
    pos_effective_mask = pos_loss_mask
    neg_effective_mask = neg_loss_mask
```

### 0.3 Config 확장

```yaml
training:
  value_loss:
    lambda_return:
      gamma: 1.0
      lam: 0.95
      warmup_steps: 1000    # EOS-only warmup steps (0이면 비활성화)
      bias_init: 0.5        # Value head bias 초기값
```

---

## Phase 1: λ-Return 핵심 함수 구현 ✅ 구현 완료

### 1.1 수정 대상
- `src/weighted_mtp/utils/pairwise_utils.py`

### 1.2 구현 내용

```python
def compute_lambda_return(
    values: torch.Tensor,           # [batch, seq] 현재 V 예측
    terminal_rewards: torch.Tensor, # [batch] 1.0 or 0.0
    loss_mask: torch.Tensor,        # [batch, seq] output 토큰 마스크
    gamma: float = 1.0,
    lam: float = 0.95,
) -> torch.Tensor:
    """Fitted λ-Return 타겟 계산 (Offline TD)

    G_t^λ = (1-λ)(r_t + γV_{t+1}) + λ(r_t + γG_{t+1}^λ)

    Terminal reward만 있을 때 (r_t=0 for t<T):
    - Terminal: G_T = R
    - Earlier: G_t = (1-λ)γV_{t+1} + λγG_{t+1}

    Args:
        values: [batch, seq] 현재 Value model 예측 (detached)
        terminal_rewards: [batch] terminal reward (1.0 or 0.0)
        loss_mask: [batch, seq] 유효 토큰 마스크
        gamma: discount factor (default: 1.0, InstructGPT 설정)
        lam: GAE smoothing factor (default: 0.95)

    Returns:
        lambda_returns: [batch, seq] 위치별 λ-return 타겟
    """
    batch_size, seq_len = values.shape
    device = values.device
    dtype = values.dtype

    # 결과 텐서 초기화
    lambda_returns = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

    # 각 시퀀스의 마지막 유효 위치
    seq_lengths = loss_mask.sum(dim=1).long()  # [batch]

    # 역방향으로 λ-return 계산
    for b in range(batch_size):
        T = seq_lengths[b].item() - 1  # 0-indexed last position
        if T < 0:
            continue

        # Terminal position: G_T = R
        # loss_mask가 1인 위치 중 마지막 찾기
        valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue
        last_pos = valid_positions[-1].item()
        lambda_returns[b, last_pos] = terminal_rewards[b]

        # 역방향 전파: t = T-1, T-2, ..., 0
        G_next = terminal_rewards[b]
        for i in range(len(valid_positions) - 2, -1, -1):
            t = valid_positions[i].item()
            t_next = valid_positions[i + 1].item()

            # G_t = (1-λ)γV_{t+1} + λγG_{t+1}
            V_next = values[b, t_next]
            td_component = (1 - lam) * gamma * V_next
            mc_component = lam * gamma * G_next
            G_t = td_component + mc_component

            lambda_returns[b, t] = G_t
            G_next = G_t

    return lambda_returns


def compute_lambda_value_loss(
    value_logits: torch.Tensor,     # [batch, seq, 1]
    terminal_rewards: torch.Tensor, # [batch]
    attention_mask: torch.Tensor,   # [batch, seq]
    loss_mask: torch.Tensor,        # [batch, seq]
    gamma: float = 1.0,
    lam: float = 0.95,
) -> torch.Tensor:
    """λ-Return 기반 Value Loss

    MC MSE를 대체하는 Fitted TD 방식의 value loss.

    Args:
        value_logits: [batch, seq, 1] Value head 출력
        terminal_rewards: [batch] terminal reward (1.0 or 0.0)
        attention_mask: [batch, seq] padding 마스크
        loss_mask: [batch, seq] output 토큰 마스크 (labels != -100)
        gamma: discount factor
        lam: GAE smoothing factor

    Returns:
        loss: scalar tensor
    """
    values = value_logits.squeeze(-1)  # [batch, seq]

    # λ-return 타겟 계산 (gradient 차단)
    with torch.no_grad():
        lambda_targets = compute_lambda_return(
            values.detach(), terminal_rewards, loss_mask, gamma, lam
        )

    # MSE loss (masked)
    combined_mask = attention_mask * loss_mask
    mse = (values - lambda_targets) ** 2
    masked_mse = (mse * combined_mask).sum() / (combined_mask.sum() + 1e-8)

    return masked_mse
```

### 1.3 테스트
- `tests/unit/test_pairwise_utils.py`에 단위 테스트 추가
- λ=1.0일 때 MC와 동일한지 검증
- λ=0.95일 때 타겟 차별화 검증

---

## Phase 2: Config 확장

### 2.1 수정 대상
- `configs/production/critic_mlp.yaml`
- `configs/local/critic_local.yaml`

### 2.2 Config 구조 변경

```yaml
training:
  value_loss:
    # 기존 옵션 (유지, 상호 배타적)
    use_pairwise: false
    pairwise_coef: 0
    use_mc_mse: false        # λ-return 사용 시 false로 설정
    mc_mse_coef: 1.0

    # λ-Return 옵션 (신규)
    use_lambda_return: true
    lambda_return:
      gamma: 1.0             # discount factor (InstructGPT: 1.0)
      lam: 0.95              # GAE smoothing (InstructGPT: 0.95)
      coef: 1.0              # loss coefficient
      warmup_steps: 1000     # EOS-only warmup (0이면 비활성화)
      bias_init: 0.5         # Value head bias 초기값
```

### 2.3 호환성 매트릭스

| use_mc_mse | use_lambda_return | 동작 |
|------------|-------------------|------|
| true | false | 기존 MC MSE |
| false | true | 신규 λ-return |
| false | false | value loss 비활성화 |
| true | true | 에러 (상호 배타적) |

---

## Phase 3: run_critic.py 통합

### 3.1 수정 위치
- Value model 로딩 (bias 초기화)
- Training loop (warmup + λ-return)
- Validation function

### 3.2 주요 변경

**Value Model 로딩 시**:
```python
# Bias 초기화 적용
bias_init = config.training.value_loss.lambda_return.get("bias_init", 0.5)
if bias_init is not None:
    apply_value_head_bias_init(value_model, bias_init)
```

**Training loop**:
```python
# Config에서 설정 추출
use_lambda_return = config.training.value_loss.get("use_lambda_return", False)
lambda_config = config.training.value_loss.get("lambda_return", {})
warmup_steps = lambda_config.get("warmup_steps", 0)

# Loss 계산
if use_lambda_return:
    # Warmup 체크
    if warmup_steps > 0 and global_step < warmup_steps:
        effective_pos_mask = create_eos_only_mask(pos_loss_mask)
        effective_neg_mask = create_eos_only_mask(neg_loss_mask)
    else:
        effective_pos_mask = pos_loss_mask
        effective_neg_mask = neg_loss_mask

    pos_lambda_loss = compute_lambda_value_loss(
        pos_value_logits, pos_rewards, pos_attention_mask, effective_pos_mask,
        gamma=lambda_config.get("gamma", 1.0),
        lam=lambda_config.get("lam", 0.95),
    )
    neg_lambda_loss = compute_lambda_value_loss(
        neg_value_logits, neg_rewards, neg_attention_mask, effective_neg_mask,
        gamma=lambda_config.get("gamma", 1.0),
        lam=lambda_config.get("lam", 0.95),
    )
    lambda_loss_val = (pos_lambda_loss + neg_lambda_loss) / 2
    value_loss = value_loss + lambda_config.get("coef", 1.0) * lambda_loss_val
```

### 3.3 로깅 추가
- `train/warmup_active`: warmup 진행 중 여부 (0/1)
- `value/lambda_loss`: λ-return loss 값
- `config/gamma`: 사용된 γ 값
- `config/lambda`: 사용된 λ 값

---

## Phase 4: utils/__init__.py 업데이트 ✅ 구현 완료

### 4.1 Export 추가

```python
from weighted_mtp.utils.pairwise_utils import (
    compute_lambda_return,
    compute_lambda_value_loss,
    compute_mc_value_loss,
    compute_pairwise_accuracy,
    compute_token_variance,
    create_eos_only_mask,         # 신규 (Phase 0.2)
    pairwise_ranking_loss,
)
```

---

## Phase 5: 테스트 및 검증 ✅ 구현 완료

### 5.1 단위 테스트
```bash
PYTHONPATH=src pytest tests/unit/test_pairwise_utils.py tests/unit/test_value_head.py -v
```

**테스트 파일 및 케이스** (총 25개):

| 파일 | 테스트 클래스 | 케이스 수 |
|------|---------------|-----------|
| `test_pairwise_utils.py` | `TestComputeLambdaReturn` | 6개 |
| `test_pairwise_utils.py` | `TestComputeLambdaValueLoss` | 4개 |
| `test_pairwise_utils.py` | `TestCreateEosOnlyMask` | 8개 |
| `test_value_head.py` | `TestInitValueHeadBias` | 7개 |

**주요 테스트 케이스**:
1. `test_lambda_1_equals_mc`: λ=1.0일 때 MC와 동일
2. `test_lambda_095_creates_gradient`: λ=0.95일 때 위치별 타겟 차이
3. `test_instruction_masked_sequence`: Instruction 마스킹 환경에서 EOS 마스크 정확성
4. `test_mlp_head_bias_init`: MLP Value head bias 초기화 검증
5. `test_linear_head_bias_init`: Linear Value head bias 초기화 검증

### 5.2 로컬 통합 테스트
```bash
PYTHONPATH=src python -m weighted_mtp.pipelines.run_critic \
    --config configs/local/critic_local.yaml
```

### 5.3 검증 체크리스트
- [ ] Bias 초기화 후 초기 예측값 ~0.5
- [ ] Warmup 중 EOS만 학습 (마스크 확인)
- [ ] Warmup 후 val acc 0.6+ 도달
- [ ] λ-return 적용 후 token variance 증가
- [ ] 최종 val acc 65%+ (목표)

---

## 구현 순서 요약

| Phase | 작업 | 파일 | 코드량 |
|-------|------|------|--------|
| **0** | Bias 초기화 + EOS warmup | value_model.py, run_critic.py | ~30줄 |
| **1** | λ-return 함수 구현 | pairwise_utils.py | ~60줄 |
| **2** | Config 확장 | critic_*.yaml | ~15줄 |
| **3** | Training/Validation 통합 | run_critic.py | ~40줄 |
| **4** | Export 업데이트 | utils/__init__.py | ~5줄 |
| **5** | 테스트 | tests/ | ~80줄 |

**총 예상 코드량**: ~230줄

---

## 학습 흐름 예시

```
Step 0-1000 (Warmup):
  - EOS 토큰만 학습
  - Target: R (1.0 or 0.0)
  - 기대: val acc 0.5 → 0.6~0.7

Step 1000+ (본 학습):
  - 전체 output 토큰 학습
  - Target: λ-return (위치별 차별화)
  - 기대: val acc 0.7 → 0.75+
```

---

## 예상 효과

### Before (MC MSE)
```
Token:    [t=0]  [t=1]  ...  [t=T]
Target:     R      R    ...    R     (동일)
Result:  Problem-level memorization
Val Acc: 59%
```

### After (Phase 0 + λ-Return)
```
Warmup:
Token:    [t=0]  [t=1]  ...  [t=T]
Target:     -      -    ...    R     (EOS만)
Result:  Terminal pattern 학습

본 학습:
Token:    [t=0]  [t=1]  ...  [t=T]
Target:   0.3R   0.5R  ...    R     (차별화)
Result:  Token-level credit
Val Acc: 65%+ (목표)
```

---

## 롤백 계획

문제 발생 시 즉시 복귀 가능:

```yaml
# 기존 방식으로 롤백
value_loss:
  use_mc_mse: true
  use_lambda_return: false
```

코드 변경 없이 config만으로 전환 가능.

---

## 참고 문헌

1. Ouyang et al. (2022). "Training language models to follow instructions with human feedback" (InstructGPT)
2. Schulman et al. (2016). "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (GAE)
3. Sutton & Barto (2018). "Reinforcement Learning: An Introduction" - TD(λ) 및 Eligibility Traces
