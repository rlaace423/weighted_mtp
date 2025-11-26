# Pairwise Sampling 파이프라인 통합 계획

## 문제 정의

현재 `use_pairwise` 옵션이 **critic 파이프라인에만** 구현되어 있음.
Stage 2 학습(verifiable)에서 Pairwise Ranking Loss를 Auxiliary Loss로 활용하려면 추가 구현 필요.

### 현재 상태

| 파이프라인 | use_pairwise | Value Head | Pairwise Loss | 용도 |
|-----------|--------------|------------|---------------|------|
| baseline | X | X | X | SFT 비교 기준선 |
| rho1 | X | X | X | Reference 모델 기반 선택적 학습 |
| verifiable | X | O | X | TD Weighting + Policy 학습 |
| critic | **O** | O | **O** | Value Head 사전학습 |

### 목표 상태

| 파이프라인 | use_pairwise | Auxiliary Pairwise Loss | 설명 |
|-----------|--------------|-------------------------|------|
| baseline | X (제외) | X | Value Head 없음, Pairwise 불필요 |
| rho1 | X (제외) | X | Reference 모델 방식, Pairwise 불필요 |
| verifiable | **O** | **O** | Positive: TD Weighting, Auxiliary: Pairwise Ranking |
| critic | O | - | 기존 유지 |

---

## 범위 분석

### 구현 대상: verifiable 파이프라인

**제외 대상:**
- `baseline`: Value Head가 없어 Pairwise Loss 계산 불가
- `rho1`: Reference 모델 기반 방식으로, Pairwise와 별개 연구 방향

### 핵심 설계 원칙

```
[Pairwise 모드 학습 흐름]

1. 데이터 로딩:
   - Difficulty 샘플링 → Pairwise 쌍 생성 (기존 구현 재사용)
   - 결과: {instruction, input, correct_output, incorrect_output}

2. Forward Pass:
   - Positive (correct): Policy Loss + Value Loss + TD Weighting 적용
   - Negative (incorrect): Value Loss만 (Policy 학습 제외)

3. Loss 구성:
   - Policy Loss: Positive sample의 output 부분만 학습
   - Value Loss: Positive + Negative 모두 학습 (TD targets)
   - Auxiliary Loss: Pairwise Ranking Loss (V_pos > V_neg)

4. Total Loss:
   total_loss = policy_loss + value_coef * value_loss + aux_coef * pairwise_loss
```

---

## 영향받는 파일

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `src/weighted_mtp/pipelines/run_verifiable.py` | 핵심 수정 | Pairwise 모드 분기 + Auxiliary Loss 추가 |
| `configs/verifiable/verifiable_pairwise.yaml` | 신규 생성 | Pairwise 모드 설정 파일 |
| `configs/verifiable/verifiable.yaml` | 수정 | use_pairwise: false 명시 |
| `tests/integration/test_pipeline_verifiable_pairwise.py` | 신규 생성 | 통합 테스트 |

---

## Phase별 수정 계획

### Phase 1: run_verifiable.py - Config 구조 업데이트

**목표**: `sampling_method` → `use_pairwise` 구조 통일

#### Step 1.1: sampling_config 처리 로직 수정

**변경 전** (`run_verifiable.py:348-367`):
```python
sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)
sampling_method = sampling_config.get("sampling_method")
logger.info(f"샘플링 방식: {sampling_method}")

# ...

if sampling_method == "difficulty":
    val_sampling_config["difficulty"] = val_sampling_config.get("difficulty", {}).copy()
    val_sampling_config["difficulty"]["n_samples"] = config.data_sampling.val_n_samples
```

**변경 후**:
```python
sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)
use_pairwise = sampling_config.get("use_pairwise", False)
logger.info(f"Pairwise 모드: {use_pairwise}")

# Validation sampling config
val_sampling_config = sampling_config.copy()
val_sampling_config["n_samples"] = config.data_sampling.val_n_samples
```

#### Step 1.2: pairwise_ranking_loss 함수 임포트

```python
# run_critic.py에서 재사용
from weighted_mtp.pipelines.run_critic import pairwise_ranking_loss, compute_pairwise_accuracy
```

**주의**: 순환 참조 방지를 위해 공통 모듈로 분리 필요시 `weighted_mtp/losses/pairwise.py` 생성

---

### Phase 2: run_verifiable.py - Pairwise 학습 루프 구현

**목표**: Pairwise 배치 처리 + Auxiliary Loss 통합

#### Step 2.1: 학습 루프 분기 추가

**변경 위치**: `run_verifiable.py:478-560` (기존 학습 루프 내부)

**변경 후 구조**:
```python
if use_pairwise:
    # === Pairwise 모드 ===

    # 1. Positive/Negative 데이터 추출
    pos_input_ids = batch["pos_input_ids"].to(device)
    pos_attention_mask = batch["pos_attention_mask"].to(device)
    pos_labels = batch["pos_labels"].to(device)
    neg_input_ids = batch["neg_input_ids"].to(device)
    neg_attention_mask = batch["neg_attention_mask"].to(device)
    neg_labels = batch["neg_labels"].to(device)

    # 2. Forward (Positive)
    pos_outputs = adapter(pos_input_ids, pos_attention_mask,
                          return_value_logits=True, return_hidden_states=True)
    pos_logits = pos_outputs["logits"]
    pos_value_logits = pos_outputs["value_logits"]

    # 3. Forward (Negative) - Value만 사용
    neg_outputs = adapter(neg_input_ids, neg_attention_mask,
                          return_value_logits=True)
    neg_value_logits = neg_outputs["value_logits"]

    # 4. Policy Loss (Positive만, TD Weighted)
    pos_rewards = torch.ones(pos_input_ids.size(0), device=device, dtype=model_dtype)
    td_errors = compute_td_errors(pos_value_logits, pos_rewards, pos_attention_mask, gamma=1.0)
    weights = build_weights(td_errors, beta, min_weight, max_weight)

    policy_loss = compute_weighted_ce_loss(pos_logits, pos_labels, weights, pos_attention_mask)

    # 5. Value Loss (Positive + Negative)
    pos_td_targets = compute_td_targets(pos_value_logits, pos_rewards, pos_attention_mask, gamma, lam)
    neg_rewards = torch.zeros(neg_input_ids.size(0), device=device, dtype=model_dtype)
    neg_td_targets = compute_td_targets(neg_value_logits, neg_rewards, neg_attention_mask, gamma, lam)

    pos_value_loss = compute_value_loss(pos_value_logits, pos_td_targets, pos_labels)
    neg_value_loss = compute_value_loss(neg_value_logits, neg_td_targets, neg_labels)
    value_loss = (pos_value_loss + neg_value_loss) / 2

    # 6. Auxiliary Pairwise Ranking Loss
    pos_mask = (pos_labels != -100).to(model_dtype)
    neg_mask = (neg_labels != -100).to(model_dtype)
    aux_loss = pairwise_ranking_loss(pos_value_logits, neg_value_logits, pos_mask, neg_mask)

    # 7. Total Loss
    aux_coef = getattr(config.training, "aux_coef", 0.1)
    total_loss = policy_loss + value_coef * value_loss + aux_coef * aux_loss

else:
    # === 기존 Pointwise 모드 ===
    # (기존 코드 유지)
```

#### Step 2.2: Validation 함수 Pairwise 지원

**변경 위치**: `validate_verifiable()` 함수

```python
def validate_verifiable(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    device: torch.device,
    # ... 기존 파라미터 ...
    use_pairwise: bool = False,  # 추가
    aux_coef: float = 0.1,       # 추가
) -> dict[str, float]:
```

---

### Phase 3: Config 파일 생성/수정

#### Step 3.1: verifiable.yaml 수정

```yaml
data_sampling:
  seed: 42
  val_n_samples: 1000
  use_pairwise: false  # 명시적 설정

  n_samples: 50000
  auto_data_balancing: true
  correct_ratio: 0.5
  difficulty_bins:
    diff_7: [7, 7]
    else: [8, 25]
  difficulty_weights:
    diff_7: 0.35
    else: 0.65
```

#### Step 3.2: verifiable_pairwise.yaml 신규 생성

```yaml
# Verifiable + Pairwise Auxiliary Loss
# Stage 2 학습에 Pairwise Ranking 보조 손실 추가

experiment:
  name: verifiable-pairwise
  description: "Stage 2 with Pairwise Auxiliary Loss"
  tags:
    - verifiable
    - pairwise-auxiliary
    - td-weighting

data_sampling:
  seed: 42
  val_n_samples: 1000
  use_pairwise: true  # Pairwise 모드 활성화

  n_samples: 50000
  auto_data_balancing: true
  correct_ratio: 0.5
  difficulty_bins:
    diff_7: [7, 7]
    else: [8, 25]
  difficulty_weights:
    diff_7: 0.35
    else: 0.65

training:
  # ... 기존 설정 ...
  aux_coef: 0.1  # Pairwise Ranking Loss 계수 (신규)
```

---

### Phase 4: 공통 모듈 분리 (Optional)

순환 참조 방지 및 코드 재사용을 위해 pairwise loss 함수 분리

#### Step 4.1: `weighted_mtp/losses/pairwise.py` 생성

```python
"""Pairwise Ranking Loss 모듈

run_critic, run_verifiable에서 공통 사용
"""

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    mask_pos: torch.Tensor,
    mask_neg: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry Pairwise Ranking Loss

    Output 토큰만 사용하여 시퀀스 평균 비교
    """
    v_pos_mean = (v_pos.squeeze(-1) * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)

    return -F.logsigmoid(v_pos_mean - v_neg_mean).mean()


def compute_pairwise_accuracy(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    mask_pos: torch.Tensor,
    mask_neg: torch.Tensor,
) -> dict[str, float]:
    """Pairwise Accuracy 계산"""
    v_pos_mean = (v_pos.squeeze(-1) * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)

    correct_pairs = (v_pos_mean > v_neg_mean).float().sum().item()
    total_pairs = v_pos_mean.size(0)

    return {
        "pairwise_accuracy": correct_pairs / total_pairs if total_pairs > 0 else 0.0,
        "mean_pos": v_pos_mean.mean().item(),
        "mean_neg": v_neg_mean.mean().item(),
        "margin": (v_pos_mean - v_neg_mean).mean().item(),
        "correct_pairs": correct_pairs,
        "total_pairs": total_pairs,
    }
```

#### Step 4.2: run_critic.py, run_verifiable.py에서 임포트 변경

```python
# Before (run_critic.py 내부 정의)
def pairwise_ranking_loss(...): ...

# After
from weighted_mtp.losses.pairwise import pairwise_ranking_loss, compute_pairwise_accuracy
```

---

### Phase 5: 테스트 코드 작성

#### Step 5.1: 통합 테스트 생성

**파일**: `tests/integration/test_pipeline_verifiable_pairwise.py`

```python
"""Verifiable Pairwise Pipeline Integration Test"""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf


@pytest.fixture
def verifiable_pairwise_config():
    """Verifiable + Pairwise 테스트용 config"""
    return OmegaConf.create({
        "data_sampling": {
            "seed": 42,
            "val_n_samples": 10,
            "use_pairwise": True,
            "n_samples": 100,
            "auto_data_balancing": True,
            "correct_ratio": 0.5,
            "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
            "difficulty_weights": {"diff_7": 0.35, "else": 0.65},
        },
        "training": {
            "aux_coef": 0.1,
            # ... 기타 설정 ...
        },
        # ... 나머지 설정 ...
    })


@pytest.mark.integration
def test_verifiable_pairwise_training_step(verifiable_pairwise_config):
    """단일 학습 스텝 검증"""
    # 테스트 구현
    pass


@pytest.mark.integration
def test_pairwise_auxiliary_loss_gradient():
    """Auxiliary Loss gradient 역전파 검증"""
    # 테스트 구현
    pass
```

---

## 구현 순서

1. **Phase 4**: 공통 모듈 분리 (선행 작업)
2. **Phase 1**: Config 구조 업데이트
3. **Phase 2**: 학습 루프 Pairwise 분기 구현
4. **Phase 3**: Config 파일 생성
5. **Phase 5**: 테스트 코드 작성

---

## 검증 계획

| 단계 | 검증 항목 | 방법 |
|------|----------|------|
| 1 | Config 로딩 | use_pairwise=true/false 분기 확인 |
| 2 | 데이터 로딩 | Pairwise 배치 구조 확인 (pos/neg) |
| 3 | Policy Loss | Positive sample output만 학습 확인 |
| 4 | Value Loss | Positive=1, Negative=0 reward 확인 |
| 5 | Auxiliary Loss | Pairwise accuracy > 0.5 수렴 확인 |
| 6 | Gradient Flow | 모든 Loss가 역전파되는지 확인 |

---

## 예상 영향도

- **Breaking Change 없음**: use_pairwise=false가 기본값이므로 기존 동작 유지
- **새로운 학습 모드**: verifiable_pairwise.yaml로 Auxiliary Loss 실험 가능
- **코드 재사용**: pairwise loss 함수를 공통 모듈로 분리하여 유지보수성 향상

---

## 승인 요청

위 계획에 대해 검토 및 승인 부탁드립니다.

- [ ] Phase 4: 공통 모듈 분리 (pairwise.py)
- [ ] Phase 1: run_verifiable.py Config 구조 업데이트
- [ ] Phase 2: run_verifiable.py Pairwise 학습 루프 구현
- [ ] Phase 3: Config 파일 생성/수정
- [ ] Phase 5: 테스트 코드 작성
