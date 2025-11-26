# Advantage Whitening 적용 개발 계획

## 상태: 구현 완료

## 배경

Pairwise Critic 학습이 성공적으로 진행됨에 따라, Stage 2 Policy Learning에서 TD error 기반 가중치 산정 방식을 개선.

### 해결한 문제

```python
# 기존: Raw TD error → exp (스케일 불안정)
weights = torch.exp(td_errors / beta)
```

Pairwise로 학습된 Critic은 **상대적 차이**에 특화되어 있어:
1. TD error의 절대적 스케일이 배치마다 불안정
2. 고정된 beta 값이 일관된 효과를 내지 못함
3. 가중치 폭발/소실 위험

### 적용한 해결책: Normalized IQL (Advantage Whitening)

```python
# 개선: Whitening → exp → 평균 정규화
A_norm = (td_errors - mean) / (std + 1e-8)
weights = torch.exp(A_norm / beta)
weights = clamp(weights, min_weight, max_weight)
weights = weights / weights.mean()  # 평균 1 유지
```

---

## 구현 내용

### 1. `td_weighting.py` - `build_weights()` 전면 교체

**파일**: `src/weighted_mtp/value_weighting/td_weighting.py`

```python
def build_weights(
    td_errors: torch.Tensor,
    attention_mask: torch.Tensor,  # 필수 인자로 변경
    beta: float = 1.0,             # 기본값 0.9 → 1.0
    min_weight: float = 0.1,
    max_weight: float = 5.0,       # 기본값 3.0 → 5.0
) -> torch.Tensor:
    """Normalized IQL weighting (Advantage Whitening 적용)"""

    # 유효 토큰만으로 통계 계산
    mask = attention_mask.bool()
    valid_td = td_errors[mask]

    # Advantage Whitening
    mean = valid_td.mean()
    std = valid_td.std()
    td_normalized = (td_errors - mean) / (std + 1e-8)

    # Exponential + Clipping
    weights = torch.exp(td_normalized / beta)
    weights = torch.clamp(weights, min=min_weight, max=max_weight)

    # 평균 1 정규화
    valid_weights = weights[mask]
    weights = weights / (valid_weights.mean() + 1e-8)

    # Padding 마스킹
    weights = weights * attention_mask.float()

    return weights
```

**변경 사항**:
- `normalize` 옵션 제거 (항상 Whitening 적용)
- `attention_mask` 필수 인자로 변경
- 기본값 업데이트: `beta=1.0`, `max_weight=5.0`

### 2. `run_verifiable.py` - 호출부 수정

**Training Loop** (line 519-525):
```python
weights = build_weights(
    td_errors=td_errors,
    attention_mask=pos_attention_mask,  # 추가
    beta=config.training.beta,
    min_weight=config.training.weight_clip_min,
    max_weight=config.training.weight_clip_max,
)
```

**Validation** (line 136-142):
```python
weights = build_weights(
    td_errors=td_errors,
    attention_mask=pos_attention_mask,  # 추가
    beta=beta,
    min_weight=weight_clip_min,
    max_weight=weight_clip_max,
)
```

### 3. Config 업데이트

**`configs/verifiable/verifiable.yaml`** 및 **`verifiable_pairwise.yaml`**:

```yaml
# Normalized IQL weighting (Advantage Whitening)
beta: 1.0           # 0.9 → 1.0 (Whitening 후 적정값)
pairwise_coef: 0.1
weight_clip_min: 0.1  # 0 → 0.1
weight_clip_max: 5.0  # 3 → 5.0
```

---

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `src/weighted_mtp/value_weighting/td_weighting.py` | `build_weights()` 전면 교체 |
| `src/weighted_mtp/pipelines/run_verifiable.py` | 호출부 2곳 수정 |
| `configs/verifiable/verifiable.yaml` | 파라미터 업데이트 |
| `configs/verifiable/verifiable_pairwise.yaml` | 파라미터 업데이트 |

---

## 검증 방법

### 로컬 테스트

```bash
# 짧은 학습 실행
torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_verifiable \
    --config configs/verifiable/verifiable.yaml \
    --override training.n_epochs=0.1
```

확인 사항:
1. Weight 분포가 안정적인가? (mean ≈ 1)
2. Loss가 정상적으로 감소하는가?
3. Gradient norm이 폭발하지 않는가?

---

## 예상 효과

| 지표 | 기존 | 개선 후 |
|------|------|---------|
| Weight 스케일 | 배치마다 변동 | 항상 평균 1 |
| Beta 일관성 | 배치 의존적 | 일관된 선별력 |
| 학습 안정성 | 가중치 폭발 위험 | 안정적 |
| Critic 호환성 | Pairwise 스케일 무시 | 상대적 중요도 활용 |
