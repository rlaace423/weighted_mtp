# Mask Naming Cleanup Plan

## 개요

파이프라인 전체에서 마스크 변수명의 일관성을 확보하기 위한 리팩토링 계획.
두 종류의 마스크를 명확히 구분하고, 함수 시그니처와 호출부의 일관성을 보장한다.

---

## 마스크 종류 정의

| 종류 | 이름 | 정의 | 용도 |
|-----|-----|-----|------|
| **Padding 마스크** | `attention_mask` | 실제 토큰=1, 패딩=0 | 모델 forward, attention 계산 |
| **학습 마스크** | `loss_mask` | labels != -100 | TD 통계, 가중치, loss 계산 |

### 핵심 차이점

```
Instruction + Response + Padding 구조:
┌─────────────────────────────────────────────────┐
│  [BOS] [Inst1] [Inst2] [Resp1] [Resp2] [PAD]   │
│                                                 │
│  attention_mask: [1, 1, 1, 1, 1, 0]            │
│  labels:         [-100, -100, -100, tok1, tok2, -100] │
│  loss_mask:      [False, False, False, True, True, False] │
│                                                 │
│  attention_mask terminal: position 4 (Resp2)   │
│  loss_mask terminal: position 4 (Resp2)        │
└─────────────────────────────────────────────────┘
```

두 마스크의 terminal 위치는 대부분 동일하지만, 의미가 다르므로 명확히 구분해야 한다.

---

## 마스크 dtype 전략

### 원칙: bool 기준으로 통일

```python
# 생성 시: bool (가장 작은 dtype, 의미 명확)
pos_loss_mask = (pos_labels != -100)  # dtype=torch.bool

# 사용 시: 함수 내부에서 필요한 dtype으로 캐스팅
# - td_weighting.py 등에서 이미 내부 캐스팅 구현됨
# - 필요 시 호출부에서 .to(model_dtype) 또는 .float() 사용
```

### 선택 근거

| 기준 | bool | long | bfloat16 |
|-----|------|------|----------|
| **메모리** | 1 byte | 8 bytes | 2 bytes |
| **의미 명확성** | ✅ True/False | ⚠️ 0/1 정수 | ⚠️ 0.0/1.0 부동소수점 |
| **인덱싱 직접 사용** | ✅ 가능 | ❌ 불가 | ❌ 불가 |
| **자동 캐스팅** | ✅ PyTorch 자동 | ⚠️ 일부만 | ✅ 연산 시 자동 |

### bfloat16 환경에서의 정밀도

```
bfloat16 마스크 값 표현:
- 0.0 = 0x0000 (정확)
- 1.0 = 0x3F80 (정확)

결론: 0/1 이진 마스크는 bfloat16에서 정밀도 손실 없음
```

### 함수 내부 dtype 캐스팅 (이미 구현됨)

```python
# td_weighting.py 내부에서 수행하는 캐스팅
bool_mask = loss_mask.bool()      # 인덱싱용 (209줄)
loss_mask.to(dtype)               # model dtype 연산 (62, 104줄)
loss_mask.float()                 # float32 통계 계산 (179, 215, 230줄)
terminal_indices.long()           # 위치 인덱스 (71, 156줄)
```

---

## 현재 상태 분석

### 1. td_weighting.py ✅ 완료

이미 `loss_mask` 파라미터로 변경 완료:
- `compute_td_targets(loss_mask=...)`
- `compute_td_errors(loss_mask=...)`
- `build_weights(loss_mask=...)`
- `compute_td_stats(loss_mask=...)`
- `compute_weight_stats(loss_mask=...)`
- `test_td_weighting.py` 테스트 수정 완료

### 2. td_accumulator.py ✅ 완료

이미 `loss_mask` 파라미터로 변경 완료:
- `accumulate(loss_mask=...)`
- 내부 리스트명 `masks_list` → `loss_masks_list`
- `test_td_accumulator.py` 테스트 수정 완료

### 3. run_verifiable.py

| 위치 | 현재 코드 | 문제 |
|-----|----------|-----|
| 142줄 | `pos_response_mask = (pos_labels != -100).long()` | 이름 불일치 + 불필요한 .long() |
| 147줄 | `compute_td_errors(..., mask=pos_attention_mask)` | attention_mask 전달 |
| 152줄 | `build_weights(..., mask=pos_response_mask)` | response_mask 전달 |
| 170줄 | `pos_mask = (pos_labels != -100).to(model_dtype)` | 중복 생성 |
| 591줄 | `pos_response_mask = (pos_labels != -100).long()` | 이름 불일치 + 불필요한 .long() |
| 594줄 | `compute_td_errors(..., attention_mask=...)` | **키워드 인자 오류!** |
| 604줄 | `build_weights(..., attention_mask=...)` | **키워드 인자 오류!** |
| 621줄 | `pos_mask = (pos_labels != -100).to(model_dtype)` | 중복 생성 |
| 658줄 | `valid_label_mask = pos_mask.unsqueeze(-1)` | 이름 불일치 |
| 735줄 | `response_mask = valid_label_mask.squeeze(-1)` | 이름 불일치 |

### 4. run_critic.py

| 위치 | 현재 코드 | 문제 |
|-----|----------|-----|
| 155줄 | `pos_mask = (pos_labels != -100).to(model_dtype)` | 이름 불일치 |
| 156줄 | `neg_mask = (neg_labels != -100).to(model_dtype)` | 이름 불일치 |
| 588줄 | `pos_mask = (pos_labels != -100).to(model_dtype)` | 이름 불일치 |
| 589줄 | `neg_mask = (neg_labels != -100).to(model_dtype)` | 이름 불일치 |

### 5. logging_utils.py

| 함수 | 파라미터 | 문제 |
|-----|---------|-----|
| `compute_weight_statistics` | `attention_mask`, `labels` | 내부에서 `labels != -100` 계산 |
| `compute_value_function_stats` | `attention_mask` | 실제로는 loss_mask 전달받음 |

### 6. rho1_weighting.py ✅ 완료

이미 `loss_mask` 파라미터로 변경 완료:
- `compute_mtp_selective_weights(loss_mask=...)` 
- `compute_rho1_stats(loss_mask=...)`
- `run_rho1.py` 호출부 수정 완료
- `test_rho1_weighting.py` 테스트 수정 완료

---

## 목표 상태

### 1. 함수 파라미터 네이밍 규칙

**value_weighting 모듈**: 모든 마스크 파라미터를 `loss_mask`로 통일

```python
# td_weighting.py
def compute_td_targets(..., loss_mask: torch.Tensor, ...)
def compute_td_errors(..., loss_mask: torch.Tensor, ...)
def build_weights(..., loss_mask: torch.Tensor, ...)
def compute_td_stats(..., loss_mask: torch.Tensor = None)
def compute_weight_stats(..., loss_mask: torch.Tensor = None)

# td_accumulator.py
def accumulate(self, td_errors: torch.Tensor, loss_mask: torch.Tensor)

# rho1_weighting.py
def compute_mtp_selective_weights(..., loss_mask: torch.Tensor, ...)
def compute_rho1_stats(..., loss_mask: torch.Tensor = None)
```

**logging_utils.py**: 파라미터 의미 명확화

```python
def compute_weight_statistics(
    weights: torch.Tensor,
    loss_mask: torch.Tensor,  # labels != -100 마스크 직접 전달
) -> dict[str, float]

def compute_value_function_stats(
    values: torch.Tensor,
    returns: torch.Tensor,
    loss_mask: torch.Tensor,  # labels != -100 마스크
) -> dict[str, float]
```

### 2. 파이프라인 변수 네이밍 규칙

| Prefix | 의미 | 예시 |
|--------|-----|------|
| `pos_` | Positive (correct) 샘플 | `pos_loss_mask` |
| `neg_` | Negative (incorrect) 샘플 | `neg_loss_mask` |
| 없음 | 단일 샘플 또는 통합 | `loss_mask` |

```python
# run_verifiable.py, run_critic.py
# bool로 생성 (한 번만)
pos_loss_mask = (pos_labels != -100)
neg_loss_mask = (neg_labels != -100)

# 필요 시 dtype 변환
pairwise_ranking_loss(mask_pos=pos_loss_mask, ...)  # 내부에서 자동 캐스팅
# 또는 명시적 변환이 필요한 경우
some_func(mask=pos_loss_mask.to(model_dtype))
```

---

## 변경 작업 목록

### Task 1: td_weighting.py 파라미터명 변경 ✅ 완료

**파일**: `src/weighted_mtp/value_weighting/td_weighting.py`

이미 완료됨:
- 5개 함수의 `mask` → `loss_mask` 변경
- docstring 및 내부 변수 모두 수정
- `test_td_weighting.py` 테스트 수정 완료

### Task 2: td_accumulator.py 파라미터명 변경 ✅ 완료

**파일**: `src/weighted_mtp/value_weighting/td_accumulator.py`

이미 완료됨:
- `accumulate(mask=...)` → `accumulate(loss_mask=...)`
- 내부 리스트명 `masks_list` → `loss_masks_list`
- `test_td_accumulator.py` 테스트 수정 완료

### Task 3: rho1_weighting.py 파라미터명 변경 ✅ 완료

**파일**: `src/weighted_mtp/value_weighting/rho1_weighting.py`

이미 완료됨:
- `compute_mtp_selective_weights(loss_mask=...)` 
- `compute_rho1_stats(loss_mask=...)`
- `run_rho1.py` 호출부 수정 완료
- `test_rho1_weighting.py` 테스트 수정 완료

### Task 4: run_verifiable.py 변수명 통일 및 호출부 수정 ✅ 완료

**파일**: `src/weighted_mtp/pipelines/run_verifiable.py`

이미 완료됨:
- `pos_loss_mask`, `neg_loss_mask` 변수명 통일
- `compute_td_errors(loss_mask=...)`, `build_weights(loss_mask=...)` 키워드 인자 수정
- 중복 마스크 생성 제거 (`pos_mask`, `neg_mask` 삭제)
- `compute_td_stats(loss_mask=...)`, `compute_value_function_stats(loss_mask=...)`, `compute_weight_statistics(loss_mask=...)` 호출부 수정

#### 4-1. validate_verifiable 함수 (59~224줄)

```python
# 변경 전 (142줄)
pos_response_mask = (pos_labels != -100).long()

# 변경 후 (bool로 생성, 한 번만)
pos_loss_mask = (pos_labels != -100)
neg_loss_mask = (neg_labels != -100)
```

```python
# 변경 전 (147-149줄)
td_errors = compute_td_errors(
    value_logits=pos_value_logits,
    rewards=pos_rewards,
    mask=pos_attention_mask,
    gamma=1.0,
)

# 변경 후
td_errors = compute_td_errors(
    value_logits=pos_value_logits,
    rewards=pos_rewards,
    loss_mask=pos_loss_mask,
    gamma=1.0,
)
```

```python
# 변경 전 (150-156줄)
weights = build_weights(
    td_errors=td_errors,
    mask=pos_response_mask,
    beta=beta,
    ...
)

# 변경 후
weights = build_weights(
    td_errors=td_errors,
    loss_mask=pos_loss_mask,
    beta=beta,
    ...
)
```

```python
# 변경 전 (170-171줄) - 중복 생성
pos_mask = (pos_labels != -100).to(model_dtype)
neg_mask = (neg_labels != -100).to(model_dtype)

# 변경 후 - 삭제 (이미 위에서 생성됨)
# pairwise_ranking_loss 등에서 pos_loss_mask, neg_loss_mask 직접 사용
```

#### 4-2. run_verifiable_training 함수 (227~1083줄)

```python
# 변경 전 (591줄)
pos_response_mask = (pos_labels != -100).long()

# 변경 후 (bool로 생성, 한 번만)
pos_loss_mask = (pos_labels != -100)
neg_loss_mask = (neg_labels != -100)
```

```python
# 변경 전 (593-598줄) - 키워드 인자 오류 수정
td_errors = compute_td_errors(
    value_logits=pos_value_logits,
    rewards=pos_rewards,
    attention_mask=pos_attention_mask,  # 잘못된 키워드!
    gamma=1.0,
)

# 변경 후
td_errors = compute_td_errors(
    value_logits=pos_value_logits,
    rewards=pos_rewards,
    loss_mask=pos_loss_mask,
    gamma=1.0,
)
```

```python
# 변경 전 (602-608줄) - 키워드 인자 오류 수정
weights = build_weights(
    td_errors=td_errors,
    attention_mask=pos_response_mask,  # 잘못된 키워드!
    beta=config.training.beta,
    ...
)

# 변경 후
weights = build_weights(
    td_errors=td_errors,
    loss_mask=pos_loss_mask,
    beta=config.training.beta,
    ...
)
```

```python
# 변경 전 (621-622줄) - 중복 생성
pos_mask = (pos_labels != -100).to(model_dtype)
neg_mask = (neg_labels != -100).to(model_dtype)

# 변경 후 - 삭제 (이미 위에서 생성됨)
# pairwise_ranking_loss 등에서 pos_loss_mask, neg_loss_mask 직접 사용
```

```python
# 변경 전 (658줄)
valid_label_mask = pos_mask.unsqueeze(-1)

# 변경 후 - 삭제 (불필요)
```

```python
# 변경 전 (735줄)
response_mask = valid_label_mask.squeeze(-1)

# 변경 후
# pos_loss_mask를 직접 사용
```

```python
# 변경 전 (738줄)
td_stats = compute_td_stats(td_errors, response_mask)

# 변경 후
td_stats = compute_td_stats(td_errors, pos_loss_mask)
```

```python
# 변경 전 (742-746줄)
value_func_stats = compute_value_function_stats(
    values=value_logits.squeeze(-1),
    returns=td_targets.squeeze(-1),
    attention_mask=response_mask,
)

# 변경 후
value_func_stats = compute_value_function_stats(
    values=value_logits.squeeze(-1),
    returns=td_targets.squeeze(-1),
    loss_mask=pos_loss_mask,
)
```

```python
# 변경 전 (749줄)
weight_dist_stats = compute_weight_statistics(weights, attention_mask, labels)

# 변경 후
weight_dist_stats = compute_weight_statistics(weights, pos_loss_mask)
```

### Task 5: run_critic.py 변수명 통일 ✅ 완료

**파일**: `src/weighted_mtp/pipelines/run_critic.py`

이미 완료됨:
- `pos_loss_mask`, `neg_loss_mask` 변수명 통일
- `.to(model_dtype)` 제거 (bool 타입 유지)
- `pairwise_ranking_loss`, `compute_mc_value_loss`, `compute_pairwise_accuracy` 호출부 수정

### Task 6: logging_utils.py 파라미터명 변경 ✅ 완료

**파일**: `src/weighted_mtp/utils/logging_utils.py`

이미 완료됨:
- `compute_weight_statistics(weights, loss_mask)` - 파라미터 단순화
- `compute_value_function_stats(values, returns, loss_mask=None)` - 파라미터 변경
- 내부 로직 단순화 (attention_mask, labels 조합 → loss_mask 직접 사용)

### Task 7: 테스트 파일 업데이트

#### 7-1. test_td_weighting.py ✅ 완료

모든 `mask=` → `loss_mask=` 변경 완료.

#### 7-2. test_td_accumulator.py ✅ 완료

모든 `mask` 변수 → `loss_mask` 변경 완료.

#### 7-3. test_rho1_weighting.py ✅ 완료

모든 `attention_mask=` → `loss_mask=` 변경 완료.

---

## 변경 후 데이터 흐름도

```
┌─────────────────────────────────────────────────────────┐
│  DataLoader                                              │
│  batch["pos_attention_mask"]  →  pos_attention_mask     │
│  batch["pos_labels"]          →  pos_labels             │
│                                                          │
│  pos_loss_mask = (pos_labels != -100)  # bool, 한 번만  │
│  neg_loss_mask = (neg_labels != -100)  # bool, 한 번만  │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│  Model Forward                                           │
│  adapter(input_ids, attention_mask=pos_attention_mask)  │
│  # attention_mask: 모델 내부 attention 계산용            │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│  TD Weighting (함수 내부에서 dtype 캐스팅)               │
│  td_errors = compute_td_errors(loss_mask=pos_loss_mask) │
│  weights = build_weights(loss_mask=pos_loss_mask)       │
│  accumulator.accumulate(td_errors, pos_loss_mask)       │
│  # 함수 내부: loss_mask.bool(), .float(), .to(dtype)    │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│  Loss Computation (PyTorch 자동 캐스팅)                  │
│  compute_mtp_ce_loss(                                   │
│      attention_mask=pos_attention_mask,  # 전체 시퀀스   │
│      weights=weights,                    # loss_mask 기반│
│  )                                                       │
│  pairwise_ranking_loss(                                 │
│      mask_pos=pos_loss_mask,  # bool → 연산 시 자동 캐스팅│
│      mask_neg=neg_loss_mask,                            │
│  )                                                       │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│  Logging                                                 │
│  compute_td_stats(td_errors, loss_mask=pos_loss_mask)   │
│  compute_weight_stats(weights, loss_mask=pos_loss_mask) │
│  compute_value_function_stats(..., loss_mask=...)       │
│  compute_weight_statistics(weights, loss_mask=...)      │
└─────────────────────────────────────────────────────────┘
```

---

## 영향받는 파일 요약

| 파일 | 변경 유형 | 상태 |
|-----|---------|------|
| `td_weighting.py` | 5개 함수 파라미터명 변경 | ✅ 완료 |
| `td_accumulator.py` | 1개 메서드 파라미터명 변경 | ✅ 완료 |
| `rho1_weighting.py` | 2개 함수 파라미터명 변경 | ✅ 완료 |
| `run_rho1.py` | 호출부 수정 + loss_mask 생성 | ✅ 완료 |
| `run_verifiable.py` | 변수명 통일 + 키워드 인자 수정 + 중복 제거 | ✅ 완료 |
| `run_critic.py` | 변수명 통일 | ✅ 완료 |
| `logging_utils.py` | 2개 함수 파라미터 단순화 | ✅ 완료 |
| `test_td_weighting.py` | 테스트 코드 키워드 인자 수정 | ✅ 완료 |
| `test_td_accumulator.py` | 테스트 코드 변수명 수정 | ✅ 완료 |
| `test_rho1_weighting.py` | 테스트 코드 키워드 인자 수정 | ✅ 완료 |

---

## 검증 계획

### 1. 단위 테스트

```bash
# rho1 테스트 (이미 완료)
PYTHONPATH=src pytest tests/unit/test_rho1_weighting.py -v

# td 관련 테스트
PYTHONPATH=src pytest tests/unit/test_td_weighting.py tests/unit/test_td_accumulator.py -v
```

### 2. 통합 테스트

```bash
PYTHONPATH=src pytest tests/integration/test_pipeline_verifiable.py tests/integration/test_pipeline_critic.py tests/integration/test_pipeline_rho1.py -v
```

### 3. 정적 분석

```bash
# 키워드 인자 불일치 검사
grep -rn "attention_mask=" src/weighted_mtp/pipelines/ | grep -E "(compute_td|build_weights)"
# 결과가 없어야 함
```

---

## 완료 기준

- [x] 모든 `labels != -100` 마스크가 `*loss_mask` 또는 `loss_mask`로 명명 ✅
- [x] 마스크 생성 시 bool 타입 사용 (`.long()` 제거) ✅
- [x] 중복 마스크 생성 제거 (한 번만 생성) ✅
- [x] `attention_mask`는 모델 forward의 padding 마스크로만 사용 ✅
- [x] TD weighting 함수들이 `loss_mask` 파라미터를 사용 ✅
- [x] TD accumulator가 `loss_mask` 파라미터를 사용 ✅
- [x] Rho1 weighting 함수들이 `loss_mask` 파라미터를 사용 ✅
- [x] logging_utils 함수들이 `loss_mask` 파라미터를 사용 ✅
- [x] 함수 호출부의 키워드 인자가 함수 시그니처와 일치 ✅
- [x] value_weighting 모듈 단위 테스트 통과 ✅
- [ ] 파이프라인 통합 테스트 검증
