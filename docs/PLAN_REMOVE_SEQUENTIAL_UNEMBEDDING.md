# Sequential Unembedding 제거 계획서

## 배경 및 목적

### 문제 정의
현재 Sequential Unembedding 구현은 다음 문제를 가지고 있음:

1. **FSDP 호환성 위반**: `transformer.py:371`에서 forward 내부에서 backward() 호출
   - FSDP의 Forward-Backward 분리 원칙 위반
   - GPU 간 gradient 동기화(All-Reduce) 실패 가능성

2. **메모리 최적화 효과 없음**: `del logits` 후에도 CrossEntropyLoss backward를 위해 logits가 saved_tensors에 유지됨
   - 메모리 절감 효과: 0%

3. **불필요한 복잡도**: 효과 없는 코드 경로가 유지되어 디버깅/유지보수 어려움

### 목표
- Sequential Unembedding 관련 코드 완전 제거
- 표준 방식(forward → loss 계산 → backward)으로 전환
- FSDP + Activation Checkpointing 완전 호환

---

## 영향 범위 분석

### 수정 대상 파일

| 파일 | 수정 내용 | 우선순위 |
|------|----------|---------|
| `models/meta_mtp/transformer.py` | compute_sequential_loss 로직 제거 | P0 |
| `models/meta_mtp/adapter.py` | compute_sequential_loss 파라미터 제거 | P0 |
| `pipelines/run_baseline.py` | 표준 방식으로 전환 | P0 |
| `pipelines/run_verifiable.py` | 표준 방식으로 전환 (2D weights) | P0 |
| `pipelines/run_rho1.py` | 표준 방식으로 전환 (3D weights) | P0 |
| `utils/loss_utils.py` | compute_head_ce_loss 삭제, compute_mtp_ce_loss 3D weights 지원 추가 | P0 |
| `utils/__init__.py` | compute_head_ce_loss export 제거 | P1 |

### 함수별 의존 관계

```
compute_sequential_loss=True 호출 경로:
├── run_baseline.py:402 → adapter.forward() → transformer.forward()
├── run_verifiable.py:513 → adapter.forward() → transformer.forward()
└── run_rho1.py:550 → adapter.forward() → transformer.forward()
    └── transformer.py:339-376 (Sequential Unembedding 로직)
        └── compute_head_ce_loss() (삭제 대상)

대체 경로 (표준 방식):
├── adapter.forward() → transformer.forward() → logits 반환
└── loss_utils.compute_mtp_ce_loss(logits, labels, ...) → loss
    └── loss.backward() (외부에서 호출)
```

---

## Phase별 실행 계획

### Phase 1: loss_utils.py 확장 (선행 작업)

**목적**: 3D weights 지원 추가하여 Rho1 파이프라인 호환성 확보

**수정 내용**:
```python
# loss_utils.py - compute_mtp_ce_loss 수정

def compute_mtp_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    weights: torch.Tensor | None = None,  # 2D [batch, seq] 또는 3D [batch, seq, n_future]
    n_future: int | None = None,
) -> dict[str, torch.Tensor]:
    ...
    for k in range(1, n_future + 1):
        ...
        # weights shape 자동 감지 (compute_head_ce_loss에서 가져온 로직)
        if weights is not None:
            if weights.dim() == 3:
                # Per-head weights [batch, seq, n_future]: Rho1
                weights_k = weights[:, :valid_len, k - 1]
            else:
                # Position-level weights [batch, seq]: Verifiable
                weights_k = weights[:, k : k + valid_len]
        else:
            weights_k = None
        ...
```

**검증**:
- 기존 2D weights 동작 유지 (run_verifiable.py)
- 3D weights 정상 동작 (run_rho1.py)

---

### Phase 2: transformer.py 수정

**목적**: compute_sequential_loss 로직 제거

**수정 전** (transformer.py:281-376):
```python
def forward(
    self,
    tokens: torch.Tensor,
    start_pos: int = 0,
    return_all_heads: bool = False,
    return_hidden_states: bool = False,
    compute_sequential_loss: bool = False,  # 삭제
    labels: Optional[torch.Tensor] = None,  # 삭제
    attention_mask: Optional[torch.Tensor] = None,  # 삭제
    weights: Optional[torch.Tensor] = None,  # 삭제
    loss_scale: float = 1.0,  # 삭제
):
    ...
    # 삭제 대상: lines 339-376 (Sequential Unembedding 블록 전체)
    if compute_sequential_loss:
        ...
```

**수정 후**:
```python
def forward(
    self,
    tokens: torch.Tensor,
    start_pos: int = 0,
    return_all_heads: bool = False,
    return_hidden_states: bool = False,
):
    """Forward pass

    Args:
        tokens: [batch, seq] 입력 토큰
        start_pos: KV cache 시작 위치 (학습 시 0)
        return_all_heads: True면 모든 MTP heads 반환
        return_hidden_states: True면 hidden_states도 함께 반환 (Value Head용)

    Returns:
        return_hidden_states=True:
            (logits, hidden_states) tuple
        기본:
            logits: [batch, seq, n_future_tokens, vocab] or [batch, seq, vocab]
    """
    # 기존 로직 유지 (lines 378-394)
    ...
```

---

### Phase 3: adapter.py 수정

**목적**: compute_sequential_loss 파라미터 전달 로직 제거

**수정 전** (adapter.py:260-302):
```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_hidden_states: bool = False,
    compute_sequential_loss: bool = False,  # 삭제
    labels: Optional[torch.Tensor] = None,  # 삭제
    weights: Optional[torch.Tensor] = None,  # 삭제
    loss_scale: float = 1.0,  # 삭제
) -> torch.Tensor | dict[str, torch.Tensor]:
    ...
    if compute_sequential_loss:  # 삭제
        return self.transformer(...)
```

**수정 후**:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_hidden_states: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Forward pass

    Args:
        input_ids: [batch, seq] 입력 토큰
        attention_mask: [batch, seq] attention mask (현재 미사용, 향후 확장용)
        return_hidden_states: True면 hidden_states도 함께 반환 (Value Head용)

    Returns:
        return_hidden_states=True:
            {"logits": tensor, "hidden_states": tensor}
        기본:
            logits: [batch, seq, n_future_tokens, vocab]
    """
    ...
```

---

### Phase 4: 파이프라인 수정

#### 4-1: run_baseline.py

**수정 전** (lines 394-406):
```python
# Forward + Sequential Unembedding (메모리 효율적 loss 계산 + backward)
# backward()는 adapter 내부에서 수행됨
loss_result = adapter(
    input_ids,
    attention_mask=attention_mask,
    compute_sequential_loss=True,
    labels=labels,
    loss_scale=1.0 / gradient_accumulation_steps,
)
ce_loss = loss_result["loss"]  # detached scalar
```

**수정 후**:
```python
# Forward
logits = adapter(input_ids)  # [batch, seq, n_future, vocab]

# Loss 계산
ce_loss = compute_mtp_ce_loss_unweighted(
    logits=logits,
    labels=labels,
    attention_mask=attention_mask,
)

# Backward (외부에서 명시적 호출)
scaled_loss = ce_loss / gradient_accumulation_steps
scaled_loss.backward()
```

#### 4-2: run_verifiable.py

**수정 전** (lines 510-518):
```python
loss_result = policy_model(
    pos_input_ids,
    attention_mask=pos_attention_mask,
    compute_sequential_loss=True,
    labels=pos_labels,
    weights=weights,  # 2D weights
    loss_scale=1.0 / gradient_accumulation_steps,
)
weighted_ce_loss = loss_result["loss"]
```

**수정 후**:
```python
# Forward
logits = policy_model(pos_input_ids)  # [batch, seq, n_future, vocab]

# Weighted Loss 계산
loss_dict = compute_mtp_ce_loss(
    logits=logits,
    labels=pos_labels,
    attention_mask=pos_attention_mask,
    weights=weights,  # 2D weights [batch, seq]
)
weighted_ce_loss = loss_dict["weighted_ce_loss"]
unweighted_ce_loss = loss_dict["unweighted_ce_loss"]

# Backward
scaled_loss = weighted_ce_loss / gradient_accumulation_steps
scaled_loss.backward()
```

#### 4-3: run_rho1.py

**수정 전** (lines 547-555):
```python
loss_result = adapter(
    input_ids,
    attention_mask=attention_mask,
    compute_sequential_loss=True,
    labels=labels,
    weights=weights,  # 3D weights [batch, seq, n_future]
    loss_scale=1.0 / gradient_accumulation_steps,
)
weighted_ce_loss = loss_result["loss"]
```

**수정 후**:
```python
# Forward
logits = adapter(input_ids)  # [batch, seq, n_future, vocab]

# Weighted Loss 계산 (3D weights)
loss_dict = compute_mtp_ce_loss(
    logits=logits,
    labels=labels,
    attention_mask=attention_mask,
    weights=weights,  # 3D weights [batch, seq, n_future]
)
weighted_ce_loss = loss_dict["weighted_ce_loss"]
unweighted_ce_loss = loss_dict["unweighted_ce_loss"]

# Backward
scaled_loss = weighted_ce_loss / gradient_accumulation_steps
scaled_loss.backward()
```

---

### Phase 5: 정리 작업

**목적**: 불필요한 코드 완전 제거

1. **loss_utils.py**: `compute_head_ce_loss` 함수 삭제 (lines 11-85)
2. **utils/__init__.py**: `compute_head_ce_loss` export 제거 (lines 60, 108)

---

## 검증 계획

### 단위 테스트
```bash
# Phase 1 검증: loss_utils.py 3D weights 지원
PYTHONPATH=src pytest tests/test_loss_utils.py -v

# Phase 4 검증: 각 파이프라인 단일 step 실행
PYTHONPATH=src python -m weighted_mtp.pipelines.run_baseline --config configs/local/baseline_local.yaml
```

### 통합 테스트
```bash
# FSDP 분산 환경 테스트 (2 GPU)
torchrun --nproc_per_node=2 -m weighted_mtp.pipelines.run_baseline --config configs/production/baseline.yaml
```

### 검증 포인트
| 항목 | 기대값 |
|------|--------|
| Loss 감소 | 10-20 steps 내 loss 변화 시작 |
| Grad Norm | Clip ratio 0.9-1.0 (거의 clipping 없음) |
| FSDP 동기화 | 모든 rank에서 동일한 loss 값 |
| 메모리 | OOM 없이 정상 실행 |

---

## 롤백 계획

문제 발생 시 git revert로 즉시 롤백 가능:
```bash
git revert HEAD~N  # N = 커밋 수
```

---

## 일정

| Phase | 작업 | 예상 시간 |
|-------|------|----------|
| Phase 1 | loss_utils.py 확장 | 15분 |
| Phase 2 | transformer.py 수정 | 10분 |
| Phase 3 | adapter.py 수정 | 10분 |
| Phase 4 | 파이프라인 수정 (3개) | 30분 |
| Phase 5 | 정리 작업 | 5분 |
| 검증 | 테스트 실행 | 20분 |
| **Total** | | **~90분** |

---

## 승인 요청

위 계획에 대해 승인 부탁드립니다.

- [ ] Phase 1-5 계획 승인
- [ ] 추가 수정 요청 사항
