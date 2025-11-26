# Policy Loss 메모리 최적화 개발 계획

## 배경

Verifiable Pipeline에서 batch_size=128 유지 시 OOM 발생. Policy Loss 계산의 for loop가 주요 메모리 병목으로 식별됨.

### 현재 문제점

```python
# run_verifiable.py line 541-565 (동일 패턴: run_baseline.py, run_rho1.py)
for k in range(1, n_future + 1):
    logits_k = pos_logits[:, :valid_len, k - 1, :]  # [batch, seq-k, vocab]
    ce_loss_k = F.cross_entropy(...)  # 중간 텐서 생성
    weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * combined_mask_k.reshape(-1)
    # ... 4번 루프 동안 모든 중간 텐서가 backward graph에 유지됨
```

**메모리 소비 원인**:
1. `F.cross_entropy` 내부에서 softmax 계산 시 [batch×seq, vocab] 크기 텐서 생성
2. 4번 루프 동안 `ce_loss_k`, `weighted_ce_k` 등이 backward graph에 누적
3. 동일 패턴이 3개 파이프라인에서 중복 구현됨

---

## 개발 원칙 적용

| 원칙 | 적용 내용 |
|------|-----------|
| 원칙 1 | 앞/뒤 흐름 파악 완료 - run_verifiable.py, run_baseline.py, run_rho1.py 분석 |
| 원칙 2 | 3개 파이프라인의 중복 로직을 공통 유틸리티로 추출 |
| 원칙 3 | 기존 인라인 코드 삭제, 새 유틸리티 함수로 대체 |
| 원칙 4 | 하위 호환성 고려하지 않음, fallback 없이 전면 교체 |

---

## Phase 1: 공통 유틸리티 함수 생성 [완료]

### 목표
`src/weighted_mtp/utils/loss_utils.py` 생성하여 MTP CE Loss 계산 로직 통합

### 구현 내용

```python
# src/weighted_mtp/utils/loss_utils.py

def compute_mtp_ce_loss(
    logits: torch.Tensor,           # [batch, seq, n_future, vocab]
    labels: torch.Tensor,           # [batch, seq]
    attention_mask: torch.Tensor,   # [batch, seq]
    weights: torch.Tensor | None = None,  # [batch, seq] (optional, for weighted loss)
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    """MTP Cross-Entropy Loss 계산 (메모리 최적화)

    각 head별로 계산 후 즉시 중간 텐서 삭제하여 메모리 절감.

    Returns:
        {
            "weighted_ce_loss": Tensor (scalar),
            "unweighted_ce_loss": Tensor (scalar),
        }
    """
```

### 메모리 최적화 전략

1. **즉시 삭제**: 각 k 루프 완료 후 `del ce_loss_k, weighted_ce_k`
2. **스칼라 누적**: 중간 텐서 대신 `.item()` 또는 `.detach()` 사용하여 graph 분리
3. **in-place 연산**: 가능한 경우 `mul_()`, `add_()` 사용

### 완료 조건
- [x] `loss_utils.py` 생성
- [x] `compute_mtp_ce_loss()` 함수 구현
- [x] `compute_mtp_ce_loss_unweighted()` 함수 구현

---

## Phase 2: run_verifiable.py 적용 [완료]

### 목표
Training loop와 Validation 함수에 새 유틸리티 적용

### 변경 대상

| 파일 | 위치 | 변경 내용 |
|------|------|-----------|
| run_verifiable.py | line 541-568 (training) | for loop → `compute_mtp_ce_loss()` 호출 |
| run_verifiable.py | line 146-174 (validation) | for loop → `compute_mtp_ce_loss()` 호출 |

### 변경 전/후 비교

**변경 전** (line 537-568):
```python
batch_weighted_ce_loss = 0.0
batch_unweighted_ce_loss = 0.0

for k in range(1, n_future + 1):
    valid_len = seq_len - k
    logits_k = pos_logits[:, :valid_len, k - 1, :]
    labels_k = pos_labels[:, k : k + valid_len]
    # ... 30줄의 인라인 코드
```

**변경 후**:
```python
ce_losses = compute_mtp_ce_loss(
    logits=pos_logits,
    labels=pos_labels,
    attention_mask=pos_attention_mask,
    weights=weights,
)
weighted_ce_loss = ce_losses["weighted_ce_loss"]
unweighted_ce_loss = ce_losses["unweighted_ce_loss"]
```

### 완료 조건
- [x] Training loop 수정
- [x] Validation 함수 수정
- [x] 기존 인라인 코드 삭제
- [x] 불필요한 import 제거 (F, logging, Any)
- [x] 로컬 테스트 통과

---

## Phase 3: 다른 파이프라인 적용 [완료]

### 목표
run_baseline.py, run_rho1.py에도 동일한 최적화 적용

### 변경 대상

| 파일 | 위치 |
|------|------|
| run_baseline.py | line 109-130, line 411-432 |
| run_rho1.py | line 170-200, line 509-540 |

### 주의사항
- run_rho1.py는 per-head weights 구조가 다르므로 `compute_mtp_ce_loss` 미적용
- 대신 메모리 최적화(del 추가)만 적용

### 완료 조건
- [x] run_baseline.py: `compute_mtp_ce_loss_unweighted` 적용
- [x] run_rho1.py: 메모리 최적화(del) 적용
- [x] 불필요한 import/변수 정리
- [x] 전체 테스트 통과

---

## Phase 4: 검증 및 정리 [완료]

### 메모리 프로파일링
```bash
# 최적화 전
PYTHONPATH=src python -c "
import torch
from weighted_mtp.pipelines.run_verifiable import run_verifiable_training
# ... 메모리 측정
"

# 최적화 후: 동일 조건에서 메모리 사용량 비교
```

### 기대 효과
- 메모리 절감: ~20-30% (per-GPU)
- 코드 중복 제거: 3개 파이프라인에서 ~90줄 인라인 코드 → 1개 유틸리티 함수

### 완료 조건
- [x] 모든 파이프라인 import 테스트 통과
- [x] 불필요한 import/변수 정리 완료
- [ ] 실제 학습 실행으로 메모리 절감 확인 (사용자 테스트 필요)

---

## 일정 (예상)

| Phase | 작업 | 예상 소요 |
|-------|------|-----------|
| 1 | 유틸리티 함수 생성 | 1시간 |
| 2 | run_verifiable.py 적용 | 30분 |
| 3 | 다른 파이프라인 적용 | 1시간 |
| 4 | 검증 및 정리 | 30분 |

---

## 롤백 계획

문제 발생 시:
1. Git revert로 변경 전 상태 복구
2. 기존 인라인 코드는 삭제하므로, commit 단위로 롤백 가능하도록 phase별 커밋

---

## 승인 요청

위 계획에 따라 Phase 1부터 순차 진행해도 될까요?
