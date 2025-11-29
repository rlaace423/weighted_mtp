# EMA 기반 TD Error 정규화 계획

## 1. 문제 정의

### 1.1 현재 상태

```python
# td_weighting.py - build_weights 함수
def build_weights(td_errors, loss_mask, beta, min_weight, max_weight):
    valid_td = td_errors[loss_mask.bool()]

    # 현재 micro-batch만으로 통계 계산
    mean = valid_td.mean()   # ← 로컬 배치
    std = valid_td.std()     # ← 로컬 배치

    td_normalized = (td_errors - mean) / (std + 1e-8)
    weights = torch.exp(td_normalized / beta)
    ...
```

### 1.2 문제점

| 문제 | 설명 | 영향 |
|------|------|------|
| Per-micro-batch 정규화 | 각 batch마다 독립적인 mean/std 사용 | Weight scale 불일치 |
| GPU 간 불일치 | 각 GPU가 로컬 배치로만 통계 계산 | 분산 학습 시 비일관성 |
| Gradient Accumulation 불일치 | accumulation step 내 batch들이 다른 기준 사용 | 학습 불안정 |

```
예시 (gradient_accumulation_steps=4):

Batch 1: td_errors 평균=0.3, std=0.1 → weights 계산
Batch 2: td_errors 평균=0.5, std=0.2 → weights 계산 (다른 기준)
Batch 3: td_errors 평균=0.2, std=0.15 → weights 계산 (다른 기준)
Batch 4: td_errors 평균=0.4, std=0.1 → weights 계산 (다른 기준)
→ optimizer.step()

문제: 4개 배치가 각각 다른 채점 기준으로 weight를 계산함
```

---

## 2. 해결 방안: EMA 기반 전역 통계

### 2.1 핵심 아이디어

BatchNorm의 Running Statistics와 동일한 원리:
- 학습 중 각 batch의 통계를 EMA로 누적
- 모든 batch에서 누적된 EMA 통계를 사용하여 정규화
- 분산 학습 시 all-reduce로 GPU 간 동기화

### 2.2 목표 상태

```python
# EMA 통계 유지 (학습 전체에서 공유)
ema_mean = 0.0
ema_std = 1.0

for batch in all_batches:
    td_errors = compute_td_errors(...)

    # EMA 통계로 weights 계산 (모든 batch가 동일한 기준)
    weights = build_weights_with_ema(td_errors, ema_mean, ema_std, beta, ...)

    loss.backward()

    # EMA 업데이트 (다음 batch를 위해)
    ema_mean, ema_std = update_ema(td_errors, ema_mean, ema_std, momentum)
```

### 2.3 장점

| 항목 | 설명 |
|------|------|
| 메모리 효율 | O(1) - scalar 2개만 유지 |
| 연산 효율 | Forward 1회 유지 (Recompute 불필요) |
| 구현 단순성 | 기존 파이프라인 구조 유지 |
| 안정성 | 학습 진행에 따라 통계가 안정화됨 |

---

## 3. 구현 계획

### Phase 1: TDStatsEMA 클래스 구현 ✅ 완료

**파일**: `src/weighted_mtp/value_weighting/td_stats_ema.py`

EMA 전용 경량 클래스 구현 (기존 TDStatsAccumulator 삭제 및 대체):

```python
class TDStatsEMA:
    """TD Error EMA 통계 추적기

    학습 전체에서 mean/std를 EMA로 추적하여 일관된 정규화 제공.
    BatchNorm의 running_mean/running_var와 동일한 원리.
    """

    def __init__(
        self,
        device: torch.device,
        momentum: float = 0.1,
        warmup_steps: int = 10,
    ):
        """
        Args:
            device: torch.device
            momentum: EMA 업데이트 계수 (0.1 = 90% 이전 + 10% 현재)
            warmup_steps: 초기 warmup 동안은 현재 batch 통계 사용
        """
        self.device = device
        self.momentum = momentum
        self.warmup_steps = warmup_steps

        self.ema_mean = torch.tensor(0.0, device=device)
        self.ema_std = torch.tensor(1.0, device=device)
        self.step_count = 0

    def get_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """현재 EMA 통계 반환"""
        return self.ema_mean, self.ema_std

    def update(
        self,
        td_errors: torch.Tensor,
        loss_mask: torch.Tensor,
        distributed: bool = True,
    ):
        """EMA 통계 업데이트

        Args:
            td_errors: [batch, seq] TD error 텐서
            loss_mask: [batch, seq] 학습 대상 토큰 마스크
            distributed: True면 all-reduce로 GPU 간 동기화
        """
        # 현재 batch 통계 계산
        valid_td = td_errors[loss_mask.bool()].detach()
        batch_mean = valid_td.mean()
        batch_std = valid_td.std()

        # 분산 학습 시 GPU 간 동기화
        if distributed and is_distributed():
            reduced = all_reduce_scalars({
                "mean": batch_mean.item(),
                "std": batch_std.item(),
            }, op="mean")
            batch_mean = torch.tensor(reduced["mean"], device=self.device)
            batch_std = torch.tensor(reduced["std"], device=self.device)

        self.step_count += 1

        # Warmup 기간: 현재 batch 통계로 초기화
        if self.step_count <= self.warmup_steps:
            # 점진적 warmup: step이 증가할수록 EMA 비중 증가
            warmup_momentum = self.step_count / self.warmup_steps
            self.ema_mean = warmup_momentum * self.ema_mean + (1 - warmup_momentum) * batch_mean
            self.ema_std = warmup_momentum * self.ema_std + (1 - warmup_momentum) * batch_std
        else:
            # 일반 EMA 업데이트
            self.ema_mean = (1 - self.momentum) * self.ema_mean + self.momentum * batch_mean
            self.ema_std = (1 - self.momentum) * self.ema_std + self.momentum * batch_std

    def state_dict(self) -> dict:
        """Checkpoint 저장용"""
        return {
            "ema_mean": self.ema_mean.item(),
            "ema_std": self.ema_std.item(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict):
        """Checkpoint 로드용"""
        self.ema_mean = torch.tensor(state_dict["ema_mean"], device=self.device)
        self.ema_std = torch.tensor(state_dict["ema_std"], device=self.device)
        self.step_count = state_dict["step_count"]
```

**작업 항목**:
- [x] TDStatsEMA 클래스 구현
- [x] 단위 테스트 작성 (`tests/unit/test_td_stats_ema.py`)
- [x] `__init__.py` export 추가

---

### Phase 2: build_weights 함수 확장 ✅ 완료

**파일**: `src/weighted_mtp/value_weighting/td_weighting.py`

기존 `build_weights` 함수에 외부 통계 사용 옵션 추가:

```python
def build_weights(
    td_errors: torch.Tensor,
    loss_mask: torch.Tensor,
    beta: float = 1.0,
    min_weight: float = 0.1,
    max_weight: float = 5.0,
    external_mean: torch.Tensor = None,
    external_std: torch.Tensor = None,
) -> torch.Tensor:
    """TD error 기반 토큰 가중치 계산

    Args:
        td_errors: [batch, seq] TD error
        loss_mask: [batch, seq] 학습 대상 토큰 마스크
        beta: Temperature 파라미터
        min_weight: 최소 가중치
        max_weight: 최대 가중치
        external_mean: 외부 제공 평균 (None이면 현재 batch 통계 사용)
        external_std: 외부 제공 표준편차 (None이면 현재 batch 통계 사용)

    Returns:
        weights: [batch, seq] Token-level weights
    """
    bool_mask = loss_mask.bool()

    if external_mean is not None and external_std is not None:
        # 외부 통계 사용 (EMA 등)
        mean = external_mean
        std = external_std
    else:
        # 기존 방식: 현재 batch 통계
        valid_td = td_errors[bool_mask]
        if valid_td.numel() <= 1:
            weights = torch.ones_like(td_errors) * loss_mask.float()
            return weights
        mean = valid_td.mean()
        std = valid_td.std()

    # Advantage Whitening
    td_normalized = (td_errors - mean) / (std + 1e-8)

    # Exponential transformation + Clipping
    weights = torch.exp(td_normalized / beta)
    weights = torch.clamp(weights, min=min_weight, max=max_weight)
    weights = weights * loss_mask.float()

    return weights
```

**작업 항목**:
- [x] `build_weights` 함수에 `external_mean`, `external_std` 파라미터 추가
- [x] 하위 호환성 유지 (파라미터 None이면 기존 동작)
- [x] 단위 테스트 업데이트 (`TestBuildWeightsExternalStats` 클래스 추가)

---

### Phase 3: run_verifiable.py 파이프라인 적용

**파일**: `src/weighted_mtp/pipelines/run_verifiable.py`

```python
from weighted_mtp.value_weighting.td_weighting import TDStatsEMA

def run_verifiable_training(config: DictConfig):
    ...

    # EMA 통계 추적기 초기화
    td_ema = TDStatsEMA(
        device=device,
        momentum=config.training.get("td_ema_momentum", 0.1),
        warmup_steps=config.training.get("td_ema_warmup_steps", 10),
    )

    for batch in batches:
        ...

        # TD error 계산
        td_errors = compute_td_errors(
            value_logits=pos_value_logits,
            rewards=pos_rewards,
            loss_mask=pos_loss_mask,
            gamma=1.0,
        )

        # EMA 통계로 weights 계산
        ema_mean, ema_std = td_ema.get_stats()
        weights = build_weights(
            td_errors=td_errors,
            loss_mask=pos_loss_mask,
            beta=config.training.beta,
            min_weight=config.training.weight_clip_min,
            max_weight=config.training.weight_clip_max,
            external_mean=ema_mean,
            external_std=ema_std,
        )

        # Loss 계산 + Backward
        ce_losses = compute_mtp_ce_loss(...)
        total_loss = weighted_ce_loss + value_loss
        scaled_loss = total_loss / gradient_accumulation_steps
        scaled_loss.backward()

        # EMA 업데이트 (backward 후)
        td_ema.update(td_errors, pos_loss_mask, distributed=True)

        ...
```

**작업 항목**:
- [ ] TDStatsEMA 인스턴스 생성
- [ ] build_weights 호출부에 EMA 통계 전달
- [ ] EMA 업데이트 로직 추가
- [ ] Config에 td_ema_momentum, td_ema_warmup_steps 파라미터 추가

---

### Phase 4: 로깅 및 모니터링

**추가할 메트릭**:

```python
# Step-level 로깅에 EMA 통계 추가
if is_main_process() and use_mlflow:
    mlflow.log_metrics({
        ...
        "td/ema_mean": td_ema.ema_mean.item(),
        "td/ema_std": td_ema.ema_std.item(),
        "td/batch_mean": batch_td_mean,  # 현재 batch 통계 (비교용)
        "td/batch_std": batch_td_std,
    }, step=global_step)
```

**작업 항목**:
- [ ] EMA 통계 로깅 추가
- [ ] Batch vs EMA 통계 비교 시각화

---

### Phase 5: Checkpoint 저장/로드

```python
# Checkpoint 저장 시
save_checkpoint(
    ...
    td_ema_state=td_ema.state_dict(),
)

# Checkpoint 로드 시
if "td_ema_state" in checkpoint:
    td_ema.load_state_dict(checkpoint["td_ema_state"])
```

**작업 항목**:
- [ ] save_checkpoint에 td_ema_state 추가
- [ ] load_checkpoint에 td_ema_state 복원 로직 추가

---

## 4. Config 파라미터

```yaml
training:
  # 기존 파라미터
  beta: 0.9
  weight_clip_min: 0.1
  weight_clip_max: 5.0

  # EMA 정규화 파라미터 (신규)
  td_ema_momentum: 0.1      # EMA 업데이트 계수 (0.1 = 90% 이전 + 10% 현재)
  td_ema_warmup_steps: 10   # 초기 warmup step 수
```

---

## 5. 구현 순서

```
Phase 1: TDStatsEMA 클래스 구현
        ├── td_accumulator.py에 클래스 추가
        ├── 단위 테스트 작성
        └── __init__.py export

Phase 2: build_weights 함수 확장
        ├── external_mean/std 파라미터 추가
        ├── 하위 호환성 검증
        └── 단위 테스트 업데이트

Phase 3: run_verifiable.py 적용
        ├── TDStatsEMA 인스턴스 생성
        ├── EMA 통계 전달
        └── EMA 업데이트 로직

Phase 4: 로깅 및 모니터링
        └── MLflow 메트릭 추가

Phase 5: Checkpoint 통합
        ├── 저장 로직
        └── 로드 로직
```

---

## 6. 검증 계획

### 단위 테스트

```bash
PYTHONPATH=src pytest tests/unit/test_td_weighting.py -v
PYTHONPATH=src pytest tests/unit/test_td_stats_ema.py -v
```

### 통합 테스트

```bash
PYTHONPATH=src pytest tests/integration/test_pipeline_verifiable.py -v
```

### 검증 항목

| 테스트 | 설명 |
|--------|------|
| EMA 수렴 | warmup 후 EMA가 안정화되는지 확인 |
| 분산 동기화 | Multi-GPU에서 EMA가 동일한지 확인 |
| 하위 호환성 | external_mean=None일 때 기존 동작 유지 |
| Checkpoint | 저장/로드 후 EMA 상태 복원 확인 |

---

## 7. 성공 기준

| 지표 | 기준 |
|------|------|
| 메모리 증가 | < 1KB (scalar 2개) |
| 연산 오버헤드 | < 1% (EMA 업데이트) |
| 정규화 일관성 | 모든 batch가 동일한 EMA 통계 사용 |
| 기존 호환성 | external_mean=None 시 기존 동작 유지 |
