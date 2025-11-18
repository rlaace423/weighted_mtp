# DDP → FSDP 완전 전환 계획서

## 문서 정보
- **작성일**: 2025-11-18
- **대상**: A100 80GB GPU 4-way 분산 학습
- **전환 방향**: DDP 완전 삭제, FSDP 통일
- **전략**: NO_SHARD (Critic) + FULL_SHARD (Verifiable/Baseline/Rho-1)

---

## Executive Summary

### 전환 배경

**현재 상황:**
- 분산 방식: DDP (DistributedDataParallel)
- 메모리 사용: 80-90GB per GPU (6.7B 모델 full finetuning)
- 문제: A100 80GB 한계 초과, batch_size=4 불가능

**전환 결정:**
- DDP 완전 삭제, FSDP로 통일
- Sharding strategy로 파이프라인별 최적화
- 개발 원칙 4 준수: "불필요한 중복 제거, 깔끔한 코드"

### FSDP 전환 효과

| Pipeline | 현재 (DDP) | 전환 후 (FSDP) | Strategy |
|----------|-----------|---------------|----------|
| **Critic** | 30GB (가능) | 30GB (동일) | NO_SHARD |
| **Verifiable** | 90GB (OOM) | 30GB (가능) | FULL_SHARD |
| **Baseline** | 90GB (OOM) | 30GB (가능) | FULL_SHARD |
| **Rho-1** | 90GB (OOM) | 35GB (가능) | FULL_SHARD |

**핵심 효과:**
- 메모리 절감: 67% (90GB → 30GB)
- Batch size 증가: 4 → 8 (FULL_SHARD)
- 속도: NO_SHARD는 DDP 동일, FULL_SHARD는 6-10% 느림
- 코드 단순화: 단일 wrapper, 단일 테스트

### 전환 범위

**삭제:**
- `src/weighted_mtp/runtime/ddp.py`
- `tests/unit/test_ddp.py`

**신규:**
- `src/weighted_mtp/runtime/fsdp.py`
- `tests/unit/test_fsdp.py`

**수정:**
- 4개 파이프라인 (run_*.py)
- 4개 config 파일 (*.yaml)
- `runtime/__init__.py`
- `utils/checkpoint_utils.py`

---

## I. 현재 DDP 아키텍처 분석

### 1. 분산학습 구조 (ARCHITECTURE.md 기준)

#### Torchrun 기반 프로세스 생성

```bash
# VESSL A100 4-GPU 환경
PYTHONPATH=src torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  src/weighted_mtp/pipelines/run_verifiable.py \
  --config configs/verifiable/verifiable.yaml
```

**환경 변수 자동 설정:**
- `RANK`: 0-3
- `WORLD_SIZE`: 4
- `MASTER_ADDR`, `MASTER_PORT`

#### DDP Wrapping 패턴

**runtime/ddp.py 구현:**
```python
def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if not is_distributed():
        return model  # Single-device 환경

    wrapped_model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )
    return wrapped_model

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model
```

#### 파이프라인 사용 패턴 (공통)

**4개 파이프라인 모두 동일:**
```python
# 1. Import
from weighted_mtp.runtime import wrap_model_ddp, unwrap_model, all_reduce_scalar

# 2. 모델 로딩
adapter = load_adapter(config, device)

# 3. DDP wrapping
adapter = wrap_model_ddp(adapter, device)

# 4. Optimizer 생성
optimizer = AdamW(adapter.parameters(), ...)

# 5. Training loop
for batch in dataloader:
    loss.backward()
    optimizer.step()

    # Metric aggregation
    avg_loss = all_reduce_scalar(loss.item())

# 6. Checkpoint 저장
if is_main_process():
    save_checkpoint(
        adapter=unwrap_model(adapter),  # DDP unwrap
        optimizer=optimizer,
        ...
    )
```

**파일별 사용 현황:**
- `run_critic.py`: Line 202 (wrap), Line 206 (unwrap), Line 551 (all_reduce)
- `run_verifiable.py`: Line 326 (wrap), Line 694 (unwrap), Line 588 (all_reduce)
- `run_baseline.py`: Line 191 (wrap), Line 503 (unwrap), Line 421 (all_reduce)
- `run_rho1.py`: Line 258 (wrap), Line 554 (unwrap), Line 469 (all_reduce)

### 2. Checkpoint 저장/로드 (checkpoint_utils.py)

#### save_checkpoint() 구현

```python
def save_checkpoint(
    adapter,  # Type hint 없음 (유연성)
    optimizer: torch.optim.Optimizer,
    epoch: int | float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    checkpoint_path: Path | str,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "adapter_state_dict": adapter.state_dict(),  # DDP는 자동 unwrap
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    if hasattr(adapter, "value_head") and adapter.value_head is not None:
        checkpoint["value_head_state_dict"] = adapter.value_head.state_dict()

    torch.save(checkpoint, checkpoint_path)
```

**DDP 특성:**
- `DDP_model.state_dict()` 호출 시 자동으로 unwrap됨
- 별도 처리 불필요

**FSDP 문제:**
- `FSDP_model.state_dict()` 호출 시 shard만 반환
- Full state dict gathering 명시적 필요

### 3. Rank-aware Data Sampling

**변경 불필요 (FSDP와 무관):**
```python
# datasets.py: 메타데이터 기반 샘플링
all_indices = _compute_sampling_indices_from_metadata(...)

if world_size > 1:
    rank_indices = all_indices[rank::world_size]
```

**특징:**
- DistributedSampler 사용 안 함
- 메타데이터 기반 커리큘럼 학습
- 75% 메모리 절약 (각 rank가 1/4만 로드)

---

## II. FSDP 전환 설계

### 1. Sharding Strategy 결정

#### NO_SHARD (Critic)

**적용 대상:** Critic 파이프라인

**이유:**
- Value Head만 학습 (4096 params)
- 메모리: ~30GB (충분)
- DDP 성능 재현 가능

**동작:**
- 모델 복제 (DDP와 동일)
- All-reduce만 사용
- 통신 오버헤드 최소

**Config:**
```yaml
# configs/critic/critic.yaml
distributed:
  fsdp:
    sharding_strategy: NO_SHARD
    mixed_precision: true
    cpu_offload: false
```

#### FULL_SHARD (Verifiable/Baseline/Rho-1)

**적용 대상:** Full finetuning 파이프라인

**이유:**
- 6.7B 모델 전체 학습
- 메모리: 90GB → 30GB (75% 절감)
- FSDP 필수

**동작:**
- 모델/Grad/Optimizer 샤딩
- All-gather (forward) + Reduce-scatter (backward)
- 통신 증가, 메모리 대폭 감소

**Config:**
```yaml
# configs/verifiable/verifiable.yaml
distributed:
  fsdp:
    sharding_strategy: FULL_SHARD
    mixed_precision: true
    cpu_offload: false
```

### 2. FSDP 래퍼 설계

#### wrap_model_fsdp() 인터페이스

**DDP와 동일한 인터페이스 유지:**
```python
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def wrap_model_fsdp(
    model: torch.nn.Module,
    device: torch.device,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision: bool = True,
    cpu_offload: bool = False,
) -> torch.nn.Module:
    """FSDP로 모델 래핑

    DDP wrap_model_ddp()와 동일한 인터페이스.
    분산 환경에서만 FSDP 적용, MPS/CPU local test는 skip.

    Args:
        model: 원본 모델
        device: torch.device (cuda:rank, mps, cpu)
        sharding_strategy: "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
        mixed_precision: FP16 mixed precision 사용 여부
        cpu_offload: CPU 오프로드 (메모리 부족 시)

    Returns:
        FSDP-wrapped model (또는 원본 model if not distributed)
    """
    if not is_distributed():
        return model  # MPS/CPU local test

    # Sharding strategy 변환
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Mixed precision 설정
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # CPU offload 설정
    cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None

    # Auto Wrap Policy 설정 (TransformerBlock 단위 wrapping)
    from weighted_mtp.models.meta_mtp.transformer import TransformerBlock

    fsdp_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # FSDP wrapping
    wrapped_model = FSDP(
        model,
        auto_wrap_policy=fsdp_auto_wrap_policy,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload_config,
        device_id=device.index if device.type == "cuda" else None,
        sync_module_states=True,
        use_orig_params=True,  # PyTorch 2.0+ 권장 (향후 LoRA 등 확장성)
    )

    return wrapped_model
```

#### unwrap_model() 구현

**FSDP wrapper 제거:**
```python
def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """FSDP wrapper 제거

    FSDP wrapped 모델이면 원본 모델 반환.
    일반 모델이면 그대로 반환.
    Checkpoint 저장 시 사용.

    Args:
        model: FSDP-wrapped model 또는 원본 model

    Returns:
        Original model
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        return model.module
    return model
```

#### all_reduce_scalar() 유지

**DDP와 동일 (변경 불필요):**
```python
def all_reduce_scalar(
    value: float,
    op: str = "mean",
) -> float:
    """GPU ranks 간 scalar 값 집계

    FSDP와 DDP 모두 동일한 dist.all_reduce() 사용
    """
    if not is_distributed():
        return value

    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor(value, device=device)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if op == "mean":
        world_size = dist.get_world_size()
        tensor /= world_size

    return tensor.item()
```

### 3. Checkpoint 저장/로드 수정

#### save_checkpoint() FSDP 지원

**utils/checkpoint_utils.py 수정:**
```python
def save_checkpoint(
    adapter,
    optimizer: torch.optim.Optimizer,
    epoch: int | float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    checkpoint_path: Path | str,
) -> None:
    """Checkpoint 저장 (FSDP 지원)"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        StateDictType,
        FullStateDictConfig,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # FSDP state dict gathering
    if isinstance(adapter, FSDP):
        with FSDP.state_dict_type(
            adapter,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            adapter_state_dict = adapter.state_dict()
    else:
        # 일반 모델 (single-device 환경)
        adapter_state_dict = adapter.state_dict()

    checkpoint = {
        "epoch": epoch,
        "adapter_state_dict": adapter_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    # Value head 저장 (Critic/Verifiable)
    if hasattr(adapter, "value_head") and adapter.value_head is not None:
        if isinstance(adapter, FSDP):
            # FSDP는 전체 모델 state_dict에 포함됨
            pass
        else:
            checkpoint["value_head_state_dict"] = adapter.value_head.state_dict()

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint 저장 완료: {checkpoint_path}")
```

**주의:**
- FSDP는 `state_dict_type` context manager 필요
- `rank0_only=True`로 Rank 0만 저장
- Single-device 환경은 일반 state_dict() 호출

---

## III. Phase별 개발 계획

### Phase 1: FSDP 래퍼 구현

**목표:** DDP와 동일한 인터페이스의 FSDP wrapper 구현

**파일 생성:** `src/weighted_mtp/runtime/fsdp.py`

**구현 내용:**
1. `wrap_model_fsdp()` - FSDP 모델 래핑 (NO_SHARD/FULL_SHARD 지원)
2. `unwrap_model()` - FSDP wrapper 제거
3. `all_reduce_scalar()` - Metric 집계 (FSDP/DDP 동일)

**테스트:** `tests/unit/test_fsdp.py`

```python
# test_fsdp.py 주요 테스트
def test_wrap_model_fsdp_single_device():
    """단일 장치 환경에서는 FSDP wrapping 하지 않음"""

def test_wrap_model_fsdp_no_shard():
    """NO_SHARD strategy 테스트"""

def test_wrap_model_fsdp_full_shard():
    """FULL_SHARD strategy 테스트"""

def test_unwrap_model_fsdp():
    """FSDP wrapper 제거 확인"""

def test_unwrap_model_plain():
    """일반 모델은 그대로 반환"""

def test_all_reduce_scalar():
    """Metric aggregation 테스트"""

def test_model_forward_after_wrap():
    """FSDP wrapping 후 forward pass"""

def test_state_dict_consistency():
    """Wrap/unwrap 후 state_dict 일관성"""
```

**검증 기준:**
- ✅ `pytest tests/unit/test_fsdp.py` 모든 테스트 통과
- ✅ `wrap_model_fsdp()` 인터페이스가 `wrap_model_ddp()`와 일치
- ✅ Single-device 환경에서 wrapping 하지 않음

**예상 시간:** 2-3시간

---

### Phase 2: Checkpoint 유틸리티 수정

**목표:** FSDP Full state dict gathering 지원

**파일 수정:** `src/weighted_mtp/utils/checkpoint_utils.py`

**수정 내용:**
1. `save_checkpoint()` - FSDP isinstance 체크 추가
2. Full state dict gathering context manager
3. DDP 하위 호환 유지

**테스트:**
```python
# tests/unit/test_checkpoint_utils.py (신규 또는 기존 확장)
def test_save_checkpoint_fsdp():
    """FSDP 모델 checkpoint 저장"""

def test_save_checkpoint_normal_model():
    """일반 모델 checkpoint 저장 (하위 호환)"""

def test_load_checkpoint_fsdp():
    """FSDP checkpoint 로드"""
```

**검증 기준:**
- ✅ FSDP 모델 checkpoint 정상 저장
- ✅ Rank 0만 저장 확인
- ✅ Single-device 모델 정상 저장

**예상 시간:** 1시간

---

### Phase 3: Config 파일 업데이트

**목표:** 4개 파이프라인 config에 FSDP 설정 추가

**파일 수정:**
1. `configs/defaults.yaml` (공통 설정)
2. `configs/critic/critic.yaml`
3. `configs/verifiable/verifiable.yaml`
4. `configs/baseline/baseline.yaml`
5. `configs/rho1/rho1.yaml`

**수정 내용:**

```yaml
# configs/defaults.yaml (신규 섹션 추가)
distributed:
  fsdp:
    mixed_precision: true
    cpu_offload: false

# configs/critic/critic.yaml (오버라이드)
distributed:
  fsdp:
    sharding_strategy: NO_SHARD

# configs/verifiable/verifiable.yaml (오버라이드)
distributed:
  fsdp:
    sharding_strategy: FULL_SHARD

# configs/baseline/baseline.yaml (오버라이드)
distributed:
  fsdp:
    sharding_strategy: FULL_SHARD

# configs/rho1/rho1.yaml (오버라이드)
distributed:
  fsdp:
    sharding_strategy: FULL_SHARD
```

**검증 기준:**
- ✅ Config 로딩 정상 (OmegaConf merge)
- ✅ `config.distributed.fsdp.sharding_strategy` 접근 가능

**예상 시간:** 30분

---

### Phase 4: 파이프라인 전환 (4개 동시)

**목표:** DDP → FSDP 일괄 전환

**파일 수정:**
1. `src/weighted_mtp/pipelines/run_critic.py`
2. `src/weighted_mtp/pipelines/run_verifiable.py`
3. `src/weighted_mtp/pipelines/run_baseline.py`
4. `src/weighted_mtp/pipelines/run_rho1.py`

**수정 패턴 (4개 파이프라인 동일):**

```python
# 기존
from weighted_mtp.runtime import (
    wrap_model_ddp,
    unwrap_model,
    all_reduce_scalar,
)

adapter = wrap_model_ddp(adapter, device)

# 변경 후
from weighted_mtp.runtime import (
    wrap_model_fsdp,
    unwrap_model,
    all_reduce_scalar,
)

sharding_strategy = config.distributed.fsdp.sharding_strategy
adapter = wrap_model_fsdp(
    adapter,
    device,
    sharding_strategy=sharding_strategy,
    mixed_precision=config.distributed.fsdp.mixed_precision,
    cpu_offload=config.distributed.fsdp.cpu_offload,
)
```

**변경 지점:**

**run_critic.py:**
- Line 42: `wrap_model_ddp` → `wrap_model_fsdp`
- Line 202: `wrap_model_ddp()` 호출 → `wrap_model_fsdp()` 호출

**run_verifiable.py:**
- Line 45: `wrap_model_ddp` → `wrap_model_fsdp`
- Line 326: `wrap_model_ddp()` 호출 → `wrap_model_fsdp()` 호출

**run_baseline.py:**
- Line 44: `wrap_model_ddp` → `wrap_model_fsdp`
- Line 191: `wrap_model_ddp()` 호출 → `wrap_model_fsdp()` 호출

**run_rho1.py:**
- Line 43: `wrap_model_ddp` → `wrap_model_fsdp`
- Line 258: `wrap_model_ddp()` 호출 → `wrap_model_fsdp()` 호출

**unwrap_model, all_reduce_scalar는 변경 불필요:**
- `unwrap_model()`: FSDP wrapper 제거 (동일 인터페이스)
- `all_reduce_scalar()`: FSDP와 동일

**검증 기준:**
- ✅ Syntax 오류 없음
- ✅ Import 오류 없음
- ✅ Config 접근 정상

**예상 시간:** 1-2시간

---

### Phase 5: DDP 완전 삭제

**목표:** DDP 관련 파일 및 import 제거

**삭제 파일:**
1. `src/weighted_mtp/runtime/ddp.py`
2. `tests/unit/test_ddp.py`

**수정 파일:**
1. `src/weighted_mtp/runtime/__init__.py`

**수정 내용:**

```python
# runtime/__init__.py

# 기존
from weighted_mtp.runtime.ddp import (
    wrap_model_ddp,
    unwrap_model,
    all_reduce_scalar,
)

# 변경 후 (완전 삭제)
from weighted_mtp.runtime.fsdp import (
    wrap_model_fsdp,
    unwrap_model,
    all_reduce_scalar,
)

__all__ = [
    # distributed.py
    "init_distributed",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_main_process",
    "is_distributed",
    "create_distributed_sampler",
    "barrier",
    "cleanup_distributed",
    # fsdp.py (DDP 제거)
    "wrap_model_fsdp",
    "unwrap_model",
    "all_reduce_scalar",
    # environment.py
    "setup_seed",
    "get_device",
    "setup_torch_backends",
    "setup_environment",
    "get_gpu_memory_info",
]
```

**검증 기준:**
- ✅ `src/weighted_mtp/runtime/ddp.py` 삭제 완료
- ✅ `tests/unit/test_ddp.py` 삭제 완료
- ✅ Import 오류 없음
- ✅ `from weighted_mtp.runtime import wrap_model_fsdp` 정상

**예상 시간:** 10분

---

### Phase 6: 통합 테스트 및 검증

**목표:** 전체 시스템 정상 동작 확인

**테스트 항목:**

1. **Unit Tests**
   ```bash
   PYTHONPATH=src pytest tests/unit/test_fsdp.py -v
   PYTHONPATH=src pytest tests/unit/test_checkpoint_utils.py -v
   ```

2. **Integration Tests**
   ```bash
   PYTHONPATH=src pytest tests/integration/test_pipeline_critic.py -v
   PYTHONPATH=src pytest tests/integration/test_pipeline_verifiable.py -v
   PYTHONPATH=src pytest tests/integration/test_pipeline_baseline.py -v
   PYTHONPATH=src pytest tests/integration/test_pipeline_rho1.py -v
   ```

3. **메모리 프로파일링**
   ```bash
   # Critic (NO_SHARD)
   nvidia-smi --query-gpu=memory.used --format=csv -l 1
   # 예상: ~30GB per GPU

   # Verifiable (FULL_SHARD)
   nvidia-smi --query-gpu=memory.used --format=csv -l 1
   # 예상: ~30GB per GPU (75% 절감)
   ```

4. **성능 벤치마크**
   ```bash
   # Critic: NO_SHARD vs DDP(baseline)
   # 예상: tokens/sec 차이 < 1%

   # Verifiable: FULL_SHARD
   # 예상: tokens/sec 6-10% 느림 (메모리 절감 trade-off)
   ```

**검증 기준:**
- ✅ 모든 unit/integration tests 통과
- ✅ Critic 메모리: ~30GB (NO_SHARD)
- ✅ Verifiable 메모리: ~30GB (FULL_SHARD, 75% 절감)
- ✅ Critic 속도: DDP 대비 < 1% 차이
- ✅ Verifiable 속도: DDP 대비 6-10% 느림 허용
- ✅ OOM 없이 3 epoch 완주

**예상 시간:** 2-3시간

---

## IV. 테스트 전략

### 1. Unit Tests

**test_fsdp.py (신규):**
```python
# Single-device 환경 테스트
def test_wrap_model_fsdp_single_device():
    """단일 장치에서는 FSDP wrapping 하지 않음"""

# NO_SHARD strategy
def test_wrap_model_fsdp_no_shard():
    """NO_SHARD는 메모리 복제 (단일 GPU처럼)"""

# FULL_SHARD strategy
def test_wrap_model_fsdp_full_shard():
    """FULL_SHARD는 메모리 샤딩"""

# Unwrap 테스트
def test_unwrap_model_fsdp():
    """FSDP wrapper 제거"""

def test_unwrap_model_plain():
    """일반 모델은 그대로 반환"""

# Metric aggregation
def test_all_reduce_scalar():
    """GPU ranks 간 metric 집계"""

# Forward pass
def test_model_forward_after_wrap():
    """FSDP wrapping 후 forward 정상 동작"""

# State dict
def test_state_dict_consistency():
    """Wrap/unwrap 후 state_dict 일관성"""
```

**test_checkpoint_utils.py (확장):**
```python
def test_save_checkpoint_fsdp():
    """FSDP 모델 checkpoint 저장"""

def test_save_checkpoint_single_device():
    """Single-device 모델 checkpoint 저장"""

def test_load_checkpoint_fsdp():
    """FSDP checkpoint 로드"""
```

### 2. Integration Tests

**기존 tests 활용 (수정 불필요):**
- `tests/integration/test_pipeline_critic.py`
- `tests/integration/test_pipeline_verifiable.py`
- `tests/integration/test_pipeline_baseline.py`
- `tests/integration/test_pipeline_rho1.py`

**자동 FSDP 전환:**
- 파이프라인 코드가 FSDP로 전환되면 integration tests도 자동 적용
- Config 기반으로 sharding strategy 결정

### 3. 메모리 검증

**VESSL 환경에서 실행:**
```bash
# GPU 메모리 모니터링
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > memory_log.csv

# Critic (NO_SHARD)
PYTHONPATH=src torchrun --nproc_per_node=4 \
  src/weighted_mtp/pipelines/run_critic.py \
  --config configs/critic/critic.yaml

# Verifiable (FULL_SHARD)
PYTHONPATH=src torchrun --nproc_per_node=4 \
  src/weighted_mtp/pipelines/run_verifiable.py \
  --config configs/verifiable/verifiable.yaml
```

**검증 목표:**

| Pipeline | Strategy | 목표 메모리 | Batch Size |
|----------|----------|-----------|-----------|
| Critic | NO_SHARD | ~30GB | 8 |
| Verifiable | FULL_SHARD | ~30GB | 4-8 |
| Baseline | FULL_SHARD | ~30GB | 4-8 |
| Rho-1 | FULL_SHARD | ~35GB | 4-8 |

### 4. 성능 벤치마크

**측정 지표:**
- Tokens/sec (throughput)
- Epoch 시간
- Loss 수렴 (정합성)

**비교 기준:**
- Critic NO_SHARD vs DDP baseline: < 1% 차이
- Verifiable FULL_SHARD: 6-10% 느림 허용

---

## V. Rollback 전략

### Phase별 Rollback

**Phase 1-2 실패:**
- FSDP 파일 삭제
- DDP 유지
- 영향 없음

**Phase 3 실패:**
- Config 복원
- 영향 없음

**Phase 4 실패:**
- 파이프라인 개별 rollback 가능
- Import만 수정하면 됨

**Phase 5 실패:**
- DDP 파일 복원 (git checkout)
- __init__.py 복원

**Phase 6 실패:**
- 전체 rollback
- Git branch 전환

### Git Branch 전략

```bash
# 각 Phase마다 commit
git checkout -b fsdp/complete-migration
git commit -m "Phase 1: FSDP wrapper 구현"
git commit -m "Phase 2: Checkpoint utils 수정"
git commit -m "Phase 3: Config 업데이트"
git commit -m "Phase 4: 파이프라인 전환"
git commit -m "Phase 5: DDP 삭제"
git commit -m "Phase 6: 통합 테스트 완료"
```

**Rollback 시:**
```bash
git reset --hard <commit_hash>
```

---

## VI. 리스크 및 대응

### 리스크 1: FSDP NO_SHARD와 DDP 성능 차이

**확률:** 낮음

**영향:** 중간 (Critic 속도 저하)

**대응:**
- 벤치마크 비교 (tokens/sec)
- 1% 이상 차이 나면 DDP 복원 고려
- 하지만 업계 사례상 차이 없음

### 리스크 2: FSDP Checkpoint 호환성

**확률:** 중간

**영향:** 높음 (Checkpoint 로드 실패)

**대응:**
- Phase 2에서 철저히 테스트
- Full state dict gathering 검증
- Rank 0 저장 확인

### 리스크 3: Config 누락

**확률:** 낮음

**영향:** 중간 (Runtime error)

**대응:**
- Phase 3에서 config validation
- Default 값 설정
- Error 메시지 명확화

### 리스크 4: 메모리 목표 미달

**확률:** 낮음

**영향:** 높음 (OOM 발생)

**대응:**
- Phase 6 메모리 프로파일링
- Batch size 조정
- CPU offload 고려

### 리스크 5: Integration Test 실패

**확률:** 중간

**영향:** 높음 (파이프라인 오작동)

**대응:**
- Phase 4에서 개별 파이프라인 수동 테스트
- Error 로그 분석
- Rollback 준비

---

## VII. 성공 기준

### Phase별 성공 기준

| Phase | 성공 기준 |
|-------|----------|
| Phase 1 | test_fsdp.py 모든 테스트 통과 |
| Phase 2 | Checkpoint 저장/로드 정상 |
| Phase 3 | Config 로딩 오류 없음 |
| Phase 4 | 4개 파이프라인 import/syntax 오류 없음 |
| Phase 5 | DDP 파일 삭제 완료, import 오류 없음 |
| Phase 6 | 메모리/성능 목표 달성 |

### 최종 성공 기준

1. ✅ **메모리**: FULL_SHARD 파이프라인 ~30GB (75% 절감)
2. ✅ **성능**: NO_SHARD는 DDP 동일, FULL_SHARD는 10% 이내 저하
3. ✅ **테스트**: 모든 unit/integration tests 통과
4. ✅ **안정성**: OOM 없이 3 epoch 완주
5. ✅ **코드 품질**: DDP 관련 파일 완전 삭제

---

## VIII. 예상 일정

| Phase | 작업 내용 | 예상 시간 | 누적 시간 |
|-------|----------|----------|----------|
| Phase 1 | FSDP 래퍼 구현 | 2-3시간 | 2-3시간 |
| Phase 2 | Checkpoint utils 수정 | 1시간 | 3-4시간 |
| Phase 3 | Config 업데이트 | 30분 | 4-5시간 |
| Phase 4 | 파이프라인 전환 | 1-2시간 | 5-7시간 |
| Phase 5 | DDP 삭제 | 10분 | 5-7시간 |
| Phase 6 | 통합 테스트 | 2-3시간 | 7-10시간 |
| **Total** | | **7-10시간** | |

**검증 및 디버깅 시간:** 각 Phase마다 +30분

---

## IX. 참고 자료

### PyTorch FSDP 공식 문서
- https://pytorch.org/docs/stable/fsdp.html
- https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html

### Sharding Strategies
- `NO_SHARD`: DDP와 동일 (모델 복제)
- `SHARD_GRAD_OP`: ZeRO-2 (Optimizer + Gradient 샤딩)
- `FULL_SHARD`: ZeRO-3 (Model + Optimizer + Gradient 샤딩)

### Meta LLaMA Training
- FSDP 기반 대규모 모델 학습
- Mixed precision (FP16/BF16)
- Activation checkpointing

### Hugging Face Accelerate
- Config 기반 DDP/FSDP 선택
- 단일 코드로 다양한 전략 지원

---

## X. 결론

### DDP 완전 삭제 결정 이유

1. **개발 원칙 준수**: "불필요한 중복 제거, 깔끔한 코드"
2. **PyTorch 권장**: FSDP의 `NO_SHARD` strategy로 DDP 재현 가능
3. **코드 단순화**: 단일 wrapper, 단일 테스트, 유지보수 용이
4. **미래 확장성**: 더 큰 모델(13B, 70B) 확장 시 FSDP 전용 유리

### FSDP 전환의 핵심 가치

1. **메모리 효율**: FULL_SHARD로 75% 절감 (90GB → 30GB)
2. **유연성**: NO_SHARD/FULL_SHARD로 파이프라인별 최적화
3. **성능**: NO_SHARD는 DDP 동일, FULL_SHARD는 6-10% trade-off
4. **확장성**: 13B, 70B 모델로 확장 가능

### 즉시 시작 가능한 이유

1. ✅ Pure PyTorch 구현 (FSDP 호환)
2. ✅ Safetensors 지원
3. ✅ 명확한 Phase별 계획
4. ✅ Rollback 전략 완비
5. ✅ 테스트 인프라 준비됨

**다음 단계**: Phase 1 시작 - `runtime/fsdp.py` 구현
