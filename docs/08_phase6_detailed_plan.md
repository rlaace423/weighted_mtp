# Phase 6: DDP 분산 학습 인프라 통합 가이드

## 문서 개요

본 문서는 **Phase 6: DDP 분산 학습 인프라 통합**을 위한 실행 가이드입니다. 기존 `runtime/distributed.py` + `runtime/environment.py` 인프라를 활용하여 3개 파이프라인 Runner(`run_critic.py`, `run_verifiable.py`, `run_rho1.py`)에 DDP(Distributed Data Parallel) 지원을 추가합니다.

**버전**: v1.0 (2025-01-17) - DDP 통합 계획
**선행 조건**: Phase 5 (Stage별 독립 실행 파이프라인) 완료
**목표**: VESSL A100 4-GPU 환경 분산 학습 + M3 Mac MPS Local Test 호환

---

## Part 1: 개요 및 맥락

### 1.1 Phase 6의 위치와 목적

Phase 6는 **분산 학습 인프라 통합**을 담당합니다.

```
Phase 5 (Pipelines)  →  [Phase 6 (DDP 통합)]  →  Production Training
 Stage별 독립 실행       4-GPU 분산 학습 지원      VESSL A100 4장 활용
```

**핵심 질문**: 어떻게 기존 파이프라인을 최소한으로 수정하여 DDP 분산 학습을 지원하고, 로컬 MPS 환경에서도 테스트 가능하게 할 것인가?

### 1.2 DDP 통합의 필요성

**현재 상황**:
- Phase 5에서 3개 파이프라인 완성 (run_critic, run_verifiable, run_rho1)
- `runtime/distributed.py` + `runtime/environment.py`에 우수한 DDP 인프라 이미 구현됨
- **문제**: 파이프라인에서 DDP model wrapping 미적용 → 4-GPU 환경에서 1개만 사용

**DDP 미적용 시 문제점**:

| 문제 | 영향 |
|------|------|
| **GPU 활용률** | 4-GPU 중 1개만 사용 (75% 낭비) |
| **학습 속도** | 4x 속도 향상 기회 상실 |
| **배치 크기** | 작은 배치 → 성능 저하 가능 |
| **비용 효율** | VESSL 4-GPU 비용 동일하나 효과 1/4 |

**DDP 적용 시 장점**:

| 장점 | 효과 |
|------|------|
| **4-GPU 활용** | Effective batch size 4x → ~4x 빠른 학습 |
| **Gradient aggregation** | 4개 GPU gradient 평균 → 안정적 학습 |
| **동일 코드** | MPS local test / 4-GPU VESSL 모두 동작 |
| **최소 수정** | 기존 runtime/ 활용 → 3-5줄 추가/파일 |

### 1.3 기존 Runtime 인프라 분석

**이미 구현된 것** (`src/weighted_mtp/runtime/distributed.py` 295줄):

```python
def init_distributed(backend="nccl") -> tuple[int, int]:
    """torch.distributed 초기화 (torchrun 환경변수 기반)"""

def is_main_process() -> bool:
    """Rank 0 체크 (logging용)"""

def create_distributed_sampler(dataset, shuffle=True, seed=42):
    """DistributedSampler 자동 생성 (distributed 환경 감지)"""

def setup_fsdp_config(...):
    """FSDP config (미사용, DDP 기반 설계)"""
```

**이미 구현된 것** (`src/weighted_mtp/runtime/environment.py` 271줄):

```python
def get_device(rank=None, force_cpu=False) -> torch.device:
    """Auto-select: cuda:{rank}, mps (M3 Mac), cpu"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def setup_seed(base_seed, rank=None) -> int:
    """Rank-aware seed: actual_seed = base_seed + rank"""

def setup_environment(base_seed=42) -> tuple[int, torch.device]:
    """Unified setup: seed + device + backends"""
```

**없는 것 (Phase 6에서 추가 필요)**:
- ❌ DDP model wrapping
- ❌ Model unwrapping (checkpoint 저장 시)
- ❌ Metric aggregation (all_reduce)

**인프라 평가**: ✅ **95% 완성**, DDP wrapper utilities만 추가하면 즉시 사용 가능

### 1.4 DDP vs FSDP 결정

**모델 크기 분석**:
- Meta-Llama-MTP: ~7B parameters
- FP16 precision: 7B × 2 bytes = 14GB (model weights)
- Optimizer states (AdamW): ~2x model = 28GB
- Gradients: 14GB
- **Total**: ~56GB per GPU

**A100 Memory**:
- A100 40GB: ❌ 부족 (56GB > 40GB)
- A100 80GB: ✅ 충분 (56GB < 80GB)

**결론: DDP 선택 (FSDP 불필요)**

| 항목 | DDP | FSDP |
|------|-----|------|
| **적합 모델 크기** | <10B params (7B 적합) | >10B params (오버킬) |
| **구현 복잡도** | 단순 (model wrapping) | 복잡 (sharding, gather/scatter) |
| **성능** | 빠름 (통신 overhead 적음) | 느림 (sharding overhead) |
| **메모리 효율** | 각 GPU에 전체 모델 | 모델 분산 저장 |
| **기존 인프라** | ✅ 기구현 (distributed.py) | ⚠️ 미완성 (setup_fsdp_config만) |

**VESSL 환경**: A100 80GB × 4장 → DDP로 충분

### 1.5 PyTorch DDP Best Practice (2024)

**표준 DDP 패턴**:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. 초기화 (torchrun이 환경변수 설정)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()

# 2. Device 설정
device = torch.device(f"cuda:{rank}")
model = MyModel().to(device)

# 3. DDP wrapping
model = DDP(model, device_ids=[rank])

# 4. DistributedSampler
train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)

# 5. Training loop
for epoch in range(n_epochs):
    train_sampler.set_epoch(epoch)  # Shuffle consistency

    for batch in train_loader:
        # Forward/backward (DDP auto-syncs gradients)
        loss = model(batch)
        loss.backward()
        optimizer.step()

# 6. Cleanup
dist.destroy_process_group()
```

**핵심 원칙**:
1. **Rank-aware device**: `cuda:{rank}` 할당
2. **DDP wrapping**: `DistributedDataParallel(model, device_ids=[rank])`
3. **DistributedSampler**: 각 GPU가 다른 데이터 샘플 처리
4. **Gradient auto-sync**: DDP가 backward 시 자동 all_reduce
5. **Rank 0 logging**: MLflow 등 I/O는 rank 0만 수행

**HuggingFace Accelerate 비교**:

| 항목 | Native DDP | HuggingFace Accelerate |
|------|-----------|------------------------|
| **의존성** | PyTorch 기본 | accelerate 패키지 추가 |
| **기존 인프라 재사용** | ✅ 100% 재사용 | ❌ 중복 발생 |
| **코드 변경** | 3-5줄/파일 | 2-3줄/파일 (유사) |
| **제어력** | Full control | 추상화로 제한적 |
| **개발원칙 준수** | ✅ 중복 없음 | ❌ runtime/ 인프라 무시 |

**결론**: **Native DDP Wrapper 방식** 채택 (개발원칙 2 준수)

---

## Part 2: 핵심 설계 결정

### 2.1 Decision 1: runtime/ddp.py 추가 (Minimal Utilities)

**문제 인식**: DDP model wrapping, unwrapping, metric aggregation 기능 필요

**해결책**: `runtime/ddp.py` 생성 (3개 utility 함수만)

**설계 원칙**:
- **최소주의**: 3개 함수만 추가 (wrap, unwrap, all_reduce)
- **기존 인프라 재사용**: `distributed.py`의 `is_main_process()`, `init_distributed()` 활용
- **자동 감지**: Distributed 환경 여부 자동 감지 (MPS local test 시 DDP skip)

**구조**:

```
src/weighted_mtp/runtime/
├── distributed.py      # 기존 (init_distributed, is_main_process, DistributedSampler)
├── environment.py      # 기존 (get_device, setup_seed, MPS/CUDA 자동 선택)
└── ddp.py              # 신규 (wrap_model_ddp, unwrap_model, all_reduce_scalar)
```

**API 설계**:

```python
# runtime/ddp.py

def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """DDP로 모델 래핑

    Distributed 환경에서만 DDP 적용, MPS/CPU local test는 skip

    Args:
        model: 원본 모델
        device: torch.device (cuda:rank, mps, cpu)
        find_unused_parameters: Unused gradient 허용 (기본 False)

    Returns:
        DDP-wrapped model (또는 원본 model if not distributed)
    """

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """DDP wrapper 제거하여 원본 모델 추출

    Checkpoint 저장 시 사용 (DDP wrapper 제외하고 state_dict 저장)

    Args:
        model: DDP-wrapped model (또는 원본 model)

    Returns:
        Original model
    """

def all_reduce_scalar(
    value: float,
    op: str = "mean",
) -> float:
    """GPU ranks 간 scalar 값 집계

    Loss, accuracy 등 metric 평균/합계 계산

    Args:
        value: 현재 rank의 scalar 값
        op: "mean" (평균) 또는 "sum" (합계)

    Returns:
        전체 ranks에서 집계된 값
    """
```

**자동 전환 메커니즘**:

```python
def wrap_model_ddp(model, device, find_unused_parameters=False):
    if not dist.is_initialized():
        # MPS/CPU local test - no wrapping
        return model

    # CUDA distributed - DDP wrapping
    device_ids = [device.index] if device.type == "cuda" else None
    return DDP(model, device_ids=device_ids, find_unused_parameters=find_unused_parameters)
```

**호환성 보장**:
- **VESSL A100 4-GPU**: `torchrun` 실행 → `dist.is_initialized() == True` → DDP 활성화
- **M3 Mac MPS**: 일반 `python` 실행 → `dist.is_initialized() == False` → DDP skip
- **코드 변경 없음**: 동일 코드로 양쪽 환경 지원

### 2.2 Decision 2: Pipelines 최소 수정 (3-5줄/파일)

**문제 인식**: DDP 통합 시 기존 파이프라인 대규모 수정 위험

**해결책**: Import + Wrapping + Unwrapping만 추가 (기존 로직 불변)

**수정 전** (run_critic.py):

```python
from weighted_mtp.runtime import setup_environment, is_main_process

# Setup
rank, device = setup_environment(config.runtime.seed)

# Model
adapter = load_adapter(config.models.policy, device)
optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-4)

# Training loop
for epoch in range(n_epochs):
    loss = train_stage1(adapter, train_loader, optimizer, config, device)

    if is_main_process():
        mlflow.log_metrics({"train/loss": loss}, step=epoch)

# Checkpoint
torch.save(adapter.state_dict(), checkpoint_path)
```

**수정 후** (run_critic.py):

```python
from weighted_mtp.runtime import (
    setup_environment,
    is_main_process,
    wrap_model_ddp,      # 추가
    unwrap_model,        # 추가
    all_reduce_scalar,   # 추가
)

# Setup (변경 없음)
rank, device = setup_environment(config.runtime.seed)

# Model
adapter = load_adapter(config.models.policy, device)
adapter = wrap_model_ddp(adapter, device)  # ⭐ DDP wrapping 추가
optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)

# Training loop
for epoch in range(n_epochs):
    loss = train_stage1(adapter, train_loader, optimizer, config, device)
    avg_loss = all_reduce_scalar(loss)  # ⭐ Metric aggregation 추가

    if is_main_process():
        mlflow.log_metrics({"train/loss": avg_loss}, step=epoch)  # 변경

# Checkpoint
torch.save(unwrap_model(adapter).state_dict(), checkpoint_path)  # ⭐ Unwrap 추가
```

**변경 요약**:
- ✅ Import 1줄 수정
- ✅ Wrapping 1줄 추가
- ✅ Metric aggregation 1줄 수정
- ✅ Unwrap 1줄 수정
- ✅ **총 4줄 변경**

**기존 로직 보존**:
- ❌ `train_stage1()` 함수 수정 불필요
- ❌ `load_adapter()` 수정 불필요
- ❌ Dataset/Dataloader 로직 수정 불필요 (`create_distributed_sampler()` 이미 구현됨)

### 2.3 Decision 3: MLflow Logging Rank 0 전용

**문제 인식**: 4개 GPU가 동시에 MLflow에 로깅 시 중복/충돌 발생

**해결책**: `is_main_process()` 체크로 Rank 0만 로깅

**기존 코드** (이미 구현됨):

```python
if is_main_process():
    mlflow.log_metrics({"train/loss": loss}, step=epoch)
```

**추가 작업**: 없음 (Phase 5에서 이미 rank 0 체크 완료)

**Metric Aggregation 필요**:

```python
# Before (각 GPU가 자기 loss만 계산)
loss = compute_loss(...)  # GPU 0: 0.5, GPU 1: 0.6, GPU 2: 0.55, GPU 3: 0.58

if is_main_process():
    mlflow.log_metrics({"train/loss": loss}, step=epoch)  # ❌ GPU 0 loss만 로깅 (0.5)

# After (4개 GPU loss 평균)
loss = compute_loss(...)
avg_loss = all_reduce_scalar(loss.item())  # (0.5+0.6+0.55+0.58)/4 = 0.5575

if is_main_process():
    mlflow.log_metrics({"train/loss": avg_loss}, step=epoch)  # ✅ 전체 평균 로깅
```

**적용 대상**:
- ✅ Training loss (step-level)
- ✅ Validation loss (epoch-level)
- ✅ TD error stats (mean, std)
- ✅ Weight stats (mean, std)
- ❌ GPU metrics (각 GPU별 개별 로깅 가능, 선택적)

### 2.4 Decision 4: torchrun 실행 표준화

**문제 인식**: DDP 실행 방식 표준화 필요 (사용자 혼란 방지)

**해결책**: `torchrun` 명령어 표준 제공

**Local Single-GPU / MPS Test** (기존 방식 유지):

```bash
# M3 Mac MPS
python -m weighted_mtp.pipelines.run_critic \
    --config configs/critic/critic_local.yaml

# Single CUDA GPU
python -m weighted_mtp.pipelines.run_critic \
    --config configs/critic/critic.yaml
```

**VESSL A100 4-GPU (DDP)**:

```bash
# torchrun으로 실행 (4-GPU)
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    -m weighted_mtp.pipelines.run_critic \
    --config configs/critic/critic.yaml
```

**torchrun 환경변수 자동 설정**:
- `RANK`: 현재 프로세스 rank (0, 1, 2, 3)
- `WORLD_SIZE`: 전체 프로세스 수 (4)
- `LOCAL_RANK`: 노드 내 rank (0, 1, 2, 3)
- `MASTER_ADDR`, `MASTER_PORT`: 통신 주소

**runtime/distributed.py가 자동 감지**:

```python
def init_distributed(backend="nccl"):
    if "RANK" in os.environ:
        # torchrun 환경 → DDP 초기화
        dist.init_process_group(backend=backend)
        return dist.get_rank(), dist.get_world_size()
    else:
        # 일반 실행 → Single-device
        return 0, 1
```

**편의 스크립트** (선택적):

```bash
# scripts/train_ddp.sh
#!/bin/bash
torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_critic "$@"

# 사용
./scripts/train_ddp.sh --config configs/critic/critic.yaml
```

---

## Part 3: runtime/ddp.py 설계

### 3.1 전체 구조

**파일 경로**: `src/weighted_mtp/runtime/ddp.py`

**역할**: DDP 최소 utilities (wrap, unwrap, all_reduce)

**전체 코드** (~100줄):

```python
"""DDP 유틸리티

Distributed Data Parallel model wrapping, unwrapping, metric aggregation
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """DDP로 모델 래핑

    Distributed 환경에서만 DDP 적용, MPS/CPU local test는 skip

    Args:
        model: 원본 모델
        device: torch.device (cuda:rank, mps, cpu)
        find_unused_parameters: Unused gradient 허용 (기본 False)

    Returns:
        DDP-wrapped model (또는 원본 model if not distributed)

    Example:
        >>> device = torch.device("cuda:0")
        >>> model = MyModel().to(device)
        >>> model = wrap_model_ddp(model, device)
    """
    if not dist.is_initialized():
        # MPS/CPU local test - no wrapping
        return model

    # CUDA distributed - DDP wrapping
    device_ids = [device.index] if device.type == "cuda" else None
    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """DDP wrapper 제거하여 원본 모델 추출

    Checkpoint 저장 시 사용 (DDP wrapper 제외하고 state_dict 저장)

    Args:
        model: DDP-wrapped model (또는 원본 model)

    Returns:
        Original model

    Example:
        >>> wrapped_model = wrap_model_ddp(model, device)
        >>> # Training...
        >>> original_model = unwrap_model(wrapped_model)
        >>> torch.save(original_model.state_dict(), "checkpoint.pt")
    """
    if isinstance(model, DDP):
        return model.module
    return model


def all_reduce_scalar(
    value: float,
    op: str = "mean",
) -> float:
    """GPU ranks 간 scalar 값 집계

    Loss, accuracy 등 metric 평균/합계 계산

    Args:
        value: 현재 rank의 scalar 값
        op: "mean" (평균) 또는 "sum" (합계)

    Returns:
        전체 ranks에서 집계된 값

    Example:
        >>> # GPU 0: loss = 0.5, GPU 1: loss = 0.6
        >>> avg_loss = all_reduce_scalar(loss.item())  # 0.55
        >>> if is_main_process():
        >>>     mlflow.log_metrics({"train/loss": avg_loss}, step=epoch)
    """
    if not dist.is_initialized():
        return value

    tensor = torch.tensor(value, device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if op == "mean":
        tensor /= dist.get_world_size()

    return tensor.item()
```

### 3.2 wrap_model_ddp() 상세

**기능**: 모델을 DDP로 래핑 (분산 환경에서만)

**핵심 로직**:

```python
def wrap_model_ddp(model, device, find_unused_parameters=False):
    # 1. Distributed 환경 체크
    if not dist.is_initialized():
        return model  # MPS/CPU → skip

    # 2. Device type에 따라 device_ids 설정
    device_ids = [device.index] if device.type == "cuda" else None
    # CUDA: device_ids=[0,1,2,3] (각 rank에서 자기 device)
    # MPS/CPU: device_ids=None (device_ids 불필요)

    # 3. DDP wrapping
    return DDP(model, device_ids=device_ids, find_unused_parameters=find_unused_parameters)
```

**find_unused_parameters 설명**:
- `False` (기본): 모든 파라미터가 gradient 계산에 사용됨 (가정)
- `True`: 일부 파라미터가 사용되지 않을 수 있음 (예: Conditional branches)
- **우리 케이스**: `False` 사용 (Adapter 전체 파라미터 항상 사용)

**MPS 호환성**:
- MPS는 `dist.is_initialized() == False` → DDP skip → 원본 모델 반환
- CUDA는 `dist.is_initialized() == True` → DDP wrapping

### 3.3 unwrap_model() 상세

**기능**: DDP wrapper 제거 (checkpoint 저장용)

**필요성**:

```python
# DDP-wrapped model state_dict (오염됨)
wrapped_model.state_dict()
# {
#     "module.value_head.weight": ...,  # ❌ "module." prefix
#     "module.mtp_heads.0.weight": ...,
# }

# Original model state_dict (깨끗함)
unwrap_model(wrapped_model).state_dict()
# {
#     "value_head.weight": ...,  # ✅ "module." prefix 없음
#     "mtp_heads.0.weight": ...,
# }
```

**로직**:

```python
def unwrap_model(model):
    if isinstance(model, DDP):
        return model.module  # DDP.module이 원본 모델
    return model  # 이미 unwrapped
```

**사용 패턴**:

```python
# Training
model = wrap_model_ddp(adapter, device)
optimizer = Adam(model.parameters())

# Checkpoint 저장
checkpoint = {
    "adapter_state_dict": unwrap_model(model).state_dict(),  # ✅ Clean state_dict
    "optimizer_state_dict": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")
```

### 3.4 all_reduce_scalar() 상세

**기능**: GPU 간 scalar metric 집계

**필요성**:

```python
# 4-GPU 환경에서 각 GPU의 loss
# GPU 0: loss = 0.50 (batch 0-7)
# GPU 1: loss = 0.60 (batch 8-15)
# GPU 2: loss = 0.55 (batch 16-23)
# GPU 3: loss = 0.58 (batch 24-31)

# Without all_reduce
if is_main_process():
    mlflow.log_metrics({"train/loss": 0.50}, step=epoch)  # ❌ GPU 0만

# With all_reduce
avg_loss = all_reduce_scalar(loss.item())  # (0.50+0.60+0.55+0.58)/4 = 0.5575
if is_main_process():
    mlflow.log_metrics({"train/loss": 0.5575}, step=epoch)  # ✅ 전체 평균
```

**로직**:

```python
def all_reduce_scalar(value, op="mean"):
    if not dist.is_initialized():
        return value  # Single-device → 그대로 반환

    # Tensor 변환 (current device에 위치)
    tensor = torch.tensor(value, device=torch.cuda.current_device())

    # All-reduce (SUM)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # 모든 GPU의 값을 합산하여 각 GPU에 동일 값 저장

    # Mean 계산 (필요 시)
    if op == "mean":
        tensor /= dist.get_world_size()

    return tensor.item()  # Python float로 반환
```

**지원 연산**:
- `op="mean"`: 평균 (기본, loss/accuracy용)
- `op="sum"`: 합계 (total samples count용)

**사용 예시**:

```python
# Training loop
for batch in train_loader:
    loss = compute_loss(...)
    loss.backward()
    optimizer.step()

    # Metric aggregation
    avg_loss = all_reduce_scalar(loss.item(), op="mean")
    total_samples = all_reduce_scalar(batch_size, op="sum")

    if is_main_process():
        mlflow.log_metrics({
            "train/loss": avg_loss,
            "train/samples_processed": total_samples,
        }, step=global_step)
```

---

## Part 4: Pipelines 수정 가이드

### 4.1 run_critic.py 수정

**목표**: DDP 지원 추가 (4줄 변경)

**수정 전** (핵심 부분만):

```python
from weighted_mtp.runtime import setup_environment, is_main_process

def run_critic_training(config_path, **override_params):
    # Setup
    rank, device = setup_environment(config.runtime.seed)

    # Model
    adapter = load_adapter(config.models.policy, device)
    optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=config.training.learning_rate)

    # Training loop
    for epoch in range(n_epochs):
        train_metrics = train_stage1(adapter, train_loader, optimizer, config, device)
        val_metrics = evaluate_stage(adapter, val_loader, config, device, stage="stage1")

        if is_main_process():
            mlflow.log_metrics({
                "train/loss": train_metrics["stage1_loss"],
                "val/loss": val_metrics["val_loss"],
            }, step=epoch)

        # Checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(adapter, optimizer, epoch, train_metrics, val_metrics, checkpoint_path)
```

**수정 후**:

```python
from weighted_mtp.runtime import (
    setup_environment,
    is_main_process,
    wrap_model_ddp,      # 추가
    unwrap_model,        # 추가
    all_reduce_scalar,   # 추가
)

def run_critic_training(config_path, **override_params):
    # Setup (변경 없음)
    rank, device = setup_environment(config.runtime.seed)

    # Model
    adapter = load_adapter(config.models.policy, device)
    adapter = wrap_model_ddp(adapter, device)  # ⭐ 추가 (1줄)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=config.training.learning_rate)

    # Training loop
    for epoch in range(n_epochs):
        train_metrics = train_stage1(adapter, train_loader, optimizer, config, device)
        val_metrics = evaluate_stage(adapter, val_loader, config, device, stage="stage1")

        # Metric aggregation (⭐ 추가 2줄)
        avg_train_loss = all_reduce_scalar(train_metrics["stage1_loss"])
        avg_val_loss = all_reduce_scalar(val_metrics["val_loss"])

        if is_main_process():
            mlflow.log_metrics({
                "train/loss": avg_train_loss,  # 수정
                "val/loss": avg_val_loss,      # 수정
            }, step=epoch)

        # Checkpoint (⭐ unwrap 추가 1줄)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(unwrap_model(adapter), optimizer, epoch, train_metrics, val_metrics, checkpoint_path)
```

**변경 요약**:
1. ✅ Import 1줄 수정 (wrap, unwrap, all_reduce 추가)
2. ✅ Wrapping 1줄 추가 (`wrap_model_ddp`)
3. ✅ Metric aggregation 2줄 추가 (`all_reduce_scalar`)
4. ✅ Unwrap 1줄 수정 (`save_checkpoint(unwrap_model(adapter), ...)`)

**총 5줄 변경/추가**

### 4.2 run_verifiable.py 수정

**동일 패턴** (run_critic.py와 유사):

```python
from weighted_mtp.runtime import (
    setup_environment,
    is_main_process,
    wrap_model_ddp,      # 추가
    unwrap_model,        # 추가
    all_reduce_scalar,   # 추가
)

def run_verifiable_training(config_path, critic_checkpoint=None, **override_params):
    rank, device = setup_environment(config.runtime.seed)

    adapter = load_adapter(config.models.policy, device)
    load_critic_checkpoint(config.experiment.critic_checkpoint, adapter, device)
    adapter = wrap_model_ddp(adapter, device)  # ⭐ DDP wrapping
    optimizer = torch.optim.Adam(adapter.parameters(), lr=config.training.learning_rate)

    for epoch in range(n_epochs):
        train_metrics = train_stage2(adapter, train_loader, optimizer, config, device)
        val_metrics = evaluate_stage(adapter, val_loader, config, device, stage="stage2")

        # Metric aggregation
        avg_train_total = all_reduce_scalar(train_metrics["stage2_total_loss"])
        avg_train_mtp = all_reduce_scalar(train_metrics["mtp_loss"])
        avg_train_value = all_reduce_scalar(train_metrics["value_loss"])
        avg_val_loss = all_reduce_scalar(val_metrics["val_loss"])

        if is_main_process():
            mlflow.log_metrics({
                "train/total_loss": avg_train_total,
                "train/mtp_loss": avg_train_mtp,
                "train/value_loss": avg_train_value,
                "val/loss": avg_val_loss,
            }, step=epoch)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(unwrap_model(adapter), optimizer, epoch, train_metrics, val_metrics, checkpoint_path)
```

**추가 고려**: TD error/weight stats aggregation

```python
# train_stage2() 내부에서 TD error stats 계산 시
td_stats = compute_td_stats(td_errors)  # {"td_mean": 0.1, "td_std": 0.3, ...}
weight_stats = compute_weight_stats(weights)  # {"weight_mean": 1.2, ...}

# Aggregation 필요
avg_td_mean = all_reduce_scalar(td_stats["td_mean"])
avg_td_std = all_reduce_scalar(td_stats["td_std"])
avg_weight_mean = all_reduce_scalar(weight_stats["weight_mean"])

if is_main_process():
    mlflow.log_metrics({
        "train/td_mean": avg_td_mean,
        "train/td_std": avg_td_std,
        "train/weight_mean": avg_weight_mean,
    }, step=global_step)
```

### 4.3 run_rho1.py 수정

**동일 패턴**:

```python
from weighted_mtp.runtime import (
    setup_environment,
    is_main_process,
    wrap_model_ddp,
    unwrap_model,
    all_reduce_scalar,
)

def run_rho1_training(config_path, **override_params):
    rank, device = setup_environment(config.runtime.seed)

    policy_adapter = load_adapter(config.models.policy, device)
    policy_adapter = wrap_model_ddp(policy_adapter, device)  # ⭐ DDP wrapping

    # Reference model은 wrapping 불필요 (inference only)
    ref_model = load_reference_model(config.models.reference, device)

    optimizer = torch.optim.Adam(policy_adapter.parameters(), lr=config.training.learning_rate)

    # Training loop (동일 패턴)
    for epoch in range(n_epochs):
        train_metrics = train_rho1(policy_adapter, ref_model, train_loader, optimizer, config, device)

        avg_loss = all_reduce_scalar(train_metrics["rho1_loss"])
        avg_excess_loss = all_reduce_scalar(train_metrics["excess_loss_mean"])

        if is_main_process():
            mlflow.log_metrics({
                "train/rho1_loss": avg_loss,
                "train/excess_loss_mean": avg_excess_loss,
            }, step=epoch)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        save_checkpoint(unwrap_model(policy_adapter), optimizer, epoch, train_metrics, val_metrics, checkpoint_path)
```

**Ref model 주의사항**:
- ❌ Ref model은 DDP wrapping 불필요 (학습 안 함, inference only)
- ✅ Policy adapter만 wrapping

### 4.4 checkpoint_utils.py 수정 (선택적)

**현재 save_checkpoint() 구조**:

```python
def save_checkpoint(adapter, optimizer, epoch, train_metrics, val_metrics, checkpoint_path):
    checkpoint = {
        "adapter_state_dict": adapter.state_dict(),
        "value_head_state_dict": adapter.value_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        ...
    }
    torch.save(checkpoint, checkpoint_path)
```

**수정 필요 여부**: ❌ 불필요

**이유**:
- Caller에서 이미 `unwrap_model(adapter)` 전달 → `adapter`는 unwrapped model
- 현재 코드 그대로 사용 가능

**대안** (더 안전한 방식):

```python
def save_checkpoint(adapter, optimizer, epoch, train_metrics, val_metrics, checkpoint_path):
    # Unwrap 내부에서 처리 (호출자 부담 감소)
    from weighted_mtp.runtime.ddp import unwrap_model
    adapter = unwrap_model(adapter)

    checkpoint = {
        "adapter_state_dict": adapter.state_dict(),
        "value_head_state_dict": adapter.value_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        ...
    }
    torch.save(checkpoint, checkpoint_path)
```

**권장**: Caller에서 unwrap (명시적, 현재 방식 유지)

---

## Part 5: Step별 구현 가이드

### Step 1: runtime/ddp.py 구현

**목표**: DDP utilities 파일 생성

**작업**:

```bash
# 1. 파일 생성
touch src/weighted_mtp/runtime/ddp.py

# 2. 코드 작성 (Part 3.1 참고)
# - wrap_model_ddp()
# - unwrap_model()
# - all_reduce_scalar()

# 3. __init__.py 업데이트
# src/weighted_mtp/runtime/__init__.py
```

**__init__.py 수정**:

```python
# src/weighted_mtp/runtime/__init__.py
from .distributed import (
    init_distributed,
    is_main_process,
    create_distributed_sampler,
    cleanup_distributed,
)
from .environment import (
    get_device,
    setup_seed,
    setup_environment,
)
from .ddp import (  # ⭐ 추가
    wrap_model_ddp,
    unwrap_model,
    all_reduce_scalar,
)

__all__ = [
    "init_distributed",
    "is_main_process",
    "create_distributed_sampler",
    "cleanup_distributed",
    "get_device",
    "setup_seed",
    "setup_environment",
    "wrap_model_ddp",      # ⭐ 추가
    "unwrap_model",        # ⭐ 추가
    "all_reduce_scalar",   # ⭐ 추가
]
```

**검증 기준**:
- [ ] `python -m py_compile src/weighted_mtp/runtime/ddp.py` 성공
- [ ] Import 테스트: `from weighted_mtp.runtime import wrap_model_ddp` 성공
- [ ] Unit test 작성 (tests/unit/test_ddp.py)

**예상 소요 시간**: 1-2시간

### Step 2: run_critic.py 수정

**목표**: Critic runner DDP 지원 추가

**작업**:

1. Import 수정
2. `wrap_model_ddp()` 추가
3. `all_reduce_scalar()` 추가
4. `unwrap_model()` 추가

**검증 기준**:
- [ ] Syntax check: `python -m py_compile src/weighted_mtp/pipelines/run_critic.py`
- [ ] MPS local test: `python -m weighted_mtp.pipelines.run_critic --config configs/critic/critic_local.yaml`
- [ ] DDP simulation test (선택적): `torchrun --nproc_per_node=2 -m weighted_mtp.pipelines.run_critic --config configs/critic/critic_local.yaml`

**예상 소요 시간**: 1-2시간

### Step 3: run_verifiable.py 수정

**목표**: Verifiable runner DDP 지원 추가

**작업**: Step 2와 동일 패턴

**추가 고려**:
- TD error/weight stats aggregation
- Critic checkpoint 로드 후 wrapping

**검증 기준**:
- [ ] Syntax check 성공
- [ ] MPS local test 성공 (critic checkpoint 로드 확인)
- [ ] DDP simulation test (선택적)

**예상 소요 시간**: 1-2시간

### Step 4: run_rho1.py 수정

**목표**: Rho-1 runner DDP 지원 추가

**작업**: Step 2와 동일 패턴

**추가 고려**:
- Policy adapter만 wrapping (ref model 제외)

**검증 기준**:
- [ ] Syntax check 성공
- [ ] MPS local test 성공
- [ ] DDP simulation test (선택적)

**예상 소요 시간**: 1-2시간

### Step 5: Unit Tests 작성

**목표**: DDP utilities 테스트

**파일 생성**: `tests/unit/test_ddp.py`

**테스트 케이스**:

```python
"""DDP utilities unit tests"""
import pytest
import torch
from weighted_mtp.runtime.ddp import wrap_model_ddp, unwrap_model, all_reduce_scalar


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)


def test_wrap_model_ddp_without_distributed():
    """Distributed 환경 없을 때 원본 모델 반환"""
    model = DummyModel()
    device = torch.device("cpu")

    wrapped = wrap_model_ddp(model, device)

    assert wrapped is model  # 원본 반환
    assert not isinstance(wrapped, torch.nn.parallel.DistributedDataParallel)


def test_unwrap_model_no_wrapping():
    """Unwrapped model 그대로 반환"""
    model = DummyModel()

    unwrapped = unwrap_model(model)

    assert unwrapped is model


def test_unwrap_model_with_ddp():
    """DDP-wrapped model unwrap"""
    # DDP mock (실제 distributed 환경 없이 테스트)
    model = DummyModel()
    # 실제 DDP는 distributed 환경 필요, mock으로 대체
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Skip if not CUDA available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    model = model.to(device)
    wrapped = DDP(model, device_ids=[0])

    unwrapped = unwrap_model(wrapped)

    assert unwrapped is model


def test_all_reduce_scalar_without_distributed():
    """Distributed 환경 없을 때 원본 값 반환"""
    value = 1.234

    result = all_reduce_scalar(value)

    assert result == value


def test_all_reduce_scalar_with_sum():
    """Sum operation"""
    value = 10.0

    result = all_reduce_scalar(value, op="sum")

    assert result == value  # Single-device → 그대로 반환
```

**검증 기준**:
- [ ] `uv run pytest tests/unit/test_ddp.py -v` 전체 통과
- [ ] Coverage ≥ 80%

**예상 소요 시간**: 1-2시간

### Step 6: Integration Test (선택적)

**목표**: End-to-end DDP 테스트

**테스트 시나리오**:

1. **2-GPU DDP simulation** (로컬 머신):
   ```bash
   torchrun --nproc_per_node=2 \
       -m weighted_mtp.pipelines.run_critic \
       --config configs/critic/critic_local.yaml
   ```

2. **Checkpoint 호환성 확인**:
   - DDP로 학습한 checkpoint
   - Single-device로 로드 가능 확인

3. **Metric aggregation 검증**:
   - 2-GPU에서 서로 다른 loss 계산
   - All-reduce 후 평균 확인

**검증 기준**:
- [ ] 2-GPU 실행 성공 (GPU 2개 있는 환경)
- [ ] Checkpoint 호환성 확인
- [ ] Metric aggregation 정확도 확인

**예상 소요 시간**: 2-3시간

### Step 7: 문서 업데이트

**목표**: README 및 사용 가이드 업데이트

**작업**:

1. **README.md 업데이트**:
   - DDP 실행 방법 추가
   - torchrun 예시 명령어

2. **docs/usage_guide.md 생성** (선택적):
   - Single-device vs DDP 실행 가이드
   - VESSL 환경 설정 가이드

**내용 예시**:

```markdown
## Distributed Training (DDP)

### Local Single-Device / MPS (M3 Mac)

```bash
# MPS (M3 Mac)
python -m weighted_mtp.pipelines.run_critic \
    --config configs/critic/critic_local.yaml

# Single CUDA GPU
python -m weighted_mtp.pipelines.run_critic \
    --config configs/critic/critic.yaml
```

### VESSL A100 4-GPU (DDP)

```bash
# torchrun으로 실행
torchrun \
    --nproc_per_node=4 \
    -m weighted_mtp.pipelines.run_critic \
    --config configs/critic/critic.yaml

# 편의 스크립트 (선택적)
./scripts/train_ddp.sh --config configs/critic/critic.yaml
```

### 주의사항

- DDP 환경에서는 `batch_size`가 per-GPU batch size입니다
- Effective batch size = `batch_size` × `nproc_per_node` × `gradient_accumulation_steps`
- 예: batch_size=8, 4-GPU, grad_accum=4 → Effective=128
```

**검증 기준**:
- [ ] README.md DDP 섹션 추가
- [ ] 명령어 예시 정확성 확인

**예상 소요 시간**: 0.5-1시간

---

## Part 6: 검증 및 완료 기준

### 6.1 기능 검증

**runtime/ddp.py**:
- [ ] `wrap_model_ddp()` 정상 동작 (distributed 환경 자동 감지)
- [ ] `unwrap_model()` 정상 동작 (DDP wrapper 제거)
- [ ] `all_reduce_scalar()` 정상 동작 (metric aggregation)
- [ ] MPS 환경에서 DDP skip 확인
- [ ] Unit test 전체 통과 (`tests/unit/test_ddp.py`)

**run_critic.py**:
- [ ] MPS local test 실행 성공 (DDP 미적용)
- [ ] Checkpoint 저장 시 unwrap 확인 (state_dict에 "module." prefix 없음)
- [ ] Metric aggregation 정상 동작 (all_reduce_scalar)

**run_verifiable.py**:
- [ ] MPS local test 실행 성공
- [ ] Critic checkpoint 로드 후 DDP wrapping 성공
- [ ] TD error/weight stats aggregation 정상 동작

**run_rho1.py**:
- [ ] MPS local test 실행 성공
- [ ] Policy adapter만 wrapping 확인 (ref model 제외)

**DDP Integration** (선택적, GPU 2개 이상 환경):
- [ ] 2-GPU torchrun 실행 성공
- [ ] Gradient synchronization 확인 (각 GPU에서 동일 weight 업데이트)
- [ ] Checkpoint 호환성 확인 (DDP → Single-device 로드 가능)

### 6.2 성능 검증

**학습 속도** (4-GPU VESSL):
- [ ] DDP 적용 시 ~3-4x 속도 향상 (이론치: 4x, 실제 통신 overhead 고려)
- [ ] GPU utilization ≥ 80% (모든 GPU 활용)

**메모리 효율**:
- [ ] 각 GPU 메모리 사용량 < 80GB (A100 80GB 기준)
- [ ] OOM 에러 없음

**Metric 정확성**:
- [ ] All-reduce 평균이 각 GPU loss 평균과 일치
- [ ] Validation loss가 single-device와 동일 (동일 seed)

### 6.3 코드 품질 검증

**Linting**:
```bash
uv run ruff check --fix src/weighted_mtp/runtime/ddp.py
uv run ruff check --fix src/weighted_mtp/pipelines/run_*.py
```

**Type checking** (선택적):
```bash
uv run mypy src/weighted_mtp/runtime/ddp.py
```

**Unit tests**:
```bash
uv run pytest tests/unit/test_ddp.py -v --cov=src/weighted_mtp/runtime/ddp
```

**Integration tests** (선택적):
```bash
# 2-GPU 환경
torchrun --nproc_per_node=2 -m weighted_mtp.pipelines.run_critic \
    --config configs/critic/critic_local.yaml
```

### 6.4 완료 기준

**필수 (Must-have)**:
- [ ] `runtime/ddp.py` 구현 완료 (3개 함수)
- [ ] `run_critic.py` DDP 지원 추가
- [ ] `run_verifiable.py` DDP 지원 추가
- [ ] `run_rho1.py` DDP 지원 추가
- [ ] Unit tests 통과 (`test_ddp.py`)
- [ ] MPS local test 성공 (모든 runner)
- [ ] Syntax check 전체 통과

**권장 (Should-have)**:
- [ ] 2-GPU integration test 성공
- [ ] Checkpoint 호환성 검증
- [ ] README 업데이트 (DDP 사용 가이드)

**선택적 (Nice-to-have)**:
- [ ] VESSL 4-GPU 실제 테스트
- [ ] 성능 벤치마크 (1-GPU vs 4-GPU 속도 비교)
- [ ] scripts/train_ddp.sh 편의 스크립트

---

## Part 7: 예상 소요 시간

| 작업 | 예상 시간 | 비고 |
|------|-----------|------|
| Step 1: runtime/ddp.py 구현 | 1-2시간 | 3개 함수 + __init__.py |
| Step 2: run_critic.py 수정 | 1-2시간 | 4줄 변경 + 검증 |
| Step 3: run_verifiable.py 수정 | 1-2시간 | 동일 패턴 |
| Step 4: run_rho1.py 수정 | 1-2시간 | 동일 패턴 |
| Step 5: Unit tests 작성 | 1-2시간 | test_ddp.py |
| Step 6: Integration test (선택적) | 2-3시간 | 2-GPU 테스트 |
| Step 7: 문서 업데이트 | 0.5-1시간 | README DDP 섹션 |
| 최종 검증 및 디버깅 | 1-2시간 | End-to-end 확인 |
| **합계 (Integration test 제외)** | **6.5-11시간** | 약 1-1.5일 |
| **합계 (Integration test 포함)** | **8.5-14시간** | 약 1-2일 |

---

## Part 8: 다음 단계 (Phase 6 완료 후)

**Phase 6 완료 기준 충족 시**:
- ✅ DDP 인프라 통합 완료
- ✅ MPS local test / 4-GPU VESSL 양쪽 호환
- ✅ Checkpoint 호환성 확인
- ✅ Metric aggregation 정상 동작

**다음 작업**:
1. **VESSL Production 실험**: 4-GPU DDP로 full training 실행
2. **성능 분석**: 1-GPU vs 4-GPU 학습 속도 비교
3. **Hyperparameter tuning**: DDP 환경에서 최적 batch size/learning rate 탐색
4. **Phase 7 계획** (선택적): Mixed precision (AMP), Gradient checkpointing 등 최적화

---

## Part 9: 개발원칙 준수 체크리스트

- [x] **원칙 1**: 앞/뒤 흐름 확인 완료 (runtime/ 인프라 → pipelines/ 통합)
- [x] **원칙 2**: 기존 구조 존중 (runtime/distributed.py 재사용), 중복 제거 (Accelerate 도입 안 함)
- [x] **원칙 3**: 잘못된 구조 없음 (기존 runtime/ 인프라 우수)
- [x] **원칙 4**: 하위 호환성 무시 (DDP 전격 도입, 구조 변경 최소화)
- [x] **원칙 4-1**: 인자명 통일 (model, device, value 일관성)
- [x] **원칙 4-2**: 단순 wrapper 금지 (3개 utility 함수만, 과도한 계층 없음)
- [x] **원칙 4-3**: 한글 주석, 이모지 없음, 핵심만
- [ ] **원칙 5**: 구현 후 계획과 비교 검토 (구현 완료 시)
- [x] **원칙 6**: 의존성 도구 활용 (PyTorch 기본 DDP, torchrun)

---

## 부록 A: torchrun 상세 가이드

### A.1 torchrun 기본 사용법

**단일 노드 4-GPU**:

```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    -m weighted_mtp.pipelines.run_critic \
    --config configs/critic/critic.yaml
```

**파라미터 설명**:
- `--nproc_per_node=4`: 노드당 프로세스 수 (=GPU 수)
- `--nnodes=1`: 전체 노드 수 (단일 머신)
- `--node_rank=0`: 현재 노드 rank (0-indexed)
- `--master_addr=localhost`: 통신 주소 (단일 머신은 localhost)
- `--master_port=29500`: 통신 포트 (임의 선택, 충돌 방지)

### A.2 환경변수 자동 설정

torchrun이 각 프로세스에 설정하는 환경변수:

```bash
# GPU 0 (Rank 0)
RANK=0
WORLD_SIZE=4
LOCAL_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

# GPU 1 (Rank 1)
RANK=1
WORLD_SIZE=4
LOCAL_RANK=1
MASTER_ADDR=localhost
MASTER_PORT=29500

# ... GPU 2, 3 동일 패턴
```

### A.3 편의 스크립트

**scripts/train_ddp.sh**:

```bash
#!/bin/bash
# DDP 학습 편의 스크립트

set -e

# Default values
NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-29500}

# Run torchrun
torchrun \
    --nproc_per_node="${NPROC}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port="${MASTER_PORT}" \
    -m weighted_mtp.pipelines.run_critic \
    "$@"
```

**사용**:

```bash
# 4-GPU (기본)
./scripts/train_ddp.sh --config configs/critic/critic.yaml

# 2-GPU (NPROC override)
NPROC=2 ./scripts/train_ddp.sh --config configs/critic/critic_local.yaml

# Custom port
MASTER_PORT=12345 ./scripts/train_ddp.sh --config configs/critic/critic.yaml
```

---

## 부록 B: Troubleshooting

### B.1 DDP 관련 에러

**에러**: `RuntimeError: Default process group has not been initialized`

**원인**: `dist.is_initialized() == False`인데 DDP 사용 시도

**해결**:
```python
# wrap_model_ddp()가 자동 처리 (is_initialized 체크)
# torchrun 없이 실행 시 DDP skip됨
```

---

**에러**: `RuntimeError: module(s) in parameter list do not match module(s) in DDP`

**원인**: DDP wrapping 후 optimizer에 원본 모델 파라미터 전달

**해결**:
```python
# Wrong
model = wrap_model_ddp(model, device)
optimizer = Adam(original_model.parameters())  # ❌

# Correct
model = wrap_model_ddp(model, device)
optimizer = Adam(model.parameters())  # ✅ DDP-wrapped model
```

---

**에러**: Checkpoint 로드 시 `"module." prefix` 에러

**원인**: DDP-wrapped model state_dict를 unwrap 없이 저장

**해결**:
```python
# Correct
torch.save(unwrap_model(model).state_dict(), "checkpoint.pt")
```

### B.2 성능 이슈

**문제**: 4-GPU인데 1-GPU와 속도 차이 없음

**원인 1**: `DistributedSampler` 미사용 (모든 GPU가 동일 데이터 처리)

**해결**:
```python
# create_distributed_sampler()가 자동 처리 (이미 구현됨)
sampler = create_distributed_sampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

**원인 2**: Batch size가 너무 작음 (통신 overhead > 계산 이득)

**해결**:
```bash
# Per-GPU batch size 증가
# Before: batch_size=2 → Effective=8 (2×4)
# After: batch_size=8 → Effective=32 (8×4)
```

---

**문서 종료**
