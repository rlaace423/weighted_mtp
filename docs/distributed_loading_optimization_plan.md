# A100 4-GPU 분산 데이터 로딩 최적화 계획

**작성일**: 2025-11-17
**버전**: 1.0.0
**목적**: Rank-aware 메타데이터 샘플링 도입 및 중복 코드 제거

---

## 목차

1. [현황 분석](#1-현황-분석)
2. [개선 방향](#2-개선-방향)
3. [Phase별 구현 계획](#3-phase별-구현-계획)
4. [테스트 전략](#4-테스트-전략)
5. [승인 필요 사항](#5-승인-필요-사항)
6. [대안 및 Trade-off](#6-대안-및-trade-off)

---

## 1. 현황 분석

### 1.1 코드 검증 결과

**✅ 정상 작동 확인**:
- Config 파라미터(n_samples, difficulty_weights, balance_correct, correct_ratio)가 모두 올바르게 작동
- 메타데이터 기반 샘플링으로 전체 데이터(3.7M, ~15GB) 로드 회피
- DistributedSampler로 각 GPU가 중복 없이 서브셋 처리
- Curriculum learning epoch별 재샘플링 정상 동작

**⚠️ 문제점 발견**:

#### 1.1.1 비효율적 데이터 로딩

현재 구조에서 **각 GPU가 동일한 n_samples를 전부 로드**한 후 1/world_size만 사용:

```python
# 모든 rank가 동일하게 실행
dataset = load_dataset(
    dataset_name="codecontests",
    split="train",
    n_samples=100000,  # 각 GPU가 100,000개 전부 로드
    ...
)

sampler = DistributedSampler(
    dataset,  # 100,000개
    num_replicas=4,
    rank=rank,  # 각 GPU가 25,000개씩만 사용
)
```

**실제 데이터 흐름** (Verifiable, 4 GPU, 100,000 샘플):
1. Rank 0: 메타데이터(217MB) → 100,000 인덱스 계산 → JSONL 100,000 라인 읽기 → **25,000개만 사용**
2. Rank 1: 메타데이터(217MB) → 100,000 인덱스 계산 → JSONL 100,000 라인 읽기 → **25,000개만 사용**
3. Rank 2: 동일 (중복)
4. Rank 3: 동일 (중복)

**비효율 정량화**:
- **메모리**: 각 GPU당 ~500MB 중복 (75,000개 불필요 샘플)
- **I/O**: JSONL 읽기 4배 중복
- **CPU**: 메타데이터 샘플링 계산 4배 중복

**Curriculum Learning 추가 비용**:
- Epoch 0.0, 0.3, 0.7에서 DataLoader 재생성 시 위 과정 반복
- 총 4회 × 4 GPU = **16회 중복 로딩**

**영향도**:
- A100 80GB VRAM 대비 0.6% (메모리 병목 아님)
- 전체 학습 시간 대비 1-2% (I/O 병목 가능)
- 대규모 데이터셋 확장 시 문제 확대

#### 1.1.2 중복 코드 (개발원칙 위반)

**create_dataloader() 중복**:
- `run_baseline.py:71` (72 lines)
- `run_critic.py:69` (80 lines)
- `run_verifiable.py:78` (81 lines)
- `run_rho1.py:103` (77 lines)

**공통 로직** (99% 동일):
```python
# 4개 파일 모두 동일한 패턴
def create_dataloader(...):
    # 1. dataset_path 파싱
    dataset_name = dataset_path_obj.parent.parent.name
    split = "train" if "train" in split_file else ...

    # 2. load_dataset() 호출
    dataset = load_dataset(...)

    # 3. AlpacaDataCollator 생성
    collator = AlpacaDataCollator(...)

    # 4. DistributedSampler 생성
    sampler = create_distributed_sampler(...)

    # 5. DataLoader 생성
    dataloader = DataLoader(...)

    return dataloader, sampler
```

**차이점**: Baseline/Rho-1은 difficulty 파라미터 없음 (None 하드코딩)

**개발원칙 위반**:
- [원칙 2] 유사 기능 중복 메서드 반복
- [원칙 4-2] 단순 wrapper 형태의 비공개 메서드 반복

---

## 2. 개선 방향

### 2.1 핵심 개선사항

#### 2.1.1 Rank-aware 메타데이터 샘플링

각 GPU가 자기 담당 서브셋만 로드:

```python
def load_dataset(
    dataset_name: str,
    split: str,
    n_samples: int,
    balance_correct: bool = False,
    correct_ratio: float = 0.5,
    difficulty_weights: Optional[dict] = None,
    difficulty_bins: Optional[dict] = None,
    seed: int = 42,
    rank: int = 0,  # 신규 파라미터
    world_size: int = 1,  # 신규 파라미터
) -> Dataset:
    """각 rank가 자기 담당 샘플만 로드"""

    # 1. 메타데이터 로드 (모든 rank 동일)
    metadata = _load_metadata(dataset_name, split)

    # 2. 전체 샘플링 인덱스 계산 (시드 동일 → 재현성 보장)
    all_indices = _compute_sampling_indices_from_metadata(
        metadata, n_samples, balance_correct, correct_ratio,
        difficulty_weights, difficulty_bins, seed
    )

    # 3. 현재 rank 담당 서브셋만 필터링
    rank_indices = all_indices[rank::world_size]

    logger.info(
        f"Rank {rank}/{world_size}: 전체 {len(all_indices)} 중 "
        f"{len(rank_indices)} 샘플 로드"
    )

    # 4. 해당 인덱스만 JSONL 읽기
    samples = _read_jsonl_by_indices(jsonl_path, rank_indices)

    return Dataset.from_list(samples)
```

**핵심 로직**:
- `all_indices[rank::world_size]` 패턴으로 균등 분산
- 예: 100,000개 샘플, 4 GPU
  - Rank 0: indices[0::4] → 0, 4, 8, ... (25,000개)
  - Rank 1: indices[1::4] → 1, 5, 9, ... (25,000개)
  - Rank 2: indices[2::4] → 2, 6, 10, ... (25,000개)
  - Rank 3: indices[3::4] → 3, 7, 11, ... (25,000개)

**효과**:
- ✅ 메모리: GPU당 ~500MB → ~125MB (75% 절감)
- ✅ I/O: JSONL 읽기 중복 제거 (4배 → 1배)
- ✅ CPU: 메타데이터 샘플링 계산은 유지 (재현성 위해 필요)
- ✅ Curriculum 재샘플링 시에도 중복 없음

#### 2.1.2 중복 코드 통합

`src/weighted_mtp/data/dataloader.py` 신규 생성:

```python
"""DataLoader 팩토리 (분산 학습 지원)

4개 파이프라인의 중복 create_dataloader() 로직을 통합.
Rank-aware 샘플링으로 DistributedSampler 불필요.
"""

from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from weighted_mtp.data.datasets import load_dataset
from weighted_mtp.data.collators import AlpacaDataCollator
from weighted_mtp.runtime.distributed import get_rank, get_world_size


def create_dataloader(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    n_samples: int,
    balance_correct: bool = False,
    correct_ratio: float = 0.5,
    difficulty_weights: Optional[dict] = None,
    difficulty_bins: Optional[dict] = None,
    seed: int = 42,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader 생성 (Rank-aware 분산 학습)

    각 GPU가 자기 담당 샘플만 로드하므로 DistributedSampler 불필요.

    Args:
        dataset_path: 데이터셋 경로 (예: storage/datasets/codecontests/processed/train.jsonl)
        tokenizer: Tokenizer
        batch_size: 배치 크기 (per GPU)
        max_length: 최대 시퀀스 길이
        n_samples: 전체 샘플 수 (분산 환경에서는 자동으로 분할)
        balance_correct: is_correct 균형 샘플링 여부
        correct_ratio: correct 샘플 비율 (0.5 = 50:50)
        difficulty_weights: 난이도별 가중치 (curriculum learning용)
        difficulty_bins: 난이도 구간 정의
        seed: 랜덤 시드
        shuffle: 셔플 여부

    Returns:
        DataLoader

    Examples:
        >>> # Baseline: 정답만 100,000개
        >>> loader = create_dataloader(
        ...     "storage/datasets/codecontests/processed/train.jsonl",
        ...     tokenizer, batch_size=4, max_length=2048,
        ...     n_samples=100000, balance_correct=False, correct_ratio=1.0
        ... )
        >>>
        >>> # Verifiable: Curriculum learning
        >>> loader = create_dataloader(
        ...     "storage/datasets/codecontests/processed/train.jsonl",
        ...     tokenizer, batch_size=4, max_length=2048,
        ...     n_samples=100000, balance_correct=True, correct_ratio=0.5,
        ...     difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
        ...     difficulty_bins={"low": [1,3], "medium": [4,7], "high": [8,11]}
        ... )
    """
    # 데이터셋 이름 및 스플릿 추출
    dataset_path_obj = Path(dataset_path)
    dataset_name = dataset_path_obj.parent.parent.name
    split_file = dataset_path_obj.name

    if "train" in split_file:
        split = "train"
    elif "valid" in split_file or "validation" in split_file:
        split = "validation"
    else:
        split = "test"

    # 분산 환경 정보 자동 추출
    rank = get_rank()
    world_size = get_world_size()

    # Rank-aware 데이터셋 로드
    dataset = load_dataset(
        dataset_name=dataset_name,
        split=split,
        n_samples=n_samples,
        balance_correct=balance_correct,
        correct_ratio=correct_ratio,
        difficulty_weights=difficulty_weights,
        difficulty_bins=difficulty_bins,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Collator 생성
    collator = AlpacaDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # DataLoader 생성 (DistributedSampler 불필요)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # 직접 shuffle 사용
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
    )

    return dataloader
```

**변경사항**:
- DistributedSampler 제거 (이미 rank-aware 분산됨)
- sampler 반환 제거 (tuple → DataLoader만 반환)
- rank, world_size 자동 추출 (명시적 전달 불필요)

#### 2.1.3 파이프라인 적용

4개 파이프라인에서 중복 create_dataloader() 삭제 및 import 변경:

```python
# 기존 (run_verifiable.py)
def create_dataloader(...):  # 81 lines 중복
    ...

train_loader, train_sampler = create_dataloader(...)
if train_sampler is not None:
    train_sampler.set_epoch(...)

# 신규
from weighted_mtp.data.dataloader import create_dataloader

train_loader = create_dataloader(...)
# sampler 로직 제거 (불필요)
```

---

## 3. Phase별 구현 계획

### Phase 1: 중복 제거 - create_dataloader 통합

**목표**: 4개 파일의 중복 코드 제거

**작업 범위**:
1. `src/weighted_mtp/data/dataloader.py` 생성
   - 기존 create_dataloader() 로직 통합
   - 파라미터 통일 (difficulty_weights/bins optional)
   - DistributedSampler 유지 (Phase 3에서 제거)
2. 4개 파이프라인 수정:
   - 중복 create_dataloader() 삭제
   - `from weighted_mtp.data.dataloader import create_dataloader` 추가
   - 호출 코드는 기존과 동일

**파일 변경**:
- 신규: `src/weighted_mtp/data/dataloader.py` (150 lines)
- 수정: `src/weighted_mtp/pipelines/run_baseline.py` (-77 lines)
- 수정: `src/weighted_mtp/pipelines/run_critic.py` (-80 lines)
- 수정: `src/weighted_mtp/pipelines/run_verifiable.py` (-81 lines)
- 수정: `src/weighted_mtp/pipelines/run_rho1.py` (-77 lines)
- **순 변화**: -165 lines (코드 중복 제거)

**리스크**: 낮음 (로직 변경 없음, 단순 이동)

**테스트**:
```bash
# 기존 integration test 재실행
pytest tests/integration/test_pipeline_*.py -v
```

**완료 조건**:
- 4개 integration test 모두 통과
- 로컬 MPS 환경에서 정상 동작

**예상 소요**: 2-3시간

---

### Phase 2: Rank-aware 샘플링 구현

**목표**: load_dataset()에 rank 기반 서브셋 로딩 추가

**작업 범위**:
1. `src/weighted_mtp/data/datasets.py` 수정:
   - `load_dataset()` 시그니처에 `rank`, `world_size` 추가
   - `all_indices[rank::world_size]` 필터링 로직 추가
   - 로그 메시지 추가 (rank별 샘플 수)
2. `dataloader.py` 수정:
   - `get_rank()`, `get_world_size()` import
   - load_dataset() 호출 시 rank, world_size 전달

**핵심 코드**:
```python
# datasets.py:25
def load_dataset(
    dataset_name: str,
    split: str,
    n_samples: int,
    balance_correct: bool = False,
    correct_ratio: float = 0.5,
    difficulty_weights: Optional[dict] = None,
    difficulty_bins: Optional[dict] = None,
    seed: int = 42,
    rank: int = 0,  # 신규
    world_size: int = 1,  # 신규
) -> Dataset:
    """메타데이터 기반 효율적 로딩 (Rank-aware 분산)

    분산 환경에서 각 GPU가 자기 담당 샘플만 로드합니다.
    재현성을 위해 모든 rank가 동일한 시드로 전체 인덱스를 계산한 후,
    rank::world_size 패턴으로 서브셋을 선택합니다.

    Args:
        ...
        rank: 현재 프로세스의 global rank (기본: 0)
        world_size: 전체 프로세스 수 (기본: 1)

    Returns:
        Dataset (각 rank가 1/world_size 크기)
    """
    metadata = _load_metadata(dataset_name, split)

    if metadata is None:
        raise FileNotFoundError(...)

    # 전체 샘플링 인덱스 계산 (모든 rank 동일)
    all_indices = _compute_sampling_indices_from_metadata(
        metadata=metadata,
        n_samples=n_samples,
        balance_correct=balance_correct,
        correct_ratio=correct_ratio,
        difficulty_weights=difficulty_weights,
        difficulty_bins=difficulty_bins,
        seed=seed,
    )

    # Rank 담당 서브셋 필터링
    if world_size > 1:
        rank_indices = all_indices[rank::world_size]
        logger.info(
            f"[Rank {rank}/{world_size}] 전체 {len(all_indices)} 샘플 중 "
            f"{len(rank_indices)} 샘플 로드 (분산 학습)"
        )
    else:
        rank_indices = all_indices
        logger.info(f"메타데이터 기반 샘플링: {len(rank_indices)} 샘플")

    # 해당 인덱스만 JSONL 읽기
    data_files = _get_dataset_paths(dataset_name)
    jsonl_path = Path(data_files[split])
    samples = _read_jsonl_by_indices(jsonl_path, rank_indices)

    dataset = Dataset.from_list(samples)

    logger.info(f"데이터셋 로드 완료: {len(dataset)} 샘플")

    return dataset
```

**파일 변경**:
- 수정: `src/weighted_mtp/data/datasets.py` (+30 lines)
- 수정: `src/weighted_mtp/data/dataloader.py` (+5 lines)

**리스크**: 중간 (로직 변경, 재현성 검증 필요)

**테스트**:
```bash
# 신규 단위 테스트
pytest tests/unit/test_datasets.py::test_rank_aware_sampling -v
pytest tests/unit/test_datasets.py::test_reproducibility -v
```

**완료 조건**:
- Rank 0~3이 중복 없이 전체 샘플 커버
- 동일 seed로 재현성 보장
- 단위 테스트 통과

**예상 소요**: 3-4시간

---

### Phase 3: DistributedSampler 제거

**목표**: Rank-aware 샘플링으로 DistributedSampler 불필요화

**작업 범위**:
1. `dataloader.py` 수정:
   - `create_distributed_sampler()` 호출 제거
   - sampler 반환 제거 (tuple → DataLoader만 반환)
   - `shuffle=True` 직접 사용
2. 4개 파이프라인 수정:
   - sampler 관련 코드 제거:
     - `train_loader, train_sampler = ...` → `train_loader = ...`
     - `if train_sampler: train_sampler.set_epoch(...)` 삭제
     - `val_loader, val_sampler = ...` → `val_loader = ...`

**핵심 변경**:
```python
# 기존 (dataloader.py)
sampler = create_distributed_sampler(
    dataset, shuffle=shuffle, seed=seed, drop_last=False
)
dataloader = DataLoader(
    dataset, batch_size=batch_size,
    sampler=sampler,
    shuffle=(sampler is None),  # sampler 있으면 shuffle 비활성화
    ...
)
return dataloader, sampler

# 신규
dataloader = DataLoader(
    dataset, batch_size=batch_size,
    shuffle=shuffle,  # 직접 shuffle
    ...
)
return dataloader  # sampler 제거
```

```python
# 기존 (run_verifiable.py)
train_loader, train_sampler = create_dataloader(...)

# DistributedSampler epoch 설정 (재현성 유지)
if train_sampler is not None:
    train_sampler.set_epoch(int(target_epoch))

# 신규
train_loader = create_dataloader(...)
# sampler 로직 제거 (shuffle은 DataLoader가 자동 처리)
```

**Shuffle 재현성 처리**:
- Curriculum learning 재샘플링 시 seed 변경:
  ```python
  seed=config.data_sampling.seed + int(current_epoch * 1000)
  ```
- DataLoader의 shuffle은 worker별로 다른 시드 사용 (자동)

**파일 변경**:
- 수정: `src/weighted_mtp/data/dataloader.py` (-10 lines)
- 수정: `src/weighted_mtp/pipelines/run_baseline.py` (-5 lines)
- 수정: `src/weighted_mtp/pipelines/run_critic.py` (-5 lines)
- 수정: `src/weighted_mtp/pipelines/run_verifiable.py` (-10 lines)
- 수정: `src/weighted_mtp/pipelines/run_rho1.py` (-5 lines)
- **순 변화**: -35 lines

**리스크**: 중간 (분산 학습 동작 변경)

**테스트**:
```bash
# 분산 테스트 (RANK 시뮬레이션)
pytest tests/integration/test_distributed_loading.py -v

# 기존 integration test
pytest tests/integration/test_pipeline_*.py -v
```

**완료 조건**:
- 4 rank 시뮬레이션으로 중복 없음 검증
- Curriculum learning 재샘플링 정상 동작
- 기존 integration test 통과

**예상 소요**: 2-3시간

---

### Phase 4: 통합 검증

**목표**: A100 4-GPU 환경에서 실제 동작 검증

**작업 범위**:
1. 로컬 멀티 프로세스 테스트:
   ```bash
   # torchrun으로 4 프로세스 시뮬레이션
   TOKENIZERS_PARALLELISM=false torchrun \
     --nproc_per_node=4 \
     --nnodes=1 \
     -m weighted_mtp.cli.train \
     --config configs/verifiable/verifiable_local.yaml \
     --use-micro-model
   ```

2. VESSL A100 실행 (선택적):
   - scripts/vessl/verifiable_4gpu.yaml 실행
   - MLflow 로그 확인 (각 rank별 샘플 수)
   - Checkpoint 정상 저장 확인

3. 성능 벤치마크 (선택적):
   ```bash
   python scripts/benchmark_loading.py \
     --baseline-branch main \
     --optimized-branch rank-aware-sampling
   ```

**검증 항목**:
- [ ] 각 GPU가 25,000개씩 로드 (로그 확인)
- [ ] JSONL 읽기 1회만 (I/O 모니터링)
- [ ] 메모리 사용량 75% 절감 (nvidia-smi)
- [ ] Loss curve가 기존과 동일 (MLflow 비교)
- [ ] Curriculum 재샘플링 정상 (로그 확인)

**리스크**: 낮음 (이전 phase 완료 시)

**완료 조건**:
- 로컬 멀티 프로세스 테스트 통과
- VESSL 실행 성공 (선택적)
- 성능 개선 수치 확인

**예상 소요**: 2-3시간 (VESSL 포함 시 +2시간)

---

## 4. 테스트 전략

### 4.1 단위 테스트

**파일**: `tests/unit/test_datasets.py` 확장

```python
import pytest
from weighted_mtp.data.datasets import load_dataset


def test_rank_aware_sampling():
    """Rank-aware 샘플링이 정확히 분산되는지 검증"""
    n_samples = 1000
    world_size = 4
    dataset_name = "codecontests"
    split = "train"
    seed = 42

    # 각 rank가 로드한 task_id 수집
    all_task_ids = []
    for rank in range(world_size):
        dataset = load_dataset(
            dataset_name=dataset_name,
            split=split,
            n_samples=n_samples,
            balance_correct=False,
            correct_ratio=1.0,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
        all_task_ids.extend(dataset["task_id"])

    # 검증: 중복 없음, 총 1000개
    assert len(all_task_ids) == n_samples, (
        f"전체 샘플 수 불일치: {len(all_task_ids)} != {n_samples}"
    )
    assert len(set(all_task_ids)) == n_samples, (
        f"중복 샘플 발견: {n_samples - len(set(all_task_ids))}개"
    )

    # 검증: 각 rank가 250개씩 로드
    for rank in range(world_size):
        dataset = load_dataset(
            dataset_name=dataset_name,
            split=split,
            n_samples=n_samples,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
        assert len(dataset) == n_samples // world_size, (
            f"Rank {rank} 샘플 수 불일치: {len(dataset)} != {n_samples // world_size}"
        )


def test_reproducibility():
    """동일 seed로 재현성 보장"""
    dataset1 = load_dataset(
        dataset_name="codecontests",
        split="train",
        n_samples=100,
        seed=42,
        rank=0,
        world_size=4,
    )
    dataset2 = load_dataset(
        dataset_name="codecontests",
        split="train",
        n_samples=100,
        seed=42,
        rank=0,
        world_size=4,
    )

    assert dataset1["task_id"] == dataset2["task_id"], (
        "동일 seed로 샘플이 달라짐 (재현성 위반)"
    )


def test_curriculum_consistency():
    """Curriculum learning에서 difficulty 분포 검증"""
    dataset = load_dataset(
        dataset_name="codecontests",
        split="train",
        n_samples=1000,
        balance_correct=True,
        correct_ratio=0.5,
        difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
        difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
        seed=42,
        rank=0,
        world_size=1,
    )

    # difficulty 분포 계산
    difficulties = [sample["metadata"]["difficulty"] for sample in dataset]
    low_count = sum(1 for d in difficulties if 1 <= d <= 3)
    medium_count = sum(1 for d in difficulties if 4 <= d <= 7)
    high_count = sum(1 for d in difficulties if 8 <= d <= 11)

    # 0.7:0.3:0.0 비율 검증 (±10% 허용)
    assert abs(low_count / len(dataset) - 0.7) < 0.1, (
        f"Low difficulty 비율 이상: {low_count / len(dataset):.2%}"
    )
    assert abs(medium_count / len(dataset) - 0.3) < 0.1, (
        f"Medium difficulty 비율 이상: {medium_count / len(dataset):.2%}"
    )
    assert high_count == 0, f"High difficulty 샘플 존재: {high_count}개"


def test_backward_compatibility():
    """Rank 파라미터 없이도 동작 (로컬 환경)"""
    dataset = load_dataset(
        dataset_name="codecontests",
        split="train",
        n_samples=100,
        seed=42,
        # rank, world_size 생략 (기본값 0, 1)
    )

    assert len(dataset) == 100, (
        f"로컬 환경 샘플 수 불일치: {len(dataset)} != 100"
    )
```

---

### 4.2 통합 테스트

**파일**: `tests/integration/test_distributed_loading.py` 신규

```python
import pytest
import os
from pathlib import Path
from weighted_mtp.data.dataloader import create_dataloader
from weighted_mtp.models.tokenizer_utils import load_tokenizer


@pytest.fixture
def tokenizer():
    """Tokenizer fixture"""
    return load_tokenizer("storage/models/micro-mtp/tokenizer")


def test_distributed_dataloader_no_overlap(tokenizer):
    """4 rank 시뮬레이션으로 중복 없음 검증"""
    dataset_path = "storage/datasets/codecontests/processed/train.jsonl"
    n_samples = 400
    world_size = 4

    # 각 rank 시뮬레이션
    all_batches = []
    for rank in range(world_size):
        # 환경변수로 rank 설정 (get_rank() 시뮬레이션)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        loader = create_dataloader(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            batch_size=4,
            max_length=512,
            n_samples=n_samples,
            balance_correct=False,
            correct_ratio=1.0,
            seed=42,
            shuffle=False,  # 순서 보장
        )

        # 배치 수집
        batches = list(loader)
        all_batches.extend(batches)

        # 검증: 각 rank가 100개 샘플 (batch_size=4 → 25 batches)
        total_samples = sum(len(batch["input_ids"]) for batch in batches)
        assert total_samples == n_samples // world_size, (
            f"Rank {rank} 샘플 수 불일치: {total_samples} != {n_samples // world_size}"
        )

    # 전체 검증: 400 샘플 (중복 없음)
    total_samples = sum(len(batch["input_ids"]) for batch in all_batches)
    assert total_samples == n_samples, (
        f"전체 샘플 수 불일치: {total_samples} != {n_samples}"
    )


def test_verifiable_curriculum_distributed(tokenizer):
    """Verifiable 파이프라인의 curriculum + 분산 테스트"""
    dataset_path = "storage/datasets/codecontests/processed/train.jsonl"
    n_samples = 400
    world_size = 4

    # Curriculum stage 1 (low: 70%)
    for rank in range(world_size):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        loader = create_dataloader(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            batch_size=4,
            max_length=512,
            n_samples=n_samples,
            balance_correct=True,
            correct_ratio=0.5,
            difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        # 각 rank가 100개 로드 (25 batches)
        batches = list(loader)
        total_samples = sum(len(batch["input_ids"]) for batch in batches)
        assert total_samples == n_samples // world_size
```

---

### 4.3 성능 벤치마크

**파일**: `scripts/benchmark_loading.py` 신규

```python
"""데이터 로딩 성능 벤치마크

기존 DistributedSampler 방식 vs Rank-aware 샘플링 비교:
- 메모리 사용량
- JSONL 읽기 시간
- 전체 데이터 로딩 시간
"""

import time
import psutil
import subprocess
from pathlib import Path


def measure_memory_usage():
    """현재 프로세스 메모리 사용량 (MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_baseline():
    """기존 DistributedSampler 방식 벤치마크"""
    print("=== Baseline (DistributedSampler) ===")

    # Git checkout main branch
    subprocess.run(["git", "checkout", "main"], check=True)

    # 4 rank 시뮬레이션
    for rank in range(4):
        start_time = time.time()
        start_memory = measure_memory_usage()

        # DataLoader 생성 (실제 코드 실행)
        # ...

        end_memory = measure_memory_usage()
        end_time = time.time()

        print(f"Rank {rank}:")
        print(f"  메모리: {end_memory - start_memory:.2f} MB")
        print(f"  시간: {end_time - start_time:.2f}s")


def benchmark_optimized():
    """Rank-aware 샘플링 벤치마크"""
    print("=== Optimized (Rank-aware) ===")

    # Git checkout rank-aware-sampling branch
    subprocess.run(["git", "checkout", "rank-aware-sampling"], check=True)

    # 4 rank 시뮬레이션
    for rank in range(4):
        start_time = time.time()
        start_memory = measure_memory_usage()

        # DataLoader 생성 (실제 코드 실행)
        # ...

        end_memory = measure_memory_usage()
        end_time = time.time()

        print(f"Rank {rank}:")
        print(f"  메모리: {end_memory - start_memory:.2f} MB")
        print(f"  시간: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    benchmark_baseline()
    benchmark_optimized()
```

---

## 5. 승인 필요 사항

개발원칙 [원칙 3]에 따라 다음 변경사항에 대한 승인이 필요합니다:

### 5.1 중복 코드 삭제

**대상**:
- `src/weighted_mtp/pipelines/run_baseline.py:71-147` (77 lines)
- `src/weighted_mtp/pipelines/run_critic.py:69-148` (80 lines)
- `src/weighted_mtp/pipelines/run_verifiable.py:78-158` (81 lines)
- `src/weighted_mtp/pipelines/run_rho1.py:103-179` (77 lines)

**사유**: 4개 파일에서 99% 동일한 create_dataloader() 중복

**대체**: `src/weighted_mtp/data/dataloader.py`로 통합

**승인 여부**: [ ] 승인 / [ ] 거부

---

### 5.2 DistributedSampler 제거

**대상**:
- `create_distributed_sampler()` 호출 중단
- `sampler.set_epoch()` 로직 제거
- 반환 타입 변경 (tuple → DataLoader)

**사유**: Rank-aware 샘플링으로 DistributedSampler 불필요

**영향**: 분산 학습 메커니즘 변경 (기능적으로는 동일)

**승인 여부**: [ ] 승인 / [ ] 거부

---

### 5.3 함수 시그니처 변경

**대상**: `load_dataset()` 파라미터 추가

**변경**:
```python
# 기존
def load_dataset(
    dataset_name: str,
    split: str,
    n_samples: int,
    ...
) -> Dataset

# 신규
def load_dataset(
    dataset_name: str,
    split: str,
    n_samples: int,
    ...
    rank: int = 0,  # 추가
    world_size: int = 1,  # 추가
) -> Dataset
```

**사유**: Rank 기반 서브셋 로딩

**하위 호환**: 기본값(0, 1)으로 로컬 환경 호환

**승인 여부**: [ ] 승인 / [ ] 거부

---

### 5.4 하위 호환성 포기 확인

**개발원칙 [원칙 4]**: 하위 호환성을 고려하지 말고 기존 코드를 전격적으로 삭제

**확인 사항**:
- [ ] 기존 DistributedSampler 기반 코드는 완전히 삭제
- [ ] 불필요한 fallback 로직 없음
- [ ] 중복 create_dataloader() 모두 제거
- [ ] 이전 코드 복원 불가 (깨끗한 리팩토링)

**승인 여부**: [ ] 승인 / [ ] 거부

---

## 6. 대안 및 Trade-off

### 6.1 대안 1: 현재 구조 유지

**내용**:
- Rank-aware 샘플링 도입 안 함
- DistributedSampler 유지
- 중복 코드만 통합 (Phase 1만 수행)

**장점**:
- 리스크 최소 (검증된 구조)
- 개발 시간 최소
- DistributedSampler 표준 패턴

**단점**:
- 비효율 유지 (메모리 중복, I/O 중복)
- 대규모 데이터셋 확장 시 병목
- 코드 의도와 실제 동작 괴리

**권장 여부**: △ (단기 프로덕션은 가능, 장기적으로 개선 필요)

---

### 6.2 대안 2: 부분 적용

**내용**:
- Phase 1, 2만 수행 (Rank-aware 샘플링)
- DistributedSampler는 유지 (Phase 3 생략)
- 이중 분산 (load_dataset + DistributedSampler)

**장점**:
- 점진적 개선 (리스크 분산)
- DistributedSampler 로직 재사용
- 기존 코드 변경 최소

**단점**:
- 이중 분산으로 복잡도 증가
- Shuffle 로직 중복
- 효율 개선 반감 (메모리는 절약, 복잡도는 증가)

**권장 여부**: × (복잡도 증가, 권장하지 않음)

---

### 6.3 대안 3: 전체 적용 (본 계획)

**내용**:
- Phase 1~4 모두 수행
- Rank-aware 샘플링 + 중복 제거 + DistributedSampler 제거
- 깨끗한 리팩토링

**장점**:
- 메모리 75% 절감
- I/O 중복 제거
- 코드 단순화 (-200 lines)
- 대규모 확장 대비

**단점**:
- 개발 시간 소요 (12-16시간)
- 테스트 복잡도 증가
- 분산 학습 동작 변경 리스크

**권장 여부**: ✅ **권장** (장기적으로 최선)

---

### 6.4 Trade-off 분석

| 항목 | 대안 1 (현상 유지) | 대안 2 (부분 적용) | 대안 3 (전체 적용) |
|------|-------------------|-------------------|-------------------|
| **메모리 효율** | ❌ 중복 유지 | ✅ 75% 절감 | ✅ 75% 절감 |
| **I/O 효율** | ❌ 4배 중복 | ✅ 중복 제거 | ✅ 중복 제거 |
| **코드 복잡도** | △ 중복 존재 | ⚠️ 이중 분산 | ✅ 단순화 |
| **개발 시간** | ✅ 2-3시간 | △ 6-8시간 | ⚠️ 12-16시간 |
| **테스트 난이도** | ✅ 낮음 | △ 중간 | ⚠️ 높음 |
| **리스크** | ✅ 낮음 | △ 중간 | ⚠️ 중간 |
| **확장성** | ❌ 제한적 | ✅ 좋음 | ✅ 좋음 |
| **유지보수** | ❌ 중복 관리 | ⚠️ 복잡 | ✅ 명확 |
| **권장도** | △ 단기만 | × 비권장 | ✅ **권장** |

---

## 7. 결론

### 7.1 권장사항

**Phase 1~4 전체 적용**을 권장합니다:

**근거**:
1. **효율 개선**: 메모리 75%, I/O 4배 절감
2. **코드 품질**: 중복 제거(-200 lines), 의도 명확화
3. **확장성**: 대규모 데이터셋 대응
4. **개발원칙 준수**: [원칙 2, 4] 중복 제거, 깨끗한 리팩토링

**실행 계획**:
1. 승인 확정 후 Phase별 순차 진행
2. 각 Phase별 테스트 통과 후 커밋
3. Phase 4 완료 후 VESSL 실행
4. 성능 개선 측정 및 문서화

**예상 일정**:
- Phase 1: 0.5일 (중복 제거)
- Phase 2: 0.5일 (Rank-aware 샘플링)
- Phase 3: 0.5일 (DistributedSampler 제거)
- Phase 4: 0.5일 (통합 검증)
- **총 2일** (집중 개발 시)

**성공 기준**:
- [ ] 모든 단위/통합 테스트 통과
- [ ] 로컬 멀티 프로세스 테스트 성공
- [ ] VESSL A100 실행 성공
- [ ] 메모리 사용량 75% 절감 확인
- [ ] I/O 중복 제거 확인

---

### 7.2 다음 단계

승인 후 다음 순서로 진행:

1. **승인 확정** (본 문서)
2. **Phase 1 구현** (중복 제거)
3. **Phase 2 구현** (Rank-aware 샘플링)
4. **Phase 3 구현** (DistributedSampler 제거)
5. **Phase 4 검증** (통합 테스트)
6. **성과 문서화** (벤치마크 결과)

---

**문서 버전**: 1.0.0
**최종 업데이트**: 2025-11-17
**작성자**: Claude Code (Weighted MTP Team)
**검토자**: 사용자 승인 대기
