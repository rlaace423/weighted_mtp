# Phase 6: CLI 파이프라인 연동 구현 가이드

## 문서 개요

본 문서는 **Phase 6: CLI 파이프라인 연동 구현**을 위한 실행 가이드입니다. Phase 5에서 구현된 학습 파이프라인을 사용자가 CLI로 실행할 수 있도록 연결하고, Validation evaluation, Best checkpoint 저장, MLflow 로깅 등 프로덕션 학습에 필수적인 기능을 추가합니다.

**버전**: v2.0 (2025-01-16)
**선행 조건**: Phase 5 (Value Training 파이프라인) 완료
**목표**: CLI → Config → Resource Loading → Training Pipeline → MLflow 전체 흐름 완성

---

## Part 1: 개요 및 맥락

### 1.1 Phase 6의 위치와 목적

Phase 6는 **사용자 진입점 → 파이프라인 실행** 연결의 핵심 구간입니다.

```
Phase 5 (pipeline)  →  [Phase 6 (CLI + Config + MLflow)]  →  실제 학습 실행
  run_training_pipeline()      사용자 커맨드 → 실행             VESSL/로컬
```

**핵심 질문**: 사용자가 `python -m weighted_mtp.cli.train --recipe configs/recipe.verifiable.yaml` 한 줄로 어떻게 전체 학습을 실행할 것인가?

### 1.2 Phase 6의 핵심 기능

#### 기능 1: Config 시스템 (defaults + recipe deep merge)

**문제 인식**:
- 모든 설정을 recipe에 반복 작성하면 유지보수 어려움
- 실험마다 다른 부분(dataset, beta, value_coef)만 override하고 싶음

**해결책**:
```yaml
# defaults.yaml (환경 고정값)
project:
  name: weighted-mtp
mlflow:
  tracking_uri: "http://13.50.240.176"
  experiment: "weighted-mtp/production"
training:
  stage1:
    n_epochs: 0.5
    learning_rate: 1.0e-4
  stage2:
    beta: 0.9  # 기본값
    value_coef: 0.5

# recipe.verifiable.yaml (실험 차이만)
experiment:
  name: verifiable-critic-wmtp
dataset:
  name: codecontests
training:
  stage2:
    beta: 1.2  # Override만 명시
```

**Deep merge 결과**:
- defaults + recipe 재귀적 병합
- recipe의 값이 defaults를 override
- recipe에 없는 키는 defaults 유지

#### 기능 2: Resource Loading (model, tokenizer, datasets)

**Phase 5 → Phase 6 연결**:
```python
# Phase 5에서 정의된 인터페이스
def run_training_pipeline(
    adapter: MetaLlamaMTPAdapter,
    stage1_dataloader: DataLoader,
    stage2_dataloader: DataLoader,
    config: dict,
    device: torch.device,
    save_dir: Path | None,
) -> dict[str, dict[str, float]]
```

**Phase 6의 역할**: 이 인터페이스에 맞는 resource를 로딩하여 전달
- `adapter` ← `_load_model(config, device)`
- `stage1_dataloader` ← `_create_dataloaders(stage1_dataset, ...)`
- `config` ← `load_config()` + `_extract_training_config()`

#### 기능 3: Validation Evaluation & Best Checkpoint

**문제 인식**:
- Phase 5는 train loss만 출력, validation 평가 없음
- Overfitting 감지 불가, best checkpoint 선택 로직 없음

**해결책**:
1. **Validation dataloader 추가**: train/val 분리
2. **Periodic evaluation**: Step interval마다 validation loss 계산
3. **Best checkpoint 저장**: Validation loss 개선 시에만 저장
4. **Early stopping 준비**: Validation loss가 개선되지 않으면 경고

#### 기능 4: MLflow 실험 추적 (WMTP EC2 + S3 재사용)

**기존 인프라 재사용**:
- EC2 Tracking Server: http://13.50.240.176 (Basic Auth)
- S3 Artifact Storage: s3://wmtp/mlflow-artifacts
- Experiment: weighted-mtp/production (ID: 8)

**Logging 항목**:
- Config parameters (flatten)
- Train/Validation metrics (step 단위)
- Checkpoint artifact upload (S3)

#### 기능 5: Distributed Training 지원

**분산학습 고려사항**:
- **Checkpoint 저장**: Rank 0 only (storage 절약)
- **Console logging**: Rank 0 only (깔끔한 출력)
- **MLflow logging**: Rank 0 only (중복 방지)
- **Metrics 평균**: All ranks에서 계산 → Rank 0에서 출력

### 1.3 기대 효과

1. **사용 편의성**: 한 줄 커맨드로 실험 실행 (`--recipe` 전환만으로 3가지 실험)
2. **재현성**: seed + config로 동일한 결과 보장
3. **안정성**: Validation 기반 best checkpoint, overfitting 조기 감지
4. **추적성**: MLflow로 모든 실험 기록, S3에 checkpoint 백업

---

## Part 2: 핵심 설계 결정

### 2.1 Decision 0: Phase 5 인터페이스 존중 (가장 중요한 결정)

**원칙 1, 2 준수**:
- Phase 5의 `run_training_pipeline()` 인터페이스는 **변경하지 않음**
- Phase 6는 이 인터페이스에 맞는 resource를 준비하는 역할만

**Rationale**:
1. **역할 분리**: Phase 5 = 파이프라인 로직, Phase 6 = 진입점 + resource 준비
2. **테스트 가능성**: Phase 5 unit test가 계속 동작
3. **유지보수성**: 파이프라인 변경 시 Phase 6 수정 불필요

### 2.2 Decision 1: Validation Evaluation 추가

**문제**: Phase 5는 train loss만 출력, overfitting 감지 불가

**해결책**: Validation dataloader 추가 + periodic evaluation

**구현 위치**:
- `pipelines/training.py`: `evaluate_stage()` 함수 추가
- `cli/train.py`: validation dataloader 로딩

**Evaluation 시점**:
- Step interval마다 (예: 100 steps)
- Epoch 종료 시
- 학습 완료 후 최종 평가

**요구사항**:
```python
def evaluate_stage(
    adapter: MetaLlamaMTPAdapter,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    stage: str,
) -> dict[str, float]:
    """Validation evaluation (no_grad)

    Returns:
        {
            "val_loss": float,
            "val_mtp_loss": float,  # Stage 2만
            "val_value_loss": float,  # Stage 2만
            "val_value_explained_variance": float,
        }
    """
```

### 2.3 Decision 2: Best Checkpoint 저장 전략

**문제**: 모든 step checkpoint 저장하면 storage 낭비, 실시간 저장은 I/O 오버헤드 심각

**해결책**: Epoch 단위 checkpoint 저장 + 학습 완료 후 best 선택 (Meta MTP 2024 방식)

**저장 정책**:
1. **Periodic checkpoint**: Epoch마다 저장 (`save_checkpoint_every: 1`)
   - `checkpoint_stage1_epoch_0.5.pt` (Stage 1: 0.5 epoch)
   - `checkpoint_stage2_epoch_1.pt`, `checkpoint_stage2_epoch_2.pt` (Stage 2: 2.5 epochs)
   - 각 checkpoint에 validation loss 메타데이터 기록
2. **Best checkpoint**: 학습 완료 후 선택 (`checkpoint_{stage}_best.pt`)
   - 저장된 모든 epoch checkpoint의 validation loss 비교
   - 최저 val_loss 가진 checkpoint를 `best.pt`로 복사/심볼릭 링크
3. **Final checkpoint**: 학습 완료 후 최종 저장 (`checkpoint_{stage}_final.pt`)

**Best checkpoint 선택 기준**:
- Stage 1: `val_loss` 최소화
- Stage 2: `val_loss` 최소화 (total_loss = weighted_ce + value_coef * value_loss)

**Meta MTP 2024 논문 방식**:
> "Model checkpoints with maximal validation ROUGE-L F1 are selected separately for each model dataset and model type."
- 학습 중 여러 checkpoint 저장 → 학습 완료 후 validation metric 기반 best 선택
- 우리는 ROUGE-L 대신 validation loss 사용

### 2.4 Decision 3: Step 기반 Logging & Evaluation

**문제**: Epoch 단위는 너무 coarse, 세밀한 제어 불가

**해결책**: Global step 기반 interval (모니터링 전용)

**Interval 역할 구분**:
```yaml
training:
  log_interval: 10      # 10 step마다 train loss 출력 (console/MLflow)
  eval_interval: 100    # 100 step마다 validation 평가 (로깅만, checkpoint 저장 안 함)
  save_checkpoint_every: 1  # 1 epoch마다 checkpoint 저장
```

**핵심 원칙**:
- `eval_interval: 100` → **모니터링 전용**
  - Validation loss 계산 → console/MLflow 로깅만
  - Overfitting 조기 감지, 학습 진행 상황 추적
  - **Checkpoint 저장 안 함** (I/O 오버헤드 방지)
- `save_checkpoint_every: 1` → **Checkpoint 저장 전용**
  - Epoch 끝날 때만 checkpoint 저장
  - 각 checkpoint에 validation loss 메타데이터 기록
- **학습 완료 후** → Best 선택
  - 저장된 모든 checkpoint의 validation loss 비교
  - 최저 val_loss 가진 checkpoint를 `best.pt`로 복사

**흐름 예시** (Stage 2: 2.5 epochs, 1000 steps/epoch):
```python
# Step 100: validation → val_loss=2.5 (로깅만, 저장 안 함)
# Step 200: validation → val_loss=2.3 (로깅만)
# ...
# Step 1000 (Epoch 1 끝):
#   - validation → val_loss=2.1
#   - checkpoint 저장 "epoch_1.pt" (val_loss=2.1 메타데이터 기록)

# Step 1100: validation → val_loss=2.0 (로깅만)
# Step 1200: validation → val_loss=1.9 (로깅만)
# ...
# Step 2000 (Epoch 2 끝):
#   - validation → val_loss=1.8
#   - checkpoint 저장 "epoch_2.pt" (val_loss=1.8 메타데이터 기록)

# Step 2500 (Final):
#   - validation → val_loss=1.85
#   - checkpoint 저장 "final.pt" (val_loss=1.85 메타데이터 기록)

# 학습 완료 후:
# epoch_2.pt (val_loss=1.8)이 최저 → "best.pt"로 복사
```

**Global step 계산**:
```python
global_step = 0
for epoch in range(n_epochs):
    for batch in dataloader:
        # Training step
        ...
        global_step += 1

        # Log (모니터링 전용)
        if global_step % log_interval == 0:
            logger.info(f"Step {global_step}, Loss: {loss:.4f}")

        # Eval (모니터링 전용, checkpoint 저장 안 함)
        if global_step % eval_interval == 0:
            val_metrics = evaluate_stage(...)
            logger.info(f"Step {global_step}, Val Loss: {val_metrics['val_loss']:.4f}")
            mlflow_manager.log_metrics({"val_loss": val_metrics["val_loss"]}, step=global_step)

    # Epoch 끝: checkpoint 저장
    val_metrics = evaluate_stage(...)
    save_checkpoint(f"epoch_{epoch}.pt", metadata={"val_loss": val_metrics["val_loss"]})
```

### 2.5 Decision 4: Rank 0 Only Operations

**문제**: 분산학습 시 모든 rank가 checkpoint 저장/logging하면 중복

**해결책**: Rank 0만 I/O operations 수행

**Rank 0 Only**:
- Checkpoint 저장
- Console logging (logger.info)
- MLflow logging (start_run, log_params, log_metrics, log_artifact)

**All Ranks**:
- Training step
- Metrics 계산 (각 rank에서 계산 후 평균)

**구현**:
```python
from weighted_mtp.runtime.distributed import is_main_process

if is_main_process():  # Rank 0 only
    _save_checkpoint(...)
    logger.info(...)
    mlflow_manager.log_metrics(...)
```

### 2.6 Decision 5: Comprehensive Logging Metrics (LLM SFT/RLHF Best Practices 2024)

**문제**: Loss만 기록하면 학습 상태 파악 어려움, 디버깅 불가

**해결책**: LLM SFT/PPO RLHF 표준 logging 항목 도입 (HuggingFace TRL, OpenRLHF, QLoRA 2024 기준)

**Logging 항목 (Stage별)**:

#### Stage 1 & 2 공통 (Train Metrics)
```yaml
# 1. Loss Metrics
train/loss: 2.1234            # Total loss
train/mtp_loss: 1.9876        # MTP loss (Stage 2만)
train/value_loss: 0.1358      # Value loss
train/head_0_loss: 2.3456     # Individual head losses (디버깅용)
train/head_1_loss: 2.4123
train/head_2_loss: 2.5678
train/head_3_loss: 2.3890

# 2. Gradient & Optimization
train/grad_norm: 0.8765       # Gradient norm (안정성 지표, QLoRA: max_grad_norm=0.3)
train/learning_rate: 1.0e-5   # Current LR (scheduler 추적)

# 3. Performance & Throughput
train/samples_per_second: 4.2 # Throughput (분산학습 효율 측정)
train/tokens_per_second: 8192 # Token throughput
train/wall_time: 3600.5       # 누적 벽시계 시간 (초)
train/epoch: 1.5              # Current epoch (소수점)
train/global_step: 1500       # Global step counter

# 4. GPU & Memory (분산학습 필수)
system/gpu_memory_allocated_gb: 35.2  # 할당된 GPU 메모리 (GB)
system/gpu_memory_reserved_gb: 40.0   # 예약된 GPU 메모리
system/gpu_utilization_pct: 95.3     # GPU 활용률 (%)

# 5. Stage 2 전용 (Weighted MTP + Value)
train/td_error_mean: 0.0234   # TD error 평균 (weighting 품질)
train/td_error_std: 0.1567    # TD error 표준편차
train/weight_entropy: 2.3456  # Weight distribution entropy
train/weight_mean: 1.0234     # Weight 평균
train/weight_min: 0.1000      # Weight 최소값 (clip 확인)
train/weight_max: 5.0000      # Weight 최대값 (clip 확인)
train/value_explained_variance: 0.8234  # Value head 성능 (1.0에 가까울수록 이상적)
```

#### Validation Metrics (100 step마다 + Epoch 끝)
```yaml
val/loss: 1.8765              # Validation total loss
val/mtp_loss: 1.7432          # Validation MTP loss
val/value_loss: 0.1333        # Validation value loss
val/value_explained_variance: 0.8456
```

**Logging 빈도**:
- `log_interval: 10` → Train metrics (loss, grad_norm, LR, throughput)
- `eval_interval: 100` → Validation metrics
- Every step → GPU memory (lightweight)

**Console 출력 형식** (간결화):
```
[Step 100/2500] [Epoch 0.04] [3.2 samples/s]
  Train Loss: 2.1234 | Val Loss: 1.8765
  Grad Norm: 0.8765 | LR: 1.0e-5
  GPU Mem: 35.2/40.0 GB (95.3%)
  MTP: 1.9876 | Value: 0.1358 | ExplVar: 0.8234
```

**MLflow 전체 로깅** (모든 항목 기록):
- Train metrics → `mlflow.log_metrics(train_metrics, step=global_step)`
- Validation metrics → `mlflow.log_metrics(val_metrics, step=global_step)`
- System metrics → `mlflow.log_metrics(system_metrics, step=global_step)`

**Best Practices 근거**:
- **Gradient Norm**: Early-stage lower gradient norms → better final performance (SFT 2024)
- **Learning Rate**: Standard 1e-4 for LoRA, 2e-4 for QLoRA (HuggingFace TRL)
- **GPU Utilization**: OPPO 2024 - 1.4×-2.1× improvement target
- **Throughput**: OpenRLHF 2024 - samples/tokens per second for bottleneck detection
- **Value Explained Variance**: PPO RLHF 2024 - stability indicator (should not spike/NaN)

---

## Part 3: 구현 요구사항

### 3.1 Step 1: Config 추출/검증 함수 (`cli/train.py`)

#### 목표
YAML config를 `run_training_pipeline()` 입력 형식으로 변환하고 유효성 검증

#### 핵심 함수

**`_extract_training_config(config: dict) -> dict`**
- 역할: 전체 config에서 training 설정 추출
- 입력: defaults + recipe merge 완료된 config
- 출력: `{"stage1": {...}, "stage2": {...}, "save_dir": Path, "log_interval": int, "eval_interval": int}`

**`_validate_config(config: dict) -> None`**
- 역할: 필수 필드 존재 여부, 값 범위 검증
- 검증 항목:
  - 필수 섹션: `project`, `models`, `dataset`, `training`, `mlflow`
  - 필수 필드: `models.policy.name`, `models.policy.path`, `dataset.train`, `dataset.validation`
  - 값 범위: `n_epochs > 0`, `learning_rate > 0`, `beta > 0`
- 예외: `ValueError` (missing field 또는 invalid value)

#### 검증 기준
- [ ] `_extract_training_config()`: stage1/stage2 설정 추출 성공
- [ ] `_extract_training_config()`: save_dir 기본값 생성 (`storage/checkpoints/{project}/{experiment}`)
- [ ] `_validate_config()`: 필수 필드 누락 시 ValueError
- [ ] `_validate_config()`: 값 범위 검증 (n_epochs > 0 등)

---

### 3.2 Step 2: Resource Loading 함수 (`cli/train.py`)

#### 목표
Model, Tokenizer, Datasets, Dataloaders를 로딩하여 `run_training_pipeline()`에 전달

#### 핵심 함수

**`_load_model(config: dict, device: torch.device) -> MetaLlamaMTPAdapter`**
- 역할: Policy model adapter 로딩
- 로딩 순서:
  1. Config에서 model path 추출
  2. Metadata 로딩 (`{path}/metadata.json`)
  3. Safetensors 로딩 (`{path}/safetensors/model.safetensors`)
  4. MetaLlamaMTPAdapter 생성 및 state_dict 로딩
  5. Device로 이동
- 출력: 초기화된 adapter

**`_load_tokenizer(config: dict) -> PreTrainedTokenizer`**
- 역할: Tokenizer 로딩
- Phase 5와 동일하게 `AutoTokenizer.from_pretrained()` 사용

**`_load_datasets(config: dict) -> tuple[Dataset, Dataset, Dataset]`**
- 역할: Stage 1 train, Stage 2 train, Validation dataset 로딩
- Phase 3의 `load_dataset()` 재사용
- 반환: `(stage1_train_dataset, stage2_train_dataset, val_dataset)`
- **Validation dataset 로딩 방식**:
  - `config["data"]["validation"]["split"]` 사용 (기본값: "validation")
  - `use_full_split: true` → 샘플링 없이 전체 validation split 사용
  - CodeContests: 14,725 samples (correct + incorrect 혼합)
  - Stage 1/2 모두 동일한 validation set 사용 (일관성)

**`_create_dataloaders(datasets: tuple, tokenizer, config) -> tuple[DataLoader, ...]`**
- 역할: 4개 dataloader 생성
- 반환: `(stage1_train_loader, stage2_train_loader, stage1_val_loader, stage2_val_loader)`
- 주의사항:
  - Train: `shuffle=True` (분산학습 시 DistributedSampler 사용)
  - Validation: `shuffle=False`

#### 검증 기준
- [ ] `_load_model()`: Adapter 로딩 성공, device 이동 확인
- [ ] `_load_tokenizer()`: Tokenizer 로딩 성공
- [ ] `_load_datasets()`: 3개 dataset 반환 (stage1_train, stage2_train, val)
- [ ] `_create_dataloaders()`: 4개 dataloader 생성 (train/val × stage1/stage2)

---

### 3.3 Step 3: Validation Evaluation 함수 (`pipelines/training.py`)

#### 목표
Validation dataset으로 현재 모델 성능 평가 (no gradient)

#### 핵심 함수

**`evaluate_stage(adapter, dataloader, config, device, stage) -> dict`**
- 역할: Validation loss 계산 (train과 동일한 loss 함수 사용)
- 입력:
  - `adapter`: MetaLlamaMTPAdapter
  - `dataloader`: Validation DataLoader
  - `config`: Stage 설정 (stage1 또는 stage2)
  - `device`: torch.device
  - `stage`: "stage1" or "stage2"
- 동작:
  1. `adapter.eval()` 설정
  2. `torch.no_grad()` context
  3. Dataloader iteration
  4. Stage 1: Value loss 계산
  5. Stage 2: MTP loss + Value loss 계산
  6. 평균 계산
- 반환:
  ```python
  {
      "val_loss": float,
      "val_mtp_loss": float,  # Stage 2만
      "val_value_loss": float,  # Stage 2만
      "val_value_explained_variance": float,
  }
  ```

#### 통합 위치

**Stage 1 학습 루프 수정**:
```python
# 기존 train_stage1() 유지
# 호출하는 곳에서 periodic evaluation 추가

for epoch in range(n_epochs):
    # Training
    train_metrics = train_stage1_epoch(...)

    # Validation (epoch 종료 시)
    val_metrics = evaluate_stage(adapter, val_dataloader, config, device, "stage1")

    # Best checkpoint 저장
    if val_metrics["val_loss"] < best_val_loss:
        best_val_loss = val_metrics["val_loss"]
        _save_checkpoint(..., "best")
```

**Stage 2도 동일한 패턴 적용**

#### 검증 기준
- [ ] `evaluate_stage()`: no_grad context에서 실행
- [ ] `evaluate_stage()`: Stage 1 val_loss 계산 성공
- [ ] `evaluate_stage()`: Stage 2 val_loss (MTP + Value) 계산 성공
- [ ] Best checkpoint: val_loss 개선 시에만 저장 확인

---

### 3.4 Step 4: Best Checkpoint 저장 로직 (`pipelines/training.py`)

#### 목표
Validation loss 기반 best checkpoint 추적 및 저장

#### 핵심 로직

**Best checkpoint tracking**:
```python
# run_training_pipeline() 내부
best_val_loss_stage1 = float('inf')
best_val_loss_stage2 = float('inf')

# Stage 1 학습 중
for epoch in range(n_epochs):
    train_metrics = train_stage1_epoch(...)
    val_metrics = evaluate_stage(..., "stage1")

    # Best 갱신
    if val_metrics["val_loss"] < best_val_loss_stage1:
        best_val_loss_stage1 = val_metrics["val_loss"]
        if is_main_process():
            _save_checkpoint(adapter, optimizer, "stage1", epoch, val_metrics, save_dir / "checkpoint_stage1_best.pt")
```

**Checkpoint 종류**:
1. **Periodic**: `checkpoint_stage1_epoch_0.5.pt` (optional)
2. **Best**: `checkpoint_stage1_best.pt` (필수)
3. **Final**: `checkpoint_stage1_final.pt` (필수)

#### 검증 기준
- [ ] Best checkpoint: val_loss 개선 시에만 저장
- [ ] Best checkpoint: 파일명 `checkpoint_{stage}_best.pt`
- [ ] Final checkpoint: 학습 완료 후 저장
- [ ] Rank 0 only: is_main_process() 체크

---

### 3.5 Step 5: MLflow 로깅 통합 (`runtime/mlflow.py` + `cli/train.py`)

#### 목표
WMTP EC2 MLflow 서버에 실험 추적 정보 로깅

#### 구현 파일
**`src/weighted_mtp/runtime/mlflow.py`** (신규 생성)

#### MLflowManager 클래스 구현

**역할**:
- EC2 MLflow 서버 연결 관리
- .env에서 Basic Auth 자동 로드
- Experiment 생성/조회
- Run lifecycle 관리 (start → log → end)
- S3 artifact 업로드

**핵심 메서드**:
- `__init__(tracking_uri, experiment_name, s3_artifacts)`: EC2 서버 연결 초기화
- `_inject_basic_auth(uri)`: .env에서 MLFLOW_TRACKING_USERNAME/PASSWORD 로드 후 URI에 주입
- `start_run(run_name, tags)`: mlflow.start_run() 호출, run_id 반환
- `log_params(params: dict)`: Config를 flatten하여 mlflow.log_params() 호출
- `log_metrics(metrics: dict, step: int)`: mlflow.log_metrics() 호출
- `log_artifact(local_path, artifact_path)`: mlflow.log_artifact() 호출 (S3 자동 업로드)
- `end_run(status)`: mlflow.end_run() 호출

**구현 핵심 로직**:
1. **Basic Auth 주입**:
   - .env에서 `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` 로드
   - tracking_uri가 http://로 시작하면 `http://user:pass@host` 형태로 변환
2. **Experiment 관리**:
   - `mlflow.get_experiment_by_name()` → 없으면 `mlflow.create_experiment()` 호출
   - artifact_location에 S3 경로 설정
3. **Config flatten**:
   - 중첩 dict를 `stage1.learning_rate` 형태로 평탄화
   - mlflow.log_params()는 flat dict만 지원
4. **Context manager 패턴**:
   - `with mlflow_manager.run(run_name): ...` 형태 지원 권장

#### CLI 통합 (`cli/train.py`)

**main() 함수에서 MLflow 사용**:
```python
# Rank 0 only
mlflow_manager = None
if is_main_process():
    mlflow_manager = create_mlflow_manager(config)
    mlflow_manager.start_run(run_name=run_name)
    mlflow_manager.log_params(config)

try:
    # Training
    metrics = run_training_pipeline(...)

    # Metrics 로깅
    if is_main_process():
        mlflow_manager.log_metrics(metrics["stage1"], step=0)
        mlflow_manager.log_metrics(metrics["stage2"], step=1)

        # Checkpoint 업로드
        mlflow_manager.log_artifact(checkpoint_path, "checkpoints")
finally:
    if mlflow_manager:
        mlflow_manager.end_run()
```

#### defaults.yaml 설정

```yaml
mlflow:
  tracking_uri: "http://13.50.240.176"  # EC2 MLflow Server
  experiment: "weighted-mtp/production"
  s3_artifacts: "s3://wmtp/mlflow-artifacts"
```

#### .env 설정

```.env
# MLflow EC2 Server Authentication
MLFLOW_TRACKING_USERNAME=wmtp_admin
MLFLOW_TRACKING_PASSWORD=your_password

# AWS S3 Credentials
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=eu-north-1
```

#### 검증 기준
- [ ] MLflowManager: EC2 서버 연결 성공
- [ ] Experiment: weighted-mtp/production 생성/로드
- [ ] Params: Config flatten 후 로깅 성공
- [ ] Metrics: Step 단위 로깅 성공
- [ ] Artifact: Checkpoint S3 업로드 성공 (s3://wmtp/mlflow-artifacts/8/{run_id}/artifacts/checkpoints/)

---

### 3.6 Step 6: Step 기반 Logging & Evaluation (`pipelines/training.py`)

#### 목표
Global step 기반 주기적 logging 및 evaluation

#### 핵심 로직

**Global step tracking**:
```python
def train_stage1(..., log_interval: int, eval_interval: int, val_dataloader: DataLoader):
    global_step = 0

    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Training step
            loss = ...
            loss.backward()
            optimizer.step()

            global_step += 1

            # Log interval
            if is_main_process() and global_step % log_interval == 0:
                logger.info(f"[Stage 1] Step {global_step}, Epoch {epoch}, Loss: {loss:.4f}")

            # Eval interval
            if is_main_process() and global_step % eval_interval == 0:
                val_metrics = evaluate_stage(adapter, val_dataloader, config, device, "stage1")
                logger.info(f"[Stage 1] Step {global_step}, Val Loss: {val_metrics['val_loss']:.4f}")

                # Best checkpoint
                if val_metrics["val_loss"] < best_val_loss:
                    ...
```

#### defaults.yaml 설정

```yaml
training:
  log_interval: 10    # 10 step마다 train loss 출력
  eval_interval: 100  # 100 step마다 validation 평가
  save_checkpoint_every: 1  # 1 epoch마다 checkpoint 저장
```

#### 검증 기준
- [ ] Log interval: 설정된 step마다 출력 확인
- [ ] Eval interval: 설정된 step마다 validation 평가 확인
- [ ] Global step: Epoch 경계 무관하게 증가 확인

---

### 3.7 Step 7: Multi-head Loss 개별 Logging (`pipelines/training.py`)

#### 목표
Stage 2에서 4개 MTP head loss를 개별적으로 출력

#### 핵심 로직

**Stage 2 학습 중**:
```python
# 기존: 평균만 계산
mtp_losses = []
for i, head_logits in enumerate(mtp_logits_list):
    loss_i = F.cross_entropy(...)
    mtp_losses.append(loss_i)

avg_mtp_loss = sum(mtp_losses) / len(mtp_losses)

# 추가: 개별 loss logging
if is_main_process() and global_step % log_interval == 0:
    for i, loss in enumerate(mtp_losses):
        logger.info(f"    Head {i} loss: {loss:.4f}")
    logger.info(f"    Avg MTP loss: {avg_mtp_loss:.4f}")
```

#### 검증 기준
- [ ] 4개 head loss 개별 출력 확인
- [ ] 평균 MTP loss 출력 확인
- [ ] Log interval에 맞춰 출력 확인

---

### 3.8 Step 8: CLI main() 함수 완성 (`cli/train.py`)

#### 목표
모든 구성 요소를 연결하여 사용자 커맨드 실행

#### main() 흐름

1. **Argument parsing**:
   - `--config`: defaults.yaml 경로
   - `--recipe`: recipe 파일 경로
   - `--preset`: local-light 등 preset
   - `--use-micro-model`: micro 모델 사용
   - `--dry-run`: 설정만 출력
   - `--run-name`: MLflow run 이름

2. **Config loading & merging**:
   - `load_config(config_path, recipe_path)`: Deep merge
   - Preset 적용 (if specified)
   - Micro model override (if specified)

3. **Config validation**:
   - `_validate_config(config)`

4. **Environment setup**:
   - Logging 설정
   - Device 설정
   - Distributed 초기화 (if multi-GPU)

5. **MLflow initialization** (Rank 0 only):
   - `create_mlflow_manager(config)`
   - `start_run(run_name)`
   - `log_params(config)`

6. **Resource loading**:
   - `_load_model(config, device)`
   - `_load_tokenizer(config)`
   - `_load_datasets(config)`
   - `_create_dataloaders(...)`

7. **Training config extraction**:
   - `_extract_training_config(config)`

8. **Pipeline execution**:
   - `run_training_pipeline(adapter, stage1_train_loader, stage2_train_loader, config, device, save_dir)`
   - Validation evaluation 포함 (내부에서)

9. **MLflow finalization** (Rank 0 only):
   - `log_metrics(final_metrics)`
   - `log_artifact(checkpoint_path)`
   - `end_run()`

10. **Cleanup**:
    - Distributed cleanup (if multi-GPU)

#### 검증 기준
- [ ] `--dry-run`: Config 출력 후 종료
- [ ] `--use-micro-model`: Micro 모델로 로컬 학습 성공
- [ ] `--recipe configs/recipe.verifiable.yaml`: Recipe 적용 성공
- [ ] MLflow: Params, metrics, artifact 모두 로깅 확인
- [ ] Checkpoint: Best checkpoint 저장 확인

---

### 3.9 Step 9: Distributed Training 지원 (`runtime/distributed.py`)

#### 목표
분산학습 환경에서 Rank 0 only operations 보장

#### 핵심 함수

**`is_main_process() -> bool`**
- 역할: 현재 프로세스가 Rank 0인지 확인
- 반환: `True` (Rank 0) or `False` (other ranks)
- 구현:
  ```python
  import torch.distributed as dist

  def is_main_process() -> bool:
      if not dist.is_available() or not dist.is_initialized():
          return True  # 단일 GPU
      return dist.get_rank() == 0
  ```

#### 적용 위치

**Checkpoint 저장**:
```python
if is_main_process():
    _save_checkpoint(...)
```

**Logging**:
```python
if is_main_process():
    logger.info(...)
```

**MLflow**:
```python
if is_main_process():
    mlflow_manager.log_metrics(...)
```

#### 검증 기준
- [ ] 단일 GPU: is_main_process() = True
- [ ] 분산학습: Rank 0만 checkpoint 저장
- [ ] 분산학습: Rank 0만 console 출력
- [ ] 분산학습: Rank 0만 MLflow 로깅

---

## Part 4: 통합 및 검증

### 4.1 통합 테스트

#### Test 1: Dry-run 모드
```bash
python -m weighted_mtp.cli.train \
    --config configs/defaults.yaml \
    --recipe configs/recipe.verifiable.yaml \
    --dry-run
```
**검증**: Config 출력, 실행 안 함

#### Test 2: Micro 모델 로컬 학습
```bash
python -m weighted_mtp.cli.train \
    --config configs/defaults.yaml \
    --recipe configs/recipe.verifiable.yaml \
    --use-micro-model \
    --preset local-light
```
**검증**:
- Model loading 성공
- Stage 1/2 학습 성공
- Validation evaluation 성공
- Best checkpoint 저장
- MLflow 로깅 성공

#### Test 3: Production 모델 학습
```bash
python -m weighted_mtp.cli.train \
    --config configs/defaults.yaml \
    --recipe configs/recipe.verifiable.yaml
```
**검증**:
- Production model (7B) 로딩
- 분산학습 동작 (4 GPU)
- MLflow S3 artifact 업로드

### 4.2 검증 체크리스트

#### 기능 검증
- [ ] Config deep merge 동작
- [ ] Resource loading 성공 (model, tokenizer, datasets)
- [ ] Validation evaluation 동작
- [ ] Best checkpoint 저장 (val_loss 기준)
- [ ] Step 기반 logging
- [ ] Step 기반 evaluation
- [ ] Multi-head loss 개별 출력
- [ ] MLflow params/metrics/artifact 로깅
- [ ] Rank 0 only operations

#### 성능 검증
- [ ] Micro 모델: 학습 완료 (<10분)
- [ ] Production 모델: 분산학습 동작 확인

#### 품질 검증
- [ ] Unit tests 통과 (config, resource loading)
- [ ] Integration test 통과 (end-to-end)
- [ ] MLflow UI에서 실험 확인 가능

---

## Part 5: Phase 5와의 연계

### 5.1 Phase 5에서 제공하는 것

**파이프라인 인터페이스**:
```python
def run_training_pipeline(
    adapter: MetaLlamaMTPAdapter,
    stage1_dataloader: DataLoader,
    stage2_dataloader: DataLoader,
    config: dict,
    device: torch.device,
    save_dir: Path | None,
) -> dict[str, dict[str, float]]
```

**Value weighting 모듈**:
- `compute_td_errors()`
- `build_weights()`
- `compute_weight_stats()`, `compute_td_stats()`

**Stage 1/2 학습 함수**:
- `train_stage1()`
- `train_stage2()`

### 5.2 Phase 6에서 추가하는 것

**사용자 진입점**:
- CLI argparse
- Config loading & merging
- Resource loading

**프로덕션 기능**:
- Validation evaluation
- Best checkpoint 저장
- MLflow 로깅
- Distributed training 지원

**Phase 5 수정사항**:
- `evaluate_stage()` 함수 추가 (`pipelines/training.py`)
- Global step 기반 logging/evaluation 추가
- Multi-head loss 개별 logging 추가
- Rank 0 only 체크 추가

---

## Part 6: 예상 소요 시간

| 작업 | 예상 시간 | 비고 |
|------|-----------|------|
| Config extraction/validation | 2-3시간 | _extract_training_config, _validate_config |
| Resource loading 함수 | 3-4시간 | _load_model, _load_tokenizer, _load_datasets, _create_dataloaders |
| Validation evaluation | 3-4시간 | evaluate_stage 함수 |
| Best checkpoint 저장 | 2-3시간 | Best tracking 로직 |
| MLflow 통합 | 3-4시간 | MLflowManager 신규 구현 + CLI 통합 |
| Step 기반 logging/eval | 2-3시간 | Global step tracking |
| Multi-head logging | 1-2시간 | 개별 loss 출력 |
| CLI main() 완성 | 3-4시간 | 전체 흐름 연결 |
| Distributed 지원 | 2-3시간 | is_main_process 적용 |
| 통합 테스트 및 디버깅 | 4-6시간 | End-to-end 테스트 |
| 문서화 | 2-3시간 | 본 문서 업데이트 |
| **합계** | **27-39시간** | 약 3.5-5일 |

---

## Part 7: 부록

### 7.1 Config 구조 요약

```yaml
# defaults.yaml
project:
  name: weighted-mtp
  version: "2.0.0"

models:
  policy:
    name: meta-llama-mtp
    path: storage/models_v2/meta-llama-mtp

dataset:
  name: codecontests
  train: storage/datasets_v2/codecontests/processed/train.jsonl
  validation: storage/datasets_v2/codecontests/processed/valid.jsonl

mlflow:
  tracking_uri: "http://13.50.240.176"
  experiment: "weighted-mtp/production"
  s3_artifacts: "s3://wmtp/mlflow-artifacts"

training:
  log_interval: 10
  eval_interval: 100
  save_checkpoint_every: 1

  stage1:
    n_epochs: 0.5
    learning_rate: 1.0e-4
    loss_type: mse

  stage2:
    n_epochs: 2.5
    learning_rate: 1.0e-5
    beta: 0.9
    value_coef: 0.5
    max_grad_norm: 0.5
    loss_type: mse
    weight_clip_min: 0.1
    weight_clip_max: 5.0
```

### 7.2 MLflow 저장 구조

```
s3://wmtp/mlflow-artifacts/
└── 8/                                    # Experiment ID
    └── {run_id}/                         # Run ID
        └── artifacts/
            └── checkpoints/
                ├── checkpoint_stage1_best.pt
                └── checkpoint_stage2_best.pt
```

### 7.3 개발 원칙 준수 체크리스트

- [x] **원칙 1**: Phase 5 파이프라인 인터페이스 분석 완료
- [x] **원칙 2**: Phase 5 구조 존중, 중복 메서드 없음
- [x] **원칙 3**: Phase 5 수정 최소화 (evaluate_stage 추가만)
- [x] **원칙 4**: 구체적 코드 구현 삭제, 핵심 설명만 유지
- [x] **원칙 5**: Phase 3/5 양식 참고하여 간결한 문서 작성
- [x] **원칙 6**: 의존성 도구 활용 (uv, mlflow, boto3)

### 7.4 참고 문서

- `docs/05_phase3_detailed_plan.md`: 데이터 파이프라인 (양식 참고)
- `docs/07_phase5_detailed_plan.md`: Value Training 파이프라인
- `docs/wmtp_research_proposal.md`: WMTP 연구 의도
- `src/weighted_mtp/pipelines/training.py`: Phase 5 파이프라인 구현
