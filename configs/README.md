# Config 구조 및 필드 설명

## 1. Config 구조

```
configs/
├── baseline/
│   ├── baseline.yaml          # Baseline MTP (VESSL A100 4-GPU)
│   └── baseline_local.yaml    # Baseline MTP (로컬 테스트용)
├── critic/
│   ├── critic.yaml            # Critic pretraining (VESSL A100 4-GPU)
│   └── critic_local.yaml      # Critic pretraining (로컬)
├── verifiable/
│   ├── verifiable.yaml        # Verifiable WMTP (VESSL A100 4-GPU)
│   └── verifiable_local.yaml  # Verifiable WMTP (로컬)
└── rho1/
    ├── rho1.yaml              # Rho-1 Weighted (VESSL A100 4-GPU)
    └── rho1_local.yaml        # Rho-1 Weighted (로컬)
```

**각 config 파일은 독립적이고 완전합니다** (no defaults.yaml dependency)

## 2. Config Override 메커니즘

모든 파이프라인은 `--override` 인자를 통해 계층 구조의 필드를 CLI에서 override할 수 있습니다.

### 기본 사용법

```bash
# 단일 필드 override
python -m weighted_mtp train \
  --config configs/baseline/baseline.yaml \
  --override experiment.name=my_experiment

# 여러 필드 override
python -m weighted_mtp train \
  --config configs/baseline/baseline.yaml \
  --override experiment.name=my_experiment \
  --override runtime.device=cuda \
  --override training.batch_size=8
```

### 계층 구조 override

```bash
# 중첩된 필드 override
python -m weighted_mtp train \
  --config configs/baseline/baseline.yaml \
  --override experiment.name=test \
  --override models.policy.path=storage/models/micro-mtp \
  --override training.learning_rate=5e-5 \
  --override data_sampling.n_samples=1000
```

### Curriculum schedule override (리스트 구조)

```bash
# Verifiable stage의 curriculum schedule override
python -m weighted_mtp train \
  --config configs/verifiable/verifiable.yaml \
  --override "data_sampling.curriculum_schedule[0].epoch_range=[0.0,0.5]" \
  --override "data_sampling.curriculum_schedule[0].difficulty_weights.low=0.8"
```

**Override 형식**: `key=value` (중첩: `key1.key2.key3=value`)

## 3. 주요 필드 설명

### 3.1 experiment (실험 메타정보)

```yaml
experiment:
  name: baseline-mtp           # 실험 이름 (MLflow run name 기본값)
  stage: baseline              # 파이프라인 종류 (baseline/critic/verifiable/rho1)
  description: "Baseline MTP"  # 실험 설명
  tags:                        # MLflow 태그
    - baseline
    - uniform-weight
```

**필수 필드**:
- `name`: 실험 이름
- `stage`: 파이프라인 종류 (baseline, critic, verifiable, rho1 중 하나)

### 3.2 models (모델 경로)

```yaml
models:
  policy:
    name: meta-llama-mtp
    path: storage/models/meta-llama-mtp  # Policy 모델 경로
    tokenizer_path: storage/models/meta-llama-mtp/tokenizer
    dtype: float16

  # Rho-1 전용 (reference model)
  reference:
    name: ref-sheared-llama-2.7b
    path: storage/models/ref-sheared-llama-2.7b
    dtype: float16
```

**필수 필드**:
- `models.policy.path`: Policy 모델 경로
- (Rho-1 전용) `models.reference.path`: Reference 모델 경로

### 3.3 dataset (데이터셋 설정)

```yaml
dataset:
  name: codecontests           # 데이터셋 이름
  train: storage/datasets/codecontests/processed/train.jsonl
  validation: storage/datasets/codecontests/processed/valid.jsonl
  max_length: 2048             # 최대 시퀀스 길이
```

**필수 필드**:
- `name`: 데이터셋 이름
- `train`: 학습 데이터셋 파일 경로
- `validation`: 검증 데이터셋 파일 경로

### 3.4 data_sampling (샘플링 전략)

```yaml
data_sampling:
  n_samples: 100000            # 샘플 개수
  correct_ratio: 1.0           # Correct 샘플 비율 (Baseline: 1.0, Critic: 0.5)
  seed: 42                     # Random seed

  # Verifiable 전용 (Curriculum learning)
  curriculum_learning: true
  difficulty_bins:
    low: [1, 3]
    medium: [4, 7]
    high: [8, 11]
  curriculum_schedule:
    - epoch_range: [0.0, 0.3]
      difficulty_weights:
        low: 0.7
        medium: 0.3
        high: 0.0
```

**Stage별 차이**:
- **Baseline**: `correct_ratio: 1.0` (정답만 학습)
- **Critic/Verifiable**: `correct_ratio: 0.5` (Correct/Incorrect 균형)
- **Verifiable**: `curriculum_learning` 설정 추가

### 3.5 training (학습 설정)

```yaml
training:
  n_epochs: 2.5                # Epoch 수
  batch_size: 8                # Batch size (per GPU)
  gradient_accumulation_steps: 2  # Gradient accumulation
  learning_rate: 1.0e-5        # Learning rate
  max_grad_norm: 1.0           # Gradient clipping
  log_interval: 5             # 로깅 간격 (steps)

  # Verifiable 전용 (TD error weighting)
  beta: 1.0                    # TD error temperature
  weight_clip_min: 0           # 최소 가중치
  weight_clip_max: 5.0         # 최대 가중치

  # Rho-1 전용
  temperature: 1.0             # Rho-1 temperature
  k_percent: 0.6               # Top-k percent for loss selection
```

**필수 필드**:
- `n_epochs`: Epoch 수 (> 0)
- `batch_size`: Batch size (> 0)
- `learning_rate`: Learning rate (0 < lr <= 1.0)

**Stage별 특수 필드**:
- **Verifiable**: `beta`, `weight_clip_min`, `weight_clip_max` 필수
- **Rho-1**: `temperature`, `k_percent` 필수
- **Critic**: pairwise 모드 사용

### 3.6 checkpoint (체크포인트 설정)

```yaml
checkpoint:
  save_dir: storage/checkpoints/baseline/${experiment.name}
  save_checkpoint_every: 0.5   # 저장 간격 (epochs)
  save_best: true              # Best checkpoint 저장
  save_final: true             # Final checkpoint 저장
  save_total_limit: 2          # 최대 보관 개수
```

### 3.7 runtime (런타임 설정)

```yaml
runtime:
  device: cuda                 # 디바이스 (cuda/cpu/mps/auto)
  seed: 42                     # Random seed
  mixed_precision: true        # Mixed precision 학습
```

### 3.8 mlflow (실험 추적)

```yaml
mlflow:
  tracking_uri: "http://13.50.240.176"  # MLflow 서버 주소
  experiment: "weighted-mtp/production"  # Experiment 이름
  s3_artifacts: "s3://wmtp/mlflow-artifacts"  # S3 artifact store
```

**로컬 테스트 시**:
```yaml
mlflow:
  tracking_uri: ""             # 빈 문자열 = MLflow 비활성화
  experiment: ""
```

## 4. VESSL vs 로컬 Config 차이

| 필드 | VESSL (baseline.yaml) | 로컬 (baseline_local.yaml) |
|------|----------------------|---------------------------|
| `training.batch_size` | 4 | 2 |
| `training.gradient_accumulation_steps` | 4 | 1 |
| `training.n_epochs` | 2.5 | 0.2 |
| `data_sampling.n_samples` | 100000 | 500 |
| `models.policy.path` | meta-llama-mtp | micro-mtp |
| `runtime.device` | cuda | mps |
| `mlflow.tracking_uri` | http://13.50.240.176 | "" (비활성화) |

**Effective batch size**:
- VESSL A100 4-GPU: 4 (batch) × 4 (accumulation) × 4 (GPUs) = 64
- 로컬: 2 (batch) × 1 (accumulation) × 1 (GPU) = 2

## 5. Config 검증

실행 전 config 검증:

```bash
python -m weighted_mtp validate-config --config configs/baseline/baseline.yaml
```

**검증 항목**:
- 필수 필드 존재 확인
- 값 범위 검증 (learning_rate > 0 등)
- 경로 존재 확인 (모델, 데이터셋)
- Stage별 특수 필드 검증
- 논리적 일관성 (batch_size <= n_samples)

**검증 예시**:
```bash
$ python -m weighted_mtp validate-config --config configs/baseline/baseline.yaml
✓ Config 검증 성공: configs/baseline/baseline.yaml
  - Experiment: baseline-mtp
  - Stage: baseline
  - Model: storage/models/meta-llama-mtp
  - Dataset: codecontests
```

## 6. CLI 사용 예시

### Train 파이프라인

```bash
# 기본 실행
python -m weighted_mtp train --config configs/baseline/baseline.yaml

# Override 적용
python -m weighted_mtp train \
  --config configs/baseline/baseline.yaml \
  --override experiment.name=my_experiment \
  --override runtime.device=cuda \
  --override training.batch_size=8

# 로컬 테스트용 micro 모델 사용
python -m weighted_mtp train \
  --config configs/baseline/baseline_local.yaml \
  --override models.policy.path=storage/models/micro-mtp \
  --override data_sampling.n_samples=100
```

### Direct 파이프라인 실행

```bash
# Baseline 파이프라인 직접 실행
python src/weighted_mtp/pipelines/run_baseline.py \
  --config configs/baseline/baseline.yaml \
  --override experiment.name=test

# Verifiable 파이프라인 실행
python src/weighted_mtp/pipelines/run_verifiable.py \
  --config configs/verifiable/verifiable.yaml \
  --override experiment.critic_checkpoint=storage/checkpoints/critic/checkpoint_best.pt
```

## 7. Stage별 Config 예시

### Baseline (균등 가중치)

```yaml
experiment:
  stage: baseline

data_sampling:
  correct_ratio: 1.0           # 정답만 학습

training:
  n_epochs: 2.5
  batch_size: 8
  learning_rate: 1.0e-5
```

**Override 예시**:
```bash
python -m weighted_mtp train \
  --config configs/baseline/baseline.yaml \
  --override training.learning_rate=5e-5
```

### Critic (Value head pretraining)

```yaml
experiment:
  stage: critic

data_sampling:
  correct_ratio: 0.5           # Correct/Incorrect 균형

training:
  n_epochs: 0.5
  batch_size: 8
  learning_rate: 1.0e-4
```

**Override 예시**:
```bash
python src/weighted_mtp/pipelines/run_critic.py \
  --config configs/critic/critic.yaml \
  --override training.loss_type=huber
```

### Verifiable (TD error weighting)

```yaml
experiment:
  stage: verifiable
  critic_checkpoint: storage/checkpoints/critic/critic-pretrain/checkpoint_best.pt

data_sampling:
  correct_ratio: 0.5
  curriculum_learning: true
  curriculum_schedule:
    - epoch_range: [0.0, 0.3]
      difficulty_weights: {low: 0.7, medium: 0.3, high: 0.0}

training:
  beta: 1.0                    # TD error temperature
  weight_clip_min: 0
  weight_clip_max: 5.0
```

**Override 예시**:
```bash
python src/weighted_mtp/pipelines/run_verifiable.py \
  --config configs/verifiable/verifiable.yaml \
  --override training.beta=0.9 \
  --override models.policy.path=storage/checkpoints/critic/my_checkpoint.pt
```

### Rho-1 (Reference-based weighting)

```yaml
experiment:
  stage: rho1

models:
  reference:
    path: storage/models/ref-sheared-llama-2.7b

data_sampling:
  correct_ratio: 1.0           # 정답만 학습

training:
  temperature: 1.0             # Rho-1 temperature
  k_percent: 0.6               # Top 60% loss selection
```

**Override 예시**:
```bash
python src/weighted_mtp/pipelines/run_rho1.py \
  --config configs/rho1/rho1.yaml \
  --override training.temperature=0.8 \
  --override training.k_percent=0.5
```

## 8. Troubleshooting

### Config 검증 실패

**문제**: `ConfigValidationError: 필수 필드 누락`

**해결**:
1. 이 문서의 "3. 주요 필드 설명" 참조
2. 필수 필드가 config 파일에 존재하는지 확인
3. 필요 시 다른 stage의 config 예시 참조

**문제**: `경로가 존재하지 않음: storage/models/...`

**해결**:
1. `ls storage/models/` 실행하여 경로 확인
2. 모델 다운로드 필요 시 setup 스크립트 실행
3. `--override models.policy.path=<존재하는_경로>`로 수정

### Stage별 검증 오류

**Verifiable**: `training.beta 필드 필수`
- 해결: `--override training.beta=0.9` 추가

**Rho-1**: `models.reference.path 필드 필수`
- 해결: Config 파일에 `models.reference` 섹션 추가

**Critic**: `잘못된 loss_type`
- 해결: `--override training.loss_type=mse` (mse/huber/mae 중 선택)

### Override 형식 오류

**문제**: `ValueError: 잘못된 override 형식`

**해결**: `key=value` 형식 준수
```bash
# 올바른 형식
--override experiment.name=test

# 잘못된 형식 (= 누락)
--override experiment.name test
```

## 9. 참조

- **Stage별 파이프라인 구현**: `src/weighted_mtp/pipelines/run_*.py`
- **Config 검증 로직**: `src/weighted_mtp/utils/config_utils.py`
- **Override 적용 로직**: `src/weighted_mtp/utils/config_utils.py:apply_overrides()`
- **CLI 사용법**: `python -m weighted_mtp --help`
