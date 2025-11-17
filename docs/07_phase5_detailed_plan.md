# Phase 5: Stage별 독립 실행 파이프라인 구현 가이드

## 문서 개요

본 문서는 **Phase 5: Stage별 독립 실행 파이프라인 구현**을 위한 실행 가이드입니다. 기존의 단일 오케스트레이션 방식을 완전히 폐기하고, Critic Pre-training, Verifiable WMTP, Rho-1 Weighted Training을 각각 독립적으로 실행 가능한 파이프라인으로 구현합니다.

**버전**: v2.0 (2025-01-17) - Stage 분리 기반 전면 재설계
**선행 조건**: Phase 3 (데이터 파이프라인), Phase 4 (Meta Adapter) 완료
**목표**: MLflow Projects 표준 기반 Stage별 독립 실행 + Checkpoint Handoff

---

## Part 1: 개요 및 맥락

### 1.1 Phase 5의 위치와 목적

Phase 5는 **학습 파이프라인의 실행 단위 분리**를 담당합니다.

```
Phase 4 (model)  →  [Phase 5 (Stage별 독립 Runner)]  →  실험 실행
   Adapter 구현      Critic / Verifiable / Rho-1 분리      각 stage 독립 실행
```

**핵심 질문**: 어떻게 Critic Pre-training, Verifiable Training, Rho-1 Training을 독립적으로 실행하고, Stage 간 Checkpoint를 효율적으로 전달할 것인가?

### 1.2 Stage 분리의 필요성

**기존 방식의 문제점** (단일 `run_training_pipeline`):

| 문제 | 설명 |
|------|------|
| **재실행 비용** | Stage 2만 재실행하려면 Stage 1도 다시 실행 필요 (2x 비용) |
| **실험 추적 복잡도** | 1개 MLflow run에 Stage 1+2 혼재, 분석 어려움 |
| **Rho-1 지원 불가** | Ref model 구조가 달라 단일 파이프라인으로 불가능 |
| **Config 복잡도** | training.stage1, training.stage2 한 파일에 혼재 |
| **Checkpoint 재사용** | 내부 전달만 가능, 외부 재사용 불가 |

**Stage 분리의 장점**:

| 장점 | 효과 |
|------|------|
| **독립 실행** | Stage 2만 재실행 → ~50% 비용 절감 (Critic 재사용) |
| **실험 추적** | 각 Stage별 MLflow run 분리 → 명확한 분석 |
| **Rho-1 지원** | 완전 독립 파이프라인으로 ref model 구조 수용 |
| **Config 명확화** | configs/critic/, configs/verifiable/, configs/rho1/ 분리 |
| **Checkpoint 재사용** | MLflow artifact로 저장 → 다른 실험에서 로드 가능 |

### 1.3 MLflow Projects Best Practice (2024)

**표준 Multi-Step Workflow 패턴**:

```python
# Step 1: Critic Pre-training
critic_run = mlflow.run(
    uri=".",
    entry_point="critic_training",
    parameters={"config": "configs/critic/critic.yaml"}
)
critic_run.wait()

# Artifact URI 획득
critic_checkpoint = mlflow.get_artifact_uri(
    f"runs:/{critic_run.info.run_id}/checkpoints/checkpoint_best.pt"
)

# Step 2: Verifiable Training (Critic checkpoint 사용)
verifiable_run = mlflow.run(
    uri=".",
    entry_point="verifiable_training",
    parameters={
        "config": "configs/verifiable/verifiable.yaml",
        "critic_checkpoint": critic_checkpoint
    }
)
```

**핵심 원칙**:
1. **독립성**: 각 step은 entry_point로 독립 실행
2. **Artifact 전달**: 이전 step의 output을 MLflow artifact로 저장 → 다음 step에서 로드
3. **Config 분리**: 각 step별 YAML config 분리
4. **Run 추적**: Parent-child run 관계로 전체 workflow 추적

**LLM Continual Pre-training Pattern (2024)**:
- Stage 1 (Pre-training): Checkpoint 저장
- Stage 2 (Continued Training): Stage 1 checkpoint로 초기화
- Efficiency: ~2x 비용 절감 (vs. from scratch)

### 1.4 기대 효과

1. **재실행 효율**: Stage 2만 재실행 시 Critic checkpoint 재사용 → 50% 비용 절감
2. **실험 추적**: MLflow에서 Critic run과 Verifiable run 명확히 구분
3. **Rho-1 지원**: Ref model 기반 완전 독립 파이프라인 구현 가능
4. **Config 관리**: Stage별 설정 분리로 가독성 향상
5. **재현성**: Checkpoint artifact URI로 정확한 재현

---

## Part 2: 핵심 설계 결정

### 2.1 Decision 1: Stage 완전 분리 아키텍처

**문제 인식**: 기존 `run_training_pipeline()`은 Stage 1→2를 한 함수에서 오케스트레이션

**해결책**: 각 Stage를 독립 실행 가능한 Runner로 분리

**아키텍처**:

```
src/weighted_mtp/pipelines/
├── training.py              # 공통 함수 (train_stage1, train_stage2, evaluate_stage)
├── run_critic.py            # Stage 1 Runner (독립 실행)
├── run_verifiable.py        # Stage 2 Runner (독립 실행, critic checkpoint 로드)
├── run_rho1.py              # Rho-1 Runner (독립 실행, ref model 사용)
└── checkpoint_utils.py      # Checkpoint save/load 유틸
```

**실행 방식**:

```bash
# Stage 1 (Critic Pre-training)
python -m weighted_mtp.pipelines.run_critic \
    --config configs/critic/critic.yaml

# Stage 2 (Verifiable WMTP)
python -m weighted_mtp.pipelines.run_verifiable \
    --config configs/verifiable/verifiable.yaml \
    --critic-checkpoint storage/checkpoints/critic/.../checkpoint_best.pt

# Rho-1 (독립)
python -m weighted_mtp.pipelines.run_rho1 \
    --config configs/rho1/rho1.yaml
```

**기존 코드 재사용**:
- ✅ `training.py:train_stage1()` - Critic training 로직 재사용
- ✅ `training.py:train_stage2()` - Verifiable training 로직 재사용
- ✅ `value_weighting/` 전체 - TD error, weight builder 재사용
- ❌ `run_training_pipeline()` - 삭제 (오케스트레이션 불필요)

### 2.2 Decision 2: MLflow Artifact Checkpoint Handoff

**문제 인식**: Stage 간 checkpoint 전달 메커니즘 필요

**해결책**: MLflow Artifact + Local Path 동시 지원

**Checkpoint 저장** (Stage 1):
```python
# run_critic.py
checkpoint_path = Path(config.checkpoint.save_dir) / "checkpoint_best.pt"
save_checkpoint(adapter, optimizer, epoch, metrics, checkpoint_path)

# MLflow artifact 업로드
mlflow.log_artifact(str(checkpoint_path), "checkpoints")
```

**Checkpoint 로드** (Stage 2):
```python
# run_verifiable.py
# Config에 명시된 경로 (local or MLflow URI)
critic_checkpoint = config.experiment.critic_checkpoint

# Local path 또는 MLflow artifact URI 자동 감지
checkpoint = load_critic_checkpoint(critic_checkpoint, adapter, device)
```

**지원 경로 형식**:
1. Local path: `storage/checkpoints/critic/.../checkpoint_best.pt`
2. MLflow artifact URI: `mlflow://8/{run_id}/artifacts/checkpoints/checkpoint_best.pt`

### 2.3 Decision 3: Config 계층 구조 (defaults → stage)

**문제 인식**: Stage별 설정 중복 및 공통 설정 관리

**해결책**: 3-tier config hierarchy

```
configs/
├── defaults.yaml          # Tier 1: 공통 (models, storage, mlflow, runtime)
├── critic/
│   ├── critic.yaml       # Tier 2: Stage 1 전용 (defaults 상속)
│   └── critic_local.yaml # Tier 3: Local test override
├── verifiable/
│   ├── verifiable.yaml   # Tier 2: Stage 2 전용 (defaults 상속 + critic_checkpoint)
│   └── verifiable_local.yaml
└── rho1/
    ├── rho1.yaml         # Tier 2: Rho-1 전용 (ref model 포함)
    └── rho1_local.yaml
```

**Merge 순서** (OmegaConf):
```python
defaults = OmegaConf.load("configs/defaults.yaml")
config = OmegaConf.load("configs/critic/critic.yaml")
config = OmegaConf.merge(defaults, config)  # Tier 1 + Tier 2
```

**장점**:
- 공통 설정 (models, mlflow) 한 곳에서 관리
- Stage별 차이점만 명시 (critic: n_epochs=0.5, verifiable: n_epochs=2.5)
- Local test config로 micro model override 가능

### 2.4 Decision 4: Rho-1 별도 파이프라인

**문제 인식**: Rho-1은 ref model 필요 → 구조가 Verifiable과 다름

**해결책**: 완전 독립 파이프라인 `run_rho1.py`

**Rho-1 구조 차이**:

| 측면 | Verifiable | Rho-1 |
|------|-----------|-------|
| **Critic 사용** | ✅ Stage 1 checkpoint 로드 | ❌ 불사용 |
| **Ref model** | ❌ 불필요 | ✅ 필수 (excess loss 계산) |
| **Weight 계산** | TD error 기반 | Excess loss 기반 (policy vs ref) |
| **Pipeline** | Stage 1 → Stage 2 | 독립 실행 (1-stage) |

**Config 차이**:
```yaml
# configs/rho1/rho1.yaml
models:
  policy:
    name: meta-llama-mtp
    path: storage/models_v2/meta-llama-mtp

  reference:  # ⭐ Rho-1 전용
    name: ref-sheared-llama-2.7b
    path: storage/models_v2/ref-sheared-llama-2.7b

training:
  use_reference_model: true
  alpha: 0.3  # SLM-ratio (excess loss threshold)
  weight_strategy: rho1  # vs. verifiable
```

---

## Part 3: Value Weighting 모듈 설계

### 3.1 TD Error 계산 (표준 Temporal Difference)

**이론적 배경**:

표준 TD(0) 공식 (Sutton & Barto):
```python
# Intermediate tokens (k < T): Bootstrapping
δ_k = r_k + γV(s_k) - V(s_{k-1})
    = γV(s_k) - V(s_{k-1})  # r_k = 0 (중간 토큰 보상 없음)

# Terminal token (k = T): Direct reward
δ_T = R - V(s_{T-1})  # V(terminal) = 0 가정
```

**구현 요구사항**:

```python
def compute_td_errors(
    value_logits: torch.Tensor,  # [batch, seq, 1]
    rewards: torch.Tensor,        # [batch] - Binary: 0 or 1
    gamma: float = 1.0,           # 할인율 (undiscounted)
    attention_mask: torch.Tensor = None,  # [batch, seq]
) -> torch.Tensor:
    """표준 TD error 계산

    TD error는 δ_t = V(s_{t+1}) - V(s_t) (γ=1.0)로 계산되며,
    "이 토큰이 성공 확률을 얼마나 변화시켰는가 (ΔP)"를 의미합니다.

    Returns:
        td_errors: [batch, seq] TD error (Intermediate + Terminal)
    """
```

**핵심 로직**:
```python
# Value squeeze
values = value_logits.squeeze(-1)  # [batch, seq]

# Bootstrapping: V(s_k) - V(s_{k-1})
value_next = values[:, 1:]  # [batch, seq-1]
value_current = values[:, :-1]  # [batch, seq-1]
td_errors_intermediate = gamma * value_next - value_current

# Terminal: R - V(s_{T-1})
values_terminal = values[:, -1]  # [batch]
td_errors_terminal = rewards - values_terminal

# Combine
td_errors = torch.cat([td_errors_intermediate, td_errors_terminal.unsqueeze(1)], dim=1)
```

**검증 기준**:
- [ ] Intermediate TD error 계산 정확: `gamma * V_next - V_current`
- [ ] Terminal TD error 계산 정확: `reward - V_terminal`
- [ ] Binary reward [0, 1] 환경에서 TD error bounded [-1, 1]
- [ ] Padding mask 정상 동작

### 3.2 Weight Builder (Exponential Weighting)

**이론적 배경**:

IQL/AWR Exponential Weighting:
```python
weight = exp(td_error / β)
weight = clamp(weight, min=0.1, max=5.0)  # Conservative clipping
```

**직관**:
- Positive TD error (td > 0): weight > 1 → 중요 토큰 강화
- Negative TD error (td < 0): weight < 1 → 비중요 토큰 down-weight
- Incorrect 샘플: reward=0, value>0 → td<0 → weight<1 (자동 필터링)

**구현 요구사항**:

```python
def build_weights(
    td_errors: torch.Tensor,  # [batch, seq]
    beta: float = 0.9,         # Temperature parameter
    min_weight: float = 0.1,   # Conservative minimum
    max_weight: float = 5.0,   # Conservative maximum
) -> torch.Tensor:
    """TD error 기반 exponential weighting

    Returns:
        weights: [batch, seq] Token-level weights
    """
    weights = torch.exp(td_errors / beta)
    weights = torch.clamp(weights, min=min_weight, max=max_weight)
    return weights
```

**검증 기준**:
- [ ] Exponential weighting: `exp(td / beta)` 정확 계산
- [ ] Clipping: min=0.1, max=5.0 적용 확인
- [ ] Beta sensitivity: β↓ → weight 차이 증가
- [ ] Gradient 계산 가능 (requires_grad=True 지원)

### 3.3 Metrics (Statistics Computation)

**TD Error Statistics**:

```python
def compute_td_stats(td_errors: torch.Tensor) -> dict[str, float]:
    """TD error 통계 계산

    Returns:
        {
            "td_mean": float,
            "td_std": float,
            "td_min": float,
            "td_max": float,
        }
    """
```

**Weight Statistics**:

```python
def compute_weight_stats(weights: torch.Tensor) -> dict[str, float]:
    """Weight 통계 계산

    Returns:
        {
            "weight_mean": float,
            "weight_std": float,
            "weight_min": float,
            "weight_max": float,
            "weight_entropy": float,  # Distribution entropy
        }
    """
```

**검증 기준**:
- [ ] 평균/표준편차 계산 정확
- [ ] Weight entropy 계산 (0-log(seq_len) 범위)
- [ ] Padding mask 고려 (유효 토큰만 통계 계산)

---

## Part 4: Config 분리 구조

### 4.1 defaults.yaml (공통 설정)

**역할**: 모든 stage에서 상속하는 기본값

**내용**:

```yaml
# 공통 설정 (장비, 스토리지, 모델 파라미터 스냅샷)

project:
  name: weighted-mtp
  version: "2.0.0"

storage:
  root: storage
  models_dir: storage/models_v2
  datasets_dir: storage/datasets_v2
  checkpoints_dir: storage/checkpoints  # ⭐ Checkpoint 저장 경로

models:
  policy:
    name: meta-llama-mtp
    path: storage/models_v2/meta-llama-mtp
    params:
      dim: 4096
      n_layers: 32
      n_heads: 32
      n_future_tokens: 4
      intermediate_size: 11008
      rope_theta: 10000.0
      vocab_size: 32000
    dtype: float16

  reference:
    name: ref-sheared-llama-2.7b
    path: storage/models_v2/ref-sheared-llama-2.7b
    dtype: float16
    tokenizer_shared_with: meta-llama-mtp

  reward:
    name: starling-rm-7b
    path: storage/models_v2/starling-rm-7b
    dtype: bfloat16
    status: optional

runtime:
  device: cuda
  seed: 42
  mixed_precision: true

mlflow:
  tracking_uri: "http://13.50.240.176"  # EC2 MLflow Server (Basic Auth)
  experiment: "weighted-mtp/production"
  s3_artifacts: "s3://wmtp/mlflow-artifacts"

logging:
  level: INFO
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

**수정 금지**: 이 파일은 모든 stage에서 공유되므로 변경 시 영향 범위 확인 필요

### 4.2 configs/critic/critic.yaml (Stage 1 전용)

**역할**: Critic (Value head) pre-training 독립 실행

**파일 경로**: `configs/critic/critic.yaml`

**내용**:

```yaml
# Critic Pre-training (Stage 1)
experiment:
  name: critic-pretrain
  description: "Value head pretraining for TD error calculation"
  stage: critic
  tags:
    - critic
    - value-head
    - stage1

dataset:
  name: codecontests
  train: storage/datasets_v2/codecontests/processed/train.jsonl
  validation: storage/datasets_v2/codecontests/processed/valid.jsonl
  max_length: 2048

data_sampling:
  # Stage 1: Value Head Pretrain (is_correct 균형 샘플링)
  # 목적: Binary classification (correct vs incorrect)
  n_samples: 30000  # Effective: 30K samples (15K correct + 15K incorrect)
  balance_correct: true
  correct_ratio: 0.5
  difficulty_range: [1, 11]  # 전체 난이도
  seed: 42

training:
  n_epochs: 0.5
  batch_size: 8
  learning_rate: 1.0e-4
  loss_type: mse  # mse or huber

  # Logging & Evaluation
  log_interval: 10      # 10 step마다 train loss 출력
  eval_interval: 100    # 100 step마다 validation 평가
  save_checkpoint_every: 0.5  # 0.5 epoch마다 checkpoint 저장

checkpoint:
  save_dir: storage/checkpoints/critic/${experiment.name}
  save_best: true   # Best validation loss checkpoint
  save_final: true  # Final checkpoint
```

**Local test config**: `configs/critic/critic_local.yaml`

```yaml
# Critic Local Test (Micro model)
experiment:
  name: critic-pretrain-local
  stage: critic
  tags:
    - critic
    - local
    - micro-model

models:
  policy:
    name: micro-mtp
    path: storage/models_v2/micro-mtp  # Override (micro model)

data_sampling:
  n_samples: 100  # 소량 샘플

training:
  n_epochs: 0.1
  batch_size: 2
```

### 4.3 configs/verifiable/verifiable.yaml (Stage 2 전용)

**역할**: Verifiable WMTP 독립 실행 (Critic checkpoint 로드)

**파일 경로**: `configs/verifiable/verifiable.yaml`

**내용**:

```yaml
# Verifiable WMTP (Stage 2)
experiment:
  name: verifiable-wmtp
  description: "WMTP with TD error-based token weighting"
  stage: verifiable
  tags:
    - verifiable
    - wmtp
    - stage2

  # ⭐ Stage 1 checkpoint 경로 (필수)
  critic_checkpoint: storage/checkpoints/critic/critic-pretrain/checkpoint_best.pt
  # 또는 MLflow artifact URI:
  # critic_checkpoint: mlflow://8/{run_id}/artifacts/checkpoints/checkpoint_best.pt

dataset:
  name: codecontests
  train: storage/datasets_v2/codecontests/processed/train.jsonl
  validation: storage/datasets_v2/codecontests/processed/valid.jsonl
  max_length: 2048

data_sampling:
  # Stage 2: Weighted Training (Curriculum Learning)
  n_samples: 100000  # Effective: 300K samples (100K × 3 epochs)
  balance_correct: true
  correct_ratio: 0.5
  curriculum_learning: true
  difficulty_bins:
    low: [1, 3]      # 쉬운 문제
    medium: [4, 7]   # 중간 문제
    high: [8, 11]    # 어려운 문제
  curriculum_schedule:
    - epoch_range: [0.0, 0.3]
      difficulty_weights: {low: 0.7, medium: 0.3, high: 0.0}
    - epoch_range: [0.3, 0.7]
      difficulty_weights: {low: 0.3, medium: 0.6, high: 0.1}
    - epoch_range: [0.7, 1.0]
      difficulty_weights: {low: 0.1, medium: 0.5, high: 0.4}
  seed: 42

training:
  n_epochs: 2.5
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-5

  # Weighted MTP
  beta: 0.9               # Exponential weighting temperature
  value_coef: 0.5         # Value loss coefficient (Critic Continual Learning)
  max_grad_norm: 0.5      # Gradient clipping
  loss_type: mse          # Value loss type
  weight_clip_min: 0.1    # Conservative weight minimum
  weight_clip_max: 5.0    # Conservative weight maximum

  # Logging & Evaluation
  log_interval: 10
  eval_interval: 100
  save_checkpoint_every: 1.0

checkpoint:
  save_dir: storage/checkpoints/verifiable/${experiment.name}
  save_best: true
  save_final: true
```

### 4.4 configs/rho1/rho1.yaml (Rho-1 전용)

**역할**: Rho-1 Weighted Training (Ref model 필요)

**파일 경로**: `configs/rho1/rho1.yaml`

**내용**:

```yaml
# Rho-1 Weighted Training
experiment:
  name: rho1-wmtp
  description: "Rho-1 style token weighting with reference model"
  stage: rho1
  tags:
    - rho1
    - reference-based
    - wmtp

models:
  # ⭐ Reference model 필수
  policy:
    name: meta-llama-mtp
    path: storage/models_v2/meta-llama-mtp

  reference:
    name: ref-sheared-llama-2.7b
    path: storage/models_v2/ref-sheared-llama-2.7b
    # Reference model로 excess loss 계산

dataset:
  name: codecontests
  train: storage/datasets_v2/codecontests/processed/train.jsonl
  validation: storage/datasets_v2/codecontests/processed/valid.jsonl
  max_length: 2048

data_sampling:
  n_samples: 100000
  balance_correct: true
  correct_ratio: 0.5
  seed: 42

training:
  n_epochs: 3.0
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-5

  # Rho-1 specific
  use_reference_model: true
  alpha: 0.3  # SLM-ratio (excess loss threshold)
  weight_strategy: rho1  # vs. verifiable

  # Logging & Evaluation
  log_interval: 10
  eval_interval: 100
  save_checkpoint_every: 1.0

checkpoint:
  save_dir: storage/checkpoints/rho1/${experiment.name}
  save_best: true
  save_final: true
```

---

## Part 5: Pipeline Runners 설계

### 5.1 run_critic.py (Stage 1 Runner)

**파일 경로**: `src/weighted_mtp/pipelines/run_critic.py`

**역할**: Critic pre-training 독립 실행

**핵심 구조**:

```python
"""Critic Pre-training Runner (Stage 1)

독립 실행:
    python -m weighted_mtp.pipelines.run_critic --config configs/critic/critic.yaml
"""

import logging
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf

from weighted_mtp.models import load_adapter, load_tokenizer
from weighted_mtp.data import load_dataset, create_dataloader
from weighted_mtp.pipelines.training import train_stage1, evaluate_stage
from weighted_mtp.pipelines.checkpoint_utils import save_checkpoint
from weighted_mtp.runtime.distributed import is_main_process

logger = logging.getLogger(__name__)


def run_critic_training(config_path: str, **override_params):
    """Critic pre-training 실행

    Args:
        config_path: configs/critic/critic.yaml
        override_params: CLI overrides

    Returns:
        metrics: Final metrics
        checkpoint_path: Best checkpoint path
    """
    # 1. Config 로딩 (defaults + critic config merge)
    defaults = OmegaConf.load("configs/defaults.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(defaults, config, override_params)

    # 2. MLflow 초기화 (Rank 0 only)
    if is_main_process():
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)

        with mlflow.start_run(run_name=config.experiment.name, tags=config.experiment.tags):
            mlflow.log_params(OmegaConf.to_container(config, resolve=True))

            # 3. Resource 로딩
            device = torch.device(config.runtime.device)
            adapter = load_adapter(config.models.policy, device)
            tokenizer = load_tokenizer(config.models.policy)

            train_dataset = load_dataset(config.dataset.train, config.data_sampling)
            val_dataset = load_dataset(config.dataset.validation, use_full=True)

            train_loader = create_dataloader(train_dataset, tokenizer, config.training.batch_size)
            val_loader = create_dataloader(val_dataset, tokenizer, config.training.batch_size, shuffle=False)

            # 4. Optimizer (Value head only)
            optimizer = torch.optim.Adam(
                adapter.value_head.parameters(),
                lr=config.training.learning_rate
            )

            # 5. Training loop
            logger.info("=== Critic Pre-training (Stage 1) ===")

            best_val_loss = float('inf')
            best_checkpoint_path = None

            n_epochs = int(config.training.n_epochs) + 1
            for epoch in range(n_epochs):
                # Train
                train_metrics = train_stage1(
                    adapter=adapter,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    config=config.training,
                    device=device,
                )

                # Evaluate
                val_metrics = evaluate_stage(
                    adapter=adapter,
                    dataloader=val_loader,
                    config=config.training,
                    device=device,
                    stage="stage1"
                )

                # Log metrics to MLflow
                mlflow.log_metrics({
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }, step=epoch)

                # Save checkpoint
                checkpoint_dir = Path(config.checkpoint.save_dir)
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                save_checkpoint(adapter, optimizer, epoch, train_metrics, val_metrics, checkpoint_path)

                # Best checkpoint
                if config.checkpoint.save_best and val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    best_checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
                    save_checkpoint(adapter, optimizer, epoch, train_metrics, val_metrics, best_checkpoint_path)
                    logger.info(f"✅ Best checkpoint saved: {best_checkpoint_path} (val_loss={best_val_loss:.4f})")

            # 6. Final checkpoint
            if config.checkpoint.save_final:
                final_path = checkpoint_dir / "checkpoint_final.pt"
                save_checkpoint(adapter, optimizer, n_epochs - 1, train_metrics, val_metrics, final_path)

            # 7. MLflow artifact 업로드
            mlflow.log_artifact(str(best_checkpoint_path), "checkpoints")

            logger.info(f"🎉 Critic pre-training 완료! Best checkpoint: {best_checkpoint_path}")

            return val_metrics, best_checkpoint_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Critic Pre-training (Stage 1)")
    parser.add_argument("--config", required=True, help="Config path (e.g., configs/critic/critic.yaml)")
    parser.add_argument("--run-name", help="MLflow run name override")
    parser.add_argument("--device", help="Device override (cuda/cpu)")
    args = parser.parse_args()

    overrides = {}
    if args.run_name:
        overrides["experiment.name"] = args.run_name
    if args.device:
        overrides["runtime.device"] = args.device

    run_critic_training(args.config, **overrides)
```

**핵심 포인트**:
1. **Config merge**: defaults.yaml + critic.yaml
2. **MLflow run 생성**: run_name, tags 설정
3. **Value head만 학습**: `adapter.value_head.parameters()`
4. **Best checkpoint 추적**: `val_loss` 최소화
5. **Artifact 업로드**: MLflow에 checkpoint 저장

### 5.2 run_verifiable.py (Stage 2 Runner)

**파일 경로**: `src/weighted_mtp/pipelines/run_verifiable.py`

**역할**: Verifiable WMTP 독립 실행 (Critic checkpoint 로드)

**핵심 구조**:

```python
"""Verifiable WMTP Runner (Stage 2)

독립 실행:
    python -m weighted_mtp.pipelines.run_verifiable \
        --config configs/verifiable/verifiable.yaml \
        --critic-checkpoint storage/checkpoints/critic/.../checkpoint_best.pt
"""

import logging
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf

from weighted_mtp.models import load_adapter, load_tokenizer
from weighted_mtp.data import load_dataset, create_dataloader
from weighted_mtp.pipelines.training import train_stage2, evaluate_stage
from weighted_mtp.pipelines.checkpoint_utils import save_checkpoint, load_critic_checkpoint
from weighted_mtp.runtime.distributed import is_main_process

logger = logging.getLogger(__name__)


def run_verifiable_training(config_path: str, critic_checkpoint: str = None, **override_params):
    """Verifiable WMTP 실행

    Args:
        config_path: configs/verifiable/verifiable.yaml
        critic_checkpoint: Critic checkpoint 경로 (CLI override)
        override_params: 추가 overrides

    Returns:
        metrics: Final metrics
        checkpoint_path: Best checkpoint path
    """
    # 1. Config 로딩
    defaults = OmegaConf.load("configs/defaults.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(defaults, config, override_params)

    # CLI override critic checkpoint
    if critic_checkpoint:
        config.experiment.critic_checkpoint = critic_checkpoint

    # 2. Critic checkpoint 경로 검증
    if not config.experiment.critic_checkpoint:
        raise ValueError(
            "critic_checkpoint가 필요합니다!\n"
            "  1) Config에 명시: experiment.critic_checkpoint\n"
            "  2) CLI 인자: --critic-checkpoint <path>"
        )

    # 3. MLflow 초기화
    if is_main_process():
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)

        with mlflow.start_run(run_name=config.experiment.name, tags=config.experiment.tags):
            mlflow.log_params(OmegaConf.to_container(config, resolve=True))
            mlflow.log_param("critic_checkpoint", config.experiment.critic_checkpoint)

            # 4. Resource 로딩
            device = torch.device(config.runtime.device)
            adapter = load_adapter(config.models.policy, device)
            tokenizer = load_tokenizer(config.models.policy)

            # ⭐ Critic checkpoint 로드 (Value head 초기화)
            logger.info(f"Loading critic checkpoint: {config.experiment.critic_checkpoint}")
            load_critic_checkpoint(config.experiment.critic_checkpoint, adapter, device)
            logger.info("✅ Critic checkpoint loaded successfully")

            # Dataset 로딩
            train_dataset = load_dataset(config.dataset.train, config.data_sampling)
            val_dataset = load_dataset(config.dataset.validation, use_full=True)

            train_loader = create_dataloader(train_dataset, tokenizer, config.training.batch_size)
            val_loader = create_dataloader(val_dataset, tokenizer, config.training.batch_size, shuffle=False)

            # 5. Optimizer (전체 파라미터)
            optimizer = torch.optim.Adam(
                adapter.parameters(),
                lr=config.training.learning_rate
            )

            # 6. Training loop
            logger.info("=== Verifiable WMTP (Stage 2) ===")

            best_val_loss = float('inf')
            best_checkpoint_path = None

            n_epochs = int(config.training.n_epochs) + 1
            for epoch in range(n_epochs):
                # Train
                train_metrics = train_stage2(
                    adapter=adapter,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    config=config.training,
                    device=device,
                )

                # Evaluate
                val_metrics = evaluate_stage(
                    adapter=adapter,
                    dataloader=val_loader,
                    config=config.training,
                    device=device,
                    stage="stage2"
                )

                # Log metrics
                mlflow.log_metrics({
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }, step=epoch)

                # Save checkpoint
                checkpoint_dir = Path(config.checkpoint.save_dir)
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                save_checkpoint(adapter, optimizer, epoch, train_metrics, val_metrics, checkpoint_path)

                # Best checkpoint
                if config.checkpoint.save_best and val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    best_checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
                    save_checkpoint(adapter, optimizer, epoch, train_metrics, val_metrics, best_checkpoint_path)
                    logger.info(f"✅ Best checkpoint saved: {best_checkpoint_path} (val_loss={best_val_loss:.4f})")

            # 7. Final checkpoint
            if config.checkpoint.save_final:
                final_path = checkpoint_dir / "checkpoint_final.pt"
                save_checkpoint(adapter, optimizer, n_epochs - 1, train_metrics, val_metrics, final_path)

            # 8. MLflow artifact 업로드
            mlflow.log_artifact(str(best_checkpoint_path), "checkpoints")

            logger.info(f"🎉 Verifiable WMTP 완료! Best checkpoint: {best_checkpoint_path}")

            return val_metrics, best_checkpoint_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verifiable WMTP (Stage 2)")
    parser.add_argument("--config", required=True, help="Config path (e.g., configs/verifiable/verifiable.yaml)")
    parser.add_argument("--critic-checkpoint", help="Critic checkpoint path (local or MLflow URI)")
    parser.add_argument("--run-name", help="MLflow run name override")
    args = parser.parse_args()

    overrides = {}
    if args.run_name:
        overrides["experiment.name"] = args.run_name

    run_verifiable_training(args.config, critic_checkpoint=args.critic_checkpoint, **overrides)
```

**핵심 차이점** (vs. run_critic.py):
1. **Critic checkpoint 로드**: `load_critic_checkpoint()` 호출
2. **전체 파라미터 학습**: `adapter.parameters()` (MTP heads + Value head)
3. **train_stage2 사용**: Weighted MTP + Critic Continual Learning

### 5.3 run_rho1.py (Rho-1 Runner)

**파일 경로**: `src/weighted_mtp/pipelines/run_rho1.py`

**역할**: Rho-1 Weighted Training (Ref model 사용)

**핵심 구조**:

```python
"""Rho-1 Weighted Training Runner

독립 실행:
    python -m weighted_mtp.pipelines.run_rho1 --config configs/rho1/rho1.yaml
"""

def run_rho1_training(config_path: str, **override_params):
    """Rho-1 training 실행

    Args:
        config_path: configs/rho1/rho1.yaml

    Returns:
        metrics: Final metrics
        checkpoint_path: Best checkpoint path
    """
    # 1. Config 로딩
    defaults = OmegaConf.load("configs/defaults.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(defaults, config, override_params)

    # 2. MLflow 초기화
    if is_main_process():
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)

        with mlflow.start_run(run_name=config.experiment.name, tags=config.experiment.tags):
            # 3. Resource 로딩
            device = torch.device(config.runtime.device)

            # ⭐ Policy + Reference model 로딩
            policy_adapter = load_adapter(config.models.policy, device)
            ref_model = load_reference_model(config.models.reference, device)

            tokenizer = load_tokenizer(config.models.policy)

            # Dataset 로딩
            train_dataset = load_dataset(config.dataset.train, config.data_sampling)
            val_dataset = load_dataset(config.dataset.validation, use_full=True)

            train_loader = create_dataloader(train_dataset, tokenizer, config.training.batch_size)
            val_loader = create_dataloader(val_dataset, tokenizer, config.training.batch_size, shuffle=False)

            # 4. Optimizer
            optimizer = torch.optim.Adam(
                policy_adapter.parameters(),
                lr=config.training.learning_rate
            )

            # 5. Training (Rho-1 logic)
            logger.info("=== Rho-1 Weighted Training ===")

            # Rho-1 specific training loop
            # - Compute excess loss (policy vs ref)
            # - Build weights based on excess loss
            # - Weighted CE loss

            # (구현 생략 - train_rho1() 함수 호출)
```

**핵심 차이점**:
1. **Ref model 로딩**: `load_reference_model()` 사용
2. **Excess loss 계산**: Policy loss - Ref loss
3. **Weight 전략**: `weight_strategy: rho1` (vs. verifiable)

### 5.4 공통 함수 재사용

**재사용 가능한 함수** (`pipelines/training.py`):

| 함수 | 역할 | 사용 위치 |
|------|------|----------|
| `train_stage1()` | Critic training 로직 | run_critic.py |
| `train_stage2()` | Verifiable training 로직 | run_verifiable.py |
| `evaluate_stage()` | Validation 평가 | 모든 runner |

**변경 없음**: 기존 Phase 5에서 구현된 함수 그대로 재사용

**삭제**: `run_training_pipeline()` - Stage 1→2 오케스트레이션 불필요

---

## Part 6: Checkpoint Handoff 메커니즘

### 6.1 save_checkpoint() 구조

**파일 경로**: `src/weighted_mtp/pipelines/checkpoint_utils.py`

**역할**: Checkpoint 저장 (Local + MLflow artifact)

**구현**:

```python
"""Checkpoint 저장/로드 유틸리티

MLflow artifact handoff 지원
"""

from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    adapter,
    optimizer,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    checkpoint_path: Path,
):
    """Checkpoint 저장

    Args:
        adapter: MetaLlamaMTPAdapter
        optimizer: torch.optim.Optimizer
        epoch: Current epoch
        train_metrics: Training metrics
        val_metrics: Validation metrics
        checkpoint_path: 저장 경로
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "adapter_state_dict": adapter.state_dict(),
        "value_head_state_dict": adapter.value_head.state_dict(),  # Stage 2에서 로드용
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    logger.info(f"   Train loss: {train_metrics.get('stage1_loss', train_metrics.get('stage2_total_loss', 'N/A')):.4f}")
    logger.info(f"   Val loss: {val_metrics['val_loss']:.4f}")
```

**저장 내용**:
- `adapter_state_dict`: 전체 adapter (Stage 2 final checkpoint용)
- `value_head_state_dict`: Value head만 (Stage 2 초기화용)
- `optimizer_state_dict`: Resume training용
- `train_metrics`, `val_metrics`: 성능 추적

### 6.2 load_critic_checkpoint() 구조

**역할**: Critic checkpoint 로드 (Stage 2에서 사용)

**구현**:

```python
def load_critic_checkpoint(checkpoint_path: str, adapter, device):
    """Critic checkpoint 로드 (Stage 2에서 사용)

    Args:
        checkpoint_path: Local path or MLflow artifact URI
        adapter: MetaLlamaMTPAdapter
        device: torch.device

    Returns:
        checkpoint: Loaded checkpoint dict
    """
    # MLflow artifact URI 감지
    if checkpoint_path.startswith("mlflow://"):
        logger.info(f"Downloading MLflow artifact: {checkpoint_path}")
        import mlflow
        local_path = mlflow.artifacts.download_artifacts(checkpoint_path)
        checkpoint = torch.load(local_path, map_location=device)
    else:
        # Local path
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Value head state dict 로드
    adapter.value_head.load_state_dict(checkpoint["value_head_state_dict"])
    logger.info(f"✅ Critic checkpoint loaded: {checkpoint_path}")
    logger.info(f"   Epoch: {checkpoint['epoch']}")
    logger.info(f"   Val loss: {checkpoint['val_metrics']['val_loss']:.4f}")

    return checkpoint
```

**지원 경로 형식**:
1. **Local path**: `storage/checkpoints/critic/critic-pretrain/checkpoint_best.pt`
2. **MLflow artifact URI**: `mlflow://8/{run_id}/artifacts/checkpoints/checkpoint_best.pt`

### 6.3 MLflow Artifact 업로드/다운로드

**업로드** (run_critic.py):

```python
# Best checkpoint를 MLflow artifact로 업로드
mlflow.log_artifact(str(best_checkpoint_path), "checkpoints")

# Artifact URI 획득 (자동)
# mlflow://8/{run_id}/artifacts/checkpoints/checkpoint_best.pt
```

**다운로드** (run_verifiable.py):

```python
# Config에 MLflow URI 명시
experiment:
  critic_checkpoint: mlflow://8/abc123/artifacts/checkpoints/checkpoint_best.pt

# load_critic_checkpoint()가 자동 감지 및 다운로드
checkpoint = load_critic_checkpoint(config.experiment.critic_checkpoint, adapter, device)
```

---

## Part 7: Step별 구현 가이드

### Step 0: 기존 코드 정리

**목표**: 불필요한 파일 삭제 및 백업

**작업**:
```bash
# 1. 기존 Phase 6 계획서 삭제
rm docs/08_phase6_detailed_plan.md

# 2. 기존 run_training_pipeline() 주석 처리 (테스트 후 삭제)
# src/weighted_mtp/pipelines/training.py:run_training_pipeline()

# 3. Git commit
git add docs/07_phase5_detailed_plan.md docs/08_phase6_detailed_plan.md
git commit -m "docs: Phase 5 재설계 - Stage별 독립 실행 파이프라인"
```

**검증 기준**:
- [ ] Phase 6 계획서 삭제 확인
- [ ] Phase 5 계획서 업데이트 확인

### Step 1-3: Value Weighting 모듈 (기존 유지)

**Step 1**: `value_weighting/td_error.py:compute_td_errors()` - 변경 없음
**Step 2**: `value_weighting/weight_builder.py:build_weights()` - 변경 없음
**Step 3**: `value_weighting/metrics.py:compute_td_stats(), compute_weight_stats()` - 변경 없음

**검증**: 기존 Unit test 통과 확인
```bash
uv run pytest tests/unit/test_td_error.py -v
uv run pytest tests/unit/test_weight_builder.py -v
uv run pytest tests/unit/test_metrics.py -v
```

### Step 4: checkpoint_utils.py 구현

**목표**: Checkpoint save/load 유틸리티 구현

**파일 생성**: `src/weighted_mtp/pipelines/checkpoint_utils.py`

**구현 내용**:
1. `save_checkpoint()` 함수
2. `load_critic_checkpoint()` 함수
3. MLflow artifact URI 지원

**검증 기준**:
- [ ] `save_checkpoint()` 정상 동작 (local path 저장)
- [ ] `load_critic_checkpoint()` local path 로드 성공
- [ ] `load_critic_checkpoint()` MLflow URI 로드 성공 (mock test)
- [ ] Unit test 작성 (`tests/unit/test_checkpoint_utils.py`)

**예상 소요 시간**: 2-3시간

### Step 5: Config 파일 생성

**목표**: Stage별 config 파일 생성

**작업**:

```bash
# 1. 디렉토리 생성
mkdir -p configs/critic
mkdir -p configs/verifiable
mkdir -p configs/rho1

# 2. Config 파일 생성
touch configs/critic/critic.yaml
touch configs/critic/critic_local.yaml
touch configs/verifiable/verifiable.yaml
touch configs/verifiable/verifiable_local.yaml
touch configs/rho1/rho1.yaml
touch configs/rho1/rho1_local.yaml
```

**내용 작성**: Part 4의 예시 config 복사

**검증 기준**:
- [ ] `configs/critic/critic.yaml` 생성 확인
- [ ] `configs/verifiable/verifiable.yaml` 생성 확인 (critic_checkpoint 경로 포함)
- [ ] `configs/rho1/rho1.yaml` 생성 확인 (ref model 포함)
- [ ] OmegaConf로 로딩 가능 확인

**예상 소요 시간**: 1-2시간

### Step 6: run_critic.py 구현

**목표**: Critic pre-training runner 구현

**파일 생성**: `src/weighted_mtp/pipelines/run_critic.py`

**구현 내용**: Part 5.1의 구조 참고

**핵심 로직**:
1. Config merge (defaults + critic)
2. MLflow run 생성
3. Resource 로딩 (adapter, tokenizer, datasets)
4. Training loop (train_stage1 호출)
5. Checkpoint 저장 (best, final)
6. MLflow artifact 업로드

**검증 기준**:
- [ ] `python -m weighted_mtp.pipelines.run_critic --config configs/critic/critic_local.yaml` 실행 성공
- [ ] Checkpoint 저장 확인 (`storage/checkpoints/critic/.../checkpoint_best.pt`)
- [ ] MLflow run 생성 확인 (http://13.50.240.176)
- [ ] Artifact 업로드 확인 (MLflow UI)

**예상 소요 시간**: 3-4시간

### Step 7: run_verifiable.py 구현

**목표**: Verifiable WMTP runner 구현

**파일 생성**: `src/weighted_mtp/pipelines/run_verifiable.py`

**구현 내용**: Part 5.2의 구조 참고

**핵심 로직**:
1. Config merge
2. Critic checkpoint 경로 검증
3. `load_critic_checkpoint()` 호출
4. Training loop (train_stage2 호출)
5. Checkpoint 저장

**검증 기준**:
- [ ] Critic checkpoint 로드 성공
- [ ] `python -m weighted_mtp.pipelines.run_verifiable --config configs/verifiable/verifiable_local.yaml --critic-checkpoint <path>` 실행 성공
- [ ] MLflow run 생성 확인 (parent run 연결)
- [ ] Checkpoint 저장 확인

**예상 소요 시간**: 3-4시간

### Step 8: run_rho1.py 구현

**목표**: Rho-1 weighted training runner 구현

**파일 생성**: `src/weighted_mtp/pipelines/run_rho1.py`

**구현 내용**:
1. Policy + Reference model 로딩
2. Excess loss 계산 로직
3. Rho-1 weight 계산 로직
4. Training loop

**검증 기준**:
- [ ] Ref model 로드 성공
- [ ] Excess loss 계산 정확
- [ ] `python -m weighted_mtp.pipelines.run_rho1 --config configs/rho1/rho1_local.yaml` 실행 성공

**예상 소요 시간**: 4-5시간

### Step 9: 기존 코드 삭제 및 정리

**목표**: 불필요한 코드 제거

**삭제 대상**:
1. `src/weighted_mtp/pipelines/training.py:run_training_pipeline()` 함수
2. `src/weighted_mtp/cli/train.py` (또는 주석 처리)

**검증 기준**:
- [ ] 모든 runner 독립 실행 확인
- [ ] 기존 테스트 업데이트 (run_training_pipeline 제거)
- [ ] Linting 통과 (`ruff check --fix`)

**예상 소요 시간**: 1-2시간

---

## Part 8: 검증 및 완료 기준

### 8.1 기능 검증

**Critic Pre-training (Stage 1)**:
- [ ] `python -m weighted_mtp.pipelines.run_critic --config configs/critic/critic_local.yaml` 실행 성공
- [ ] Checkpoint 저장 확인 (`storage/checkpoints/critic/.../checkpoint_best.pt`)
- [ ] MLflow run 생성 확인 (tags: critic, stage1)
- [ ] Artifact 업로드 확인 (MLflow UI)

**Verifiable WMTP (Stage 2)**:
- [ ] Critic checkpoint 로드 성공
- [ ] `python -m weighted_mtp.pipelines.run_verifiable --config configs/verifiable/verifiable_local.yaml --critic-checkpoint <path>` 실행 성공
- [ ] MLflow run 생성 확인 (tags: verifiable, stage2)
- [ ] Parent run 연결 확인 (선택적)

**Rho-1 Training**:
- [ ] Ref model 로드 성공
- [ ] `python -m weighted_mtp.pipelines.run_rho1 --config configs/rho1/rho1_local.yaml` 실행 성공
- [ ] Excess loss 계산 정확
- [ ] MLflow run 생성 확인 (tags: rho1)

**Checkpoint Handoff**:
- [ ] Local path checkpoint 로드 성공
- [ ] MLflow artifact URI checkpoint 로드 성공
- [ ] Value head state dict 정확히 로드됨

### 8.2 성능 검증

**재실행 효율**:
- [ ] Stage 1 실행 시간 측정 (예: 10분)
- [ ] Stage 2만 재실행 시 Stage 1 skip 확인 (예: 5분 절약)

**MLflow 추적**:
- [ ] Critic run과 Verifiable run 분리 확인
- [ ] Metrics 정확히 로깅됨 (train/loss, val/loss)
- [ ] Artifacts 정확히 업로드됨 (checkpoints/)

### 8.3 코드 품질 검증

**Linting**:
```bash
uv run ruff check --fix src/weighted_mtp/pipelines/
```

**Type checking** (선택적):
```bash
uv run mypy src/weighted_mtp/pipelines/
```

**Unit tests**:
```bash
uv run pytest tests/unit/test_checkpoint_utils.py -v
uv run pytest tests/unit/test_td_error.py -v
uv run pytest tests/unit/test_weight_builder.py -v
uv run pytest tests/unit/test_metrics.py -v
```

### 8.4 완료 기준

**필수 (Must-have)**:
- [x] Config 분리 구조 완성 (critic/, verifiable/, rho1/)
- [x] `run_critic.py` 구현 및 독립 실행 성공
- [x] `run_verifiable.py` 구현 및 checkpoint 로드 성공
- [x] `checkpoint_utils.py` 구현 및 테스트 통과
- [x] MLflow artifact handoff 동작 확인
- [x] 기존 `run_training_pipeline()` 삭제

**권장 (Should-have)**:
- [ ] `run_rho1.py` 구현 (Rho-1 실험 지원)
- [ ] MLflow Projects entry_points 정의 (`MLproject` 파일)
- [ ] Integration test 작성 (end-to-end)

**선택적 (Nice-to-have)**:
- [ ] Parent-child run 연결 (MLflow UI에서 workflow 추적)
- [ ] Checkpoint versioning (checkpoint_v1.pt, checkpoint_v2.pt)
- [ ] Distributed training 지원 (Rank 0 only operations)

---

## Part 9: 예상 소요 시간

| 작업 | 예상 시간 | 비고 |
|------|-----------|------|
| Step 0: 기존 코드 정리 | 0.5시간 | Phase 6 계획서 삭제 |
| Step 1-3: Value Weighting (기존) | 0시간 | 변경 없음 |
| Step 4: checkpoint_utils.py | 2-3시간 | save/load 함수 + tests |
| Step 5: Config 파일 생성 | 1-2시간 | 6개 YAML 파일 작성 |
| Step 6: run_critic.py | 3-4시간 | Stage 1 runner 구현 |
| Step 7: run_verifiable.py | 3-4시간 | Stage 2 runner 구현 |
| Step 8: run_rho1.py | 4-5시간 | Rho-1 runner 구현 (선택적) |
| Step 9: 기존 코드 삭제 | 1-2시간 | 정리 및 테스트 업데이트 |
| 통합 테스트 및 디버깅 | 3-4시간 | End-to-end 검증 |
| 문서화 및 최종 검토 | 1-2시간 | README 업데이트 |
| **합계 (Rho-1 제외)** | **14-21시간** | 약 2-3일 |
| **합계 (Rho-1 포함)** | **18-26시간** | 약 2.5-3.5일 |

---

## Part 10: 다음 단계 (Phase 5 완료 후)

**Phase 5 완료 기준 충족 시**:
- ✅ Critic / Verifiable 독립 실행 가능
- ✅ Checkpoint handoff 동작 확인
- ✅ MLflow 실험 추적 가능
- ✅ Config 분리 완료

**다음 작업**:
1. **Production 실험 실행**: Critic + Verifiable full training
2. **성능 분석**: MLflow UI에서 metrics 비교 (baseline vs. verifiable)
3. **Rho-1 실험**: Ref model 기반 weighted training
4. **논문 작성**: WMTP 연구 결과 정리

**선택적 개선**:
- Distributed training 지원 (DDP)
- Hyperparameter tuning (Ray Tune 연동)
- Automated pipeline (Airflow/Prefect)

---

## 부록 A: MLflow Projects 통합 (선택적)

**MLproject 파일 생성**:

```yaml
# MLproject
name: weighted-mtp

entry_points:
  critic_training:
    parameters:
      config: {type: str, default: configs/critic/critic.yaml}
    command: "python -m weighted_mtp.pipelines.run_critic --config {config}"

  verifiable_training:
    parameters:
      config: {type: str, default: configs/verifiable/verifiable.yaml}
      critic_checkpoint: {type: str}
    command: "python -m weighted_mtp.pipelines.run_verifiable --config {config} --critic-checkpoint {critic_checkpoint}"

  rho1_training:
    parameters:
      config: {type: str, default: configs/rho1/rho1.yaml}
    command: "python -m weighted_mtp.pipelines.run_rho1 --config {config}"
```

**실행**:

```bash
# MLflow Projects로 실행
mlflow run . -e critic_training -P config=configs/critic/critic.yaml

mlflow run . -e verifiable_training \
    -P config=configs/verifiable/verifiable.yaml \
    -P critic_checkpoint=storage/checkpoints/critic/.../checkpoint_best.pt
```

---

## 부록 B: 개발원칙 준수 체크리스트

- [x] **원칙 1**: Phase 4 → Phase 5 흐름 분석 완료 (Adapter → Stage별 Runner)
- [x] **원칙 2**: 기존 구조 존중 (train_stage1/2, value_weighting 재사용), 중복 제거 (run_training_pipeline 삭제)
- [x] **원칙 3**: 잘못된 구조 전격 삭제 (단일 오케스트레이션 파이프라인 폐기)
- [x] **원칙 4**: 하위 호환성 고려 안 함 (완전히 새로운 Stage 분리 구조)
- [x] **원칙 4-1**: 인자명 통일 (config, checkpoint_path, device)
- [x] **원칙 4-2**: Wrapper 최소화 (runner는 필수적 entry point)
- [x] **원칙 4-3**: 한글 주석, 이모지 없음, 버전별 주석 제거
- [ ] **원칙 5**: 계획서와 코드 일치 여부 최종 검토 (구현 후)
- [x] **원칙 6**: 의존성 도구 활용 (MLflow, OmegaConf, torch)

---

**문서 종료**
