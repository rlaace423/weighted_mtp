# Weighted Multi-Token Prediction (WMTP)

Meta LLaMA MTP 네이티브 파이프라인을 사용하는 Weighted MTP 구현.

## 프로젝트 구조

```
weighted_mtp/
├── vendor/meta_llama/        # Meta 레퍼런스 코드
├── src/weighted_mtp/         # 프로젝트 소스
│   ├── pipelines/            # 4개 학습 파이프라인
│   ├── models/               # Pure PyTorch Transformer
│   ├── data/                 # 메타데이터 기반 데이터 로딩
│   └── value_weighting/      # TD error/Rho-1 weighting
├── configs/                  # 설정 파일
│   ├── baseline/             # Baseline MTP
│   ├── critic/               # Critic pre-training
│   ├── verifiable/           # Verifiable WMTP
│   └── rho1/                 # Rho-1 WMTP
├── storage/                  # 모델 및 데이터 자산
├── tests/                    # 테스트
├── scripts/                  # 유틸리티 스크립트
└── docs/                     # 문서
```

## 빠른 시작

```bash
# 의존성 설치
uv pip install -e ".[dev]"

# 로컬 테스트 (Micro 모델)
PYTHONPATH=src python src/weighted_mtp/pipelines/run_baseline.py \
  --config configs/baseline/baseline_local.yaml

# 테스트 실행
pytest tests/unit/
pytest tests/integration/  # DDP 테스트는 torchrun 필요
```

## 파이프라인

4개의 독립 실행 가능한 학습 파이프라인:

### 1. Critic Pre-training (Stage 1)
Value head를 사전 학습하여 TD error 계산 기반 마련:

```bash
# VESSL 4-GPU 실행
bash scripts/vessl/critic_4gpu.sh

# 로컬 실행
PYTHONPATH=src python src/weighted_mtp/pipelines/run_critic.py \
  --config configs/critic/critic.yaml
```

**설정 (`configs/critic/critic.yaml`)**:
```yaml
training:
  n_epochs: 1.0
  batch_size: 8  # Per GPU
  learning_rate: 1.0e-4
  loss_type: mse
```

### 2. Baseline MTP
표준 MTP (균등 가중치, 정답만 학습):

```bash
PYTHONPATH=src python src/weighted_mtp/pipelines/run_baseline.py \
  --config configs/baseline/baseline.yaml
```

**설정 (`configs/baseline/baseline.yaml`)**:
```yaml
data_sampling:
  correct_ratio: 1.0  # 정답만
  curriculum_learning: true

training:
  n_epochs: 3.0
  learning_rate: 1.0e-5
```

### 3. Verifiable WMTP (Stage 2)
TD error 기반 토큰 가중치 + Value head continual learning:

```bash
PYTHONPATH=src python src/weighted_mtp/pipelines/run_verifiable.py \
  --config configs/verifiable/verifiable.yaml
```

**설정 (`configs/verifiable/verifiable.yaml`)**:
```yaml
data_sampling:
  balance_correct: true
  correct_ratio: 0.5  # 정답/오답 50:50
  curriculum_learning: true

training:
  beta: 0.9              # TD error temperature
  value_coef: 0.5        # Value loss coefficient
  weight_clip_min: 0.1   # 최소 토큰 가중치
  weight_clip_max: 5.0   # 최대 토큰 가중치
  loss_type: mse
```

**핵심 메커니즘**:
- TD error 기반 토큰 가중치: `weight = exp(td_error / beta)`
- Value head continual learning으로 TD error 정확도 유지
- 정답/오답 균형 샘플링으로 robustness 향상

### 4. Rho-1 WMTP
Reference 모델 기반 excess loss weighting:

```bash
PYTHONPATH=src python src/weighted_mtp/pipelines/run_rho1.py \
  --config configs/rho1/rho1.yaml
```

**설정 (`configs/rho1/rho1.yaml`)**:
```yaml
models:
  reference:
    name: ref-sheared-llama-2.7b
    path: storage/models/ref-sheared-llama-2.7b

training:
  temperature: 1.0  # Excess loss temperature
  k_percent: 0.6    # Top-k selection threshold
```

## 평가

```bash
# HumanEval 평가
PYTHONPATH=src python src/weighted_mtp/pipelines/run_evaluation.py \
  --checkpoint storage/checkpoints/verifiable/checkpoint_best.pt \
  --model-path storage/models/meta-llama-mtp \
  --dataset humaneval \
  --n-samples 1 \
  --max-length 2048

# MBPP 평가
PYTHONPATH=src python src/weighted_mtp/pipelines/run_evaluation.py \
  --checkpoint storage/checkpoints/baseline/checkpoint_best.pt \
  --model-path storage/models/meta-llama-mtp \
  --dataset mbpp \
  --n-samples 1
```

## 문서

### 핵심 문서
- [ARCHITECTURE.md](docs/ARCHITECTURE.md): 코드베이스 아키텍처 및 핵심 구현
- [SETUP.md](docs/SETUP.md): 환경 설정 및 데이터 준비
- [VESSL.md](docs/VESSL.md): VESSL A100 4-GPU 실행 가이드
- [MLFLOW.md](docs/MLFLOW.md): MLflow 추적 및 S3 연동
- [RESEARCH.md](docs/RESEARCH.md): 연구 배경 및 이론

### 참조 문서
- [VESSL_CHEATSHEET.md](docs/VESSL_CHEATSHEET.md): VESSL YAML 참조
- [gpu_parallel_testing_summary.md](docs/gpu_parallel_testing_summary.md): DDP 테스팅 결과 (Phase 1-5)
- [data_loading_strategy.md](docs/data_loading_strategy.md): Rank-aware 샘플링 전략

## 라이선스

MIT License