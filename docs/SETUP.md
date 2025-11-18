# 환경 설정

Weighted MTP 프로젝트의 로컬 개발 환경 구성 및 데이터 준비 가이드.

---

## 의존성 설치

### uv 패키지 관리자

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치
uv sync

# 개발 의존성 포함
uv sync --dev
```

### 환경변수 설정

프로젝트 루트에 `.env` 파일 생성:

```bash
# AWS S3 (MLflow artifacts)
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
AWS_DEFAULT_REGION=eu-north-1

# HuggingFace
HF_TOKEN=<your-token>

# MLflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<password>
```

---

## 모델 준비

### 자동 설정 스크립트

```bash
# 전체 모델 준비 (다운로드 + 변환 + 검증)
uv run python scripts/setup_models.py --model all --steps all

# 개별 모델
uv run python scripts/setup_models.py --model meta-llama-mtp --steps all
uv run python scripts/setup_models.py --model micro-mtp --steps all
```

### 모델 디렉터리 구조

```
storage/models/
├── meta-llama-mtp/          # Base MTP (25GB)
│   ├── configs/
│   │   └── params.json
│   ├── safetensors/
│   │   ├── model.safetensors
│   │   └── SHA256SUMS
│   ├── tokenizer/
│   │   ├── tokenizer.model
│   │   └── tokenizer_config.json
│   └── metadata.json
├── micro-mtp/               # 로컬 테스트 (177MB)
├── ref-sheared-llama-2.7b/  # Rho-1 reference (10GB)
└── starling-rm-7b/          # Reward model (선택)
```

---

## 데이터셋 준비

### 자동 전처리 스크립트

```bash
# 전체 데이터셋 준비 (다운로드 + 전처리 + 메타데이터 + small + stats)
uv run python scripts/setup_datasets.py --datasets all --steps all

# 개별 데이터셋
uv run python scripts/setup_datasets.py --datasets codecontests --steps all

# 단계별 실행
uv run python scripts/setup_datasets.py --datasets codecontests --steps process
uv run python scripts/setup_datasets.py --datasets codecontests --steps metadata
uv run python scripts/setup_datasets.py --datasets codecontests --steps small
uv run python scripts/setup_datasets.py --steps stats
```

### 데이터셋 구조

```
storage/datasets/
├── codecontests/
│   ├── processed/
│   │   ├── train.jsonl              # 3.7M samples (~15GB)
│   │   ├── train_metadata.json      # is_correct, difficulty (~217MB)
│   │   ├── validation.jsonl
│   │   ├── validation_metadata.json
│   │   └── schema.json
│   └── stats/
│       └── 2025-11-17_summary.json
├── mbpp/
└── humaneval/
```

### 메타데이터 파일 형식

```json
{
  "metadata": [
    {"is_correct": true, "difficulty": 7},
    {"is_correct": false, "difficulty": 2}
  ],
  "stats": {
    "total": 3691981,
    "correct": 1754404,
    "incorrect": 1937577
  }
}
```

---

## 검증

### Storage 무결성 검증

```bash
# 전체 검증 (모델 + 데이터셋)
uv run python scripts/verify_storage.py --check all

# 모델만 검증
uv run python scripts/verify_storage.py --check models

# 데이터셋만 검증
uv run python scripts/verify_storage.py --check datasets

# 검증 리포트 생성
uv run python scripts/verify_storage.py --check all --generate-report
```

### 단위 테스트

```bash
# 전체 테스트
uv run pytest tests/unit/

# 특정 모듈
uv run pytest tests/unit/test_adapter.py
uv run pytest tests/unit/test_datasets.py
```

### 통합 테스트 (DDP 멀티프로세스)

```bash
# DDP 통합 테스트 (torchrun 필요)
PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
torchrun --nproc_per_node=2 --nnodes=1 \
  -m pytest tests/integration/test_ddp_multiprocess.py -v -s

# Checkpoint 동기화 테스트
torchrun --nproc_per_node=2 --nnodes=1 \
  -m pytest tests/integration/test_checkpoint_sync.py -v -s
```

---

## 로컬 실행

### 직접 실행 (단일 파이프라인)

```bash
# Baseline 파이프라인
PYTHONPATH=src python src/weighted_mtp/pipelines/run_baseline.py \
  --config configs/baseline/baseline_local.yaml

# Critic 파이프라인 (로컬)
PYTHONPATH=src python src/weighted_mtp/pipelines/run_critic.py \
  --config configs/critic/critic_local.yaml

# Verifiable 파이프라인
PYTHONPATH=src python src/weighted_mtp/pipelines/run_verifiable.py \
  --config configs/verifiable/verifiable_local.yaml
```

### DDP 분산 학습 (로컬 2-GPU 시뮬레이션)

```bash
# 2-process CPU Gloo backend
PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  src/weighted_mtp/pipelines/run_critic.py \
  --config configs/critic/critic_local.yaml
```

**로컬 설정 특징**:
- `*_local.yaml`: 작은 batch size, 짧은 epoch (0.1)
- Micro 모델: 177MB (4-layer, 512-dim)
- Small 데이터: 100 samples (train), 32 (validation)

---

## Troubleshooting

### uv 설치 오류
```bash
# macOS Homebrew 대안
brew install uv
```

### HuggingFace 다운로드 실패
```bash
# .env 파일의 HF_TOKEN 확인
echo $HF_TOKEN

# 토큰 재설정
huggingface-cli login
```

### 메모리 부족
```bash
# Micro 모델 사용
--use-micro-model

# Batch size 감소
# configs/*_local.yaml에서 batch_size 조정
```

### Safetensors 변환 오류
```bash
# 기존 safetensors 삭제 후 재생성
rm -rf storage/models/*/safetensors/
uv run python scripts/setup_models.py --model all --steps convert
```

---

## 참고 스크립트

- `scripts/setup_models.py`: 모델 준비 통합 스크립트
- `scripts/setup_datasets.py`: 데이터셋 준비 통합 스크립트
- `scripts/verify_storage.py`: 무결성 검증
- `scripts/regenerate_micro_model.py`: Micro 모델 재생성
