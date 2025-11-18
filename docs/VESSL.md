# VESSL 실행 가이드

VESSL A100 4-GPU 클러스터에서 분산학습 실행 가이드.

---

## VESSL CLI 설정

### 설치 및 로그인

```bash
# 설치
brew install vessl  # macOS

# 로그인
vessl login

# 현재 설정 확인
vessl whoami
# → Username: ssikssik2
# → Organization: wooshikwon
# → Project: wmtp
```

### Project 설정

```bash
# Organization/Project 설정
vessl configure --organization wooshikwon
vessl configure project wmtp

# Project 목록 확인
vessl project list
```

---

## VESSL Storage 준비

### Volume 생성

```bash
# Models volume
vessl storage create-volume weighted-mtp-models \
  --storage-name vessl-storage \
  --tag weighted-mtp --tag models

# Datasets volume
vessl storage create-volume weighted-mtp-datasets \
  --storage-name vessl-storage \
  --tag weighted-mtp --tag datasets

# Checkpoints volume
vessl storage create-volume weighted-mtp-checkpoints \
  --storage-name vessl-storage \
  --tag weighted-mtp --tag checkpoints

# Volume 목록 확인
vessl storage list-volumes --storage-name vessl-storage
```

### 데이터 업로드

```bash
# 전체 업로드 (models, datasets, checkpoints)
./scripts/03_upload_to_vessl.sh

# Datasets만 업로드
./scripts/04_upload_datasets.sh
```

**경로 형식**:
```
로컬: storage/models/meta-llama-mtp/safetensors/model.safetensors
→ VESSL: volume://vessl-storage/weighted-mtp-models/meta-llama-mtp/safetensors/model.safetensors
```

### 단일 파일 업로드 (주의사항)

**올바른 방법** (디렉토리 경로만, trailing slash 필수):
```bash
vessl storage copy-file \
  "storage/models/meta-llama-mtp/safetensors/model.safetensors" \
  "volume://vessl-storage/weighted-mtp-models/meta-llama-mtp/safetensors/"
```

**잘못된 방법** (파일명까지 포함하면 안 됨):
```bash
# ✗ 파일명이 디렉토리가 됨
vessl storage copy-file \
  "file.txt" \
  "volume://vessl-storage/weighted-mtp-models/file.txt"
```

### 파일 목록 확인

```bash
# Volume 파일 목록
vessl storage list-files volume://vessl-storage/weighted-mtp-models

# 특정 경로
vessl storage list-files volume://vessl-storage/weighted-mtp-models/meta-llama-mtp/
```

---

## 파이프라인 실행

### Baseline MTP

```bash
./scripts/vessl/baseline_4gpu.sh
```

### Critic Pretraining

```bash
./scripts/vessl/critic_4gpu.sh
```

### Verifiable WMTP

```bash
./scripts/vessl/verifiable_4gpu.sh
# Critic checkpoint 경로 입력 프롬프트
```

### Rho-1 WMTP

```bash
./scripts/vessl/rho1_4gpu.sh
```

---

## VESSL YAML 구조

### 기본 템플릿

```yaml
name: weighted-mtp-baseline-4gpu
description: Baseline MTP A100 4-GPU 학습

resources:
  cluster: vessl-gcp-oregon
  preset: gpu-a100-large-spot

image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

import:
  /workspace:
    git:
      url: github.com/wooshikwon/weighted_mtp.git
      ref: main
  /workspace/storage/models: volume://vessl-storage/weighted-mtp-models
  /workspace/storage/datasets: volume://vessl-storage/weighted-mtp-datasets
  /workspace/storage/checkpoints: volume://vessl-storage/weighted-mtp-checkpoints

env:
  MLFLOW_TRACKING_USERNAME: "{{MLFLOW_TRACKING_USERNAME}}"
  MLFLOW_TRACKING_PASSWORD: "{{MLFLOW_TRACKING_PASSWORD}}"
  AWS_ACCESS_KEY_ID: "{{AWS_ACCESS_KEY_ID}}"
  AWS_SECRET_ACCESS_KEY: "{{AWS_SECRET_ACCESS_KEY}}"
  AWS_DEFAULT_REGION: "{{AWS_DEFAULT_REGION}}"

run:
  - workdir: /workspace
    command: |
      # uv 설치
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"

      # 의존성 설치
      uv sync --frozen

      # 4-GPU 분산 학습
      torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        -m weighted_mtp train \
        --config configs/baseline/baseline.yaml
```

### Volume Mount 경로

```
/workspace/
├── storage/
│   ├── models/      → volume://vessl-storage/weighted-mtp-models
│   ├── datasets/    → volume://vessl-storage/weighted-mtp-datasets
│   └── checkpoints/ → volume://vessl-storage/weighted-mtp-checkpoints
├── configs/
├── src/
└── [git repo 파일들]
```

---

## 분산학습 설정

### Torchrun 명령어

```bash
torchrun \
  --nproc_per_node=4 \    # GPU 개수
  --nnodes=1 \             # 노드 개수
  -m weighted_mtp train \
  --config configs/baseline/baseline.yaml
```

### 자동 환경변수

Torchrun이 자동 설정:
- `RANK`: 전체 프로세스 순위 (0-3)
- `LOCAL_RANK`: 노드 내 순위 (0-3)
- `WORLD_SIZE`: 전체 프로세스 개수 (4)
- `MASTER_ADDR`, `MASTER_PORT`: 통신 설정

파이프라인 코드에서 자동 감지:
```python
if "RANK" in os.environ:
    rank, world_size = init_distributed()
else:
    rank, world_size = 0, 1  # 단일 GPU
```

---

## Run 관리

### Run 생성

```bash
# YAML 파일로 실행
vessl run create -f scripts/vessl/baseline_4gpu.yaml
```

### Run 상태 확인

```bash
# Run 목록
vessl run list

# 특정 run 상태
vessl run get <run-id>

# 로그 확인 (실시간)
vessl run logs <run-id> --follow
```

---

## Troubleshooting

### VESSL CLI 오류

**오류**: `Invalid project`
```bash
# 해결: Project 설정
vessl configure project wmtp
```

### Volume 업로드 실패

**오류**: 파일이 잘못된 경로에 업로드됨
```bash
# 원인: Destination에 파일명까지 포함
# 해결: Destination은 디렉토리만, trailing slash 포함
vessl storage copy-file "file.txt" "volume://vessl-storage/models/"
```

**오류**: `timeout` 또는 `connection error`
```bash
# 해결 1: 파일 분할 업로드
# 해결 2: 백그라운드 실행
vessl storage copy-file "large-file.bin" "volume://vessl-storage/models/" &
```

### Run 실패

**로그 확인**:
```bash
vessl run logs <run-id>
```

**일반적인 원인**:
1. Volume mount 실패 → Volume 존재 확인
2. Out of Memory → batch_size 감소
3. Git clone 실패 → GitHub repo 접근 권한
4. 환경변수 누락 → YAML env 섹션 확인

---

## 참고 명령어

### Storage 관리

```bash
# Storage 목록
vessl storage list

# Volume 생성
vessl storage create-volume <name> --storage-name vessl-storage --tag <tag>

# Volume 목록
vessl storage list-volumes --storage-name vessl-storage

# 파일 업로드
vessl storage copy-file <local> volume://<storage>/<volume>/<path>/

# 파일 목록
vessl storage list-files volume://<storage>/<volume>/<path>
```

### Run 관리

```bash
# Run 생성
vessl run create -f <yaml-file>

# Run 목록
vessl run list

# Run 상태
vessl run get <run-id>

# Run 로그
vessl run logs <run-id> --follow
```

---

## 추가 문서

- **YAML Cheatsheet**: `docs/vessl_yaml_cheatsheet.md`
- **공식 문서**: https://docs.vessl.ai/
