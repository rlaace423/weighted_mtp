# VESSL 실행 가이드

Weighted MTP 프로젝트를 VESSL A100 4-GPU 클러스터에서 실행하기 위한 완전한 가이드입니다.

최종 업데이트: 2025-11-17 (Phase 8 완료 후)

---

## 1. 개요

### 지원 파이프라인
- **Baseline MTP**: 균등 가중치 학습
- **Critic Pretraining**: Value head 사전학습
- **Verifiable WMTP**: TD error 기반 가중치 학습
- **Rho-1 WMTP**: Reference model 기반 가중치 학습

### 인프라
- **MLflow Tracking**: http://13.50.240.176 (EC2 + PostgreSQL backend)
- **Artifact Store**: S3 (s3://wmtp/mlflow-artifacts)
- **VESSL Storage**: vessl-storage (organization: wooshikwon, project: wmtp)
- **GPU**: NVIDIA A100 80GB × 4

---

## 2. VESSL CLI 설정

### 2.1 CLI 설치 및 로그인

```bash
# 설치
brew install vessl  # macOS
# 또는
pip install vessl

# 로그인
vessl login

# 현재 설정 확인
vessl whoami
```

**출력 예시:**
```
Username: ssikssik2
Email: wonwooshik@gmail.com
Default organization: wooshikwon
Default project: wmtp
```

### 2.2 Organization 및 Project 설정

```bash
# Organization 리스트 확인
vessl organization list

# Project 리스트 확인
vessl project list

# 기본 organization/project 변경 (필요시)
vessl configure --organization wooshikwon
vessl configure project wmtp
```

---

## 3. VESSL Storage 준비

### 3.1 Storage 명령어

#### Storage 목록 확인
```bash
vessl storage list
```

**출력 예시:**
```
Name           IsVESSLManaged  Path  Created                    Updated
vessl-storage  True                  2024-10-25 05:06:54+00:00  2024-10-25 05:06:54+00:00
```

#### Volume 생성
```bash
# Models volume
vessl storage create-volume weighted-mtp-models \
  --storage-name vessl-storage \
  --tag weighted-mtp \
  --tag models

# Datasets volume
vessl storage create-volume weighted-mtp-datasets \
  --storage-name vessl-storage \
  --tag weighted-mtp \
  --tag datasets

# Checkpoints volume (빈 볼륨)
vessl storage create-volume weighted-mtp-checkpoints \
  --storage-name vessl-storage \
  --tag weighted-mtp \
  --tag checkpoints
```

#### Volume 목록 확인
```bash
vessl storage list-volumes --storage-name vessl-storage
```

**출력 예시:**
```
Name                      Updated                    Tags
weighted-mtp-models       2025-11-17 14:21:33+00:00  ['models', 'weighted-mtp']
weighted-mtp-datasets     2025-11-17 14:21:40+00:00  ['datasets', 'weighted-mtp']
weighted-mtp-checkpoints  2025-11-17 14:21:48+00:00  ['checkpoints', 'weighted-mtp']
```

### 3.2 데이터 업로드

#### 단일 파일 업로드
```bash
vessl storage copy-file \
  <local-file-path> \
  volume://vessl-storage/weighted-mtp-models/<destination-path>
```

#### 디렉토리 업로드 (스크립트 사용)
```bash
./scripts/02_setup_vessl_storage.sh
```

스크립트는 다음을 자동으로 수행합니다:
1. Volume 생성 (이미 존재하면 스킵)
2. storage/models/ 디렉토리의 모든 파일 업로드
3. storage/datasets/ 디렉토리의 모든 파일 업로드

**업로드 형식:**
```
로컬: storage/models/meta-llama-mtp/model.safetensors
→ VESSL: volume://vessl-storage/weighted-mtp-models/meta-llama-mtp/model.safetensors
```

#### Volume 파일 목록 확인
```bash
vessl storage list-files \
  --storage-name vessl-storage \
  --volume-name weighted-mtp-models
```

---

## 4. 환경변수 설정

### 4.1 로컬 .env 파일

프로젝트 루트의 `.env` 파일에서 환경변수를 설정합니다.

**경로**: `/Users/wesley/Desktop/wooshikwon/weighted_mtp/.env`

**필수 환경변수**:
```bash
# AWS S3 (MLflow artifacts)
AWS_ACCESS_KEY_ID=<your-aws-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-aws-secret-access-key>
AWS_DEFAULT_REGION=<your-aws-region>
S3_BUCKET_NAME=<your-s3-bucket-name>

# HuggingFace
HF_TOKEN=<your-huggingface-token>

# MLflow
MLFLOW_TRACKING_USERNAME=<your-mlflow-username>
MLFLOW_TRACKING_PASSWORD=<your-mlflow-password>
```

**참고**: 실제 값은 프로젝트 루트의 `.env` 파일을 참조하세요.

### 4.2 VESSL YAML 환경변수

YAML 파일에서 템플릿 사용:
```yaml
env:
  MLFLOW_TRACKING_USERNAME: "{{MLFLOW_TRACKING_USERNAME}}"
  MLFLOW_TRACKING_PASSWORD: "{{MLFLOW_TRACKING_PASSWORD}}"
  AWS_ACCESS_KEY_ID: "{{AWS_ACCESS_KEY_ID}}"
  AWS_SECRET_ACCESS_KEY: "{{AWS_SECRET_ACCESS_KEY}}"
  AWS_DEFAULT_REGION: "{{AWS_DEFAULT_REGION}}"
  HF_TOKEN: "{{HF_TOKEN}}"
```

Shell script에서 치환:
```bash
sed -i "s|{{MLFLOW_TRACKING_USERNAME}}|$MLFLOW_TRACKING_USERNAME|g" config.yaml
```

---

## 5. VESSL Run 실행

### 5.1 Run 명령어

#### YAML 파일로 실행
```bash
vessl run create -f <yaml-file>
```

#### 실행 상태 확인
```bash
# Run 목록
vessl run list

# 특정 run 상태
vessl run get <run-id>

# 로그 확인 (실시간)
vessl run logs <run-id> --follow
```

### 5.2 파이프라인별 실행

#### Baseline MTP
```bash
./scripts/vessl/baseline_4gpu.sh
```

#### Critic Pretraining
```bash
./scripts/vessl/critic_4gpu.sh
```

#### Verifiable WMTP (Critic checkpoint 필요)
```bash
./scripts/vessl/verifiable_4gpu.sh
# 실행 시 checkpoint 경로 입력 프롬프트
```

#### Rho-1 WMTP
```bash
./scripts/vessl/rho1_4gpu.sh
```

---

## 6. VESSL YAML 구조

### 6.1 기본 구조

```yaml
name: weighted-mtp-baseline-4gpu
description: Baseline MTP A100 4-GPU 프로덕션 학습
tags:
  - weighted-mtp
  - baseline
  - a100
  - 4gpu

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
  # ... (환경변수)

run:
  - workdir: /workspace
    command: |
      set -e
      # uv 설치
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"

      # 의존성 설치
      uv sync --frozen

      # 학습 실행 (4-GPU distributed)
      torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        -m weighted_mtp train \
        --config configs/baseline/baseline.yaml
```

### 6.2 Volume Mount 경로

VESSL에서의 경로 구조:
```
/workspace/
  ├── storage/
  │   ├── models/          → volume://vessl-storage/weighted-mtp-models
  │   ├── datasets/        → volume://vessl-storage/weighted-mtp-datasets
  │   └── checkpoints/     → volume://vessl-storage/weighted-mtp-checkpoints
  ├── configs/
  ├── src/
  └── [기타 git repo 파일들]
```

Config 파일에서 사용하는 경로:
```yaml
models:
  policy:
    path: storage/models/meta-llama-mtp  # /workspace/storage/models/meta-llama-mtp

dataset:
  train: storage/datasets/codecontests/processed/train.jsonl
```

---

## 7. 분산 학습 설정

### 7.1 Torchrun 명령어

```bash
torchrun \
  --nproc_per_node=4 \     # GPU 개수
  --nnodes=1 \              # 노드 개수
  --node_rank=0 \           # 현재 노드 순위
  -m weighted_mtp train \
  --config configs/baseline/baseline.yaml
```

### 7.2 환경변수 (자동 설정)

Torchrun이 자동으로 설정하는 환경변수:
- `RANK`: 전체 프로세스 순위 (0-3)
- `LOCAL_RANK`: 노드 내 프로세스 순위 (0-3)
- `WORLD_SIZE`: 전체 프로세스 개수 (4)
- `MASTER_ADDR`: Master 주소 (기본값: localhost)
- `MASTER_PORT`: Master 포트 (기본값: 29500)

파이프라인 코드에서 자동 감지:
```python
if "RANK" in os.environ:
    rank, world_size = init_distributed()
else:
    rank, world_size = 0, 1  # 단일 GPU
```

---

## 8. MLflow 연동

### 8.1 Tracking Server

- **URL**: http://13.50.240.176
- **Backend**: PostgreSQL (EC2 내부)
- **Artifact Store**: S3 (s3://wmtp/mlflow-artifacts)

### 8.2 인증

```bash
# Health check
curl -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD" \
  http://13.50.240.176/health

# 웹 UI 접속
# URL: http://13.50.240.176
# Username: wmtp_admin
# Password: (from .env)
```

### 8.3 Experiment 구조

```
weighted-mtp/
  └── production/
      ├── baseline-mtp
      ├── critic-pretrain
      ├── verifiable-wmtp
      └── rho1-wmtp
```

Config 파일 설정:
```yaml
mlflow:
  tracking_uri: "http://13.50.240.176"
  experiment: "weighted-mtp/production"
  s3_artifacts: "s3://wmtp/mlflow-artifacts"
```

---

## 9. Troubleshooting

### 9.1 VESSL CLI 오류

**오류**: `Invalid project`
```bash
# 해결: Project 설정
vessl configure project wmtp
```

**오류**: `Missing option '--storage-name'`
```bash
# 해결: storage-name 추가
vessl storage list-volumes --storage-name vessl-storage
```

### 9.2 Volume 업로드 실패

**오류**: `timeout` 또는 `connection error`
```bash
# 해결: 파일을 분할하여 업로드
# 또는 VESSL 웹 UI에서 수동 업로드
```

### 9.3 MLflow 연결 실패

**오류**: `401 Unauthorized`
```bash
# 해결: .env 파일 확인
cat .env | grep MLFLOW

# Credentials 테스트
curl -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD" \
  http://13.50.240.176/health
```

### 9.4 Run 실패

**로그 확인:**
```bash
vessl run logs <run-id>
```

**일반적인 원인:**
1. Volume mount 실패 → Volume 존재 확인
2. Out of Memory → batch_size 감소
3. Git clone 실패 → GitHub repo 접근 권한
4. 환경변수 누락 → YAML env 섹션 확인

---

## 10. 참고 명령어 요약

### Storage 관리
```bash
# Storage 목록
vessl storage list

# Volume 생성
vessl storage create-volume <name> --storage-name vessl-storage --tag <tag>

# Volume 목록
vessl storage list-volumes --storage-name vessl-storage

# 파일 업로드
vessl storage copy-file <local> volume://<storage>/<volume>/<path>

# 파일 목록
vessl storage list-files --storage-name <storage> --volume-name <volume>

# Volume 삭제
vessl storage delete-volume <name> --storage-name vessl-storage
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

### 설정
```bash
# 현재 설정 확인
vessl whoami

# Organization 목록
vessl organization list

# Project 목록
vessl project list

# 기본 설정 변경
vessl configure --organization <org>
vessl configure project <project>
```

---

## 11. 추가 문서

- **전체 실행 가이드**: `scripts/VESSL_README.md`
- **YAML Cheatsheet**: `docs/vessl_yaml_cheatsheet.md`
- **Config 사용법**: `configs/README.md`
- **공식 문서**: https://docs.vessl.ai/
