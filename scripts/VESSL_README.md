# VESSL 실행 가이드

Weighted MTP 프로젝트를 VESSL A100 4-GPU 클러스터에서 실행하기 위한 완전한 가이드입니다.

## 목차

1. [사전 준비](#사전-준비)
2. [실행 순서](#실행-순서)
3. [파이프라인별 실행](#파이프라인별-실행)
4. [모니터링](#모니터링)
5. [Troubleshooting](#troubleshooting)

## 사전 준비

### 1. 환경변수 설정

프로젝트 루트의 `.env` 파일에 다음 환경변수가 설정되어 있어야 합니다:

```bash
# AWS S3 (MLflow artifacts 저장용)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=eu-north-1
S3_BUCKET_NAME=wmtp

# HuggingFace (모델 다운로드용)
HF_TOKEN=your_hf_token

# MLflow (실험 추적용)
MLFLOW_TRACKING_USERNAME=wmtp_admin
MLFLOW_TRACKING_PASSWORD=your_password
```

### 2. VESSL CLI 설치

```bash
# macOS
brew install vessl

# 또는 pip
pip install vessl

# 로그인
vessl login
```

### 3. 필요한 도구 확인

```bash
# Git
git --version

# AWS CLI (선택)
aws --version
```

## 실행 순서

### Step 1: GitHub Push

코드를 GitHub에 push하여 VESSL에서 가져올 수 있도록 합니다.

```bash
./scripts/01_push_to_github.sh
```

**확인 사항:**
- main 브랜치에 최신 코드가 push되었는지
- GitHub repo: `git@github.com:wooshikwon/weighted_mtp.git`

### Step 2: VESSL Storage 준비

VESSL volume을 생성하고 모델/데이터셋을 업로드합니다.

```bash
./scripts/02_setup_vessl_storage.sh
```

**생성되는 volumes:**
- `weighted-mtp-models`: 모델 파일
  - meta-llama-mtp (7B)
  - micro-mtp (테스트용 소형 모델)
  - ref-sheared-llama-2.7b (Rho-1 reference)
  - starling-rm-7b (critic용)
- `weighted-mtp-datasets`: 데이터셋
  - codecontests
  - humaneval
  - mbpp
- `weighted-mtp-checkpoints`: 체크포인트 저장용 (빈 볼륨)

**주의사항:**
- 업로드는 시간이 오래 걸릴 수 있습니다 (수십 분 소요)
- 네트워크 상태에 따라 재시도 필요할 수 있음
- storage/ 디렉토리 크기: 약 50GB

**업로드 확인:**
```bash
vessl volume ls weighted-mtp-models
vessl volume ls weighted-mtp-datasets
vessl volume ls weighted-mtp-checkpoints
```

### Step 3: MLflow/S3 연결 확인

MLflow 서버와 S3 bucket 접근을 확인합니다.

```bash
./scripts/03_verify_mlflow_s3.sh
```

**확인 항목:**
- MLflow health check (http://13.50.240.176)
- MLflow API 접근
- S3 bucket 접근 (s3://wmtp)
- 인증 정보 유효성

**예상 출력:**
```
MLflow Tracking Server: 정상
S3 Bucket: 정상
인증 정보: 유효
```

## 파이프라인별 실행

### Baseline MTP

**특징:**
- 정답만 학습 (uniform weighting)
- 독립 실행 가능
- 학습 시간: ~2.5 epochs

**실행:**
```bash
./scripts/vessl/baseline_4gpu.sh
```

**Config:**
- batch_size: 4 (per GPU)
- gradient_accumulation_steps: 4
- effective_batch: 64 (4 * 4 * 4 GPUs)
- n_epochs: 2.5
- learning_rate: 1e-5

### Critic Pretraining

**특징:**
- Value head pretraining
- Correct/Incorrect 균형 샘플링
- **중요**: Verifiable 파이프라인에서 사용할 checkpoint 생성
- 학습 시간: ~0.5 epochs

**실행:**
```bash
./scripts/vessl/critic_4gpu.sh
```

**Config:**
- batch_size: 8
- gradient_accumulation_steps: 2
- effective_batch: 64
- n_epochs: 0.5
- learning_rate: 1e-4 (higher for critic)
- loss_type: mse

**Checkpoint 저장 위치:**
- `/workspace/storage/checkpoints/critic/critic-pretrain/checkpoint_best.pt`

### Verifiable WMTP

**특징:**
- TD error weighting
- **필수**: Critic checkpoint 필요
- Curriculum learning 옵션
- 학습 시간: ~2.0 epochs

**실행:**
```bash
./scripts/vessl/verifiable_4gpu.sh
```

실행 시 critic checkpoint 경로 입력 프롬프트:
```
Critic checkpoint 경로를 입력하세요 (기본값: storage/checkpoints/critic/critic-pretrain/checkpoint_best.pt):
```

**Config:**
- batch_size: 4
- gradient_accumulation_steps: 4
- effective_batch: 64
- n_epochs: 2.0
- beta: 0.9 (TD error temperature)
- value_coef: 0.5
- weight_clip_min: 0.1, weight_clip_max: 5.0

### Rho-1 WMTP

**특징:**
- Reference model 기반 weighting
- 독립 실행 가능
- 학습 시간: ~2.5 epochs

**실행:**
```bash
./scripts/vessl/rho1_4gpu.sh
```

**Config:**
- batch_size: 4
- gradient_accumulation_steps: 4
- effective_batch: 64
- n_epochs: 2.5
- temperature: 1.0
- k_percent: 0.6 (top 60% loss selection)

## 실행 구성

### 리소스 설정

모든 파이프라인은 동일한 리소스를 사용합니다:

```yaml
resources:
  cluster: vessl-gcp-oregon
  preset: gpu-a100-large-spot
```

- **GPU**: NVIDIA A100 80GB × 4
- **Spot Instance**: 비용 절감 (중단 가능성 있음)
- **Region**: GCP Oregon

### Docker Image

```yaml
image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

- PyTorch 2.5.1
- CUDA 12.4
- cuDNN 9

### Volume Mount 경로

```
/workspace/
  ├── storage/
  │   ├── models_v2/      → volume://weighted-mtp-models
  │   ├── datasets_v2/    → volume://weighted-mtp-datasets
  │   └── checkpoints/    → volume://weighted-mtp-checkpoints
  ├── configs/
  ├── src/
  └── [기타 git repo 파일들]
```

### 분산 학습

모든 파이프라인은 `torchrun`을 사용한 4-GPU 분산 학습:

```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  -m weighted_mtp train \
  --config configs/<stage>/<stage>.yaml
```

## 모니터링

### VESSL 웹 UI

**URL**: https://vessl.ai

**확인 항목:**
- Run 상태 (Running, Completed, Failed)
- 실시간 로그
- GPU 사용률
- 메모리 사용량
- 실행 시간

### MLflow UI

**URL**: http://13.50.240.176

**로그인:**
- Username: `.env`의 `MLFLOW_TRACKING_USERNAME`
- Password: `.env`의 `MLFLOW_TRACKING_PASSWORD`

**Experiment 구조:**
```
weighted-mtp/
  └── production/
      ├── baseline-mtp
      ├── critic-pretrain
      ├── verifiable-wmtp
      └── rho1-wmtp
```

**주요 메트릭:**
- `train/loss`: 학습 loss
- `val/loss`: 검증 loss (best checkpoint 기준)
- `train/learning_rate`: Learning rate schedule
- Stage별 특수 메트릭:
  - Critic: `value_head_loss`, `value_norm`
  - Verifiable: `td_error_mean`, `weight_entropy`
  - Rho-1: `excess_loss`, `weighted_ce_loss`

### CLI 로그 확인

```bash
# 실행 중인 run 리스트
vessl run list

# 특정 run 로그 (실시간)
vessl run logs <run-id> --follow

# 특정 run 상태
vessl run get <run-id>
```

## Config Override

VESSL 실행 시 config 필드를 override하려면 YAML 파일의 `run` 섹션 명령어에 `--override` 추가:

```yaml
run:
  - workdir: /workspace
    command: |
      torchrun -m weighted_mtp train \
        --config configs/baseline/baseline.yaml \
        --override training.batch_size=8 \
        --override experiment.name=custom-run \
        --override data_sampling.n_samples=50000
```

**또는** Shell script 수정:
```bash
# scripts/vessl/baseline_4gpu.yaml의 command 섹션 수정
torchrun -m weighted_mtp train \
  --config configs/baseline/baseline.yaml \
  --override training.learning_rate=5e-5
```

## Troubleshooting

### 1. Volume 업로드 실패

**증상**: `vessl volume upload` 시 timeout 또는 connection error

**해결:**
```bash
# Volume 상태 확인
vessl volume ls

# Volume 삭제 후 재생성
vessl volume delete weighted-mtp-models
vessl volume create weighted-mtp-models

# 개별 파일 업로드 (큰 파일 분리)
cd storage/models_v2/meta-llama-mtp
vessl volume upload weighted-mtp-models . meta-llama-mtp/
```

### 2. MLflow 연결 실패

**증상**: `ConnectionRefusedError` 또는 `401 Unauthorized`

**해결:**
```bash
# Health check
curl -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD" \
  http://13.50.240.176/health

# .env 파일 확인
cat .env | grep MLFLOW

# MLflow 서버 상태 확인 (EC2 접속 필요)
ssh -i <key.pem> ubuntu@13.50.240.176
sudo systemctl status mlflow
```

### 3. S3 접근 실패

**증상**: `NoCredentialsError` 또는 `AccessDenied`

**해결:**
```bash
# Credentials 확인
aws s3 ls s3://wmtp/

# .env 파일 확인
cat .env | grep AWS

# IAM 권한 확인 (AWS Console)
# - s3:ListBucket
# - s3:GetObject
# - s3:PutObject
```

### 4. VESSL Run 실패

**증상**: Run이 `Failed` 상태

**해결:**
```bash
# 로그 확인
vessl run logs <run-id>

# 일반적인 원인:
# 1. Volume mount 실패 → Volume 존재 확인
# 2. Out of Memory → batch_size 감소
# 3. Git clone 실패 → GitHub repo 접근 권한 확인
# 4. 환경변수 누락 → YAML의 env 섹션 확인
```

### 5. GPU Out of Memory

**증상**: `CUDA out of memory` 에러

**해결:**
```bash
# Config override로 batch size 감소
--override training.batch_size=2 \
--override training.gradient_accumulation_steps=8

# 또는 YAML 파일 직접 수정
# configs/baseline/baseline.yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

### 6. Checkpoint 로드 실패

**증상**: `FileNotFoundError: checkpoint not found`

**해결:**
```bash
# Checkpoint volume 확인
vessl volume ls weighted-mtp-checkpoints

# Verifiable의 경우: Critic checkpoint 경로 확인
ls /workspace/storage/checkpoints/critic/critic-pretrain/

# Override로 경로 수정
--override experiment.critic_checkpoint=<correct-path>
```

## 비용 최적화

### Spot Instance 사용

현재 설정은 이미 Spot instance 사용 (`gpu-a100-large-spot`):
- 비용: On-demand 대비 60-70% 절감
- 단점: 중단 가능성 (하지만 checkpoint 저장으로 재개 가능)

### Checkpoint 전략

```yaml
checkpoint:
  save_checkpoint_every: 1.0  # 1 epoch마다 저장
  save_best: true             # Best checkpoint 자동 저장
  save_total_limit: 3         # 최대 3개만 보관
```

### 실행 시간 제한

장시간 실행 방지:
```bash
# Config override로 epoch 감소
--override training.n_epochs=1.0
```

## 참고 문서

- [VESSL 공식 문서](https://docs.vessl.ai/)
- [VESSL 실행 가이드](../docs/vessl_execution_guide.md)
- [VESSL YAML Cheatsheet](../docs/vessl_yaml_cheatsheet.md)
- [Config 사용법](../configs/README.md)
- [MLflow 사용법](https://mlflow.org/docs/latest/index.html)
