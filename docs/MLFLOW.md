# MLflow 연동

MLflow Tracking Server 및 S3 Artifact 연동 가이드.

---

## MLflow 인프라

### Tracking Server

- **URL**: http://13.50.240.176
- **Backend**: PostgreSQL (EC2 내부)
- **Artifact Store**: S3 (s3://wmtp/mlflow-artifacts)
- **인증**: Basic Auth

### 연결 확인

```bash
# Health check
curl -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD" \
  http://13.50.240.176/health

# 웹 UI 접속
# URL: http://13.50.240.176
# Username: wmtp_admin
# Password: (from .env)
```

---

## 환경변수 설정

### 로컬 .env 파일

```bash
# MLflow
MLFLOW_TRACKING_USERNAME=wmtp_admin
MLFLOW_TRACKING_PASSWORD=<your-password>

# AWS S3 (Artifact Store)
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
AWS_DEFAULT_REGION=eu-north-1
```

### Config 파일 설정

`configs/defaults.yaml`:

```yaml
mlflow:
  tracking_uri: "http://13.50.240.176"
  experiment: "weighted-mtp/production"
  s3_artifacts: "s3://wmtp/mlflow-artifacts"
  enable_logging: true
```

실험별 설정 (`configs/verifiable/verifiable.yaml`):

```yaml
mlflow:
  experiment: "weighted-mtp/production"
  run_name: "verifiable-a100-4gpu"
  tags:
    pipeline: "verifiable"
    gpu: "a100"
    nodes: "4"
```

---

## Experiment 구조

```
weighted-mtp/
└── production/
    ├── baseline-mtp
    ├── critic-pretrain
    ├── verifiable-wmtp
    └── rho1-wmtp
```

각 파이프라인은 자동으로 해당 experiment에 run을 생성합니다.

---

## 자동 로깅 항목

### Config Parameters

모든 설정이 flatten되어 자동 로깅:
```python
# 예시
training.learning_rate: 1e-5
training.batch_size: 8
experiment.name: verifiable-a100-4gpu
data_sampling.n_samples: 100000
```

### Metrics

파이프라인별 자동 로깅:

**공통**:
- `train/loss`: 학습 손실
- `train/learning_rate`: 학습률
- `val/loss`: 검증 손실

**Verifiable**:
- `train/td_error_mean`: TD error 평균
- `train/weight_mean`: 가중치 평균
- `train/weight_entropy`: 가중치 엔트로피
- `train/value_loss`: Value head 손실

**Rho-1**:
- `train/excess_loss_mean`: Reference 차이 평균
- `train/weight_sparsity`: 가중치 희소성

### Artifacts

자동 S3 업로드:
- `checkpoints/checkpoint_best.pt`: Best checkpoint
- `checkpoints/checkpoint_final.pt`: Final checkpoint
- `logs/training.log`: 학습 로그

---

## 실행 예시

### 로컬 실행

```bash
# .env 로드
source .env

# 학습 실행 (MLflow 자동 로깅)
uv run python -m weighted_mtp train \
  --config configs/verifiable/verifiable_local.yaml \
  --use-micro-model
```

**로깅 흐름**:
1. Config에서 tracking URI 로드
2. Experiment 생성/선택
3. Run 시작 (run_name 설정)
4. Parameters 로깅
5. 학습 중 metrics 로깅
6. Checkpoint S3 업로드
7. Run 종료

### VESSL 실행

VESSL YAML에서 환경변수 템플릿 사용:

```yaml
env:
  MLFLOW_TRACKING_USERNAME: "{{MLFLOW_TRACKING_USERNAME}}"
  MLFLOW_TRACKING_PASSWORD: "{{MLFLOW_TRACKING_PASSWORD}}"
  AWS_ACCESS_KEY_ID: "{{AWS_ACCESS_KEY_ID}}"
  AWS_SECRET_ACCESS_KEY: "{{AWS_SECRET_ACCESS_KEY}}"
  AWS_DEFAULT_REGION: "{{AWS_DEFAULT_REGION}}"
```

Shell script에서 치환:
```bash
sed -i "s|{{MLFLOW_TRACKING_USERNAME}}|$MLFLOW_TRACKING_USERNAME|g" config.yaml
```

---

## S3 Checkpoint 최적화

### 비동기 업로드

```python
# src/utils/s3_utils.py
upload_to_s3_async(
    checkpoint_path="checkpoint_best.pt",
    mlflow_enabled=True
)
```

**특징**:
- ThreadPoolExecutor로 non-blocking 업로드
- 학습 루프 블로킹 없음
- Best checkpoint 실시간 백업

### 자동 정리

```python
cleanup_s3_checkpoints(
    experiment_id="exp_123",
    run_id="run_456",
    save_total_limit=3
)
```

**보존 정책**:
- `checkpoint_best.pt`: 항상 유지
- `checkpoint_final.pt`: 항상 유지
- 중간 checkpoint: save_total_limit 개수만 유지 (오래된 것부터 삭제)

---

## 분산학습에서의 MLflow

### Rank 0 책임

분산학습 시 **Rank 0만** MLflow 로깅 수행:

```python
if rank == 0 and config.mlflow.enable_logging:
    # MLflow run 시작
    mlflow.start_run(run_name=config.experiment.name)

    # Parameters 로깅
    mlflow.log_params(flatten_dict(config))

    # Metrics 로깅 (학습 중)
    mlflow.log_metric("train/loss", loss.item(), step)

    # Checkpoint 저장 및 S3 업로드
    save_checkpoint("checkpoint_best.pt", model, optimizer)
    upload_to_s3_async("checkpoint_best.pt")

    # Run 종료
    mlflow.end_run()
```

### Gradient Synchronization

Rank 0 이외의 프로세스:
- MLflow 로깅 **수행 안 함**
- Gradient 계산 및 all-reduce만 수행
- Rank 0에서 통합된 checkpoint 저장 대기

---

## Troubleshooting

### 401 Unauthorized

```bash
# .env 파일 확인
cat .env | grep MLFLOW

# Credentials 테스트
curl -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD" \
  http://13.50.240.176/health
```

### S3 업로드 실패

```bash
# AWS credentials 확인
aws s3 ls s3://wmtp/mlflow-artifacts/

# .env 파일의 AWS 설정 확인
cat .env | grep AWS
```

### Run이 보이지 않음

```bash
# Experiment 이름 확인
# configs/defaults.yaml의 mlflow.experiment 확인

# MLflow UI에서 Experiment 필터 확인
```

---

## 참고

- **S3 Utils**: `src/weighted_mtp/utils/s3_utils.py`
- **Pipeline 통합**: 각 `src/weighted_mtp/pipelines/run_*.py`
- **MLflow UI**: http://13.50.240.176
