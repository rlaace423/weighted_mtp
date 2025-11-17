# MLflow 통합 가이드 (리팩토링 버전)

본 문서는 `docs/00_ideal_structure.md`와 `docs/02_implementation_plan.md`에서 정의한 아키텍처에 맞추어, 
기존 WMTP EC2 MLflow 서버 + S3 아티팩트 구성을 **동일하게 재사용**하면서 리팩토링된 `weighted_mtp/` 프로젝트에 통합하는 절차를 상세히 설명한다.

---

## 1. 개요

- **재사용 자산**
  - EC2 MLflow Tracking Server (nginx+Basic Auth, PostgreSQL backend)
  - S3 Bucket `s3://wmtp/mlflow-artifacts/` (artifact store)
  - 기존 `.env`/Secrets 관리 방식 (묵시적 환경변수 주입)
- **리팩토링 적용 지점**
  - 파이프라인 Stage 3(평가/로깅)에서 MLflow 로깅
  - `src/runtime/mlflow.py` → MLflow 세션 초기화
  - `configs/defaults.yaml` → MLflow 설정 섹션
  - `scripts/sync_to_vessl_storage.py` → VESSL 환경에서도 동일한 tracking URI 사용

---

## 2. 시스템 아키텍처 (리팩토링 반영)

```
weighted_mtp/
    ├─ src/pipelines/training.py      # Stage 0~3 orchestrator
    ├─ src/runtime/mlflow.py          # MLflow session helper
    ├─ src/value_weighting/...        # TD error/weight 계산 모듈
    ├─ storage/                       # 모델/데이터 staging (v2 구조)
    └─ scripts/sync_to_vessl_storage.py

학습 환경 (로컬 M3 / VESSL GPU / CI)
    ↓  HTTP(S)
EC2 MLflow Tracking Server (기존 구성)
    ├─ PostgreSQL (run metadata)
    └─ S3 wmtp/mlflow-artifacts/ (artifacts)
```

> **핵심 변화**: 기존 `wmtp/` 프로젝트와 동일한 서버/버킷을 사용하되, 리팩토링된 파이프라인에서 경량 config와 storage v2 구조를 통해 접근한다.

---

## 3. Phase별 요구사항 정리

| Phase (implementation_plan) | MLflow 관련 작업 |
|-----------------------------|------------------|
| **P1 (storage 자산 변환)** | 모델/데이터 변환 후 MLflow artifact path에 반영할 버전명 정리 |
| **P2~P5 (코드/파이프라인)** | `src/runtime/mlflow.py`, Stage 3 로깅 로직 구현 및 테스트 |
| **P6 (CLI & Config)** | `configs/defaults.yaml`의 `mlflow` 섹션 업데이트, `.env` 로딩 확인 |
| **P7 (테스트 & 검증)** | CI용 `config.ci_test.yaml`에서 `file://` backend 테스트 |
| **P8 (배포 & VESSL)** | VESSL Secrets 설정, `scripts/sync_to_vessl_storage.py` dry-run 이후 실운영 |
| **P9 (문서화)** | 본 가이드 및 `docs/migration_notes.md`에 최종 확인 사항 기록 |

---

## 4. 설정 파일 구조

### 4.1 `configs/defaults.yaml` (예시)
```yaml
mlflow:
  tracking_uri: "http://<EC2_PUBLIC_IP>"   # 기존 서버
  experiment_prefix: "weighted_mtp"        # 실험 네임스페이스
  artifact_root: null                      # 서버 default 사용
  enable_logging: true
runtime:
  mlflow:
    auto_log_metrics: true
    log_stage1_metrics: true
```

### 4.2 recipe 파일 (예: `configs/recipe.verifiable.yaml`)
```yaml
run:
  name: "verifiable_critic_v1"
  tags:
    - "algo:verifiable"
    - "env:vessl"
mlflow:
  experiment: "weighted_mtp/verifiable"
  tags:
    algo: "verifiable"
    horizon: 4
```

> `experiment_prefix` + recipe별 `experiment` 조합으로 최종 experiment 이름이 결정된다.

---

## 5. 환경 변수 & Secrets

| 변수 | 용도 | 설정 위치 |
|------|------|-----------|
| `MLFLOW_TRACKING_URI` | (선택) tracking URI override | 로컬 shell / VESSL env |
| `MLFLOW_TRACKING_USERNAME` | Basic Auth | `.env`, VESSL Secrets |
| `MLFLOW_TRACKING_PASSWORD` | Basic Auth | `.env`, VESSL Secrets |
| `AWS_ACCESS_KEY_ID` | S3 artifact access | `.env`, VESSL Secrets |
| `AWS_SECRET_ACCESS_KEY` | S3 artifact access | `.env`, VESSL Secrets |
| `AWS_DEFAULT_REGION` | S3 region | `.env`, VESSL Secrets |

`.env` 유지 예시:
```bash
MLFLOW_TRACKING_USERNAME=<USER>
MLFLOW_TRACKING_PASSWORD=<PASS>
MLFLOW_TRACKING_URI=http://<EC2_PUBLIC_IP>
AWS_ACCESS_KEY_ID=<KEY>
AWS_SECRET_ACCESS_KEY=<SECRET>
AWS_DEFAULT_REGION=eu-north-1
```

---

## 6. 코드 통합 포인트

### 6.1 `src/runtime/mlflow.py` (예상 구조)
- `create_mlflow_manager(config_dict: dict) -> MLflowManager`
- 기능:
  - `.env` 로딩
  - tracking URI 설정
  - experiment 생성/선택
  - run start/stop 관리
  - stage별 metrics/artifact 기록

### 6.2 `src/pipelines/training.py`
- Phase 0: MLflow 초기화 (`runtime.mlflow.create_mlflow_manager`)
- Stage 1/2:
  - `mlflow_manager.log_metrics({"td_error_mean": ..., "weight_entropy": ...}, step)`
  - 주요 하이퍼파라미터 `log_params`
- Stage 3:
  - Checkpoint/로그 artifact 업로드 (`mlflow_manager.log_artifact(path, subdir)`)
  - 최종 metrics 기록 후 run 종료

### 6.3 테스트 (Phase 7)
- `tests/integration/test_stage2_local.py`: `mlflow.set_tracking_uri("file:///tmp/mlflow")`로 대체 backend 검증
- CI: `.github/workflows/test.yaml`에서 `export MLFLOW_TRACKING_URI=file:///tmp/mlflow`

---

## 7. 실행 플로우

### 7.1 로컬 (M3 Mac)
```bash
source .env
uv run python -m weighted_mtp.cli.train \
  --config configs/defaults.yaml \
  --recipe configs/recipe.verifiable.yaml \
  --preset local-light \
  --use-micro-model true
```
- tracking URI: `.env` → EC2 서버
- artifacts: S3 `weighted_mtp/...`
- Stage 종료 후 `logs/<run_id>/training.log` 유지, best checkpoint는 S3 업로드 후 로컬 삭제

### 7.2 CI (GitHub Actions)
```yaml
- name: Run tests
  env:
    MLFLOW_TRACKING_URI: file:///tmp/mlflow
  run: |
    uv run pytest tests/ -m "not gpu"
```
- `config.ci_test.yaml`에서 artifact 로깅, run 기록은 로컬 디렉터리

### 7.3 VESSL GPU
```bash
vessl run create \
  --cluster vessl-gcp-oregon \
  --resource a100-1gpu \
  --image ghcr.io/wooshikwon/weighted-mtp:latest \
  --env-file .env.vessl \
  --command "uv run python -m weighted_mtp.cli.train \
      --config configs/defaults.yaml \
      --recipe configs/recipe.verifiable.yaml \
      --run-name vessl_verifiable_001"
```
- `.env.vessl`에는 MLflow/S3 credentials 포함
- VESSL 로그에서 run_id 확인 → MLflow UI로 추적

---

## 8. Storage v2와 MLflow 연동 지침

- `storage/models_v2/*`: 모델 버전 변경 시 MLflow run tag `model_version` 업데이트.
- `storage/datasets_v2/*`: 데이터셋 버전과 checksum을 MLflow params에 로그.
- `scripts/sync_to_vessl_storage.py` 실행 시:
  - 업로드 성공 후 MLflow run에 `storage_sync=True` 태그 남김.
  - 문제가 발생하면 run `status`를 `FAILED`로 종료하여 알람 근거 확보.

---

## 9. 운영 체크리스트

- [ ] `.env` 최신화 (credentials, tracking URI)
- [ ] MLflow 서버 헬스 체크: `curl -u user:pass http://<ip>/health`
- [ ] S3 버킷 권한 확인 (`aws s3 ls s3://wmtp/mlflow-artifacts/`)
- [ ] Stage 1/2 실행 후 MLflow Metrics에 `td_error_mean`, `weight_entropy`, `value_loss` 등이 기록되는지 확인
- [ ] Artifact(`checkpoints/best_model.pt`, `logs/training.log`) 업로드 여부
- [ ] 실패 run에 대한 S3 정리 정책 검토 (필요 시 Lifecycle Rule 설정)

---

## 10. 참고 문서 및 후속 업데이트

- `legacy_docs/WMTP_MLFLOW_가이드.md`: 세부 EC2 설정/트러블슈팅 정보 그대로 참고.
- `docs/02_implementation_plan.md`: Phase 8 완료 시 본 문서에 맞춰 검증 항목 업데이트.
- 변경 사항 발견 시:
  - `docs/migration_notes.md`에 기록
  - `mlflow_integration.md` 최신화

---

이 문서는 리팩토링된 `weighted_mtp/` 프로젝트가 기존 MLflow 인프라를 재활용하면서도 이상적 구조·Phase 계획과 완전히 호환되도록 하는 기준 문서로 사용한다.

