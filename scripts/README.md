# Weighted MTP Scripts

프로젝트 설정, 평가, 배포를 위한 스크립트 모음

## 📁 스크립트 구성

### Setup Scripts (Phase 1 자산 준비)

#### 1. `setup_models.py` - 모델 설정
모델 다운로드, 변환, Micro 생성, 검증을 통합 처리

**기능:**
- HuggingFace에서 모델 다운로드
- PyTorch → SafeTensors 변환 (단일/Sharded 자동 감지)
- Micro 모델 생성 (Meta/HuggingFace 아키텍처 지원)
- MTP Config 동기화 (params.json → meta_adapter.yaml)
- 모델 무결성 검증 (SHA256, Config, SafeTensors)

**사용 예:**
```bash
# Meta MTP 전체 설정 (다운로드 → 변환 → config → micro → 검증)
uv run python scripts/setup_models.py --model meta-llama-mtp --steps all --create-micro

# Sheared-LLaMA 변환만
uv run python scripts/setup_models.py --model ref-sheared-llama --steps convert,verify

# 검증만
uv run python scripts/setup_models.py --model meta-llama-mtp --steps verify
```

**지원 모델:**
- `meta-llama-mtp`: Meta LLaMA MTP 7B_1T_4
- `ref-sheared-llama`: Sheared-LLaMA 2.7B (Rho-1 reference)
- `starling-rm`: Starling-RM-7B-alpha (선택)

---

#### 2. `setup_datasets.py` - 데이터셋 설정
데이터셋 다운로드, Small 버전 생성, 검증, 통계 생성을 통합 처리

**기능:**
- HuggingFace에서 데이터셋 다운로드
- Small 버전 생성 (train_small, validation_small, test_small)
- 데이터셋 검증 (schema, 길이, 필수 필드)
- 통계 생성 (샘플 수, 평균 길이 등)

**사용 예:**
```bash
# 전체 데이터셋 설정
uv run python scripts/setup_datasets.py --datasets all --steps all

# MBPP만 small 생성
uv run python scripts/setup_datasets.py --datasets mbpp --steps small

# 검증만
uv run python scripts/setup_datasets.py --datasets all --steps validate

# 커스텀 크기 지정
uv run python scripts/setup_datasets.py --datasets codecontests --steps small --train-size 50 --val-size 16
```

**지원 데이터셋:**
- `codecontests`: DeepMind CodeContests
- `mbpp`: MBPP (Mostly Basic Python Problems)
- `humaneval`: OpenAI HumanEval

---

#### 3. `verify_storage.py` - 무결성 검증
전체 storage 디렉터리 검증 및 Phase1 체크리스트 생성

**기능:**
- 모든 모델 무결성 검증 (SHA256, Config, SafeTensors)
- 모든 데이터셋 검증 (파일 존재, Schema)
- Phase1 체크리스트 자동 생성
- 검증 리포트 생성 (JSON)

**사용 예:**
```bash
# 전체 검증
uv run python scripts/verify_storage.py --check all

# 모델만 검증
uv run python scripts/verify_storage.py --check models

# Phase1 체크리스트 생성
uv run python scripts/verify_storage.py --check all --phase1-checklist

# 리포트 생성
uv run python scripts/verify_storage.py --check all --generate-report
```

---

### Evaluation Scripts (Phase 7 평가)

#### 4. `compare_evaluation_results.py` - MLflow 평가 결과 비교

MLflow에서 여러 모델의 평가 결과를 조회하여 시각화 및 비교

**기능:**
- MLflow experiment 조회 및 run 필터링
- Pass@K 메트릭 비교 (Pass@1, Pass@5, Pass@10, Pass@20)
- DataFrame으로 결과 추출 및 CSV 저장
- matplotlib 차트 생성 (bar chart)

**사용 예:**
```bash
# HumanEval 결과 비교
python scripts/compare_evaluation_results.py \
  --experiment weighted-mtp-evaluation \
  --dataset humaneval \
  --output-dir results

# MBPP 결과 비교
python scripts/compare_evaluation_results.py \
  --experiment weighted-mtp-evaluation \
  --dataset mbpp \
  --output-dir results
```

**출력:**
- `comparison_{dataset}.csv`: 모델별 Pass@K 결과
- `comparison_{dataset}.png`: 시각화 차트

**환경변수:**
- `MLFLOW_TRACKING_URI`: MLflow tracking server URL (필수)

---

### Deployment Scripts (배포)

#### 5. `download_s3_checkpoints.py` - S3 checkpoint 다운로드

MLflow artifact store (S3)에서 학습된 checkpoint를 다운로드하여 로컬 또는 VESSL에서 평가

**기능:**
- MLflow experiment 및 run 조회
- S3에서 checkpoint 다운로드
- 로컬 storage 저장 또는 VESSL 업로드
- 대화형/배치 모드 지원

**사용 예 (대화형):**
```bash
# 대화형 모드로 실행
python scripts/download_s3_checkpoints.py --interactive

# Experiment 선택 → Run 선택 → Checkpoint 선택 → 다운로드 모드 선택
```

**사용 예 (배치):**
```bash
# Best checkpoint 다운로드 (로컬)
python scripts/download_s3_checkpoints.py \
  --experiment weighted-mtp-baseline \
  --run baseline_run_1 \
  --checkpoint best \
  --output-dir storage/checkpoints/baseline

# Latest checkpoint 다운로드 (VESSL)
python scripts/download_s3_checkpoints.py \
  --experiment weighted-mtp-baseline \
  --run baseline_run_1 \
  --checkpoint latest \
  --vessl
```

**Checkpoint 타입:**
- `best`: checkpoint_best.pt (가장 낮은 validation loss)
- `final`: checkpoint_final.pt (마지막 epoch)
- `latest`: checkpoint_epoch_*.pt 중 가장 최근

**환경변수:**
- `MLFLOW_TRACKING_URI`: MLflow tracking server URL (필수)
- `AWS_ACCESS_KEY_ID`: AWS access key (필수)
- `AWS_SECRET_ACCESS_KEY`: AWS secret key (필수)
- `AWS_DEFAULT_REGION`: AWS region (필수)

---

## 🚀 원클릭 실행

### Phase 1 설정

전체 Phase1 설정을 한 번에 실행:

```bash
# 1. Meta MTP 모델 설정
uv run python scripts/setup_models.py \
  --model meta-llama-mtp \
  --steps all \
  --create-micro

# 2. Sheared-LLaMA 모델 설정
uv run python scripts/setup_models.py \
  --model ref-sheared-llama \
  --steps all \
  --create-micro \
  --micro-type reference

# 3. 데이터셋 설정
uv run python scripts/setup_datasets.py \
  --datasets all \
  --steps all

# 4. 전체 검증 및 체크리스트 생성
uv run python scripts/verify_storage.py \
  --check all \
  --phase1-checklist \
  --generate-report
```

---

### Phase 7 평가 및 비교

```bash
# 1. 모델 평가 실행 (CLI)
python -m weighted_mtp evaluate \
  --checkpoint storage/checkpoints/baseline/checkpoint_best.pt \
  --dataset humaneval \
  --num-samples 20

# 2. MLflow 결과 비교
python scripts/compare_evaluation_results.py \
  --experiment weighted-mtp-evaluation \
  --dataset humaneval \
  --output-dir results
```

### 배포 및 재평가

```bash
# 1. S3에서 checkpoint 다운로드
python scripts/download_s3_checkpoints.py \
  --experiment weighted-mtp-baseline \
  --run baseline_run_1 \
  --checkpoint best \
  --output-dir storage/checkpoints/downloaded

# 2. 다운로드된 checkpoint 평가
python -m weighted_mtp evaluate \
  --checkpoint storage/checkpoints/downloaded/checkpoint_best.pt \
  --dataset humaneval
```

---

## 📋 스크립트 요약

| 스크립트 | 용도 | Phase | 주요 기능 |
|---------|------|-------|----------|
| `setup_models.py` | 모델 설정 | Phase 1 | 다운로드, 변환, Micro 생성, 검증 |
| `setup_datasets.py` | 데이터셋 설정 | Phase 1 | 다운로드, Small 생성, 검증 |
| `verify_storage.py` | 무결성 검증 | Phase 1 | 전체 검증, 리포트 생성 |
| `compare_evaluation_results.py` | 평가 비교 | Phase 7 | MLflow 결과 시각화 |
| `download_s3_checkpoints.py` | Checkpoint 배포 | 배포 | S3 다운로드, VESSL 업로드 |

---

## ⚠️ 주의사항

### 환경변수 설정

**Phase 1 (Setup):**
```bash
export HF_TOKEN=hf_...  # HuggingFace 토큰
```

**Phase 7 (Evaluation):**
```bash
export MLFLOW_TRACKING_URI=http://...  # MLflow tracking server
```

**배포 (S3 Download):**
```bash
export MLFLOW_TRACKING_URI=http://...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-north-1
```

### 디스크 용량

- 모델 다운로드: 약 50GB
- Checkpoint 다운로드: 모델당 약 7GB

### 실행 순서

1. **Setup**: `setup_models.py` → `setup_datasets.py` → `verify_storage.py`
2. **Training**: `python -m weighted_mtp --config configs/...`
3. **Evaluation**: `python -m weighted_mtp evaluate --checkpoint ...`
4. **Analysis**: `compare_evaluation_results.py`
5. **Deployment**: `download_s3_checkpoints.py` → 재평가

### 멱등성

모든 setup 스크립트는 멱등성(idempotent) 보장:
- 이미 존재하는 파일은 건너뛰거나 덮어쓰기
- 재실행 시 안전

---

## 📖 추가 문서

- **Phase 1**: `docs/03_phase1_detailed_plan.md` (자산 준비)
- **Phase 7**: `docs/07_phase7_detailed_plan.md` (SFT 평가)
- **Storage 구조**: `docs/01_storage_preparation_plan.md`
- **이상적 구조**: `docs/00_ideal_structure.md`

---

## 🗑️ 변경 이력

**Phase 1 통합 (9개 → 3개)**:
- `setup_mtp_model.sh`, `convert_*.py`, `sync_mtp_config.py`, `prepare_*.py` → `setup_models.py`
- `prepare_dataset.py` → `setup_datasets.py`
- `verify_mtp_model.py` → `verify_storage.py`

**Phase 7 추가**:
- `compare_evaluation_results.py`: MLflow 평가 결과 비교 및 시각화
- `download_s3_checkpoints.py`: S3 checkpoint 다운로드 및 배포
