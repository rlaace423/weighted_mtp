# VESSL 스크립트 사용 가이드

VESSL A100 클러스터에서 Weighted MTP 파이프라인을 실행하기 위한 통합 스크립트입니다.

## 구조

각 파이프라인당 2개 파일:
- `{pipeline}.yaml.template`: VESSL run 설정 템플릿
- `{pipeline}.sh`: 실행 스크립트 (GPU 수를 CLI 인자로 받음)

```
scripts/vessl/
├── critic.yaml.template
├── critic.sh
├── baseline.yaml.template
├── baseline.sh
├── rho1.yaml.template
├── rho1.sh
├── verifiable.yaml.template
├── verifiable.sh
├── ntp.yaml.template
└── ntp.sh
```

---

## 사용법

### 기본 사용법

```bash
./scripts/vessl/{pipeline}.sh --ngpus <1|2|4> [--batch-size N] [--grad-accum N]
```

**CLI 인자:**
- `--ngpus`: GPU 개수 (필수, 1/2/4)
- `--batch-size`: Batch size override (선택)
- `--grad-accum`: Gradient accumulation steps override (선택)
- `--critic-checkpoint`: Critic checkpoint 경로 (Verifiable만, 선택)

### 파이프라인별 예시

#### 1. Critic Pretraining (Stage 1)

```bash
# 4 GPU (권장)
./scripts/vessl/critic.sh --ngpus 4

# 1 GPU (테스트용)
./scripts/vessl/critic.sh --ngpus 1

# 2 GPU
./scripts/vessl/critic.sh --ngpus 2

# Batch size override (Effective batch size 조정)
./scripts/vessl/critic.sh --ngpus 4 --batch-size 8 --grad-accum 3
# Effective batch size = 8 × 3 = 24

# Config 예시: training.batch_size=6, gradient_accumulation_steps=4
# Override: training.batch_size=8, gradient_accumulation_steps=3
```

**출력 checkpoint**: `storage/checkpoints/critic/critic-pretrain/checkpoint_best.pt`

#### 2. Verifiable Training (Stage 2)

```bash
# 4 GPU (권장)
./scripts/vessl/verifiable.sh --ngpus 4

# Critic checkpoint 명시
./scripts/vessl/verifiable.sh --ngpus 4 \
  --critic-checkpoint storage/checkpoints/critic/critic-pretrain/checkpoint_best.pt

# Batch size + Checkpoint override
./scripts/vessl/verifiable.sh --ngpus 4 \
  --critic-checkpoint storage/checkpoints/critic/best.pt \
  --batch-size 4 --grad-accum 6

# 1 GPU (테스트용)
./scripts/vessl/verifiable.sh --ngpus 1
```

**주의**: Critic checkpoint 필요 (Stage 1 완료 후 실행)

#### 3. Baseline MTP

```bash
# 4 GPU (권장)
./scripts/vessl/baseline.sh --ngpus 4

# 1 GPU with custom batch size
./scripts/vessl/baseline.sh --ngpus 1 --batch-size 2 --grad-accum 8

# 2 GPU
./scripts/vessl/baseline.sh --ngpus 2 --batch-size 6 --grad-accum 2
```

#### 4. Rho-1 WMTP

```bash
# 4 GPU (권장)
./scripts/vessl/rho1.sh --ngpus 4

# 2 GPU with override
./scripts/vessl/rho1.sh --ngpus 2 --batch-size 4 --grad-accum 3
```

#### 5. NTP Baseline

```bash
# 1 GPU (기본)
./scripts/vessl/ntp.sh --ngpus 1

# 4 GPU with custom settings
./scripts/vessl/ntp.sh --ngpus 4 --batch-size 8 --grad-accum 2
```

---

## 자동 설정

스크립트가 GPU 수에 따라 자동으로 설정:

| GPU 수 | VESSL Preset | Torchrun | Config |
|--------|-------------|----------|--------|
| **1** | `gpu-a100-80g-small` | `python -m` (torchrun 없음) | `*_1gpu.yaml` (Critic만) |
| **2** | `gpu-a100-80g-medium` | `torchrun --nproc_per_node=2` | `*.yaml` |
| **4** | `gpu-a100-80g-large` | `torchrun --nproc_per_node=4` | `*.yaml` |

---

## 환경변수

`.env` 파일에 다음 환경변수 설정 필요:

```bash
# MLflow
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...

# AWS S3
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-north-1

# HuggingFace
HF_TOKEN=...
```

---

## 실행 후 확인

### VESSL 웹 UI
https://vessl.ai

### MLflow UI
http://13.50.240.176

### 로그 확인
```bash
# VESSL CLI로 로그 스트리밍
vessl run logs <run-id> --follow
```

---

## 워크플로우 예시

### 전체 파이프라인 실행

```bash
# Step 1: Critic Pretraining (Stage 1)
./scripts/vessl/critic.sh --ngpus 4

# [Critic 학습 완료 대기]

# Step 2: Verifiable Training (Stage 2)
./scripts/vessl/verifiable.sh --ngpus 4

# Step 3: Baseline 비교
./scripts/vessl/baseline.sh --ngpus 4

# Step 4: Rho-1 비교
./scripts/vessl/rho1.sh --ngpus 4
```

### 빠른 테스트 (1 GPU, 작은 배치)

```bash
# Critic 테스트 (Effective batch size 8)
./scripts/vessl/critic.sh --ngpus 1 --batch-size 2 --grad-accum 4

# Baseline 테스트
./scripts/vessl/baseline.sh --ngpus 1 --batch-size 2 --grad-accum 4
```

### Effective Batch Size 조정 예시

```bash
# 동일한 Effective Batch Size (24) 유지하면서 GPU 수 변경

# 1 GPU: batch=6, accum=4 (6×4=24)
./scripts/vessl/critic.sh --ngpus 1 --batch-size 6 --grad-accum 4

# 2 GPU: batch=3, accum=4 (3×4×2=24)
./scripts/vessl/critic.sh --ngpus 2 --batch-size 3 --grad-accum 4

# 4 GPU: batch=2, accum=3 (2×3×4=24)
./scripts/vessl/critic.sh --ngpus 4 --batch-size 2 --grad-accum 3
```

---

## 주의사항

1. **Critic → Verifiable 순서**: Verifiable은 Critic checkpoint 필요
2. **GPU 메모리**: 1 GPU는 테스트용, 프로덕션은 4 GPU 권장
3. **환경변수**: `.env` 파일 누락 시 오류 발생
4. **VESSL 클러스터**: `vessl-kr-a100-80g-sxm` 클러스터 사용

---

## 문제 해결

### "오류: .env 파일을 찾을 수 없습니다"
```bash
# .env 파일 생성
cp .env.example .env
# .env 파일 편집하여 환경변수 입력
```

### "오류: NGPUS=X (1, 2, 4만 지원)"
```bash
# 지원되는 GPU 수만 사용
./scripts/vessl/critic.sh --ngpus 4  # ✅
./scripts/vessl/critic.sh --ngpus 8  # ❌
```

### VESSL 인증 문제
```bash
# VESSL CLI 재인증
vessl configure
```

---

## 기술적 세부사항

### YAML 템플릿 변수

템플릿에서 사용되는 변수:
- `{{NGPUS}}`: GPU 개수 (1, 2, 4)
- `{{PRESET}}`: VESSL preset (small, medium, large)
- `{{TRAIN_COMMAND}}`: 학습 명령어 (python -m 또는 torchrun + override 인자)
- `{{NCCL_DEBUG}}`: NCCL 디버그 설정 (2+ GPU만)
- `{{MLFLOW_TRACKING_USERNAME}}`: MLflow 사용자명
- `{{AWS_ACCESS_KEY_ID}}`: AWS 액세스 키
- 등

### Shell 스크립트 로직

1. CLI 인자 파싱 (`--ngpus`, `--batch-size`, `--grad-accum`, etc.)
2. Preset 자동 선택 (GPU 수 기반)
3. Config 경로 결정
4. **Override 인자 생성** (batch size, grad accum)
   ```bash
   # 예시: --override training.batch_size=8 --override training.gradient_accumulation_steps=2
   ```
5. Train command 생성 (torchrun vs python -m + override)
6. `.env` 로드
7. 템플릿에서 YAML 생성 (변수 치환)
8. 실행 설정 요약 출력
9. `vessl run create` 실행
10. 임시 파일 정리

### Config Override 메커니즘

Shell 스크립트는 CLI 인자를 Python `--override` 인자로 변환:

```bash
# Shell 입력
./scripts/vessl/critic.sh --ngpus 4 --batch-size 8 --grad-accum 3

# 생성되는 Python 명령어
uv run torchrun ... -m weighted_mtp.pipelines.run_critic \
  --config configs/critic/critic.yaml \
  --override training.batch_size=8 \
  --override training.gradient_accumulation_steps=3
```

Python 파이프라인에서 `apply_overrides()` 함수가 config를 동적으로 수정:
```python
# configs/critic/critic.yaml
training:
  batch_size: 6          # 기본값
  gradient_accumulation_steps: 2  # 기본값

# Override 후
training:
  batch_size: 8          # 덮어쓰기됨
  gradient_accumulation_steps: 3  # 덮어쓰기됨
```

---

## 참고

- **ARCHITECTURE.md**: 시스템 아키텍처
- **RESEARCH.md**: 연구 배경 및 이론
- **configs/**: 각 파이프라인별 설정 파일
