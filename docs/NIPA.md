# NIPA H200 서버 마이그레이션 가이드

KT Cloud 기반 NIPA AI 지원 프로그램 H200 4-GPU 서버 활용 가이드.

## 서버 정보

| 항목 | 값 |
|------|-----|
| Host | `$NIPA_HOST` (`.env` 참조) |
| Port | `$NIPA_PORT` |
| User | `$NIPA_USER` |
| Password | `$NIPA_PASSWORD` |
| GPU | NVIDIA H200 x 4 |
| 작업 디렉토리 | `$NIPA_WORK_DIR` |
| 공유 캐시 | `/home/work/.cache` |
| 사용 기간 | 11/23 - 11/26 (4일) |

> 민감 정보는 `.env` 파일에서 관리. 실제 값은 `.env` 파일 참조.

## 데이터 현황

### 필수 데이터 (업로드 필요)

| 항목 | 용량 | 용도 |
|------|------|------|
| `storage/models/meta-llama-mtp` | 50GB | MTP 정책 모델 (프로덕션) |
| `storage/datasets/codecontests` | 7.8GB | 학습/검증 데이터셋 |
| **총 필수** | **~58GB** | |

### 선택적 데이터

| 항목 | 용량 | 용도 |
|------|------|------|
| `storage/models/ref-sheared-llama-2.7b` | 20GB | Reference 모델 |
| `storage/models/micro-mtp` | 354MB | 테스트용 소형 모델 |
| `storage/datasets/humaneval` | 232KB | 평가 데이터셋 |
| `storage/datasets/mbpp` | 620KB | 평가 데이터셋 |
| `storage/checkpoints/*` | 15GB | 기존 체크포인트 (선택) |

---

## Phase 1: 로컬 사전 준비

### 1.1 SSH Config 설정

```bash
# .env에서 값 확인 후 ~/.ssh/config 추가
cat >> ~/.ssh/config << 'EOF'
Host nipa
   User work
   Hostname proxy1.nipa2025.ktcloud.com
   Port 10507
   StrictHostKeyChecking no
   UserKnownHostsFile /dev/null
EOF
```

### 1.2 접속 테스트

```bash
ssh nipa
# 비밀번호: .env 파일의 NIPA_PASSWORD 값 사용
```

### 1.3 서버 디렉토리 생성

```bash
ssh nipa "mkdir -p ~/grad_school/wooshikwon/weighted_mtp/storage/{models,datasets,checkpoints}"
```

---

## Phase 2: 데이터 업로드

### 2.1 업로드 전략

대용량(58GB+) 전송 시 rsync 사용을 권장. 중단 시 이어받기 가능.

### 2.2 필수 데이터 업로드

> **주의**: 아래 rsync 명령어는 **로컬 맥에서 실행**해야 함. NIPA 서버에 접속한 상태에서 실행하면 안 됨. 서버에 접속 중이라면 먼저 `exit`로 나온 후 실행.

```bash
# 로컬에서 프로젝트 디렉토리로 이동
cd /path/to/weighted_mtp

# 1. 모델 업로드 (50GB, 약 1-2시간 소요)
rsync -avz --progress -e "ssh -p 10507" \
  ./storage/models/meta-llama-mtp/ \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/models/meta-llama-mtp/

# 2. 데이터셋 업로드 (7.8GB)
rsync -avz --progress -e "ssh -p 10507" \
  ./storage/datasets/codecontests/ \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/datasets/codecontests/

# 3. 평가 데이터셋 (소용량)
rsync -avz --progress -e "ssh -p 10507" \
  ./storage/datasets/humaneval/ \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/datasets/humaneval/

rsync -avz --progress -e "ssh -p 10507" \
  ./storage/datasets/mbpp/ \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/datasets/mbpp/
```

### 2.3 선택적 데이터 업로드

```bash
# Reference 모델 (20GB, 필요시)
rsync -avz --progress -e "ssh -p 10507" \
  ./storage/models/ref-sheared-llama-2.7b/ \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/models/ref-sheared-llama-2.7b/

# 기존 체크포인트 (필요시)
rsync -avz --progress -e "ssh -p 10507" \
  ./storage/checkpoints/ \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/checkpoints/
```

### 2.4 업로드 검증

```bash
ssh nipa "du -sh ~/grad_school/wooshikwon/weighted_mtp/storage/*/"
```

예상 결과:
```
50G  storage/models/meta-llama-mtp/
7.8G storage/datasets/codecontests/
...
```

---

## Phase 3: 코드 배포

### 3.1 코드 업로드

> **중요**: 먼저 코드를 서버에 올려야 requirements_nipa.txt 등의 파일을 사용할 수 있음.

```bash
# 로컬에서 실행
cd /path/to/weighted_mtp

rsync -avz --progress \
  --exclude 'storage/' \
  --exclude 'tests/' \
  --exclude '.github/' \
  --exclude 'docs/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.venv/' \
  --exclude 'mlruns/' \
  --exclude '.env' \
  --exclude '.coverage' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.DS_Store' \
  --exclude 'results/' \
  --exclude 'uv.lock' \
  --exclude '.claude/' \
  -e "ssh -p 10507" \
  ./ \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/
```

### 3.2 환경변수 설정

```bash
ssh nipa
cd ~/grad_school/wooshikwon/weighted_mtp

# .env 파일 생성 (MLflow, AWS 등)
cat > .env << 'EOF'
MLFLOW_TRACKING_URI=http://13.50.240.176
MLFLOW_TRACKING_USERNAME=wmtp_admin
MLFLOW_TRACKING_PASSWORD=wmtp_secure_2025
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY>
AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_KEY>
AWS_DEFAULT_REGION=eu-north-1
S3_BUCKET_NAME=wmtp
HF_TOKEN=<YOUR_HF_TOKEN>
TOKENIZERS_PARALLELISM=false
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mlflow
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_secure_password_123
EOF
```

### 3.3 PYTHONPATH 설정

```bash
# ~/.bashrc에 추가
echo 'export PYTHONPATH="$HOME/grad_school/wooshikwon/weighted_mtp/src:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Phase 4: 서버 환경 설정

### 4.1 uv 설치 및 환경 생성

```bash
ssh nipa
cd ~/grad_school/wooshikwon/weighted_mtp

# uv 설치 (이미 설치되어 있으면 생략)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Python 3.10 + 가상환경 생성 및 의존성 설치
uv sync
```

### 4.2 Flash Attention 설치 (선택)

```bash
cd ~/grad_school/wooshikwon/weighted_mtp

# Flash Attention 2 (H200 Hopper 아키텍처 최적화)
uv pip install flash-attn --no-build-isolation
```

### 4.3 환경 검증

```bash
cd ~/grad_school/wooshikwon/weighted_mtp

# PyTorch 및 GPU 확인
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

예상 결과:
```
PyTorch: 2.x.x
CUDA: True
GPUs: 4
```

### 4.4 Flash Attention 검증

```bash
uv run python -c "
import torch
print(f'Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}')
print(f'Memory efficient SDP: {torch.backends.cuda.mem_efficient_sdp_enabled()}')
try:
    import flash_attn
    print(f'flash-attn version: {flash_attn.__version__}')
except ImportError:
    print('flash-attn not installed (PyTorch SDPA will use fallback)')
"
```

예상 결과:
```
Flash SDP enabled: True
Memory efficient SDP: True
flash-attn version: 2.x.x
```

> Flash Attention이 설치되면 `F.scaled_dot_product_attention`이 자동으로 최적화된 커널 사용. 코드 변경 불필요.

---

## Phase 5: 실행

### 5.1 단일 GPU 테스트

```bash
ssh nipa
cd ~/grad_school/wooshikwon/weighted_mtp

# 테스트 실행
uv run python -m weighted_mtp.pipelines.run_baseline \
  --config configs/baseline/baseline.yaml \
  --override training.n_epochs=0.01
```

### 5.2 분산 학습 (4-GPU)

```bash
cd ~/grad_school/wooshikwon/weighted_mtp

# MLflow 로컬 파일 저장 경로 (EC2 서버 대신 사용)
MLFLOW_URI="file:///home/work/grad_school/wooshikwon/weighted_mtp/mlruns"

# Baseline MTP
uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_port=29501 \
  -m weighted_mtp.pipelines.run_baseline \
  --config configs/baseline/baseline.yaml \
  --override mlflow.tracking_uri=$MLFLOW_URI

MLFLOW_URI="file:///home/work/grad_school/wooshikwon/weighted_mtp/mlruns"

# Critic 사전학습
uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_port=29501 \
  -m weighted_mtp.pipelines.run_critic \
  --config configs/critic/critic_linear.yaml \
  --override mlflow.tracking_uri=$MLFLOW_URI

MLFLOW_URI="file:///home/work/grad_school/wooshikwon/weighted_mtp/mlruns"

# Verifiable Reward
uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_port=29501 \
  -m weighted_mtp.pipelines.run_verifiable \
  --config configs/verifiable/verifiable.yaml \
  --override mlflow.tracking_uri=$MLFLOW_URI

MLFLOW_URI="file:///home/work/grad_school/wooshikwon/weighted_mtp/mlruns"

# Rho-1
uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_port=29501 \
  -m weighted_mtp.pipelines.run_rho1 \
  --config configs/rho1/rho1.yaml \
  --override mlflow.tracking_uri=$MLFLOW_URI
```

### 5.3 MLflow UI로 실험 결과 확인

학습 중 또는 완료 후 `mlruns/` 실험 데이터를 시각화.

**1. NIPA 서버에서 MLflow UI 실행:**

```bash
cd ~/grad_school/wooshikwon/weighted_mtp

# 백그라운드 실행 (터미널 닫아도 유지)
nohup mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 > mlflow_ui.log 2>&1 &
```

**2. 로컬에서 SSH 포트포워딩:**

```bash
ssh -L 5000:localhost:5000 -p 10507 work@proxy1.nipa2025.ktcloud.com
```

**3. 브라우저 접속:** `http://localhost:5000`

### 5.4 tmux 사용 (권장)

```bash
# 세션 생성
tmux new -s project

# 학습 실행
cd ~/grad_school/wooshikwon/weighted_mtp
uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_port=29501 \
  -m weighted_mtp.pipelines.run_baseline \
  --config configs/baseline/baseline.yaml

# 세션 분리: Ctrl+B, D
# 세션 재접속: tmux attach -t project
# 세션 목록: tmux ls
# 세션 종료: tmux kill-session -t project
```

### 5.5 백그라운드 실행 (nohup)

```bash
mkdir -p logs
nohup uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  -m weighted_mtp.pipelines.run_baseline \
  --config configs/baseline/baseline.yaml \
  > logs/baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 로그 확인
tail -f logs/baseline_*.log
```

---

## Phase 6: 체크포인트 다운로드

### 6.1 결과물 로컬로 가져오기

> **주의**: 아래 명령어는 **로컬 터미널**에서 실행. NIPA 서버에 접속한 상태라면 먼저 `exit`로 나온 후 실행.

**rsync로 다운로드 (권장)**

```bash
# 특정 디렉토리 다운로드
rsync -avz --progress -e "ssh -p 10507" \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/checkpoints/baseline/ \
  ./storage/checkpoints_nipa/baseline/

# mlruns 전체 다운로드
rsync -avz --progress -e "ssh -p 10507" \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/mlruns/ \
  ./mlruns_nipa/
```

옵션 설명:
- `-a`: 아카이브 모드 (권한, 심볼릭 링크 등 보존)
- `-v`: 상세 출력
- `-z`: 압축 전송
- `--progress`: 진행률 표시
- `-e "ssh -p 10507"`: SSH 포트 지정

> **주의**: rsync/scp가 "channel 0: rcvd too much data" 또는 "Received message too long" 오류로 실패할 경우, 아래 근본 원인 해결 또는 대안 방법을 사용.

**근본 원인 해결 (권장)**

이 오류는 서버의 `.bashrc`가 비대화형 세션에서도 출력을 생성하기 때문에 발생. 서버에서 아래와 같이 수정:

```bash
# 서버에 접속
ssh nipa

# .bashrc 파일 상단에 아래 코드 추가
# 비대화형 세션에서는 나머지 .bashrc 실행을 건너뜀
nano ~/.bashrc
# 파일 최상단에 추가:
# [[ $- != *i* ]] && return
```

이후 rsync/scp가 정상 작동.

**방법 1: 서버에서 압축 후 전송 (권장)**

```bash
# 1. 서버에서 압축
ssh -p 10507 work@proxy1.nipa2025.ktcloud.com \
  "cd ~/grad_school/wooshikwon/weighted_mtp/storage/checkpoints/baseline && tar czvf baseline-mtp.tar.gz baseline-mtp"

# 2. 로컬에서 다운로드
mkdir -p ./storage/checkpoints_nipa
scp -P 10507 \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/checkpoints/baseline/baseline-mtp.tar.gz \
  ./storage/checkpoints_nipa/

# 3. 압축 해제
cd ./storage/checkpoints_nipa && tar xzf baseline-mtp.tar.gz && rm baseline-mtp.tar.gz
```

**방법 2: sftp 사용**

```bash
  mkdir -p ./storage/checkpoints_nipa/baseline-mtp
  cd ./storage/checkpoints_nipa/baseline-mtp
  sftp -oPort=10507 work@proxy1.nipa2025.ktcloud.com
  
# 접속 후:
cd /home/work/grad_school/wooshikwon/weighted_mtp/storage/checkpoints/baseline/baseline-mtp
get checkpoint_epoch_1.50.pt
bye
```

**방법 3: 단일파일만**

```bash
  mkdir -p ./storage/checkpoints_nipa/baseline-mtp
  scp -P 10507 \
work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/checkpoints/baseline/baseline-mtp/checkpoint_epoch_1.50.pt \
    ./storage/checkpoints_nipa/baseline-mtp/
```

---

## Phase 7: 정리

### 7.1 사용 완료 후 정리

```bash
ssh nipa
cd ~/grad_school/wooshikwon

# 대용량 데이터 삭제 (모델/데이터셋)
rm -rf weighted_mtp/storage/models/*
rm -rf weighted_mtp/storage/datasets/*

# 가상환경 삭제
rm -rf weighted_mtp/.venv
```

---

## 스크립트 파일

NIPA 서버용 실행 스크립트는 `scripts/nipa/` 디렉토리에 위치:

```
scripts/nipa/
├── setup_env.sh      # 환경 설정
├── baseline.sh       # Baseline MTP 실행
├── critic.sh         # Critic 사전학습 실행
├── verifiable.sh     # Verifiable Reward 실행
└── rho1.sh          # Rho-1 실행
```

---

## 트러블슈팅

### CUDA Out of Memory

```bash
# batch_size 줄이기
torchrun ... --override training.batch_size=4

# gradient_accumulation 늘리기
torchrun ... --override training.gradient_accumulation_steps=4
```

### NCCL 타임아웃

```bash
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

### 네트워크 끊김

rsync 재실행 시 자동으로 이어받기됨.

---

## VESSL vs NIPA 비교

| 항목 | VESSL | NIPA |
|------|-------|------|
| 실행 명령 | `uv run python` | `uv run python` |
| 분산 학습 | `uv run torchrun` | `uv run torchrun` |
| 스토리지 | `vessl storage` | 로컬 rsync |
| 환경 | Docker 이미지 | uv |
| 작업 제출 | `vessl run create` | SSH 직접 실행 |
| 모니터링 | VESSL Web UI | tmux + tail |

---

## 체크리스트

### 업로드 전

- [ ] SSH config 설정
- [ ] 서버 접속 테스트
- [ ] 디렉토리 생성

### 데이터 업로드

- [ ] meta-llama-mtp 모델 (50GB)
- [ ] codecontests 데이터셋 (7.8GB)
- [ ] 평가 데이터셋 (humaneval, mbpp)

### 코드 배포

- [ ] 코드 rsync
- [ ] .env 파일 설정
- [ ] PYTHONPATH 설정

### 환경 설정

- [ ] uv 설치
- [ ] uv sync (의존성 설치)
- [ ] flash-attn 설치 (선택)

### 실행 검증

- [ ] 단일 GPU 테스트
- [ ] 4-GPU 분산 학습 테스트

### 완료 후

- [ ] 체크포인트 다운로드
- [ ] 서버 데이터 정리
