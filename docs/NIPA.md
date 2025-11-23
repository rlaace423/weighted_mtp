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
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/samkim/weighted_mtp/
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

### 4.1 Conda 환경 생성

```bash
ssh nipa
cd ~/grad_school/wooshikwon

# conda 환경 생성
conda create -n weighted_mtp python=3.10 -y
conda activate weighted_mtp

# PyTorch (CUDA 12.1 권장, H200 호환)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# CUDA toolkit
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
```

### 4.2 Python 패키지 설치

```bash
cd ~/grad_school/wooshikwon/weighted_mtp

# requirements.txt 사용
pip install -r requirements_nipa.txt

# Flash Attention 2 (H200 Hopper 아키텍처 최적화)
# PyTorch SDPA가 자동으로 Flash Attention 커널 사용
pip install flash-attn --no-build-isolation
```

### 4.3 환경 검증

```bash
# PyTorch 및 GPU 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

예상 결과:
```
PyTorch: 2.x.x
CUDA: True
GPUs: 4
```

### 4.4 Flash Attention 검증

```bash
python -c "
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
conda activate weighted_mtp
source .env

# 테스트 실행
python -m weighted_mtp.pipelines.run_baseline \
  --config configs/baseline/baseline.yaml \
  --override training.n_epochs=0.01
```

### 5.2 분산 학습 (4-GPU)

```bash
# Baseline MTP
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  -m weighted_mtp.pipelines.run_baseline \
  --config configs/baseline/baseline.yaml

# Critic 사전학습
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  -m weighted_mtp.pipelines.run_critic \
  --config configs/critic/critic.yaml

# Verifiable Reward
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  -m weighted_mtp.pipelines.run_verifiable \
  --config configs/verifiable/verifiable.yaml

# Rho-1
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  -m weighted_mtp.pipelines.run_rho1 \
  --config configs/rho1/rho1.yaml
```

### 5.3 백그라운드 실행 (nohup)

```bash
nohup torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  -m weighted_mtp.pipelines.run_baseline \
  --config configs/baseline/baseline.yaml \
  > logs/baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 로그 확인
tail -f logs/baseline_*.log
```

### 5.4 tmux 사용 (권장)

```bash
# 세션 생성
tmux new -s training

# 학습 실행
conda activate weighted_mtp
source .env
torchrun --nproc_per_node=4 ...

# 세션 분리: Ctrl+B, D
# 세션 재접속: tmux attach -t training
```

---

## Phase 6: 체크포인트 다운로드

### 6.1 결과물 로컬로 가져오기

```bash
# 로컬에서 실행
rsync -avz --progress -e "ssh -p 10507" \
  work@proxy1.nipa2025.ktcloud.com:~/grad_school/wooshikwon/weighted_mtp/storage/checkpoints/ \
  ./storage/checkpoints_nipa/
```

---

## Phase 7: 정리

### 7.1 사용 완료 후 정리

```bash
ssh nipa
cd ~/grad_school/wooshik

# 대용량 데이터 삭제 (모델/데이터셋)
rm -rf weighted_mtp/storage/models/*
rm -rf weighted_mtp/storage/datasets/*

# conda 환경 삭제
conda deactivate
conda env remove -n weighted_mtp -y
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
| 실행 명령 | `uv run python` | `python` |
| 분산 학습 | `uv run torchrun` | `torchrun` |
| 스토리지 | `vessl storage` | 로컬 rsync |
| 환경 | Docker 이미지 | conda |
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

- [ ] conda 환경 생성
- [ ] PyTorch + CUDA 설치
- [ ] requirements 설치
- [ ] flash-attn 설치

### 실행 검증

- [ ] 단일 GPU 테스트
- [ ] 4-GPU 분산 학습 테스트

### 완료 후

- [ ] 체크포인트 다운로드
- [ ] 서버 데이터 정리
