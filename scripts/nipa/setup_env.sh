#!/bin/bash
# NIPA H200 서버 환경 설정 스크립트
# 사용법: source scripts/nipa/setup_env.sh

set -e

echo "=== NIPA H200 환경 설정 ==="

# 작업 디렉토리 확인
WORK_DIR="${NIPA_WORK_DIR:-$HOME/grad_school/wooshik/weighted_mtp}"
if [ ! -d "$WORK_DIR" ]; then
    echo "오류: 작업 디렉토리가 없습니다: $WORK_DIR"
    exit 1
fi

cd "$WORK_DIR"

# Conda 환경 확인/생성
if ! conda info --envs | grep -q "weighted_mtp"; then
    echo "=== Conda 환경 생성 ==="
    conda create -n weighted_mtp python=3.10 -y
fi

# 환경 활성화
echo "=== Conda 환경 활성화 ==="
conda activate weighted_mtp

# PyTorch 설치 확인
if ! python -c "import torch" 2>/dev/null; then
    echo "=== PyTorch 설치 ==="
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
    conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
fi

# Python 패키지 설치
echo "=== Python 패키지 설치 ==="
pip install -r requirements_nipa.txt

# Flash Attention
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "=== Flash Attention 설치 ==="
    pip install flash-attn --no-build-isolation
fi

# PYTHONPATH 설정
export PYTHONPATH="$WORK_DIR/src:$PYTHONPATH"

# 환경변수 로드
if [ -f .env ]; then
    source .env
    echo "환경변수 로드 완료"
else
    echo "경고: .env 파일이 없습니다"
fi

# GPU 확인
echo ""
echo "=== GPU 상태 확인 ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "=== 환경 설정 완료 ==="
echo "작업 디렉토리: $WORK_DIR"
