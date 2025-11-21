#!/bin/bash
# NIPA H200: Rho-1 실행
# 사용법: ./scripts/nipa/rho1.sh [--ngpus N] [--batch-size N] [--test]

set -e

# 기본값
NGPUS=4
BATCH_SIZE=""
GRAD_ACCUM=""
TEST_MODE=false

# CLI 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --ngpus) NGPUS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --test) TEST_MODE=true; shift ;;
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
done

# 환경 설정
WORK_DIR="${NIPA_WORK_DIR:-$HOME/grad_school/wooshik/weighted_mtp}"
cd "$WORK_DIR"

# Conda 활성화
source $(conda info --base)/etc/profile.d/conda.sh
conda activate weighted_mtp

# 환경변수
export PYTHONPATH="$WORK_DIR/src:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
if [ -f .env ]; then source .env; fi

# Override 인자
OVERRIDE_ARGS=""
if [ -n "$BATCH_SIZE" ]; then
    OVERRIDE_ARGS="$OVERRIDE_ARGS --override training.batch_size=$BATCH_SIZE"
fi
if [ -n "$GRAD_ACCUM" ]; then
    OVERRIDE_ARGS="$OVERRIDE_ARGS --override training.gradient_accumulation_steps=$GRAD_ACCUM"
fi
if [ "$TEST_MODE" = true ]; then
    OVERRIDE_ARGS="$OVERRIDE_ARGS --override training.n_epochs=0.01"
fi

echo "=== Rho-1 실행 ($NGPUS GPU) ==="

# 실행
if [ "$NGPUS" -eq 1 ]; then
    python -m weighted_mtp.pipelines.run_rho1 \
        --config configs/rho1/rho1.yaml \
        $OVERRIDE_ARGS
else
    torchrun --nproc_per_node=$NGPUS --nnodes=1 --node_rank=0 \
        -m weighted_mtp.pipelines.run_rho1 \
        --config configs/rho1/rho1.yaml \
        $OVERRIDE_ARGS
fi
