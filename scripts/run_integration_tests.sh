#!/bin/bash
# Integration Test Runner
# DDP 테스트는 torchrun으로, 나머지는 pytest로 실행

set -e  # 에러 발생 시 중단

echo "======================================"
echo "Running Integration Tests"
echo "======================================"

# 환경변수 설정
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

# 1. 단일 프로세스 테스트 (일반 pytest)
echo ""
echo "[1/3] Running single-process integration tests..."
echo "--------------------------------------"
pytest tests/integration/test_checkpoint_race_condition.py \
       tests/integration/test_pipeline_baseline.py \
       tests/integration/test_pipeline_critic.py \
       tests/integration/test_pipeline_rho1.py \
       tests/integration/test_pipeline_verifiable.py \
       tests/integration/test_pipeline_evaluation.py \
       -v --tb=short

# 2. DDP 테스트 (torchrun)
echo ""
echo "[2/3] Running DDP multiprocess tests..."
echo "--------------------------------------"
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
         -m pytest tests/integration/test_ddp_multiprocess.py \
         -v --tb=short

# 3. Checkpoint sync 테스트 (torchrun)
echo ""
echo "[3/3] Running checkpoint sync tests..."
echo "--------------------------------------"
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
         -m pytest tests/integration/test_checkpoint_sync.py \
         -v --tb=short

echo ""
echo "======================================"
echo "All integration tests completed!"
echo "======================================"
