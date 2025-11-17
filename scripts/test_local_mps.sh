#!/bin/bash
# MPS Local Test Script for M3 Mac
# Tests all 3 pipelines (critic, verifiable, rho1) with micro-mtp model

set -e

echo "======================================"
echo "MPS Local Testing - Micro-MTP Model"
echo "======================================"
echo ""
echo "Model: micro-mtp (46M params, 4 layers)"
echo "Device: MPS (Apple Silicon M3)"
echo "Samples: 500 per stage"
echo "Epochs: 0.1 per stage"
echo ""

# 색상 코드
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 에러 핸들링
handle_error() {
    echo -e "${RED}✗ Error occurred in $1${NC}"
    echo -e "${RED}Stopping test execution${NC}"
    exit 1
}

# Stage 1: Critic Pre-training
echo -e "${YELLOW}[1/3] Critic Pre-training${NC}"
echo "--------------------------------------"
echo "Config: configs/critic/critic_local.yaml"
echo "Output: storage/checkpoints/critic/critic-pretrain-local/"
echo ""

PYTHONPATH=src python src/weighted_mtp/pipelines/run_critic.py \
    --config configs/critic/critic_local.yaml \
    || handle_error "Critic Pre-training"

echo -e "${GREEN}✓ Critic Pre-training completed${NC}"
echo ""

# Stage 2: Verifiable WMTP
echo -e "${YELLOW}[2/3] Verifiable WMTP${NC}"
echo "--------------------------------------"
echo "Config: configs/verifiable/verifiable_local.yaml"
echo "Output: storage/checkpoints/verifiable/verifiable-wmtp-local/"
echo ""

PYTHONPATH=src python src/weighted_mtp/pipelines/run_verifiable.py \
    --config configs/verifiable/verifiable_local.yaml \
    || handle_error "Verifiable WMTP"

echo -e "${GREEN}✓ Verifiable WMTP completed${NC}"
echo ""

# Stage 3: Rho-1 WMTP
echo -e "${YELLOW}[3/3] Rho-1 WMTP${NC}"
echo "--------------------------------------"
echo "Config: configs/rho1/rho1_local.yaml"
echo "Output: storage/checkpoints/rho1/rho1-wmtp-local/"
echo ""

PYTHONPATH=src python src/weighted_mtp/pipelines/run_rho1.py \
    --config configs/rho1/rho1_local.yaml \
    || handle_error "Rho-1 WMTP"

echo -e "${GREEN}✓ Rho-1 WMTP completed${NC}"
echo ""

# 결과 요약
echo "======================================"
echo -e "${GREEN}All Tests Completed Successfully!${NC}"
echo "======================================"
echo ""
echo "Checkpoint Locations:"
echo "  1. Critic:     storage/checkpoints/critic/critic-pretrain-local/"
echo "  2. Verifiable: storage/checkpoints/verifiable/verifiable-wmtp-local/"
echo "  3. Rho-1:      storage/checkpoints/rho1/rho1-wmtp-local/"
echo ""
echo "Next Steps:"
echo "  - Review checkpoint files for consistency"
echo "  - Check GPU memory usage logs"
echo "  - Verify loss curves"
echo ""
