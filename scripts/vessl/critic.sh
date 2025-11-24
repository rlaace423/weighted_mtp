#!/bin/bash
# VESSL Run: Critic Pretraining (Unified)
#
# 사용법:
#   ./scripts/vessl/critic.sh --ngpus 4
#   ./scripts/vessl/critic.sh --ngpus 4 --value-head sigmoid
#   ./scripts/vessl/critic.sh --ngpus 4 --value-head mlp
#   ./scripts/vessl/critic.sh --ngpus 1 --batch-size 8 --grad-accum 2
#   ./scripts/vessl/critic.sh --ngpus 4 --max-samples 50000

set -e

# 기본값
NGPUS=4
BATCH_SIZE=""
GRAD_ACCUM=""
MAX_SAMPLES=""
VALUE_HEAD="sigmoid"  # sigmoid, linear, mlp
USE_CUSTOM_IMAGE=false

# CLI 인자 파싱
while [[ $# -gt 0 ]]; do
  case $1 in
    --ngpus)
      NGPUS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad-accum)
      GRAD_ACCUM="$2"
      shift 2
      ;;
    --use-custom-image)
      USE_CUSTOM_IMAGE=true
      shift 1
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --value-head)
      VALUE_HEAD="$2"
      shift 2
      ;;
    *)
      echo "알 수 없는 옵션: $1"
      echo "사용법: $0 --ngpus <1|2|4> [--value-head sigmoid|linear|mlp] [--batch-size N] [--grad-accum N] [--max-samples N] [--use-custom-image]"
      exit 1
      ;;
  esac
done

# value_head 검증
case $VALUE_HEAD in
  sigmoid|linear|mlp) ;;
  *)
    echo "오류: VALUE_HEAD=$VALUE_HEAD (sigmoid, linear, mlp만 지원)"
    exit 1
    ;;
esac

# Preset 자동 선택
case $NGPUS in
  1) PRESET="gpu-a100-80g-small" ;;
  2) PRESET="gpu-a100-80g-medium" ;;
  4) PRESET="gpu-a100-80g-large" ;;
  *)
    echo "오류: NGPUS=$NGPUS (1, 2, 4만 지원)"
    exit 1
    ;;
esac

# Config 설정 (value_head에 따라 선택)
CONFIG="configs/critic/critic_${VALUE_HEAD}.yaml"
if [ ! -f "$CONFIG" ]; then
  echo "오류: Config 파일을 찾을 수 없습니다: $CONFIG"
  exit 1
fi

# Override 인자 생성
OVERRIDE_ARGS=""
if [ -n "$BATCH_SIZE" ]; then
  OVERRIDE_ARGS="$OVERRIDE_ARGS --override training.batch_size=$BATCH_SIZE"
fi
if [ -n "$GRAD_ACCUM" ]; then
  OVERRIDE_ARGS="$OVERRIDE_ARGS --override training.gradient_accumulation_steps=$GRAD_ACCUM"
fi
if [ -n "$MAX_SAMPLES" ]; then
  OVERRIDE_ARGS="$OVERRIDE_ARGS --override data_sampling.problems.max_samples=$MAX_SAMPLES"
fi

# Train command 생성
if [ "$NGPUS" -eq 1 ]; then
  TRAIN_CMD="uv run python -m weighted_mtp.pipelines.run_critic --config $CONFIG$OVERRIDE_ARGS"
  NCCL_DEBUG=""
else
  TRAIN_CMD="uv run torchrun --nproc_per_node=$NGPUS --nnodes=1 --node_rank=0 -m weighted_mtp.pipelines.run_critic --config $CONFIG$OVERRIDE_ARGS"
  # NCCL 환경변수 (Phase 1: 최소 침습적, 성능 저하 없음)
  NCCL_DEBUG="\n  NCCL_TIMEOUT: \"3600\"\n  NCCL_IB_DISABLE: \"1\"\n  NCCL_SOCKET_IFNAME: \"eth0\"\n  NCCL_DEBUG: \"INFO\""
fi

# .env 파일 로드
if [ -f .env ]; then
    source .env
    echo "환경변수 로드 완료: .env"
else
    echo "오류: .env 파일을 찾을 수 없습니다"
    exit 1
fi

# 임시 YAML 파일 생성
TEMP_YAML=$(mktemp)
cp scripts/vessl/critic.yaml.template "$TEMP_YAML"

# 변수 치환
# Image 및 Setup Command 설정
if [ "$USE_CUSTOM_IMAGE" = true ]; then
  IMAGE="ghcr.io/wooshikwon/weighted_mtp:main"
  SETUP_COMMANDS="
      echo \"=== Storage 심볼릭 링크 생성 ===\"
      mkdir -p storage
      ln -s /vessl/models storage/models
      ln -s /vessl/datasets storage/datasets
      ln -s /vessl/checkpoints storage/checkpoints
      ls -lh storage/
  "
else
  IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
  SETUP_COMMANDS="
      echo \"=== 시스템 패키지 설치 ===\"
      apt-get update && apt-get install -y curl git

      echo \"=== UV 설치 ===\"
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH=\"\$HOME/.local/bin:\$PATH\"
      uv --version

      echo \"=== Storage 심볼릭 링크 생성 ===\"
      mkdir -p storage
      ln -s /vessl/models storage/models
      ln -s /vessl/datasets storage/datasets
      ln -s /vessl/checkpoints storage/checkpoints
      ls -lh storage/

      echo \"=== Python 의존성 설치 ===\"
      uv sync --frozen
  "
fi

# 변수 치환
sed -i.bak "s|{{NGPUS}}|$NGPUS|g" "$TEMP_YAML"
sed -i.bak "s|{{PRESET}}|$PRESET|g" "$TEMP_YAML"
sed -i.bak "s|{{TRAIN_COMMAND}}|$TRAIN_CMD|g" "$TEMP_YAML"
sed -i.bak "s|{{NCCL_DEBUG}}|$NCCL_DEBUG|g" "$TEMP_YAML"
sed -i.bak "s|{{IMAGE}}|$IMAGE|g" "$TEMP_YAML"

# SETUP_COMMANDS 치환 (임시 파일 사용 - 개행 및 특수문자 안전 처리)
SETUP_FILE=$(mktemp)
printf "%s" "$SETUP_COMMANDS" > "$SETUP_FILE"
sed -i.bak "/{{SETUP_COMMANDS}}/{
r $SETUP_FILE
d
}" "$TEMP_YAML"
rm -f "$SETUP_FILE"

# 환경변수 치환
sed -i.bak "s|{{MLFLOW_TRACKING_USERNAME}}|$MLFLOW_TRACKING_USERNAME|g" "$TEMP_YAML"
sed -i.bak "s|{{MLFLOW_TRACKING_PASSWORD}}|$MLFLOW_TRACKING_PASSWORD|g" "$TEMP_YAML"
sed -i.bak "s|{{AWS_ACCESS_KEY_ID}}|$AWS_ACCESS_KEY_ID|g" "$TEMP_YAML"
sed -i.bak "s|{{AWS_SECRET_ACCESS_KEY}}|$AWS_SECRET_ACCESS_KEY|g" "$TEMP_YAML"
sed -i.bak "s|{{AWS_DEFAULT_REGION}}|$AWS_DEFAULT_REGION|g" "$TEMP_YAML"
sed -i.bak "s|{{HF_TOKEN}}|$HF_TOKEN|g" "$TEMP_YAML"

echo "환경변수 치환 완료"
echo "임시 YAML: $TEMP_YAML"

# 설정 요약 출력
echo ""
echo "=== 실행 설정 ==="
echo "GPU 개수: $NGPUS"
echo "Value Head: $VALUE_HEAD"
[ -n "$BATCH_SIZE" ] && echo "Batch Size: $BATCH_SIZE (override)"
[ -n "$GRAD_ACCUM" ] && echo "Gradient Accumulation: $GRAD_ACCUM (override)"
[ -n "$MAX_SAMPLES" ] && echo "Max Samples: $MAX_SAMPLES (override)"

# VESSL Run 실행
echo ""
echo "=== VESSL Run 실행: Critic Pretraining ($NGPUS-GPU) ==="
vessl run create -f "$TEMP_YAML"

# 정리
rm -f "$TEMP_YAML" "${TEMP_YAML}.bak"

echo ""
echo "=== 실행 완료 ==="
echo "VESSL 웹 UI에서 실행 상태를 확인하세요: https://vessl.ai"
echo "MLflow UI: http://13.50.240.176"
echo ""
echo "참고: Critic 학습 완료 후 checkpoint를 Verifiable 파이프라인에서 사용합니다."
