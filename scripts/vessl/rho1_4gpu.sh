#!/bin/bash
# VESSL Run: Rho-1 WMTP (A100 4-GPU)

set -e

# .env 파일 로드
if [ -f .env ]; then
    source .env
    echo "환경변수 로드 완료: .env"
else
    echo "오류: .env 파일을 찾을 수 없습니다"
    exit 1
fi

# 임시 YAML 파일 생성 (환경변수 치환)
TEMP_YAML=$(mktemp)
cp scripts/vessl/rho1_4gpu.yaml "$TEMP_YAML"

# 환경변수 치환
sed -i.bak "s|{{MLFLOW_TRACKING_USERNAME}}|$MLFLOW_TRACKING_USERNAME|g" "$TEMP_YAML"
sed -i.bak "s|{{MLFLOW_TRACKING_PASSWORD}}|$MLFLOW_TRACKING_PASSWORD|g" "$TEMP_YAML"
sed -i.bak "s|{{AWS_ACCESS_KEY_ID}}|$AWS_ACCESS_KEY_ID|g" "$TEMP_YAML"
sed -i.bak "s|{{AWS_SECRET_ACCESS_KEY}}|$AWS_SECRET_ACCESS_KEY|g" "$TEMP_YAML"
sed -i.bak "s|{{AWS_DEFAULT_REGION}}|$AWS_DEFAULT_REGION|g" "$TEMP_YAML"
sed -i.bak "s|{{HF_TOKEN}}|$HF_TOKEN|g" "$TEMP_YAML"

echo "환경변수 치환 완료"
echo "임시 YAML: $TEMP_YAML"

# VESSL Run 실행
echo ""
echo "=== VESSL Run 실행: Rho-1 WMTP (4-GPU) ==="
vessl run create -f "$TEMP_YAML"

# 정리
rm -f "$TEMP_YAML" "${TEMP_YAML}.bak"

echo ""
echo "=== 실행 완료 ==="
echo "VESSL 웹 UI에서 실행 상태를 확인하세요: https://vessl.ai"
echo "MLflow UI: http://13.50.240.176"
echo ""
echo "참고: Reference model을 사용하여 loss 기반 weighting을 수행합니다."
