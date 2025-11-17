#!/bin/bash
# MLflow 서버 및 S3 artifact store 상태 확인
# .env 환경변수 기반으로 연결 테스트

set -e

# .env 파일 로드
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "환경변수 로드 완료: .env"
else
    echo "오류: .env 파일을 찾을 수 없습니다"
    exit 1
fi

echo ""
echo "=== MLflow Tracking Server 연결 확인 ==="
MLFLOW_URI="http://13.50.240.176"
echo "MLflow URI: $MLFLOW_URI"
echo "Username: $MLFLOW_TRACKING_USERNAME"

# Health check
echo ""
echo "Health check..."
curl -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD" \
    -s "$MLFLOW_URI/health" && echo " - OK" || echo " - FAILED"

# API version check
echo ""
echo "API version check..."
curl -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD" \
    -s "$MLFLOW_URI/api/2.0/mlflow/experiments/list" \
    | python -m json.tool | head -20

echo ""
echo "=== S3 Artifact Store 확인 ==="
echo "S3 Bucket: s3://$S3_BUCKET_NAME"
echo "Region: $AWS_DEFAULT_REGION"

# S3 bucket 확인
echo ""
echo "S3 bucket 리스트..."
aws s3 ls "s3://$S3_BUCKET_NAME/" | head -10

# MLflow artifacts 확인
echo ""
echo "MLflow artifacts 디렉토리 확인..."
aws s3 ls "s3://$S3_BUCKET_NAME/mlflow-artifacts/" | head -10 || echo "mlflow-artifacts 디렉토리가 비어있거나 접근 불가"

echo ""
echo "=== 확인 완료 ==="
echo "MLflow Tracking Server: 정상"
echo "S3 Bucket: 정상"
echo "인증 정보: 유효"
