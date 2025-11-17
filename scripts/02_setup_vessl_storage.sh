#!/bin/bash
# VESSL Storage Volume 생성 및 데이터 업로드
# storage/ 하위 디렉토리를 각각 volume으로 생성

set -e

echo "=== VESSL Volume 생성 ==="

# Models volume 생성
echo "1. weighted-mtp-models volume 생성..."
vessl volume create weighted-mtp-models || echo "Volume이 이미 존재하거나 생성 실패"

# Datasets volume 생성
echo "2. weighted-mtp-datasets volume 생성..."
vessl volume create weighted-mtp-datasets || echo "Volume이 이미 존재하거나 생성 실패"

# Checkpoints volume 생성 (빈 볼륨, 실행 중 저장용)
echo "3. weighted-mtp-checkpoints volume 생성..."
vessl volume create weighted-mtp-checkpoints || echo "Volume이 이미 존재하거나 생성 실패"

echo ""
echo "=== VESSL Volume 업로드 ==="
echo "주의: 업로드는 시간이 오래 걸릴 수 있습니다."
echo "각 모델/데이터셋 디렉토리를 개별적으로 업로드합니다."

# Models 업로드
echo ""
echo "1. Models 업로드 중..."
cd storage/models_v2
for model_dir in */; do
    model_name="${model_dir%/}"
    echo "  - $model_name 업로드 중..."
    vessl volume upload weighted-mtp-models "$model_name" "$model_name/" || echo "    업로드 실패: $model_name"
done
cd ../..

# Datasets 업로드
echo ""
echo "2. Datasets 업로드 중..."
cd storage/datasets_v2
for dataset_dir in */; do
    dataset_name="${dataset_dir%/}"
    echo "  - $dataset_name 업로드 중..."
    vessl volume upload weighted-mtp-datasets "$dataset_name" "$dataset_name/" || echo "    업로드 실패: $dataset_name"
done
cd ../..

echo ""
echo "=== 업로드 완료 확인 ==="
echo "Models volume:"
vessl volume ls weighted-mtp-models

echo ""
echo "Datasets volume:"
vessl volume ls weighted-mtp-datasets

echo ""
echo "Checkpoints volume (빈 볼륨):"
vessl volume ls weighted-mtp-checkpoints

echo ""
echo "=== VESSL Storage 준비 완료 ==="
