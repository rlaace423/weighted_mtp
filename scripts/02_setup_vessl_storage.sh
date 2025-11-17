#!/bin/bash
# VESSL Storage Volume 생성 및 데이터 업로드
# storage/ 하위 디렉토리를 각각 volume으로 생성

set -e

STORAGE_NAME="vessl-storage"

echo "=== VESSL Volume 생성 ==="

# Models volume 생성
echo "1. weighted-mtp-models volume 생성..."
vessl storage create-volume weighted-mtp-models --storage-name "$STORAGE_NAME" --tag weighted-mtp --tag models 2>/dev/null || echo "  Volume이 이미 존재하거나 생성 실패"

# Datasets volume 생성
echo "2. weighted-mtp-datasets volume 생성..."
vessl storage create-volume weighted-mtp-datasets --storage-name "$STORAGE_NAME" --tag weighted-mtp --tag datasets 2>/dev/null || echo "  Volume이 이미 존재하거나 생성 실패"

# Checkpoints volume 생성 (빈 볼륨, 실행 중 저장용)
echo "3. weighted-mtp-checkpoints volume 생성..."
vessl storage create-volume weighted-mtp-checkpoints --storage-name "$STORAGE_NAME" --tag weighted-mtp --tag checkpoints 2>/dev/null || echo "  Volume이 이미 존재하거나 생성 실패"

echo ""
echo "=== 생성된 Volume 확인 ==="
vessl storage list-volumes --storage-name "$STORAGE_NAME" | grep weighted-mtp

echo ""
echo "=== VESSL Volume 업로드 ==="
echo "주의: 업로드는 시간이 오래 걸릴 수 있습니다 (수십 분 소요)."
echo ""

# Models 업로드
echo "1. Models 업로드 중..."
for model_dir in storage/models/*/; do
    model_name=$(basename "$model_dir")
    echo "  - $model_name 업로드 중..."

    # 디렉토리 내 모든 파일 재귀적으로 업로드
    find "$model_dir" -type f | while read -r file; do
        # 상대 경로 계산
        rel_path="${file#storage/models/}"
        dest_path="volume://$STORAGE_NAME/weighted-mtp-models/$rel_path"

        echo "    업로드: $rel_path"
        vessl storage copy-file "$file" "$dest_path" 2>/dev/null || echo "      실패: $file"
    done
done

echo ""
echo "2. Datasets 업로드 중..."
for dataset_dir in storage/datasets/*/; do
    dataset_name=$(basename "$dataset_dir")
    echo "  - $dataset_name 업로드 중..."

    # 디렉토리 내 모든 파일 재귀적으로 업로드
    find "$dataset_dir" -type f | while read -r file; do
        # 상대 경로 계산
        rel_path="${file#storage/datasets/}"
        dest_path="volume://$STORAGE_NAME/weighted-mtp-datasets/$rel_path"

        echo "    업로드: $rel_path"
        vessl storage copy-file "$file" "$dest_path" 2>/dev/null || echo "      실패: $file"
    done
done

echo ""
echo "=== 업로드 완료 확인 ==="
echo "Models volume:"
vessl storage list-files --storage-name "$STORAGE_NAME" --volume-name weighted-mtp-models | head -20

echo ""
echo "Datasets volume:"
vessl storage list-files --storage-name "$STORAGE_NAME" --volume-name weighted-mtp-datasets | head -20

echo ""
echo "Checkpoints volume (빈 볼륨):"
vessl storage list-files --storage-name "$STORAGE_NAME" --volume-name weighted-mtp-checkpoints 2>/dev/null || echo "  (빈 볼륨)"

echo ""
echo "=== VESSL Storage 준비 완료 ==="
