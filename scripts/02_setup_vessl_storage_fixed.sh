#!/bin/bash
# VESSL Storage Volume 업로드 (수정본)
# 파일명을 디렉토리로 인식하는 문제 해결

set -e

STORAGE_NAME="vessl-storage"

echo "=== VESSL Volume 확인 ==="
vessl storage list-volumes --storage-name "$STORAGE_NAME" | grep weighted-mtp

echo ""
echo "=== VESSL Volume 업로드 ==="
echo "주의: 업로드는 시간이 오래 걸릴 수 있습니다 (수 시간 소요)."
echo "총 크기: ~129GB (models 121GB + datasets 7.8GB + checkpoints 177MB)"
echo ""

# Models 업로드
echo "1. Models 업로드 중..."
for model_dir in storage/models/*/; do
    model_name=$(basename "$model_dir")
    echo "  - $model_name 업로드 시작..."

    # 디렉토리 내 모든 파일 재귀적으로 업로드
    find "$model_dir" -type f | while read -r file; do
        # 상대 경로 계산 (storage/models/ 제거)
        rel_path="${file#storage/models/}"

        # 디렉토리 경로만 추출 (파일명 제외)
        dir_path=$(dirname "$rel_path")

        # destination은 디렉토리까지만 지정 (파일명 제외)
        dest_dir="volume://$STORAGE_NAME/weighted-mtp-models/$dir_path"

        echo "    업로드: $rel_path → $dest_dir/"
        vessl storage copy-file "$file" "$dest_dir/" 2>&1 || echo "      실패: $file"
    done

    echo "  ✓ $model_name 업로드 완료"
done

echo ""
echo "2. Datasets 업로드 중..."
for dataset_dir in storage/datasets/*/; do
    dataset_name=$(basename "$dataset_dir")
    echo "  - $dataset_name 업로드 시작..."

    # 디렉토리 내 모든 파일 재귀적으로 업로드
    find "$dataset_dir" -type f | while read -r file; do
        # 상대 경로 계산
        rel_path="${file#storage/datasets/}"

        # 디렉토리 경로만 추출
        dir_path=$(dirname "$rel_path")

        # destination은 디렉토리까지만
        dest_dir="volume://$STORAGE_NAME/weighted-mtp-datasets/$dir_path"

        echo "    업로드: $rel_path → $dest_dir/"
        vessl storage copy-file "$file" "$dest_dir/" 2>&1 || echo "      실패: $file"
    done

    echo "  ✓ $dataset_name 업로드 완료"
done

echo ""
echo "3. Checkpoints 업로드 중..."
if [ -d "storage/checkpoints" ]; then
    find storage/checkpoints -type f | while read -r file; do
        # 상대 경로 계산
        rel_path="${file#storage/checkpoints/}"

        # 디렉토리 경로만 추출
        dir_path=$(dirname "$rel_path")

        # destination은 디렉토리까지만
        dest_dir="volume://$STORAGE_NAME/weighted-mtp-checkpoints/$dir_path"

        echo "  업로드: $rel_path → $dest_dir/"
        vessl storage copy-file "$file" "$dest_dir/" 2>&1 || echo "    실패: $file"
    done
    echo "  ✓ Checkpoints 업로드 완료"
else
    echo "  (checkpoints 디렉토리가 비어있음)"
fi

echo ""
echo "=== 업로드 완료 확인 ==="
echo "Models volume:"
vessl storage list-files volume://vessl-storage/weighted-mtp-models | head -20

echo ""
echo "Datasets volume:"
vessl storage list-files volume://vessl-storage/weighted-mtp-datasets | head -20

echo ""
echo "Checkpoints volume:"
vessl storage list-files volume://vessl-storage/weighted-mtp-checkpoints | head -20

echo ""
echo "=== VESSL Storage 업로드 완료 ==="
