#!/bin/bash
# VESSL Storage 업로드 (최종 확정본)
# Volume: models, datasets, checkpoints (단순화된 이름)
# 파일명이 디렉토리로 인식되는 문제 해결: destination은 디렉토리 경로만 (trailing slash)

set -e

STORAGE_NAME="vessl-storage"

echo "=== VESSL 업로드 시작 ==="
echo "총 예상 크기: ~129GB (models 121GB + datasets 7.8GB + checkpoints 177MB)"
echo "예상 소요 시간: 네트워크 속도에 따라 수 시간"
echo ""

# Models 업로드
echo "=== 1. Models 업로드 ==="
cd storage/models
for model_name in */; do
    model_name=${model_name%/}  # trailing slash 제거
    echo "[$model_name] 업로드 시작..."

    find "$model_name" -type f | while read -r file; do
        # 디렉토리 경로 추출 (파일명 제외)
        dir_path=$(dirname "$file")

        # Destination: 디렉토리 경로만, trailing slash 필수
        dest_dir="volume://$STORAGE_NAME/models/$dir_path/"

        echo "  ↑ $file"
        if vessl storage copy-file "$file" "$dest_dir" 2>&1 >/dev/null; then
            :  # 성공, 아무것도 안 함
        else
            echo "    ✗ 실패"
        fi
    done

    echo "[$model_name] ✓ 완료"
    echo ""
done
cd ../..

echo "=== 2. Datasets 업로드 ==="
cd storage/datasets
for dataset_name in */; do
    dataset_name=${dataset_name%/}  # trailing slash 제거
    echo "[$dataset_name] 업로드 시작..."

    find "$dataset_name" -type f | while read -r file; do
        dir_path=$(dirname "$file")
        dest_dir="volume://$STORAGE_NAME/datasets/$dir_path/"

        echo "  ↑ $file"
        if vessl storage copy-file "$file" "$dest_dir" 2>&1 >/dev/null; then
            :  # 성공, 아무것도 안 함
        else
            echo "    ✗ 실패"
        fi
    done

    echo "[$dataset_name] ✓ 완료"
    echo ""
done
cd ../..

echo "=== 3. Checkpoints 업로드 ==="
if [ -d "storage/checkpoints" ]; then
    cd storage/checkpoints
    checkpoint_files=$(find . -type f)

    if [ -n "$checkpoint_files" ]; then
        echo "$checkpoint_files" | while read -r file; do
            # ./ 제거
            file=${file#./}
            dir_path=$(dirname "$file")
            dest_dir="volume://$STORAGE_NAME/checkpoints/$dir_path/"

            echo "  ↑ $file"
            if vessl storage copy-file "$file" "$dest_dir" 2>&1 >/dev/null; then
                :  # 성공, 아무것도 안 함
            else
                echo "    ✗ 실패"
            fi
        done
        cd ../..
        echo "[Checkpoints] ✓ 완료"
    else
        cd ../..
        echo "[Checkpoints] (비어있음, 스킵)"
    fi
else
    echo "[Checkpoints] (디렉토리 없음, 스킵)"
fi

echo ""
echo "=== 업로드 완료 확인 ==="

echo "[Models Volume]"
vessl storage list-files volume://$STORAGE_NAME/models | head -30

echo ""
echo "[Datasets Volume]"
vessl storage list-files volume://$STORAGE_NAME/datasets | head -30

echo ""
echo "[Checkpoints Volume]"
vessl storage list-files volume://$STORAGE_NAME/checkpoints

echo ""
echo "=== ✓ VESSL Storage 업로드 완료 ==="
