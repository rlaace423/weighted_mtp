#!/bin/bash
# GitHub에 코드 push
# VESSL에서 repo를 가져올 수 있도록 준비

set -e

echo "=== Git Status 확인 ==="
git status

echo ""
echo "=== Main 브랜치로 Push ==="
read -p "main 브랜치에 push하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main
    echo "Push 완료: git@github.com:wooshikwon/weighted_mtp.git"
else
    echo "Push 취소"
    exit 0
fi

echo ""
echo "=== 최신 커밋 확인 ==="
git log --oneline -3
