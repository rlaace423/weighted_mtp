"""Weighted MTP CLI 진입점

Usage:
    # 학습
    python -m weighted_mtp --config configs/baseline/baseline.yaml

    # 평가
    python -m weighted_mtp evaluate --checkpoint storage/checkpoints/baseline/checkpoint_best.pt
"""

import sys

if __name__ == "__main__":
    # evaluate 서브커맨드 체크
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        from weighted_mtp.cli.evaluate import main

        # 'evaluate' 제거하여 argparse가 나머지 인자 파싱하도록 함
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main()
    else:
        # 기본: train CLI
        from weighted_mtp.cli.train import main

        main()
