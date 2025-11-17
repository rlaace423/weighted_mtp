"""평가 파이프라인 CLI 진입점

학습된 checkpoint를 로드하여 벤치마크 데이터셋에서 Pass@K 평가 수행
"""

import argparse
import sys
from pathlib import Path

from weighted_mtp.core.env import ensure_env_loaded
from weighted_mtp.core.logging import setup_logging
from weighted_mtp.pipelines.run_evaluation import run_evaluation

# 환경변수 로드 (MLflow credentials 등)
ensure_env_loaded()

logger = setup_logging("EVALUATE")


def main():
    parser = argparse.ArgumentParser(description="Weighted MTP 모델 평가")

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint 경로 (예: storage/checkpoints/baseline/checkpoint_best.pt)",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp", "codecontests"],
        help="평가 데이터셋 (기본: humaneval)",
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="문제당 생성 샘플 수 (Pass@K 계산용, 기본: 20)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0=greedy, 기본: 0.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="최대 생성 토큰 수 (기본: 512)",
    )

    # Environment
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="디바이스 (cuda, cpu, mps, auto, 기본: auto)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="MLflow 로깅 비활성화",
    )

    # Test parameters
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="최대 평가 태스크 수 (테스트용, 기본: 전체)",
    )

    args = parser.parse_args()

    # Checkpoint 파일 존재 확인
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint 파일을 찾을 수 없습니다: {args.checkpoint}")
        sys.exit(1)

    logger.info("=== 평가 파이프라인 시작 ===")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples per task: {args.num_samples}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Device: {args.device}")
    logger.info(f"MLflow: {'disabled' if args.no_mlflow else 'enabled'}")

    # Run evaluation
    try:
        results = run_evaluation(
            checkpoint_path=args.checkpoint,
            dataset_name=args.dataset,
            num_samples_per_task=args.num_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            device=args.device,
            mlflow_enabled=not args.no_mlflow,
            max_tasks=args.max_tasks,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("평가 결과 요약")
        print("=" * 50)
        for k, v in results["pass_at_k"].items():
            print(f"{k}: {v:.2%}")
        print("=" * 50)
        print(f"총 평가 태스크: {len(results['per_task'])}")
        print(f"Checkpoint epoch: {results['checkpoint_metadata']['epoch']}")
        print(f"Checkpoint val_loss: {results['checkpoint_metadata']['val_metrics']['val_loss']:.4f}")
        print("=" * 50)

        logger.info("평가 완료")

    except Exception as e:
        logger.error(f"평가 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
