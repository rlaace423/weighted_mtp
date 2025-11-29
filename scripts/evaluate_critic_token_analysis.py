"""Critic Token Analysis 평가 스크립트

학습된 Critic 모델의 토큰 단위 value 분석
오답 코드에서 value 급락 지점을 탐지하여 에러 인지 능력 검증

사용법:
    PYTHONPATH=src python scripts/evaluate_critic_token_analysis.py \
        --checkpoint storage/checkpoints/critic/s3-downloaded/checkpoint_epoch_0.20.pt \
        --tokenizer_path storage/models/meta-llama-mtp/tokenizer \
        --n_samples 100 \
        --incorrect_only \
        --output_dir results/token_analysis

메모리 최적화:
    - .pt 파일을 직접 로드하여 메모리 1회만 사용
    - tokenizer_path를 별도 인자로 받아 checkpoint 재로드 방지
    - MPS 환경에서는 자동으로 float32로 변환
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from weighted_mtp.analysis import (
    TokenValueAnalyzer,
    plot_single_sample,
    plot_multiple_samples,
    plot_correct_vs_incorrect,
    plot_value_distribution,
    plot_drop_analysis,
    generate_report,
)
from weighted_mtp.data.datasets import load_dataset
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Critic Token Analysis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Critic checkpoint 경로 (전체 모델 포함)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="codecontests",
        help="데이터셋 이름",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="데이터 스플릿",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="분석할 샘플 수",
    )
    parser.add_argument(
        "--incorrect_only",
        action="store_true",
        help="오답 샘플만 분석",
    )
    parser.add_argument(
        "--correct_only",
        action="store_true",
        help="정답 샘플만 분석",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="최대 시퀀스 길이",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/token_analysis",
        help="출력 디렉터리",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="디바이스 (cuda, mps, cpu, auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="모델 dtype (bfloat16, float16)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="storage/models/meta-llama-mtp/tokenizer",
        help="Tokenizer 경로 (기본값: meta-llama-mtp tokenizer)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="시각화 생성",
    )
    parser.add_argument(
        "--n_plot_samples",
        type=int,
        default=20,
        help="개별 plot 생성할 샘플 수",
    )

    args = parser.parse_args()

    # 출력 디렉터리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 디바이스 설정
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # MPS 환경에서는 float32가 더 안정적 (bfloat16 numpy 변환 이슈)
    dtype = args.dtype
    if device.type == "mps" and dtype == "bfloat16":
        logger.warning("MPS 환경에서 bfloat16 → float32로 변경 (numpy 호환성)")
        dtype = "float32"

    logger.info(f"Device: {device}, dtype: {dtype}")

    # 1. Checkpoint에서 전체 모델 로딩 (메모리 효율적 직접 로드)
    checkpoint_path = Path(args.checkpoint)
    logger.info(f"Checkpoint 로딩: {checkpoint_path}")

    # .pt 파일인 경우 직접 로드 (메모리 1회 사용)
    # 디렉토리인 경우 safetensors 로드
    adapter = MetaLlamaMTPAdapter.from_pretrained(
        model_path=str(checkpoint_path),
        device=str(device),
        dtype=dtype,
    )
    adapter.eval()
    logger.info("모델 로드 완료")

    # 2. Tokenizer 로딩 (CLI 인자로 직접 지정, checkpoint 재로드 불필요)
    tokenizer_path = args.tokenizer_path
    logger.info(f"Tokenizer 로딩: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Analyzer 초기화
    analyzer = TokenValueAnalyzer(
        adapter=adapter,
        tokenizer=tokenizer,
        device=device,
    )

    # 4. 데이터 로딩
    # 샘플링 설정
    if args.incorrect_only:
        use_pairwise = True
        logger.info("Pairwise 로딩 (오답 분석용)")
    elif args.correct_only:
        use_pairwise = False
        logger.info("정답 샘플만 로딩")
    else:
        use_pairwise = True
        logger.info("정답/오답 pair 샘플링")

    logger.info(f"데이터셋 로딩: {args.dataset}/{args.split}")
    dataset = load_dataset(
        dataset_name=args.dataset,
        split=args.split,
        sampling_config={
            "n_samples": args.n_samples,
            "use_pairwise": use_pairwise,
            "difficulty_bins": {"all": [0, 25]},
            "difficulty_weights": {"all": 1.0},
        },
        seed=42,
    )

    logger.info(f"로드된 샘플 수: {len(dataset)}")

    # 5. 분석 실행
    results = []
    for i, sample in enumerate(dataset):
        # Alpaca 템플릿 구성 요소 추출
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
        is_correct = sample.get("is_correct", False)

        # 분석 수행 (학습 시와 동일한 Alpaca 템플릿 적용)
        result = analyzer.analyze_sample_full(
            code_text=output_text,
            is_correct=is_correct,
            max_length=args.max_length,
            instruction=instruction,
            input_text=input_text,
        )

        # 메타데이터 추가
        result["task_id"] = sample.get("task_id", f"sample_{i}")
        result["sample_idx"] = i

        results.append(result)

        # 진행 상황 로깅
        if (i + 1) % 10 == 0:
            logger.info(f"분석 진행: {i + 1}/{len(dataset)}")

    # 6. 결과 저장
    output_path = output_dir / "analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"분석 결과 저장: {output_path}")

    # 7. 시각화 생성
    if args.plot:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        logger.info("시각화 생성 중...")

        # 개별 샘플 plot
        for i, result in enumerate(results[: args.n_plot_samples]):
            plot_single_sample(result, plot_dir / f"sample_{i}.png")

        # 여러 샘플 비교
        plot_multiple_samples(results, plot_dir / "comparison.png")

        # 정답/오답 분리
        correct_results = [r for r in results if r["is_correct"]]
        incorrect_results = [r for r in results if not r["is_correct"]]

        # 정답 vs 오답 비교
        if correct_results and incorrect_results:
            plot_correct_vs_incorrect(
                correct_results, incorrect_results, plot_dir / "correct_vs_incorrect.png"
            )

        # Value 분포
        plot_value_distribution(results, plot_dir / "distribution.png")

        # 급락 분석
        if incorrect_results:
            plot_drop_analysis(results, plot_dir / "drop_analysis.png")

        logger.info(f"시각화 저장 완료: {plot_dir}")

    # 8. 평가 지표 계산 및 저장
    report = generate_report(results)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"평가 지표 저장: {metrics_path}")

    # 9. 요약 통계 출력
    print_summary(results, report)


def print_summary(results: list[dict], report: dict = None):
    """분석 결과 요약 출력

    Args:
        results: analyze_sample_full() 결과 리스트
        report: generate_report() 결과 (선택)
    """
    if not results:
        logger.warning("분석 결과가 없습니다.")
        return

    print("\n" + "=" * 50)
    print("Critic Token Analysis 결과")
    print("=" * 50)

    # 기본 통계
    if report:
        summary = report["summary"]
        print(f"\n전체 샘플: {summary['total_samples']}")
        print(f"  정답: {summary['correct_samples']}")
        print(f"  오답: {summary['incorrect_samples']}")

        # 그룹 비교
        gc = report["group_comparison"]

        if gc["incorrect"]["n_samples"] > 0:
            print("\n오답 샘플 통계:")
            print(f"  Drop rate: {gc['incorrect']['drop_rate']:.2%}")
            print(f"  Early drop rate: {gc['incorrect']['early_drop_rate']:.2%}")
            print(f"  Mean value: {gc['incorrect']['mean_value']:.3f}")
            print(f"  Mean max drop: {gc['incorrect']['mean_max_drop']:.3f}")

        if gc["correct"]["n_samples"] > 0:
            print("\n정답 샘플 통계:")
            print(f"  Drop rate: {gc['correct']['drop_rate']:.2%}")
            print(f"  Mean value: {gc['correct']['mean_value']:.3f}")

        # 평가 지표
        if "evaluation" in report:
            ev = report["evaluation"]
            print("\n평가 지표:")
            print(f"  Discrimination (drop rate gap): {ev['drop_discrimination']:.2%}")
            print(f"  Value gap (correct - incorrect): {ev['value_gap']:.3f}")
            print(f"  False positive rate: {ev['false_positive_rate']:.2%}")
            print(f"  Is effective: {ev['is_effective']}")
    else:
        # Fallback: report 없이 기본 출력
        correct_results = [r for r in results if r["is_correct"]]
        incorrect_results = [r for r in results if not r["is_correct"]]

        print(f"\n전체 샘플: {len(results)}")
        print(f"  정답: {len(correct_results)}")
        print(f"  오답: {len(incorrect_results)}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
