"""MLflow 평가 결과 비교 스크립트

4개 모델의 평가 결과를 MLflow에서 조회하여 시각화 및 비교
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def compare_evaluation_results(
    experiment_name: str,
    dataset: str = "humaneval",
    output_dir: str = ".",
):
    """MLflow에서 평가 결과 비교

    Args:
        experiment_name: MLflow experiment 이름
        dataset: 데이터셋 이름 (humaneval, mbpp, codecontests)
        output_dir: 출력 디렉터리 (차트 및 CSV 저장)

    Returns:
        DataFrame: 모델별 평가 결과

    Examples:
        >>> df = compare_evaluation_results(
        ...     experiment_name="weighted-mtp-evaluation",
        ...     dataset="humaneval",
        ... )
        >>> print(df)
    """
    # MLflow tracking URI 설정
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Warning: MLFLOW_TRACKING_URI 환경변수가 설정되지 않았습니다.")
        print("로컬 MLflow (./mlruns) 사용")
        tracking_uri = "./mlruns"

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    # Experiment 조회
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Error: Experiment '{experiment_name}'을 찾을 수 없습니다.")
            print("\n사용 가능한 experiments:")
            for exp in mlflow.search_experiments():
                print(f"  - {exp.name}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: MLflow experiment 조회 실패: {e}")
        sys.exit(1)

    print(f"Experiment: {experiment.name} (ID: {experiment.experiment_id})")

    # Evaluation runs 조회
    try:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.dataset = '{dataset}'",
            order_by=["start_time DESC"],
        )
    except Exception as e:
        print(f"Error: MLflow runs 조회 실패: {e}")
        sys.exit(1)

    if runs.empty:
        print(f"Error: Dataset '{dataset}'에 대한 평가 결과가 없습니다.")
        print(f"Filter: params.dataset = '{dataset}'")
        sys.exit(1)

    print(f"Found {len(runs)} evaluation runs for dataset '{dataset}'")

    # 결과 추출
    results = []
    for _, run in runs.iterrows():
        run_name = run.get("tags.mlflow.runName", "Unknown")
        checkpoint = run.get("params.checkpoint", "Unknown")

        result = {
            "run_name": run_name,
            "checkpoint": checkpoint,
            "pass@1": run.get("metrics.pass@1", 0.0),
            "pass@5": run.get("metrics.pass@5", 0.0),
            "pass@10": run.get("metrics.pass@10", 0.0),
        }

        # pass@20이 있으면 추가
        if "metrics.pass@20" in run:
            result["pass@20"] = run["metrics.pass@20"]

        results.append(result)

    results_df = pd.DataFrame(results)

    # CSV 저장
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir_path / f"comparison_{dataset}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nCSV 저장: {csv_path}")

    # 테이블 출력
    print("\n=== 평가 결과 비교 ===")
    print(results_df.to_string(index=False))

    # 차트 생성
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(results_df))
    width = 0.2

    # Pass@1, Pass@5, Pass@10 bar chart
    ax.bar(
        [i - width for i in x],
        results_df["pass@1"] * 100,
        width,
        label="Pass@1",
        alpha=0.8,
    )
    ax.bar(
        x,
        results_df["pass@5"] * 100,
        width,
        label="Pass@5",
        alpha=0.8,
    )
    ax.bar(
        [i + width for i in x],
        results_df["pass@10"] * 100,
        width,
        label="Pass@10",
        alpha=0.8,
    )

    # pass@20이 있으면 추가
    if "pass@20" in results_df.columns:
        ax.bar(
            [i + 2 * width for i in x],
            results_df["pass@20"] * 100,
            width,
            label="Pass@20",
            alpha=0.8,
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_title(f"Evaluation Results Comparison ({dataset.upper()})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["run_name"], rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # 차트 저장
    chart_path = output_dir_path / f"comparison_{dataset}.png"
    plt.savefig(chart_path, dpi=150)
    print(f"차트 저장: {chart_path}")

    plt.close()

    return results_df


def main():
    parser = argparse.ArgumentParser(description="MLflow 평가 결과 비교")

    parser.add_argument(
        "--experiment",
        type=str,
        default="weighted-mtp-evaluation",
        help="MLflow experiment 이름 (기본: weighted-mtp-evaluation)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp", "codecontests"],
        help="비교할 데이터셋 (기본: humaneval)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="출력 디렉터리 (기본: 현재 디렉터리)",
    )

    args = parser.parse_args()

    # 비교 실행
    results_df = compare_evaluation_results(
        experiment_name=args.experiment,
        dataset=args.dataset,
        output_dir=args.output_dir,
    )

    print("\n완료!")


if __name__ == "__main__":
    main()
