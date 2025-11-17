"""S3 checkpoint 다운로드 스크립트

MLflow artifact store (S3)에서 checkpoint를 다운로드하여:
1. 로컬 storage/checkpoints/에 저장 (로컬 평가용)
2. VESSL storage에 업로드 (VESSL 평가용)
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
import mlflow
from mlflow.tracking import MlflowClient


def list_experiments() -> list[dict]:
    """MLflow experiments 목록 조회

    Returns:
        [{"name": "exp-name", "id": "exp-id"}, ...]
    """
    client = MlflowClient()
    experiments = client.search_experiments()
    return [{"name": exp.name, "id": exp.experiment_id} for exp in experiments]


def list_runs(experiment_name: str, filter_string: str = "") -> list[dict]:
    """Experiment의 runs 조회

    Args:
        experiment_name: Experiment 이름
        filter_string: MLflow filter (예: "tags.model = 'baseline'")

    Returns:
        [{"run_id": "...", "run_name": "...", "params": {...}, "metrics": {...}}, ...]
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found")
        sys.exit(1)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
    )

    run_list = []
    for _, run in runs.iterrows():
        run_list.append({
            "run_id": run.get("run_id", ""),
            "run_name": run.get("tags.mlflow.runName", "Unknown"),
            "start_time": run.get("start_time", ""),
            "params": {k.replace("params.", ""): v for k, v in run.items() if k.startswith("params.")},
            "metrics": {k.replace("metrics.", ""): v for k, v in run.items() if k.startswith("metrics.")},
        })

    return run_list


def list_checkpoints(run_id: str) -> list[str]:
    """Run의 checkpoint 목록 조회

    Args:
        run_id: MLflow run ID

    Returns:
        ["checkpoint_best.pt", "checkpoint_final.pt", "checkpoint_epoch_5.00.pt", ...]
    """
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path="checkpoints")
    checkpoint_names = [Path(a.path).name for a in artifacts if a.path.startswith("checkpoints/")]
    return sorted(checkpoint_names)


def download_checkpoint(
    experiment_id: str,
    run_id: str,
    checkpoint_name: str,
    output_dir: Path,
) -> Path:
    """S3에서 checkpoint 다운로드

    Args:
        experiment_id: MLflow experiment ID
        run_id: MLflow run ID
        checkpoint_name: Checkpoint 파일명 (예: checkpoint_best.pt)
        output_dir: 저장할 디렉터리

    Returns:
        다운로드된 파일 경로
    """
    # S3 key 구성
    s3_key = f"mlflow-artifacts/{experiment_id}/{run_id}/artifacts/checkpoints/{checkpoint_name}"
    bucket = "wmtp"

    # 출력 경로
    output_path = output_dir / checkpoint_name

    # S3 다운로드
    print(f"Downloading from S3: s3://{bucket}/{s3_key}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, s3_key, str(output_path))
    print(f"Downloaded to: {output_path}")

    return output_path


def upload_to_vessl(local_path: Path, vessl_path: str) -> None:
    """VESSL storage에 업로드

    Args:
        local_path: 로컬 파일 경로
        vessl_path: VESSL storage 경로 (예: /input/checkpoints/checkpoint_best.pt)

    Note:
        VESSL CLI 또는 S3 API를 사용하여 업로드
        현재는 placeholder (실제 구현 필요 시 VESSL CLI 사용)
    """
    print(f"Uploading to VESSL: {local_path} -> {vessl_path}")
    # TODO: VESSL storage 업로드 구현
    # vessl.upload(local_path, vessl_path)
    print("Warning: VESSL upload not implemented yet. Use VESSL CLI manually:")
    print(f"  vessl upload {local_path} {vessl_path}")


def interactive_mode():
    """대화형 모드로 실행"""
    # MLflow tracking URI 설정
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: MLFLOW_TRACKING_URI 환경변수가 설정되지 않았습니다.")
        print("Example: export MLFLOW_TRACKING_URI=http://13.50.240.176")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}\n")

    # 1. Experiment 선택
    print("=== Experiments ===")
    experiments = list_experiments()
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']} (ID: {exp['id']})")

    exp_idx = int(input("\nSelect experiment number: ")) - 1
    selected_exp = experiments[exp_idx]
    print(f"Selected: {selected_exp['name']}\n")

    # 2. Run 선택
    print("=== Runs ===")
    runs = list_runs(selected_exp["name"])
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run['run_name']} (ID: {run['run_id']})")
        if "val_loss" in run["metrics"]:
            print(f"   val_loss: {run['metrics']['val_loss']:.4f}")

    run_idx = int(input("\nSelect run number: ")) - 1
    selected_run = runs[run_idx]
    print(f"Selected: {selected_run['run_name']}\n")

    # 3. Checkpoint 선택
    print("=== Checkpoints ===")
    checkpoints = list_checkpoints(selected_run["run_id"])
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"{i}. {ckpt}")

    ckpt_idx = int(input("\nSelect checkpoint number: ")) - 1
    selected_checkpoint = checkpoints[ckpt_idx]
    print(f"Selected: {selected_checkpoint}\n")

    # 4. 다운로드 모드 선택
    print("=== Download Mode ===")
    print("1. Local (storage/checkpoints/)")
    print("2. VESSL (upload to VESSL storage)")

    mode = int(input("\nSelect mode: "))

    # 5. 다운로드 실행
    if mode == 1:
        output_dir = Path("storage/checkpoints") / selected_run["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = download_checkpoint(
            experiment_id=selected_exp["id"],
            run_id=selected_run["run_id"],
            checkpoint_name=selected_checkpoint,
            output_dir=output_dir,
        )

        print(f"\nSuccess! Downloaded to: {downloaded_path}")

    elif mode == 2:
        # 임시 다운로드
        temp_dir = Path("/tmp/checkpoints")
        temp_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = download_checkpoint(
            experiment_id=selected_exp["id"],
            run_id=selected_run["run_id"],
            checkpoint_name=selected_checkpoint,
            output_dir=temp_dir,
        )

        # VESSL 업로드
        vessl_path = f"/input/checkpoints/{selected_run['run_name']}/{selected_checkpoint}"
        upload_to_vessl(downloaded_path, vessl_path)


def batch_mode(
    experiment_name: str,
    run_id: str,
    checkpoint_type: str,
    output_dir: str,
    vessl: bool,
):
    """배치 모드로 실행

    Args:
        experiment_name: Experiment 이름
        run_id: Run ID (또는 run name으로도 검색 가능)
        checkpoint_type: "best", "final", "latest" 중 하나
        output_dir: 저장할 디렉터리
        vessl: VESSL 업로드 여부
    """
    # MLflow tracking URI 설정
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: MLFLOW_TRACKING_URI 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}\n")

    # Experiment 조회
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found")
        sys.exit(1)

    # Run 조회
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"run_id = '{run_id}'",
    )

    if runs.empty:
        # run_id로 못 찾으면 run_name으로 시도
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_id}'",
        )

    if runs.empty:
        print(f"Error: Run '{run_id}' not found")
        sys.exit(1)

    selected_run_id = runs.iloc[0]["run_id"]

    # Checkpoint 목록 조회
    checkpoints = list_checkpoints(selected_run_id)

    # Checkpoint 선택
    if checkpoint_type == "best":
        checkpoint_name = "checkpoint_best.pt"
    elif checkpoint_type == "final":
        checkpoint_name = "checkpoint_final.pt"
    elif checkpoint_type == "latest":
        # checkpoint_epoch_*.pt 중 가장 최근 것
        epoch_checkpoints = [c for c in checkpoints if c.startswith("checkpoint_epoch_")]
        if not epoch_checkpoints:
            print("Error: No epoch checkpoints found")
            sys.exit(1)
        checkpoint_name = epoch_checkpoints[-1]  # Already sorted
    else:
        print(f"Error: Invalid checkpoint type '{checkpoint_type}'")
        print("Valid types: best, final, latest")
        sys.exit(1)

    if checkpoint_name not in checkpoints:
        print(f"Error: Checkpoint '{checkpoint_name}' not found")
        print(f"Available checkpoints: {checkpoints}")
        sys.exit(1)

    # 다운로드 실행
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded_path = download_checkpoint(
        experiment_id=experiment.experiment_id,
        run_id=selected_run_id,
        checkpoint_name=checkpoint_name,
        output_dir=output_path,
    )

    print(f"\nSuccess! Downloaded to: {downloaded_path}")

    # VESSL 업로드
    if vessl:
        vessl_path = f"/input/checkpoints/{checkpoint_name}"
        upload_to_vessl(downloaded_path, vessl_path)


def main():
    parser = argparse.ArgumentParser(description="S3 checkpoint 다운로드")

    # Interactive vs Batch
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="대화형 모드로 실행",
    )

    # Batch mode arguments
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment 이름 (배치 모드)",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Run ID 또는 Run name (배치 모드)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=["best", "final", "latest"],
        default="best",
        help="Checkpoint 타입 (기본: best)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="storage/checkpoints",
        help="저장할 디렉터리 (기본: storage/checkpoints)",
    )
    parser.add_argument(
        "--vessl",
        action="store_true",
        help="VESSL storage에 업로드",
    )

    args = parser.parse_args()

    # 환경변수 체크 (boto3 credentials)
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("Error: AWS credentials 환경변수가 설정되지 않았습니다.")
        print("Required:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_DEFAULT_REGION")
        sys.exit(1)

    # Interactive or Batch
    if args.interactive:
        interactive_mode()
    else:
        if not args.experiment or not args.run:
            print("Error: --experiment and --run are required in batch mode")
            print("Use --interactive for interactive mode")
            sys.exit(1)

        batch_mode(
            experiment_name=args.experiment,
            run_id=args.run,
            checkpoint_type=args.checkpoint,
            output_dir=args.output_dir,
            vessl=args.vessl,
        )


if __name__ == "__main__":
    main()
