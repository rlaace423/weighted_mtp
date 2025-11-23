"""평가 파이프라인

Checkpoint를 로드하여 벤치마크 데이터셋에서 Pass@K 평가 수행
"""

import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from tqdm import tqdm

from weighted_mtp.core.logging import setup_logging
from weighted_mtp.data import load_evaluation_dataset
from weighted_mtp.models.tokenizer_utils import load_tokenizer, resolve_tokenizer_path
from weighted_mtp.utils import (
    evaluate_pass_at_k,
    execute_code_with_tests,
    generate_with_mtp,
    load_checkpoint_for_evaluation,
)


def run_evaluation(
    checkpoint_path: str,
    dataset_name: str = "humaneval",
    num_samples_per_task: int = 20,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    device: str = "auto",
    mlflow_enabled: bool = True,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """평가 파이프라인 실행

    Args:
        checkpoint_path: Checkpoint 경로
        dataset_name: "humaneval", "mbpp", "codecontests"
        num_samples_per_task: 각 문제당 생성 개수 (Pass@K 계산용)
        temperature: Sampling temperature
        max_new_tokens: 최대 생성 토큰 수
        device: 디바이스
        mlflow_enabled: MLflow 로깅 여부
        max_tasks: 최대 평가 태스크 수 (테스트용, None=전체)

    Returns:
        {
            "pass_at_k": {"pass@1": 0.2, "pass@5": 0.65, ...},
            "per_task": [{"task_id": ..., "pass@1": ..., ...}, ...],
            "checkpoint_metadata": {...}
        }

    Examples:
        >>> results = run_evaluation(
        ...     checkpoint_path="storage/checkpoints/baseline/checkpoint_best.pt",
        ...     dataset_name="humaneval",
        ...     num_samples_per_task=20,
        ...     temperature=0.2,
        ... )
        >>> print(results["pass_at_k"])
        {'pass@1': 0.24, 'pass@5': 0.68, 'pass@10': 0.85}
    """
    # 1. Setup
    logger = setup_logging("EVALUATION")
    device_obj = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Samples per task: {num_samples_per_task}")
    logger.info(f"Device: {device_obj}")

    # 2. Load checkpoint
    logger.info("Loading checkpoint...")
    model, checkpoint_metadata = load_checkpoint_for_evaluation(
        checkpoint_path=Path(checkpoint_path),
        device=device_obj,
    )
    logger.info(f"Checkpoint epoch: {checkpoint_metadata['epoch']}")
    logger.info(f"Validation loss: {checkpoint_metadata['val_metrics']['val_loss']:.4f}")

    # 3. Load tokenizer
    tokenizer_path = resolve_tokenizer_path(checkpoint_metadata['config']['model']['path'])
    tokenizer = load_tokenizer(tokenizer_path)

    # 4. Load evaluation dataset
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = load_evaluation_dataset(dataset_name, split="test")
    logger.info(f"Dataset size: {len(dataset)}")

    # 5. MLflow setup
    if mlflow_enabled:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.start_run(run_name=f"eval_{dataset_name}_{Path(checkpoint_path).stem}")
            mlflow.log_params({
                "checkpoint": checkpoint_path,
                "dataset": dataset_name,
                "num_samples_per_task": num_samples_per_task,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "checkpoint_epoch": checkpoint_metadata["epoch"],
                "checkpoint_val_loss": checkpoint_metadata["val_metrics"]["val_loss"],
            })
            logger.info(f"MLflow tracking enabled: {tracking_uri}")
            logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        else:
            logger.warning("MLflow enabled but MLFLOW_TRACKING_URI not set. Skipping MLflow.")
            mlflow_enabled = False

    # 6. Evaluation loop
    logger.info("Starting evaluation...")
    all_results = []
    generated_samples = []  # 생성 샘플 저장 (처음 10개)

    # 테스트용 제한
    eval_dataset = dataset if max_tasks is None else dataset.select(range(min(max_tasks, len(dataset))))
    logger.info(f"Evaluating {len(eval_dataset)} tasks")

    for idx, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        task_id = sample["task_id"]
        prompt = sample["instruction"]
        test_code = sample["metadata"]["test"]
        entry_point = sample["metadata"]["entry_point"]

        # Generate N samples
        generated_codes = generate_with_mtp(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_samples_per_task,
            device=device_obj,
        )

        # Execute and check
        task_results = []
        for code in generated_codes:
            passed = execute_code_with_tests(
                code=code,
                test_code=test_code,
                entry_point=entry_point,
                timeout=5,
            )
            task_results.append(passed)

        # Record
        all_results.append({
            "task_id": task_id,
            "results": task_results,
            "num_correct": sum(task_results),
            "num_total": len(task_results),
        })

        # 처음 10개 task의 생성 샘플 저장 (MLflow artifact용)
        if idx < 10:
            generated_samples.append({
                "task_id": task_id,
                "prompt": prompt,
                "generated_codes": generated_codes,
                "results": task_results,
            })

    # 7. Compute Pass@K
    logger.info("Computing Pass@K metrics...")

    # Flatten all results
    flat_results = []
    for task in all_results:
        flat_results.extend(task["results"])

    pass_at_k_metrics = evaluate_pass_at_k(
        flat_results,
        k_values=[1, 5, 10, 20] if num_samples_per_task >= 20 else [1, 5, 10],
    )

    # Per-task Pass@K
    per_task_pass_at_k = []
    for task in all_results:
        task_metrics = evaluate_pass_at_k(
            task["results"],
            k_values=[1, 5, 10],
        )
        per_task_pass_at_k.append({
            "task_id": task["task_id"],
            **task_metrics,
        })

    # 8. Log results
    logger.info("=== Evaluation Results ===")
    for k, v in pass_at_k_metrics.items():
        logger.info(f"{k}: {v:.2%}")

    if mlflow_enabled:
        mlflow.log_metrics(pass_at_k_metrics)

        # Save per-task results
        results_df = pd.DataFrame(per_task_pass_at_k)
        results_csv = f"results_{dataset_name}.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)

        # Save generated samples (처음 10개)
        if generated_samples:
            import json
            samples_jsonl = f"samples_{dataset_name}.jsonl"
            with open(samples_jsonl, "w", encoding="utf-8") as f:
                for sample in generated_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            mlflow.log_artifact(samples_jsonl)

        mlflow.end_run()

    # 9. Return results
    return {
        "pass_at_k": pass_at_k_metrics,
        "per_task": per_task_pass_at_k,
        "checkpoint_metadata": checkpoint_metadata,
    }
