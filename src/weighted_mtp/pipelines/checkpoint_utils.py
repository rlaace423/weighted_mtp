"""Checkpoint 저장 및 로드 유틸리티

Stage 분리 파이프라인을 위한 checkpoint handoff 지원
MLflow artifact URI 및 local path 모두 지원
"""

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    adapter,
    optimizer: torch.optim.Optimizer,
    epoch: int | float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    checkpoint_path: Path | str,
) -> None:
    """Checkpoint 저장

    Args:
        adapter: MetaLlamaMTPAdapter (Value head 포함)
        optimizer: torch.optim.Optimizer
        epoch: 현재 epoch (fractional epoch 지원)
        train_metrics: Training metrics
        val_metrics: Validation metrics
        checkpoint_path: 저장 경로

    Saved checkpoint format:
        {
            "epoch": float,
            "adapter_state_dict": dict,  # 전체 adapter (Stage 2 final checkpoint용)
            "value_head_state_dict": dict,  # Value head만 (Stage 2 초기화용)
            "optimizer_state_dict": dict,
            "train_metrics": dict,
            "val_metrics": dict,
        }
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "adapter_state_dict": adapter.state_dict(),
        "value_head_state_dict": adapter.value_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint 저장 완료: {checkpoint_path}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Val loss: {val_metrics.get('val_loss', 'N/A')}")


def load_critic_checkpoint(
    checkpoint_path: str,
    adapter,
    device: torch.device,
) -> dict[str, Any]:
    """Critic checkpoint 로드 (Stage 2 초기화용)

    Local path 또는 MLflow artifact URI 자동 감지 및 로드
    Value head state dict만 adapter에 로드

    Args:
        checkpoint_path: Checkpoint 경로
            - Local path: "storage/checkpoints/critic/.../checkpoint_best.pt"
            - MLflow URI: "mlflow://8/{run_id}/artifacts/checkpoints/checkpoint_best.pt"
        adapter: MetaLlamaMTPAdapter (Value head에 로드)
        device: torch.device

    Returns:
        checkpoint: Loaded checkpoint dict (로깅용)

    Raises:
        FileNotFoundError: Checkpoint 파일이 존재하지 않음
        KeyError: checkpoint에 "value_head_state_dict" 키가 없음
    """
    # MLflow artifact URI 감지
    if checkpoint_path.startswith("mlflow://"):
        logger.info(f"MLflow artifact 다운로드 중: {checkpoint_path}")
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "MLflow artifact URI를 사용하려면 mlflow 설치 필요: pip install mlflow"
            ) from e

        # MLflow artifact 다운로드
        local_path = mlflow.artifacts.download_artifacts(checkpoint_path)
        checkpoint = torch.load(local_path, map_location=device, weights_only=False)
        logger.info(f"MLflow artifact 다운로드 완료: {local_path}")
    else:
        # Local path
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint 파일이 존재하지 않습니다: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logger.info(f"Local checkpoint 로드 완료: {checkpoint_path}")

    # Value head state dict 검증
    if "value_head_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint에 'value_head_state_dict' 키가 없습니다. "
            f"사용 가능한 키: {list(checkpoint.keys())}"
        )

    # Value head에만 로드 (전체 adapter는 로드하지 않음)
    adapter.value_head.load_state_dict(checkpoint["value_head_state_dict"])

    # 로딩 정보 출력
    logger.info("Critic checkpoint 로드 성공")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if "val_metrics" in checkpoint:
        val_loss = checkpoint["val_metrics"].get("val_loss", "N/A")
        logger.info(f"  Val loss: {val_loss}")

    return checkpoint
