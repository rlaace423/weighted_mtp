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
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    # Value head가 있는 경우만 저장 (Baseline은 value_head=None)
    if hasattr(adapter, "value_head") and adapter.value_head is not None:
        checkpoint["value_head_state_dict"] = adapter.value_head.state_dict()

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


def load_checkpoint_for_evaluation(
    checkpoint_path: Path,
    device: torch.device,
):
    """평가용 checkpoint 로드 (전체 adapter 로드)

    학습된 모델 전체를 로드하여 평가 모드로 설정합니다.
    load_critic_checkpoint()와 달리 전체 adapter를 로드합니다.

    Args:
        checkpoint_path: Checkpoint 파일 경로
        device: torch.device

    Returns:
        (model, checkpoint_metadata)
        - model: MetaLlamaMTPAdapter (eval 모드, value_head 없음)
        - checkpoint_metadata: {
            "epoch": float,
            "config": dict,  # 학습 설정 (모델 경로 등)
            "val_metrics": dict,
          }

    Raises:
        FileNotFoundError: Checkpoint 파일이 존재하지 않음
        KeyError: checkpoint에 필수 키가 없음

    Examples:
        >>> model, metadata = load_checkpoint_for_evaluation(
        ...     checkpoint_path=Path("storage/checkpoints/baseline/checkpoint_best.pt"),
        ...     device=torch.device("cpu"),
        ... )
        >>> print(metadata["epoch"])
        5.0
        >>> print(metadata["val_metrics"]["val_loss"])
        2.34
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 파일이 존재하지 않습니다: {checkpoint_path}")

    # Checkpoint 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"Checkpoint 로드 완료: {checkpoint_path}")

    # 필수 키 검증
    required_keys = ["adapter_state_dict", "epoch", "val_metrics"]
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        raise KeyError(
            f"Checkpoint에 필수 키가 없습니다: {missing_keys}\n"
            f"사용 가능한 키: {list(checkpoint.keys())}"
        )

    # Config 정보 추출 (checkpoint에 저장된 경우)
    # 없으면 checkpoint 경로에서 추론
    config_info = checkpoint.get("config", {})
    if not config_info:
        # Fallback: checkpoint 경로에서 모델 경로 추론
        # storage/checkpoints/{experiment}/checkpoint_*.pt
        checkpoint_dir = checkpoint_path.parent
        experiment_name = checkpoint_dir.name

        # 기본 모델 경로 추정
        config_info = {
            "model": {
                "path": "storage/models_v2/meta-llama-mtp"  # 기본값
            }
        }
        logger.warning(
            f"Checkpoint에 config 정보가 없습니다. 기본값 사용: {config_info['model']['path']}"
        )

    # Adapter 로드 (MetaLlamaMTPAdapter)
    from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter

    model = MetaLlamaMTPAdapter.from_pretrained(
        model_path=config_info["model"]["path"],
        device=device,
        initialize_value_head=False,  # 평가에는 value head 불필요
    )

    # State dict 로드
    model.load_state_dict(checkpoint["adapter_state_dict"])
    model.eval()  # 평가 모드 설정

    # Metadata 구성
    checkpoint_metadata = {
        "epoch": checkpoint["epoch"],
        "config": config_info,
        "val_metrics": checkpoint["val_metrics"],
    }

    logger.info("평가용 모델 로드 성공")
    logger.info(f"  Epoch: {checkpoint_metadata['epoch']}")
    logger.info(f"  Val loss: {checkpoint_metadata['val_metrics'].get('val_loss', 'N/A')}")

    return model, checkpoint_metadata


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    save_total_limit: int,
) -> None:
    """오래된 중간 checkpoint 삭제

    checkpoint_best.pt와 checkpoint_final.pt는 절대 삭제하지 않음
    checkpoint_epoch_*.pt만 save_total_limit 개수만큼 유지

    Args:
        checkpoint_dir: Checkpoint 디렉터리
        save_total_limit: 유지할 최대 개수
    """
    if not checkpoint_dir.exists():
        return

    # 중간 checkpoint 파일만 수집 (checkpoint_epoch_*.pt)
    epoch_checkpoints = sorted(
        [f for f in checkpoint_dir.glob("checkpoint_epoch_*.pt")],
        key=lambda x: x.stat().st_mtime,
    )

    # 삭제할 파일 개수 계산
    n_to_delete = len(epoch_checkpoints) - save_total_limit

    if n_to_delete > 0:
        for checkpoint_path in epoch_checkpoints[:n_to_delete]:
            logger.info(f"오래된 checkpoint 삭제: {checkpoint_path.name}")
            checkpoint_path.unlink()
