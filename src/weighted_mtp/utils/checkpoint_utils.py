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
    config: dict | None = None,
    s3_upload: bool = False,
    mlflow_run_id: str | None = None,
    td_ema_state: dict | None = None,
) -> None:
    """Checkpoint 저장 (FSDP 지원, S3 업로드 옵션)

    FSDP 환경에서는 모든 rank가 state_dict gathering에 참여해야 하며,
    실제 파일 저장 및 S3 업로드는 rank 0만 수행합니다.

    Args:
        adapter: MetaLlamaMTPAdapter (FSDP-wrapped 또는 일반 모델)
        optimizer: torch.optim.Optimizer
        epoch: 현재 epoch (fractional epoch 지원)
        train_metrics: Training metrics
        val_metrics: Validation metrics
        checkpoint_path: 저장 경로
        config: 학습 설정 정보 (모델 경로 등, 평가 시 필요)
        s3_upload: S3 업로드 여부 (MLflow artifact로 업로드)
        mlflow_run_id: MLflow run ID (S3 업로드 시 스레드 안전을 위해 필요)
        td_ema_state: TD EMA 통계 state dict (TDStatsEMA.state_dict())

    Saved checkpoint format:
        {
            "epoch": float,
            "adapter_state_dict": dict,
            "optimizer_state_dict": dict,
            "train_metrics": dict,
            "val_metrics": dict,
            "config": dict,
            "td_ema_state": dict,  # TD EMA 통계 (선택적)
        }
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        StateDictType,
        FullStateDictConfig,
    )

    checkpoint_path = Path(checkpoint_path)

    # FSDP Full state dict gathering (모든 rank가 참여해야 함)
    if isinstance(adapter, FSDP):
        with FSDP.state_dict_type(
            adapter,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            adapter_state_dict = adapter.state_dict()

        # rank 0만 실제 저장 수행
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    else:
        # 일반 모델 (single-device 환경)
        adapter_state_dict = adapter.state_dict()

    # 이하 저장 로직 (rank 0만 실행)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "adapter_state_dict": adapter_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": config,
        "td_ema_state": td_ema_state,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint 저장 완료: {checkpoint_path}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Val loss: {val_metrics.get('val_loss', 'N/A')}")

    # S3 업로드 (비동기, MlflowClient 사용)
    if s3_upload and mlflow_run_id:
        from weighted_mtp.utils.s3_utils import s3_upload_executor, upload_to_s3_async
        s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, mlflow_run_id)
        logger.info(f"S3 업로드 예약: {checkpoint_path.name} -> run_id={mlflow_run_id}")
    elif s3_upload and not mlflow_run_id:
        logger.warning(f"S3 업로드 건너뜀 (run_id 없음): {checkpoint_path.name}")


def load_checkpoint_for_evaluation(
    checkpoint_path: Path,
    device: torch.device,
    initialize_value_head: bool = False,
):
    """평가용 checkpoint 로드 (전체 adapter 로드)

    학습된 모델 전체를 로드하여 평가 모드로 설정합니다.
    load_critic_checkpoint()와 달리 전체 adapter를 로드합니다.

    Args:
        checkpoint_path: Checkpoint 파일 경로
        device: torch.device
        initialize_value_head: Value head 초기화 여부 (critic 평가 시 True)

    Returns:
        (model, checkpoint_metadata)
        - model: MetaLlamaMTPAdapter (eval 모드)
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
                "path": "storage/models/meta-llama-mtp"  # 기본값
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
        initialize_value_head=initialize_value_head,
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


def save_lora_checkpoint(
    adapter,
    optimizer: torch.optim.Optimizer,
    epoch: int | float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    checkpoint_path: Path | str,
    config: dict | None = None,
    s3_upload: bool = False,
    mlflow_run_id: str | None = None,
    save_value_head: bool = True,
    td_ema_state: dict | None = None,
) -> None:
    """LoRA 전용 checkpoint 저장 (LoRA 파라미터 + value head만)

    전체 모델 대신 LoRA 파라미터와 value head만 저장하여 checkpoint 크기를 대폭 줄임.
    Base model은 별도로 유지하고, 추론 시 base model + LoRA checkpoint 조합으로 사용.

    Args:
        adapter: MetaLlamaMTPAdapter (FSDP-wrapped 또는 일반 모델, LoRA 적용됨)
        optimizer: torch.optim.Optimizer
        epoch: 현재 epoch (fractional epoch 지원)
        train_metrics: Training metrics
        val_metrics: Validation metrics
        checkpoint_path: 저장 경로
        config: 학습 설정 정보 (base model 경로 등)
        s3_upload: S3 업로드 여부
        mlflow_run_id: MLflow run ID
        save_value_head: Value head도 함께 저장할지 여부
        td_ema_state: TD EMA 통계 state dict (TDStatsEMA.state_dict())

    Saved checkpoint format:
        {
            "epoch": float,
            "lora_state_dict": dict,  # LoRA 파라미터만
            "value_head_state_dict": dict,  # Value head (선택적)
            "optimizer_state_dict": dict,
            "train_metrics": dict,
            "val_metrics": dict,
            "config": dict,
            "lora_config": dict,  # LoRA 설정 (rank, alpha 등)
            "td_ema_state": dict,  # TD EMA 통계 (선택적)
        }
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        StateDictType,
        FullStateDictConfig,
    )

    checkpoint_path = Path(checkpoint_path)

    # FSDP Full state dict gathering (모든 rank가 참여해야 함)
    if isinstance(adapter, FSDP):
        with FSDP.state_dict_type(
            adapter,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            full_state_dict = adapter.state_dict()

        # rank 0만 실제 저장 수행
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        unwrapped = adapter.module
    else:
        full_state_dict = adapter.state_dict()
        unwrapped = adapter

    # LoRA 파라미터만 추출
    lora_state_dict = {
        k: v for k, v in full_state_dict.items()
        if "lora_A" in k or "lora_B" in k
    }

    # Value head 파라미터 추출 (선택적)
    value_head_state_dict = {}
    if save_value_head:
        value_head_state_dict = {
            k: v for k, v in full_state_dict.items()
            if "value_head" in k
        }

    # LoRA config 추출 (adapter에서)
    lora_config = None
    if hasattr(unwrapped, "lora_enabled") and unwrapped.lora_enabled:
        # transformer의 첫 번째 LoRALinear에서 config 추출
        for name, module in unwrapped.transformer.named_modules():
            if hasattr(module, "rank") and hasattr(module, "alpha"):
                lora_config = {
                    "rank": module.rank,
                    "alpha": module.alpha,
                }
                break

    # 크기 비교 로깅
    full_size = sum(v.numel() for v in full_state_dict.values())
    lora_size = sum(v.numel() for v in lora_state_dict.values())
    vh_size = sum(v.numel() for v in value_head_state_dict.values())
    saved_size = lora_size + vh_size

    logger.info(f"LoRA checkpoint 저장:")
    logger.info(f"  전체 모델: {full_size:,} params")
    logger.info(f"  LoRA: {lora_size:,} params ({lora_size/full_size*100:.2f}%)")
    if save_value_head:
        logger.info(f"  Value head: {vh_size:,} params ({vh_size/full_size*100:.2f}%)")
    logger.info(f"  저장 크기: {saved_size:,} params ({saved_size/full_size*100:.2f}%)")

    # 저장
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "lora_state_dict": lora_state_dict,
        "value_head_state_dict": value_head_state_dict if save_value_head else {},
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": config,
        "lora_config": lora_config,
        "checkpoint_type": "lora",
        "td_ema_state": td_ema_state,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"LoRA checkpoint 저장 완료: {checkpoint_path}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Val loss: {val_metrics.get('val_loss', 'N/A')}")

    # S3 업로드 (비동기)
    if s3_upload and mlflow_run_id:
        from weighted_mtp.utils.s3_utils import s3_upload_executor, upload_to_s3_async
        s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, mlflow_run_id)
        logger.info(f"S3 업로드 예약: {checkpoint_path.name}")
    elif s3_upload and not mlflow_run_id:
        logger.warning(f"S3 업로드 건너뜀 (run_id 없음): {checkpoint_path.name}")


def load_lora_checkpoint(
    adapter,
    checkpoint_path: Path | str,
    device: torch.device,
    load_value_head: bool = True,
    strict: bool = True,
) -> dict:
    """LoRA checkpoint 로드 (LoRA 파라미터 + value head를 기존 adapter에 적용)

    Base model이 이미 로드된 adapter에 LoRA 가중치와 value head를 로드합니다.

    Args:
        adapter: MetaLlamaMTPAdapter (LoRA가 적용된 상태)
        checkpoint_path: LoRA checkpoint 경로
        device: torch.device
        load_value_head: Value head도 함께 로드할지 여부
        strict: 누락된 키가 있으면 에러 발생 여부

    Returns:
        checkpoint metadata (epoch, config, val_metrics 등)

    Raises:
        FileNotFoundError: Checkpoint 파일이 존재하지 않음
        ValueError: Checkpoint이 LoRA 타입이 아님
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 파일이 존재하지 않습니다: {checkpoint_path}")

    # Checkpoint 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"LoRA checkpoint 로드: {checkpoint_path}")

    # LoRA checkpoint 타입 확인
    checkpoint_type = checkpoint.get("checkpoint_type", "full")
    if checkpoint_type != "lora":
        raise ValueError(
            f"LoRA checkpoint가 아닙니다. checkpoint_type={checkpoint_type}\n"
            f"일반 checkpoint는 load_state_dict()를 사용하세요."
        )

    # LoRA state dict 로드
    lora_state_dict = checkpoint.get("lora_state_dict", {})
    if not lora_state_dict:
        logger.warning("LoRA state_dict가 비어있습니다.")

    # Value head state dict 로드
    value_head_state_dict = checkpoint.get("value_head_state_dict", {})

    # 현재 adapter의 state_dict 가져오기
    current_state_dict = adapter.state_dict()

    # LoRA 가중치 업데이트
    for key, value in lora_state_dict.items():
        if key in current_state_dict:
            current_state_dict[key] = value
        elif strict:
            raise KeyError(f"LoRA key not found in model: {key}")
        else:
            logger.warning(f"LoRA key not found, skipping: {key}")

    # Value head 가중치 업데이트
    if load_value_head and value_head_state_dict:
        for key, value in value_head_state_dict.items():
            if key in current_state_dict:
                current_state_dict[key] = value
            elif strict:
                raise KeyError(f"Value head key not found in model: {key}")
            else:
                logger.warning(f"Value head key not found, skipping: {key}")

    # State dict 적용
    adapter.load_state_dict(current_state_dict)

    logger.info(f"LoRA checkpoint 로드 완료:")
    logger.info(f"  LoRA params: {len(lora_state_dict)} keys")
    if load_value_head:
        logger.info(f"  Value head params: {len(value_head_state_dict)} keys")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    # Metadata 반환
    return {
        "epoch": checkpoint.get("epoch"),
        "config": checkpoint.get("config"),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "train_metrics": checkpoint.get("train_metrics", {}),
        "lora_config": checkpoint.get("lora_config"),
    }


def save_hf_checkpoint(
    model,
    tokenizer,
    save_dir: Path | str,
    epoch: float,
    val_metrics: dict[str, float],
) -> None:
    """HuggingFace 형식 checkpoint 저장

    AutoModelForCausalLM.from_pretrained()로 로드 가능한 형식으로 저장합니다.
    FSDP 환경에서는 모든 rank가 state_dict gathering에 참여하고,
    실제 저장은 rank 0만 수행합니다.

    Args:
        model: HuggingFace 모델 (FSDP-wrapped 또는 일반 모델)
        tokenizer: HuggingFace 토크나이저
        save_dir: 저장 디렉터리 경로
        epoch: 현재 epoch
        val_metrics: Validation metrics

    저장되는 파일:
        save_dir/
        ├── config.json
        ├── model.safetensors (또는 pytorch_model.bin)
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── training_state.json  # epoch, val_metrics 등
    """
    import json
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        StateDictType,
        FullStateDictConfig,
    )

    save_dir = Path(save_dir)

    # FSDP Full state dict gathering (모든 rank가 참여해야 함)
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()

        # rank 0만 실제 저장 수행
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        # FSDP unwrap된 원본 모델 구조 필요
        unwrapped_model = model.module
    else:
        # 일반 모델 (single-device 환경)
        state_dict = model.state_dict()
        unwrapped_model = model

    # 저장 디렉터리 생성
    save_dir.mkdir(parents=True, exist_ok=True)

    # HuggingFace 형식으로 저장 (FSDP 안전: state_dict를 직접 저장)
    # 모델 config 저장
    unwrapped_model.config.save_pretrained(save_dir)

    # state_dict를 safetensors로 저장
    from safetensors.torch import save_file
    save_file(state_dict, save_dir / "model.safetensors")

    # 토크나이저 저장
    tokenizer.save_pretrained(save_dir)

    # 학습 상태 저장 (별도 파일)
    training_state = {
        "epoch": epoch,
        "val_metrics": val_metrics,
    }
    with open(save_dir / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2)

    logger.info(f"HuggingFace checkpoint 저장 완료: {save_dir}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Val loss: {val_metrics.get('val_loss', 'N/A')}")
