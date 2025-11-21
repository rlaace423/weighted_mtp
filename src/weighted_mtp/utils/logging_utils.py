"""MLflow 로깅 유틸리티

연구 분석을 위한 추가 메트릭 계산 함수
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_weight_statistics(
    weights: torch.Tensor,
    attention_mask: torch.Tensor = None,
) -> dict[str, float]:
    """Weight 분포 통계 계산

    Args:
        weights: Token weights [batch, seq, n_heads] or [batch, seq]
        attention_mask: 유효 토큰 마스크 [batch, seq] (None이면 전체 사용)

    Returns:
        통계 딕셔너리:
            - weight_mean: 평균
            - weight_std: 표준편차
            - weight_min: 최소값
            - weight_max: 최대값
            - weight_entropy: 엔트로피
    """
    if attention_mask is not None:
        # 유효한 토큰만 선택 (padding 제외)
        # weights가 [batch, seq, n_heads]인 경우 처리
        if weights.dim() == 3:
            batch, seq, n_heads = weights.shape
            mask = attention_mask.view(batch, seq, 1).expand(-1, -1, n_heads)
            weights_flat = weights[mask.bool()]
        else:
            mask = attention_mask.view(-1).bool()
            weights_flat = weights.view(-1)[mask]
    else:
        weights_flat = weights.view(-1)

    # Entropy 계산 (log base e)
    # 0에 가까운 값 방지를 위해 epsilon 추가
    epsilon = 1e-10
    entropy = -(weights_flat * torch.log(weights_flat + epsilon)).mean()

    return {
        "weight_mean": weights_flat.mean().item(),
        "weight_std": weights_flat.std().item(),
        "weight_min": weights_flat.min().item(),
        "weight_max": weights_flat.max().item(),
        "weight_entropy": entropy.item(),
    }


def compute_gradient_clip_stats(
    model: torch.nn.Module,
    max_grad_norm: float,
) -> dict[str, float]:
    """Gradient clipping 전후 통계 계산 (FSDP 호환)

    Args:
        model: 모델 (FSDP wrapped 또는 일반 모델)
        max_grad_norm: Gradient clipping threshold

    Returns:
        통계 딕셔너리:
            - grad_norm_pre_clip: Clipping 전 gradient norm
            - grad_norm_post_clip: Clipping 후 gradient norm
            - grad_clip_ratio: Clipping 비율 (post/pre)
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    import torch.distributed as dist

    is_fsdp = isinstance(model, FSDP)

    if is_fsdp:
        # FSDP: all-reduce로 전체 gradient norm 계산
        parameters = [p for p in model.parameters() if p.grad is not None]
        local_norm_sq = sum(p.grad.data.norm(2).item() ** 2 for p in parameters)

        # All-reduce to get global norm
        device = parameters[0].device if parameters else torch.device("cuda")
        tensor = torch.tensor(local_norm_sq, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        grad_norm_pre = tensor.item() ** 0.5

        # FSDP clip_grad_norm_ (전체 gradient에 대해 clipping)
        grad_norm_post = model.clip_grad_norm_(max_grad_norm).item()
    else:
        # Non-FSDP: 기존 방식
        params_list = [p for p in model.parameters() if p.grad is not None]

        # Pre-clip norm
        total_norm_pre = sum(p.grad.data.norm(2).item() ** 2 for p in params_list)
        grad_norm_pre = total_norm_pre ** 0.5

        # Clipping
        torch.nn.utils.clip_grad_norm_(params_list, max_grad_norm)

        # Post-clip norm
        total_norm_post = sum(p.grad.data.norm(2).item() ** 2 for p in params_list)
        grad_norm_post = total_norm_post ** 0.5

    # Clipping 비율 계산
    clip_ratio = grad_norm_post / grad_norm_pre if grad_norm_pre > 0 else 1.0

    return {
        "grad_norm_pre_clip": grad_norm_pre,
        "grad_norm_post_clip": grad_norm_post,
        "grad_clip_ratio": clip_ratio,
    }


def compute_value_function_stats(
    values: torch.Tensor,
    returns: torch.Tensor,
    attention_mask: torch.Tensor = None,
) -> dict[str, float]:
    """Value function 품질 통계 계산

    Args:
        values: Predicted values [batch, seq]
        returns: Target returns [batch, seq]
        attention_mask: 유효 토큰 마스크 [batch, seq] (None이면 전체 사용)

    Returns:
        통계 딕셔너리:
            - value_mse: Mean squared error
            - value_mean: 평균 predicted value
            - value_std: Predicted value 표준편차
            - return_mean: 평균 return
    """
    if attention_mask is not None:
        # 유효한 토큰만 선택 (padding 제외)
        mask = attention_mask.reshape(-1).bool()
        values_flat = values.reshape(-1)[mask]
        returns_flat = returns.reshape(-1)[mask]
    else:
        values_flat = values.reshape(-1)
        returns_flat = returns.reshape(-1)

    # MSE
    mse = F.mse_loss(values_flat, returns_flat)

    return {
        "value_mse": mse.item(),
        "value_mean": values_flat.mean().item(),
        "value_std": values_flat.std().item(),
        "return_mean": returns_flat.mean().item(),
    }
