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

    is_fsdp = isinstance(model, FSDP)

    if is_fsdp:
        # FSDP clip_grad_norm_: 내부적으로 all-reduce 수행, 클리핑 전 norm 반환
        grad_norm_pre = model.clip_grad_norm_(max_grad_norm).item()
    else:
        # Non-FSDP clip_grad_norm_: 내부적으로 norm 계산, 클리핑 전 norm 반환
        params_list = [p for p in model.parameters() if p.grad is not None]
        grad_norm_pre = torch.nn.utils.clip_grad_norm_(params_list, max_grad_norm).item()

    # 클리핑 후 norm: 클리핑이 적용되면 max_grad_norm을 초과하지 않음
    grad_norm_post = min(grad_norm_pre, max_grad_norm)

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


def compute_critic_classification_counts(
    value_logits: torch.Tensor,
    is_correct: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Critic 분류 성능을 위한 count 계산 (토큰 단위, micro-average용)

    각 토큰의 value prediction이 해당 시퀀스의 정답 여부를 맞추는지 평가
    V(s_t) → P(Success | s_t) 학습 목표에 부합

    Args:
        value_logits: Value predictions [batch, seq, 1]
        is_correct: Binary labels [batch] - 시퀀스별 정답 여부
        attention_mask: 유효 토큰 마스크 [batch, seq]
        threshold: Binary classification threshold

    Returns:
        count 딕셔너리:
            - tp: True Positives (토큰 단위)
            - fp: False Positives (토큰 단위)
            - fn: False Negatives (토큰 단위)
            - correct_sum: correct 시퀀스 내 토큰 예측값 합
            - correct_count: correct 시퀀스 내 유효 토큰 개수
            - incorrect_sum: incorrect 시퀀스 내 토큰 예측값 합
            - incorrect_count: incorrect 시퀀스 내 유효 토큰 개수
    """
    # value_logits: [batch, seq, 1] -> [batch, seq]
    values = value_logits.squeeze(-1)
    batch_size, seq_len = values.shape

    # 시퀀스 정답을 토큰에 broadcast: [batch] -> [batch, seq]
    token_labels = is_correct.view(-1, 1).expand(-1, seq_len)

    # 유효 토큰 마스크
    valid_mask = attention_mask.bool()

    # 토큰 단위 예측
    pred_positive = (values > threshold)
    actual_positive = token_labels.bool()

    # TP/FP/FN 계산 (유효 토큰만)
    tp = ((pred_positive & actual_positive) & valid_mask).sum().item()
    fp = ((pred_positive & ~actual_positive) & valid_mask).sum().item()
    fn = ((~pred_positive & actual_positive) & valid_mask).sum().item()

    # pred_gap 계산용: correct/incorrect 시퀀스 내 토큰 예측값
    correct_seq_mask = is_correct.bool()  # [batch]
    incorrect_seq_mask = ~correct_seq_mask

    # correct 시퀀스 내 유효 토큰들의 예측값 합계
    correct_token_mask = correct_seq_mask.view(-1, 1).expand(-1, seq_len) & valid_mask
    correct_sum = values[correct_token_mask].sum().item() if correct_token_mask.any() else 0.0
    correct_count = correct_token_mask.sum().item()

    # incorrect 시퀀스 내 유효 토큰들의 예측값 합계
    incorrect_token_mask = incorrect_seq_mask.view(-1, 1).expand(-1, seq_len) & valid_mask
    incorrect_sum = values[incorrect_token_mask].sum().item() if incorrect_token_mask.any() else 0.0
    incorrect_count = incorrect_token_mask.sum().item()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "correct_sum": correct_sum,
        "correct_count": correct_count,
        "incorrect_sum": incorrect_sum,
        "incorrect_count": incorrect_count,
    }


def compute_classification_metrics_from_counts(
    tp: float,
    fp: float,
    fn: float,
    correct_sum: float,
    correct_count: float,
    incorrect_sum: float,
    incorrect_count: float,
) -> dict[str, float]:
    """누적된 count로부터 최종 분류 메트릭 계산 (micro-average)

    Args:
        tp, fp, fn: 누적된 True Positives, False Positives, False Negatives
        correct_sum, correct_count: correct 시퀀스 예측값 합계와 개수
        incorrect_sum, incorrect_count: incorrect 시퀀스 예측값 합계와 개수

    Returns:
        메트릭 딕셔너리:
            - pred_gap: correct - incorrect 평균 예측값 차이
            - mean_correct: correct 토큰 평균 예측값
            - mean_incorrect: incorrect 토큰 평균 예측값
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1: 2 * P * R / (P + R)
    """
    # 평균 예측값 계산
    mean_correct = correct_sum / (correct_count + 1e-8)
    mean_incorrect = incorrect_sum / (incorrect_count + 1e-8)
    pred_gap = mean_correct - mean_incorrect if correct_count > 0 and incorrect_count > 0 else 0.0

    # Precision/Recall/F1 계산
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "pred_gap": pred_gap,
        "mean_correct": mean_correct,
        "mean_incorrect": mean_incorrect,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
