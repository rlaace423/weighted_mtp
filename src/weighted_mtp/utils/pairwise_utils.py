"""Pairwise Ranking Loss 유틸리티

Bradley-Terry 모델 기반 Pairwise Ranking Loss 및 메트릭 계산
run_critic, run_verifiable에서 공통 사용
"""

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    mask_pos: torch.Tensor,
    mask_neg: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry Pairwise Ranking Loss

    P(pos > neg) = sigmoid(V_pos - V_neg)
    Loss = -log(sigmoid(V_pos - V_neg))

    Output 토큰만 사용하여 시퀀스 평균 비교 (Instruction 제외)

    Args:
        v_pos: [batch, seq, 1] positive sample values
        v_neg: [batch, seq, 1] negative sample values
        mask_pos: [batch, seq] valid token mask for positive (labels != -100)
        mask_neg: [batch, seq] valid token mask for negative (labels != -100)

    Returns:
        Scalar loss
    """
    # 시퀀스 평균 value 계산 (Output 토큰만)
    v_pos_mean = (v_pos.squeeze(-1) * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)

    # Pairwise ranking loss: -log(sigmoid(v_pos - v_neg))
    return -F.logsigmoid(v_pos_mean - v_neg_mean).mean()


def compute_pairwise_accuracy(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    mask_pos: torch.Tensor,
    mask_neg: torch.Tensor,
) -> dict[str, float]:
    """Pairwise Accuracy 및 관련 메트릭 계산

    Args:
        v_pos: [batch, seq, 1] positive sample values
        v_neg: [batch, seq, 1] negative sample values
        mask_pos: [batch, seq] valid token mask for positive
        mask_neg: [batch, seq] valid token mask for negative

    Returns:
        {pairwise_accuracy, mean_pos, mean_neg, margin, correct_pairs, total_pairs}
    """
    # 시퀀스 평균 value 계산
    v_pos_mean = (v_pos.squeeze(-1) * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)

    # V(correct) > V(incorrect)인 쌍의 비율
    correct_pairs = (v_pos_mean > v_neg_mean).float().sum()
    total_pairs = v_pos_mean.size(0)
    pairwise_accuracy = (correct_pairs / total_pairs).item()

    # 평균 값
    mean_pos = v_pos_mean.mean().item()
    mean_neg = v_neg_mean.mean().item()
    margin = mean_pos - mean_neg

    return {
        "pairwise_accuracy": pairwise_accuracy,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "margin": margin,
        "correct_pairs": correct_pairs.item(),
        "total_pairs": total_pairs,
    }
