"""Value Loss 유틸리티

Pairwise Ranking Loss 및 MC MSE Loss 계산
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


def compute_mc_value_loss(
    value_logits: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Monte Carlo Value Loss (Tokenwise MSE)

    Terminal reward를 모든 토큰에 전파하여 MSE loss 계산
    V(s_t) → R (correct: 1.0, incorrect: 0.0)

    Args:
        value_logits: [batch, seq, 1] Value head 출력
        rewards: [batch] Binary reward (0.0: incorrect, 1.0: correct)
        attention_mask: [batch, seq] 유효 토큰 마스크 (padding 제외)
        loss_mask: [batch, seq] 학습 대상 토큰 마스크 (labels != -100)

    Returns:
        mc_loss: scalar tensor
    """
    _, seq_len, _ = value_logits.shape
    dtype = value_logits.dtype

    # MC targets: 모든 토큰에 동일한 terminal reward 할당
    mc_targets = rewards.view(-1, 1).expand(-1, seq_len).to(dtype)

    # 학습 대상 토큰만으로 MSE 계산 (Instruction 제외)
    combined_mask = attention_mask * loss_mask

    # Tokenwise MSE
    values = value_logits.squeeze(-1)
    mse = (values - mc_targets) ** 2

    # Masked mean
    masked_mse = (mse * combined_mask).sum() / (combined_mask.sum() + 1e-8)

    return masked_mse
