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


def compute_token_variance(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """시퀀스 내 토큰별 Value Variance 계산

    모델이 토큰별 차별화를 학습하는지 진단하는 메트릭.
    Variance ≈ 0: 모든 토큰에 동일한 값 예측 (problem-level 학습)
    Variance > 0: 토큰별 차별화 (token-level 학습)

    Args:
        values: [batch, seq, 1] Value head 출력
        mask: [batch, seq] 유효 토큰 마스크 (labels != -100)

    Returns:
        배치 평균 variance (scalar)
    """
    values = values.squeeze(-1)  # [batch, seq]

    # 시퀀스별 평균 (masked)
    seq_mean = (values * mask).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + 1e-8)

    # 시퀀스별 variance (masked)
    sq_diff = ((values - seq_mean) ** 2) * mask
    seq_variance = sq_diff.sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    return seq_variance.mean().item()


def get_scheduled_lambda(
    current_step: int,
    warmup_steps: int = 250,
    decay_steps: int = 500,
    lam_start: float = 1.0,
    lam_end: float = 0.95,
) -> float:
    """Lambda scheduling (Hold & Decay)

    EOS-only warmup 대신 λ 값을 동적으로 조절하여 학습 안정성 확보.
    초기에 Pure MC (λ=1.0)로 방향성을 잡고, 점진적으로 TD mixture로 전환.

    Phase 1 (Hold): warmup_steps 동안 λ=start 유지
        - 모델이 terminal reward 패턴을 확실히 학습
        - V=0.5 편향에서 벗어나도록 강제

    Phase 2 (Decay): decay_steps 동안 start → end 선형 감소
        - 부드러운 전환으로 학습 안정성 유지
        - 부트스트랩 비중 점진적 증가

    Phase 3 (Stable): 이후 λ=end 유지
        - 토큰별 차별화된 타겟으로 일반화 성능 향상

    Args:
        current_step: 현재 global step
        warmup_steps: Hold 기간 (λ=start 유지)
        decay_steps: Decay 기간 (start → end)
        lam_start: 초기 λ 값 (기본값 1.0, Pure MC)
        lam_end: 최종 λ 값 (기본값 0.95)

    Returns:
        현재 step의 λ 값
    """
    if current_step < warmup_steps:
        return lam_start
    elif current_step < warmup_steps + decay_steps:
        progress = (current_step - warmup_steps) / decay_steps
        return lam_start - progress * (lam_start - lam_end)
    else:
        return lam_end


def create_eos_only_mask(loss_mask: torch.Tensor) -> torch.Tensor:
    """마지막 유효 토큰(Output 끝)만 1인 마스크 생성

    Warmup 학습에서 terminal position만 학습하기 위한 마스크.
    loss_mask가 [0,0,1,1,1,0,0] 형태(Instruction 마스킹)일 때
    마지막 1의 위치를 정확히 찾음.

    Args:
        loss_mask: [batch, seq] Output 토큰 마스크 (labels != -100)

    Returns:
        eos_mask: [batch, seq] 마지막 유효 토큰만 1
    """
    batch_size, seq_len = loss_mask.shape
    device = loss_mask.device

    # 인덱스 [1, 2, ..., seq_len] 생성 (0이 아닌 1부터 시작하여 0값과 구분)
    seq_indices = torch.arange(1, seq_len + 1, device=device).unsqueeze(0)

    # 유효 위치만 인덱스 값 유지, 나머지는 0
    valid_indices = seq_indices * loss_mask.float()

    # 각 배치의 최대 인덱스 = 마지막 유효 토큰 위치
    last_valid_pos = valid_indices.argmax(dim=1)

    # EOS 마스크 생성
    eos_mask = torch.zeros_like(loss_mask, dtype=torch.float32)
    batch_indices = torch.arange(batch_size, device=device)

    # 유효 토큰이 있는 배치만 마킹
    has_valid_tokens = loss_mask.sum(dim=1) > 0
    eos_mask[batch_indices[has_valid_tokens], last_valid_pos[has_valid_tokens]] = 1.0

    return eos_mask


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


def compute_lambda_return(
    values: torch.Tensor,
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> torch.Tensor:
    """Fitted λ-Return 타겟 계산 (Offline TD)

    GAE 스타일의 λ-return을 offline 환경에 적용.
    Terminal reward만 있을 때 역방향으로 타겟 전파.

    수식:
        G_t^λ = (1-λ)γV_{t+1} + λγG_{t+1}^λ  (r_t=0 for t<T)
        G_T = R (terminal)

    γ=1.0, λ=0.95 (InstructGPT 설정):
        - 후반 토큰: R에 가까운 타겟
        - 전반 토큰: V 예측에 의존하는 타겟

    Args:
        values: [batch, seq] Value model 예측 (detached 상태로 전달)
        rewards: [batch] Terminal reward (1.0: correct, 0.0: incorrect)
        loss_mask: [batch, seq] 유효 토큰 마스크 (labels != -100)
        gamma: Discount factor (기본값 1.0)
        lam: GAE smoothing factor (기본값 0.95)

    Returns:
        lambda_returns: [batch, seq] 위치별 λ-return 타겟
    """
    batch_size, seq_len = values.shape
    device = values.device
    dtype = values.dtype

    lambda_returns = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

    for b in range(batch_size):
        # 유효한 토큰 위치 추출
        valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue

        # Terminal position: G_T = R
        last_pos = valid_positions[-1].item()
        lambda_returns[b, last_pos] = rewards[b]

        # 역방향 전파: G_t = (1-λ)γV_{t+1} + λγG_{t+1}
        G_next = rewards[b]
        for i in range(len(valid_positions) - 2, -1, -1):
            t = valid_positions[i].item()
            t_next = valid_positions[i + 1].item()

            V_next = values[b, t_next]
            td_component = (1 - lam) * gamma * V_next
            mc_component = lam * gamma * G_next
            G_t = td_component + mc_component

            lambda_returns[b, t] = G_t
            G_next = G_t

    return lambda_returns


def compute_lambda_value_loss(
    value_logits: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> torch.Tensor:
    """λ-Return 기반 Value Loss

    MC MSE를 대체하는 Fitted TD 방식의 value loss.
    타겟 계산 시 현재 V 예측을 활용하여 위치별 차별화된 타겟 생성.

    Args:
        value_logits: [batch, seq, 1] Value head 출력
        rewards: [batch] Terminal reward (1.0: correct, 0.0: incorrect)
        attention_mask: [batch, seq] Padding 마스크
        loss_mask: [batch, seq] Output 토큰 마스크 (labels != -100)
        gamma: Discount factor (기본값 1.0)
        lam: GAE smoothing factor (기본값 0.95)

    Returns:
        loss: Scalar tensor
    """
    values = value_logits.squeeze(-1)  # [batch, seq]

    # λ-return 타겟 계산 (gradient 차단)
    with torch.no_grad():
        lambda_targets = compute_lambda_return(
            values.detach(), rewards, loss_mask, gamma, lam
        )

    # MSE loss (masked)
    combined_mask = attention_mask * loss_mask
    mse = (values - lambda_targets) ** 2
    masked_mse = (mse * combined_mask).sum() / (combined_mask.sum() + 1e-8)

    return masked_mse
