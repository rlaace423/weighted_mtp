"""MSE 계산 디버그 스크립트

verifiable 파이프라인에서 MSE가 1.0이 되는 원인 분석
"""
import torch
import torch.nn.functional as F

# TD targets 계산 함수 (run_verifiable.py와 동일)
def compute_td_targets(value_logits, rewards, attention_mask, gamma=1.0, lam=1.0):
    """GAE 기반 TD targets 계산"""
    batch_size, seq_len, _ = value_logits.shape
    device = value_logits.device
    dtype = value_logits.dtype

    values = value_logits.squeeze(-1).detach()

    seq_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    masked_indices = seq_indices * attention_mask
    terminal_indices = masked_indices.max(dim=1).values.long()

    td_targets = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

    for b in range(batch_size):
        last_gae = 0.0
        term_idx = terminal_indices[b].item()

        for t in range(int(term_idx), -1, -1):
            if t == term_idx:
                next_value = 0.0
                reward = rewards[b].item()
            else:
                next_value = values[b, t + 1].item()
                reward = 0.0

            delta = reward + gamma * next_value - values[b, t].item()
            gae = delta + gamma * lam * last_gae
            td_targets[b, t] = values[b, t].item() + gae
            last_gae = gae

    td_targets = td_targets * attention_mask.to(dtype)
    return td_targets.unsqueeze(-1)


def main():
    print("=" * 60)
    print("MSE 계산 시뮬레이션")
    print("=" * 60)

    # 시뮬레이션 파라미터
    batch_size = 4
    seq_len = 10
    response_start = 3  # 처음 3개 토큰은 instruction

    # 시나리오 1: Correct sample (is_correct=True, reward=1)
    print("\n[시나리오 1] Correct 샘플 (reward=1.0)")
    print("-" * 40)

    # 예측값이 ~0.5인 경우
    value_logits = torch.full((1, seq_len, 1), 0.5)
    rewards = torch.tensor([1.0])
    attention_mask = torch.ones(1, seq_len)

    # labels 마스크 (instruction 제외)
    labels_mask = torch.zeros(1, seq_len)
    labels_mask[:, response_start:] = 1

    td_targets = compute_td_targets(value_logits, rewards, attention_mask, gamma=1.0, lam=1.0)

    print(f"Value logits (response만): {value_logits[0, response_start:, 0].tolist()}")
    print(f"TD targets (response만):   {td_targets[0, response_start:, 0].tolist()}")
    print(f"TD targets mean (response): {td_targets[0, response_start:, 0].mean().item():.4f}")

    # MSE 계산 (response 토큰만)
    values_flat = value_logits[0, response_start:, 0]
    returns_flat = td_targets[0, response_start:, 0]
    mse = F.mse_loss(values_flat, returns_flat)
    print(f"MSE: {mse.item():.4f}")

    # 시나리오 2: Incorrect sample (is_correct=False, reward=0)
    print("\n[시나리오 2] Incorrect 샘플 (reward=0.0)")
    print("-" * 40)

    value_logits = torch.full((1, seq_len, 1), 0.5)
    rewards = torch.tensor([0.0])

    td_targets = compute_td_targets(value_logits, rewards, attention_mask, gamma=1.0, lam=1.0)

    print(f"Value logits (response만): {value_logits[0, response_start:, 0].tolist()}")
    print(f"TD targets (response만):   {td_targets[0, response_start:, 0].tolist()}")
    print(f"TD targets mean (response): {td_targets[0, response_start:, 0].mean().item():.4f}")

    values_flat = value_logits[0, response_start:, 0]
    returns_flat = td_targets[0, response_start:, 0]
    mse = F.mse_loss(values_flat, returns_flat)
    print(f"MSE: {mse.item():.4f}")

    # 시나리오 3: 50/50 배치 (correct_ratio=0.5)
    print("\n[시나리오 3] 50/50 배치 (2 correct, 2 incorrect)")
    print("-" * 40)

    value_logits = torch.full((4, seq_len, 1), 0.5)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])  # 2 correct, 2 incorrect
    attention_mask = torch.ones(4, seq_len)

    td_targets = compute_td_targets(value_logits, rewards, attention_mask, gamma=1.0, lam=1.0)

    print(f"Rewards: {rewards.tolist()}")
    print(f"TD targets means per sample:")
    for i in range(4):
        print(f"  Sample {i} (reward={rewards[i].item():.1f}): target_mean={td_targets[i, response_start:, 0].mean().item():.4f}")

    # 전체 MSE
    values_flat = value_logits[:, response_start:, 0].reshape(-1)
    returns_flat = td_targets[:, response_start:, 0].reshape(-1)
    mse = F.mse_loss(values_flat, returns_flat)
    print(f"\n전체 MSE: {mse.item():.4f}")
    print(f"Value mean: {values_flat.mean().item():.4f}")
    print(f"Return mean: {returns_flat.mean().item():.4f}")

    # 시나리오 4: Value head가 반대로 예측하는 경우
    print("\n[시나리오 4] Value head 반대 예측")
    print("-" * 40)

    # Correct 샘플에서 0, Incorrect 샘플에서 1 예측
    value_logits = torch.zeros((4, seq_len, 1))
    value_logits[0, :, :] = 0.0  # correct → 예측 0
    value_logits[1, :, :] = 0.0  # correct → 예측 0
    value_logits[2, :, :] = 1.0  # incorrect → 예측 1
    value_logits[3, :, :] = 1.0  # incorrect → 예측 1

    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    td_targets = compute_td_targets(value_logits, rewards, attention_mask, gamma=1.0, lam=1.0)

    print(f"Value predictions per sample (response만):")
    for i in range(4):
        print(f"  Sample {i} (reward={rewards[i].item():.1f}): pred={value_logits[i, response_start, 0].item():.1f}, target={td_targets[i, response_start, 0].item():.1f}")

    values_flat = value_logits[:, response_start:, 0].reshape(-1)
    returns_flat = td_targets[:, response_start:, 0].reshape(-1)
    mse = F.mse_loss(values_flat, returns_flat)
    print(f"\n전체 MSE: {mse.item():.4f}")
    print(f"Value mean: {values_flat.mean().item():.4f}")
    print(f"Return mean: {returns_flat.mean().item():.4f}")

    # 시나리오 5: MSE = 1.0이 되려면?
    print("\n" + "=" * 60)
    print("결론: MSE = 1.0이 되려면?")
    print("=" * 60)
    print("""
예측값 0.5, 타겟 0 또는 1 → MSE = 0.25
예측값 0.0, 타겟 1.0 → MSE = 1.0
예측값 1.0, 타겟 0.0 → MSE = 1.0

MSE = 1.0이 나오려면:
  - 모든 예측이 타겟과 정확히 1.0 차이나야 함
  - 또는 td_targets 자체가 이상한 값을 갖고 있어야 함
""")


if __name__ == "__main__":
    main()
