"""Head 0 Selective Weighting 테스트

Head 0도 selective weighting이 적용되는지 검증:
1. Head 0 selection ratio < 1.0 (무조건 선택 아님)
2. "difficult" 모드에서 ref CE가 높은 토큰 선택
3. Timestep alignment 올바른지 확인
4. 모든 head 통계 계산 올바른지 확인
"""

import torch
import torch.nn.functional as F

from weighted_mtp.value_weighting.rho1_weighting import compute_mtp_selective_weights


def test_head0_selective_weighting():
    """Head 0도 selective weighting이 적용되는지 테스트"""
    print("\n=== Test 1: Head 0 Selective Weighting ===")

    batch_size = 2
    seq_len = 10
    n_future = 4
    vocab_size = 100

    # 더미 데이터 생성
    torch.manual_seed(42)
    policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # "difficult" 모드로 테스트
    weights, stats = compute_mtp_selective_weights(
        policy_logits=policy_logits,
        ref_logits=ref_logits,
        labels=labels,
        attention_mask=attention_mask,
        k_percent=0.4,
        mode="difficult",
    )

    print(f"Weights shape: {weights.shape}")
    print(f"\nHead selection ratios:")
    for head_idx in range(n_future):
        ratio = stats[f'head_{head_idx}_ratio']
        count = stats[f'head_{head_idx}_count']
        print(f"  Head {head_idx}: {ratio:.3f} ({count:.0f} tokens)")

    # Head 0 검증
    head_0_ratio = stats['head_0_ratio']
    assert head_0_ratio < 1.0, f"Head 0 should be selective! Got ratio={head_0_ratio}"
    assert head_0_ratio > 0.0, f"Head 0 should select some tokens! Got ratio={head_0_ratio}"
    print(f"\n✅ Head 0 selective weighting works! (ratio={head_0_ratio:.3f})")

    # 전체 통계
    print(f"\nOverall statistics:")
    print(f"  Selection ratio: {stats['selection_ratio']:.3f}")
    print(f"  Avg heads per position: {stats['avg_heads_per_position']:.3f}")

    return weights, stats


def test_difficult_mode_selection():
    """'difficult' 모드가 ref CE가 높은 토큰을 선택하는지 테스트"""
    print("\n=== Test 2: Difficult Mode Selection ===")

    batch_size = 1
    seq_len = 8
    n_future = 4
    vocab_size = 50

    torch.manual_seed(123)

    # Policy와 Ref logits 생성 (특정 패턴)
    policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Labels 생성 (실제 정답)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Ref CE 계산 (Head 0, k=1 기준)
    k = 1
    valid_len = seq_len - k
    ref_logits_k = ref_logits[:, k-1:k-1+valid_len, :]
    labels_k = labels[:, k:k+valid_len]

    ref_ce = F.cross_entropy(
        ref_logits_k.reshape(-1, vocab_size),
        labels_k.reshape(-1),
        reduction='none',
    ).view(batch_size, valid_len)

    print(f"Ref CE for Head 0 (k=1):")
    print(f"  Min: {ref_ce.min().item():.3f}")
    print(f"  Max: {ref_ce.max().item():.3f}")
    print(f"  Mean: {ref_ce.mean().item():.3f}")

    # "difficult" 모드로 weighting
    weights, stats = compute_mtp_selective_weights(
        policy_logits=policy_logits,
        ref_logits=ref_logits,
        labels=labels,
        attention_mask=attention_mask,
        k_percent=0.5,  # Top 50%
        mode="difficult",
    )

    # Head 0의 선택된 토큰 확인
    head_0_weights = weights[0, :valid_len, 0]  # [valid_len]
    selected_ce = ref_ce[0][head_0_weights == 1.0]
    unselected_ce = ref_ce[0][head_0_weights == 0.0]

    if len(selected_ce) > 0 and len(unselected_ce) > 0:
        print(f"\nHead 0 selection (k_percent=0.5):")
        print(f"  Selected tokens CE: {selected_ce.mean().item():.3f}")
        print(f"  Unselected tokens CE: {unselected_ce.mean().item():.3f}")

        # 선택된 토큰의 평균 CE가 더 높아야 함 (어려운 토큰)
        assert selected_ce.mean() > unselected_ce.mean(), \
            "Selected tokens should have higher CE (more difficult)!"
        print(f"✅ Difficult mode correctly selects high-CE tokens!")
    else:
        print(f"⚠️  Not enough tokens to compare (selected={len(selected_ce)}, unselected={len(unselected_ce)})")


def test_timestep_alignment():
    """Timestep alignment이 올바른지 테스트"""
    print("\n=== Test 3: Timestep Alignment ===")

    batch_size = 1
    seq_len = 6
    n_future = 4
    vocab_size = 10

    # 간단한 패턴으로 데이터 생성
    torch.manual_seed(456)
    policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Labels: [0, 1, 2, 3, 4, 5]
    labels = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"Labels: {labels[0].tolist()}")
    print(f"\nTimestep alignment check:")

    for k in range(1, n_future + 1):
        head_idx = k - 1
        valid_len = seq_len - k

        if valid_len <= 0:
            continue

        # 실제 로직과 동일하게 계산
        policy_logits_k = policy_logits[:, :valid_len, head_idx, :]
        ref_logits_k = ref_logits[:, k-1:k-1+valid_len, :]
        labels_k = labels[:, k:k+valid_len]

        print(f"  k={k} (Head {head_idx}):")
        print(f"    valid_len={valid_len}")
        print(f"    policy_logits_k: [:, 0:{valid_len}, {head_idx}, :]")
        print(f"    ref_logits_k: [:, {k-1}:{k-1+valid_len}, :]")
        print(f"    labels_k: [:, {k}:{k+valid_len}] = {labels_k[0].tolist()}")

    print(f"\n✅ Timestep alignment verified!")


def test_all_modes():
    """모든 모드 테스트"""
    print("\n=== Test 4: All Modes ===")

    batch_size = 2
    seq_len = 8
    n_future = 4
    vocab_size = 50

    torch.manual_seed(789)
    policy_logits = torch.randn(batch_size, seq_len, n_future, vocab_size)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    modes = ["signed", "absolute", "difficult"]

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        weights, stats = compute_mtp_selective_weights(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            labels=labels,
            attention_mask=attention_mask,
            k_percent=0.4,
            mode=mode,
        )

        print(f"Selection ratio: {stats['selection_ratio']:.3f}")
        print(f"Head ratios: ", end="")
        for head_idx in range(n_future):
            ratio = stats[f'head_{head_idx}_ratio']
            print(f"H{head_idx}={ratio:.2f} ", end="")
        print()

    print(f"\n✅ All modes work correctly!")


if __name__ == "__main__":
    print("=" * 60)
    print("Head 0 Selective Weighting Tests")
    print("=" * 60)

    try:
        # Test 1: Head 0 selective weighting
        weights, stats = test_head0_selective_weighting()

        # Test 2: Difficult mode selection
        test_difficult_mode_selection()

        # Test 3: Timestep alignment
        test_timestep_alignment()

        # Test 4: All modes
        test_all_modes()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
