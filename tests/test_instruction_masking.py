"""Instruction Masking 검증 테스트

3개 파이프라인(baseline, critic, verifiable)에서 output 토큰만 학습되는지 검증
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from weighted_mtp.data.collators import AlpacaDataCollator, apply_alpaca_template


def test_instruction_masking_basic():
    """기본 instruction masking 테스트"""
    print("=" * 60)
    print("테스트 1: 기본 Instruction Masking")
    print("=" * 60)

    # 1. Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Collator 생성
    collator = AlpacaDataCollator(
        tokenizer=tokenizer,
        max_length=256,
        padding="max_length",
    )

    # 3. 테스트 샘플
    sample = {
        "instruction": "Write a function to add two numbers.",
        "input": "",
        "output": "def add(a, b):\n    return a + b",
        "is_correct": True,
    }

    # 4. Collate
    batch = collator([sample])

    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]

    # 5. 분석
    seq_len = attention_mask.sum().item()
    n_masked = (labels == -100).sum().item()
    n_trainable = (labels != -100).sum().item()

    print(f"\n시퀀스 길이: {seq_len}")
    print(f"마스킹된 토큰 (labels=-100): {n_masked}")
    print(f"학습 대상 토큰 (labels!=-100): {n_trainable}")

    # 6. 토큰별 상세 분석
    print("\n--- 토큰별 분석 ---")

    # Prompt 부분
    prompt_text = apply_alpaca_template(
        sample["instruction"], sample["input"], output="", include_response_header=True
    )
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=True)
    len_prompt = len(prompt_tokens["input_ids"])

    print(f"\nPrompt 길이 (instruction+input+header): {len_prompt}")

    # Output 시작 위치 확인
    output_start_idx = len_prompt

    # 실제 토큰 디코딩
    print("\n[Instruction/Input 부분 - 마스킹됨]")
    instruction_tokens = input_ids[:len_prompt]
    instruction_text = tokenizer.decode(instruction_tokens)
    print(f"  토큰 수: {len_prompt}")
    print(f"  labels: 모두 -100 -> {(labels[:len_prompt] == -100).all().item()}")

    print("\n[Output 부분 - 학습 대상]")
    output_tokens = input_ids[len_prompt:seq_len]
    output_text = tokenizer.decode(output_tokens)
    print(f"  토큰 수: {seq_len - len_prompt}")
    print(f"  텍스트: {output_text}")
    print(f"  labels: 실제 토큰 ID -> {(labels[len_prompt:seq_len] != -100).all().item()}")

    print("\n[Padding 부분 - 마스킹됨]")
    padding_tokens = input_ids[seq_len:]
    print(f"  토큰 수: {256 - seq_len}")
    print(f"  labels: 모두 -100 -> {(labels[seq_len:] == -100).all().item()}")

    # 7. 검증
    assert (labels[:len_prompt] == -100).all(), "Instruction 부분이 마스킹되지 않음!"
    assert (labels[len_prompt:seq_len] != -100).all(), "Output 부분이 마스킹됨!"
    assert (labels[seq_len:] == -100).all(), "Padding 부분이 마스킹되지 않음!"

    print("\n✓ 기본 테스트 통과!")
    return True


def test_ce_loss_masking():
    """CE Loss에서 masking이 제대로 적용되는지 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: CE Loss Masking 적용")
    print("=" * 60)

    # 1. Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Collator 생성
    collator = AlpacaDataCollator(
        tokenizer=tokenizer,
        max_length=128,
        padding="max_length",
    )

    # 3. 테스트 샘플
    sample = {
        "instruction": "Add numbers.",
        "input": "",
        "output": "return a + b",
        "is_correct": True,
    }

    batch = collator([sample])

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    batch_size, seq_len = input_ids.shape
    vocab_size = tokenizer.vocab_size

    # 4. 가짜 logits 생성 (모든 토큰에 대해 동일한 예측)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # 5. Baseline 스타일 CE Loss 계산
    # k=1 (다음 토큰 예측)
    k = 1
    valid_len = seq_len - k

    logits_k = logits[:, :valid_len, :]
    labels_k = labels[:, k : k + valid_len]
    mask_k = attention_mask[:, k : k + valid_len]

    # CE Loss (reduction="none")
    ce_loss_k = F.cross_entropy(
        logits_k.reshape(-1, vocab_size),
        labels_k.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )

    # Combined mask
    valid_label_mask_k = (labels_k != -100).float()
    combined_mask_k = mask_k.float() * valid_label_mask_k

    # Masked loss
    masked_ce_k = ce_loss_k * combined_mask_k.reshape(-1)

    # 6. 분석
    print(f"\n시퀀스 길이: {seq_len}")
    print(f"k={k}에서 valid_len: {valid_len}")

    # 각 위치별 loss 확인
    n_zero_loss = (ce_loss_k == 0).sum().item()
    n_masked_by_combined = (combined_mask_k.reshape(-1) == 0).sum().item()
    n_final_zero = (masked_ce_k == 0).sum().item()

    print(f"\nCE Loss=0인 위치 (ignore_index): {n_zero_loss}")
    print(f"Combined mask=0인 위치: {n_masked_by_combined}")
    print(f"최종 Loss=0인 위치: {n_final_zero}")

    # 학습되는 토큰 수
    n_trainable = (combined_mask_k.reshape(-1) > 0).sum().item()
    print(f"학습되는 토큰 수: {n_trainable}")

    # 7. Output 토큰만 학습되는지 검증
    prompt_text = apply_alpaca_template(
        sample["instruction"], sample["input"], output="", include_response_header=True
    )
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=True)
    len_prompt = len(prompt_tokens["input_ids"])

    # k=1이므로 output 시작은 len_prompt, 예측 대상은 len_prompt부터
    # labels_k는 labels[:, 1:valid_len+1]이므로
    # 학습 대상은 labels_k에서 -100이 아닌 부분

    actual_seq_len = attention_mask.sum().item()
    expected_trainable = actual_seq_len - len_prompt  # output 토큰 수

    print(f"\n예상 학습 토큰 수: {expected_trainable}")
    print(f"실제 학습 토큰 수: {n_trainable}")

    # 오차 범위 내에서 일치 확인 (경계 조건으로 인해 1-2개 차이 가능)
    assert abs(n_trainable - expected_trainable) <= 1, f"학습 토큰 수 불일치: {n_trainable} vs {expected_trainable}"

    print("\n✓ CE Loss Masking 테스트 통과!")
    return True


def test_mtp_head_masking():
    """MTP 각 head에서 masking이 일관되게 적용되는지 테스트"""
    print("\n" + "=" * 60)
    print("테스트 3: MTP Head별 Masking 일관성")
    print("=" * 60)

    # 1. Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Collator 생성
    collator = AlpacaDataCollator(
        tokenizer=tokenizer,
        max_length=128,
        padding="max_length",
    )

    # 3. 테스트 샘플
    sample = {
        "instruction": "Test instruction.",
        "input": "",
        "output": "def func(): pass",
        "is_correct": True,
    }

    batch = collator([sample])

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    batch_size, seq_len = input_ids.shape
    vocab_size = tokenizer.vocab_size
    n_future = 4

    # 4. 가짜 MTP logits 생성
    logits = torch.randn(batch_size, seq_len, n_future, vocab_size)

    print(f"\n시퀀스 길이: {seq_len}")
    print(f"n_future: {n_future}")

    # 5. 각 head별로 학습 토큰 수 계산
    trainable_per_head = []

    for k in range(1, n_future + 1):
        valid_len = seq_len - k

        if valid_len <= 0:
            trainable_per_head.append(0)
            continue

        logits_k = logits[:, :valid_len, k - 1, :]
        labels_k = labels[:, k : k + valid_len]
        mask_k = attention_mask[:, k : k + valid_len]

        # Combined mask
        valid_label_mask_k = (labels_k != -100).float()
        combined_mask_k = mask_k.float() * valid_label_mask_k

        n_trainable = (combined_mask_k > 0).sum().item()
        trainable_per_head.append(n_trainable)

        print(f"\nHead {k-1} (k={k}):")
        print(f"  valid_len: {valid_len}")
        print(f"  학습 토큰 수: {n_trainable}")

    # 6. 검증: 각 head에서 학습 토큰 수가 점진적으로 감소해야 함
    for i in range(len(trainable_per_head) - 1):
        # k가 증가하면 valid_len이 감소하므로 학습 토큰도 감소
        diff = trainable_per_head[i] - trainable_per_head[i + 1]
        assert diff >= 0, f"Head {i}의 학습 토큰이 Head {i+1}보다 적음!"

    print("\n✓ MTP Head별 Masking 테스트 통과!")
    return True


def test_value_loss_masking():
    """Value Loss에서 masking이 제대로 적용되는지 테스트 (critic, verifiable)"""
    print("\n" + "=" * 60)
    print("테스트 4: Value Loss Masking (Critic/Verifiable)")
    print("=" * 60)

    # 1. Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Collator 생성
    collator = AlpacaDataCollator(
        tokenizer=tokenizer,
        max_length=128,
        padding="max_length",
    )

    # 3. 테스트 샘플
    sample = {
        "instruction": "Test value masking.",
        "input": "",
        "output": "result = 42",
        "is_correct": True,
    }

    batch = collator([sample])

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    batch_size, seq_len = input_ids.shape

    # 4. 가짜 value logits 생성
    value_logits = torch.randn(batch_size, seq_len, 1)
    td_targets = torch.randn(batch_size, seq_len, 1)

    # 5. Value Loss 계산 (critic/verifiable 스타일)
    # Mask padded tokens AND instruction tokens (labels != -100)
    valid_label_mask = (labels != -100).unsqueeze(-1).float()
    attn_mask_expanded = attention_mask.unsqueeze(-1).float()
    loss_mask = valid_label_mask * attn_mask_expanded

    # MSE Loss
    loss_per_token = F.mse_loss(value_logits, td_targets, reduction="none")
    masked_loss = loss_per_token * loss_mask
    value_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)

    # 6. 분석
    n_masked = (loss_mask == 0).sum().item()
    n_trainable = (loss_mask > 0).sum().item()

    print(f"\n시퀀스 길이: {seq_len}")
    print(f"마스킹된 토큰: {n_masked}")
    print(f"학습 대상 토큰: {n_trainable}")

    # 7. Output 토큰만 학습되는지 검증
    prompt_text = apply_alpaca_template(
        sample["instruction"], sample["input"], output="", include_response_header=True
    )
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=True)
    len_prompt = len(prompt_tokens["input_ids"])

    actual_seq_len = attention_mask.sum().item()
    expected_trainable = actual_seq_len - len_prompt

    print(f"\n예상 학습 토큰 수: {expected_trainable}")
    print(f"실제 학습 토큰 수: {n_trainable}")

    assert n_trainable == expected_trainable, f"학습 토큰 수 불일치: {n_trainable} vs {expected_trainable}"

    print("\n✓ Value Loss Masking 테스트 통과!")
    return True


def test_detailed_example():
    """상세 예시와 함께 masking 결과 시각화"""
    print("\n" + "=" * 60)
    print("테스트 5: 상세 예시 시각화")
    print("=" * 60)

    # 1. Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Collator 생성 (짧은 max_length로 시각화)
    collator = AlpacaDataCollator(
        tokenizer=tokenizer,
        max_length=100,
        padding="max_length",
    )

    # 3. 테스트 샘플
    sample = {
        "instruction": "Add two numbers.",
        "input": "",
        "output": "a+b",
        "is_correct": True,
    }

    batch = collator([sample])

    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]

    # 4. 토큰별 시각화
    print("\n--- 토큰별 Masking 상태 ---")
    print(f"{'Idx':>4} | {'Token':>20} | {'Label':>8} | {'AttnMask':>8} | Status")
    print("-" * 60)

    actual_len = attention_mask.sum().item()

    # Prompt 길이 계산
    prompt_text = apply_alpaca_template(
        sample["instruction"], sample["input"], output="", include_response_header=True
    )
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=True)
    len_prompt = len(prompt_tokens["input_ids"])

    for i in range(min(int(actual_len) + 5, 100)):  # 실제 토큰 + padding 일부
        token_id = input_ids[i].item()
        token_str = tokenizer.decode([token_id])
        label = labels[i].item()
        attn = attention_mask[i].item()

        # Status 결정
        if i < len_prompt:
            status = "INSTRUCTION (masked)"
        elif i < actual_len:
            status = "OUTPUT (trained)" if label != -100 else "OUTPUT (masked?)"
        else:
            status = "PADDING (masked)"

        # 토큰 문자열 정리
        token_display = token_str.replace("\n", "\\n")[:20]
        label_display = str(label) if label != -100 else "-100"

        print(f"{i:4} | {token_display:>20} | {label_display:>8} | {attn:>8} | {status}")

        if i == len_prompt - 1:
            print("-" * 60 + " <- Prompt 끝")
        if i == int(actual_len) - 1:
            print("-" * 60 + " <- Sequence 끝")

    # 5. 요약
    n_instruction = len_prompt
    n_output = int(actual_len) - len_prompt
    n_padding = 100 - int(actual_len)

    print(f"\n--- 요약 ---")
    print(f"Instruction/Input 토큰: {n_instruction} (모두 labels=-100)")
    print(f"Output 토큰: {n_output} (학습 대상)")
    print(f"Padding 토큰: {n_padding} (모두 labels=-100)")

    # 검증
    instruction_masked = (labels[:len_prompt] == -100).all().item()
    output_trained = (labels[len_prompt:int(actual_len)] != -100).all().item()
    padding_masked = (labels[int(actual_len):] == -100).all().item()

    print(f"\n검증 결과:")
    print(f"  Instruction 마스킹: {'✓' if instruction_masked else '✗'}")
    print(f"  Output 학습: {'✓' if output_trained else '✗'}")
    print(f"  Padding 마스킹: {'✓' if padding_masked else '✗'}")

    assert instruction_masked and output_trained and padding_masked, "Masking 검증 실패!"

    print("\n✓ 상세 예시 테스트 통과!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Instruction Masking 검증 테스트 시작")
    print("=" * 60)

    results = []

    results.append(("기본 Instruction Masking", test_instruction_masking_basic()))
    results.append(("CE Loss Masking", test_ce_loss_masking()))
    results.append(("MTP Head별 Masking", test_mtp_head_masking()))
    results.append(("Value Loss Masking", test_value_loss_masking()))
    results.append(("상세 예시 시각화", test_detailed_example()))

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n모든 테스트 통과! Masking이 올바르게 구현되어 있습니다.")
    else:
        print("\n일부 테스트 실패! 확인이 필요합니다.")
