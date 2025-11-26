"""PairwiseDataCollator 핵심 기능 검증 테스트

핵심 검증 항목:
- Pairwise 데이터 구조 (pos_*, neg_* 분리)
- Loss masking (instruction, input, padding 제외)
- Output 토큰만 학습 대상
- 배치 처리
"""

import pytest
import torch
from pathlib import Path

from weighted_mtp.data.collators import PairwiseDataCollator, apply_alpaca_template


@pytest.fixture(scope="module")
def tokenizer():
    """실제 LlamaTokenizer 로딩 (없으면 skip)"""
    try:
        from transformers import AutoTokenizer

        tokenizer_path = Path("storage/models/meta-llama-mtp/tokenizer")

        if not tokenizer_path.exists():
            pytest.skip("Tokenizer not found: storage/models/meta-llama-mtp/tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    except ImportError:
        pytest.skip("transformers 라이브러리 필요")


class TestPairwiseDataCollator:
    """PairwiseDataCollator 핵심 기능 테스트"""

    def test_collator_initialization(self, tokenizer):
        """Collator 초기화"""
        collator = PairwiseDataCollator(
            tokenizer=tokenizer,
            max_length=2048,
            padding="max_length",
        )

        assert collator.max_length == 2048
        assert collator.padding == "max_length"

    def test_single_pair_output_structure(self, tokenizer):
        """단일 쌍의 출력 구조 검증"""
        collator = PairwiseDataCollator(tokenizer, max_length=256)

        sample = {
            "instruction": "Add two numbers.",
            "input": "",
            "correct_output": "def add(a, b): return a + b",
            "incorrect_output": "def add(a, b): return a - b",
        }

        batch = collator([sample])

        # 필수 키 존재 확인
        required_keys = [
            "pos_input_ids", "pos_attention_mask", "pos_labels",
            "neg_input_ids", "neg_attention_mask", "neg_labels",
        ]
        for key in required_keys:
            assert key in batch, f"Missing key: {key}"

        # Shape 검증 (batch_size=1, seq_len=256)
        assert batch["pos_input_ids"].shape == (1, 256)
        assert batch["pos_attention_mask"].shape == (1, 256)
        assert batch["pos_labels"].shape == (1, 256)
        assert batch["neg_input_ids"].shape == (1, 256)
        assert batch["neg_attention_mask"].shape == (1, 256)
        assert batch["neg_labels"].shape == (1, 256)

    def test_pos_neg_different_content(self, tokenizer):
        """Positive와 Negative가 다른 내용을 가지는지 확인"""
        collator = PairwiseDataCollator(tokenizer, max_length=256)

        sample = {
            "instruction": "Add two numbers.",
            "input": "",
            "correct_output": "def add(a, b): return a + b",
            "incorrect_output": "def add(a, b): return a - b",  # '-' vs '+'
        }

        batch = collator([sample])

        # pos와 neg의 input_ids가 달라야 함
        assert not torch.equal(batch["pos_input_ids"], batch["neg_input_ids"]), \
            "pos와 neg는 서로 다른 토큰을 가져야 함"

    def test_instruction_masking(self, tokenizer):
        """Instruction 부분이 마스킹되는지 확인"""
        collator = PairwiseDataCollator(tokenizer, max_length=256)

        sample = {
            "instruction": "Add two numbers.",
            "input": "",
            "correct_output": "def add(a, b): return a + b",
            "incorrect_output": "def add(a, b): return a - b",
        }

        batch = collator([sample])

        pos_labels = batch["pos_labels"][0]
        neg_labels = batch["neg_labels"][0]

        # Instruction 부분 (최소 첫 10개 토큰)은 -100이어야 함
        assert (pos_labels[:10] == -100).all(), "pos Instruction 마스킹 실패"
        assert (neg_labels[:10] == -100).all(), "neg Instruction 마스킹 실패"

    def test_output_not_masked(self, tokenizer):
        """Output 부분은 마스킹되지 않아야 함"""
        collator = PairwiseDataCollator(tokenizer, max_length=256)

        sample = {
            "instruction": "Task",
            "input": "",
            "correct_output": "correct output here",
            "incorrect_output": "wrong output here",
        }

        batch = collator([sample])

        pos_labels = batch["pos_labels"][0]
        neg_labels = batch["neg_labels"][0]

        # Non-padding 영역 중 일부는 -100이 아니어야 함 (output 영역)
        pos_non_masked = pos_labels[pos_labels != -100]
        neg_non_masked = neg_labels[neg_labels != -100]

        assert len(pos_non_masked) > 0, "pos output이 모두 마스킹됨"
        assert len(neg_non_masked) > 0, "neg output이 모두 마스킹됨"

    def test_padding_masking(self, tokenizer):
        """Padding 영역이 마스킹되는지 확인"""
        collator = PairwiseDataCollator(tokenizer, max_length=512, padding="max_length")

        sample = {
            "instruction": "Short task.",
            "input": "",
            "correct_output": "short",
            "incorrect_output": "wrong",
        }

        batch = collator([sample])

        pos_attention = batch["pos_attention_mask"][0]
        pos_labels = batch["pos_labels"][0]
        neg_attention = batch["neg_attention_mask"][0]
        neg_labels = batch["neg_labels"][0]

        # Padding 영역 (attention_mask == 0)은 labels도 -100이어야 함
        pos_padding_mask = pos_attention == 0
        neg_padding_mask = neg_attention == 0

        if pos_padding_mask.any():
            assert (pos_labels[pos_padding_mask] == -100).all(), "pos padding 마스킹 실패"
        if neg_padding_mask.any():
            assert (neg_labels[neg_padding_mask] == -100).all(), "neg padding 마스킹 실패"

    def test_batch_processing(self, tokenizer):
        """배치 처리 검증"""
        collator = PairwiseDataCollator(tokenizer, max_length=256)

        batch_samples = [
            {
                "instruction": "Add numbers.",
                "input": "",
                "correct_output": "def add(a,b): return a+b",
                "incorrect_output": "def add(a,b): return a-b",
            },
            {
                "instruction": "Multiply numbers.",
                "input": "Example: 2 * 3",
                "correct_output": "def mul(a,b): return a*b",
                "incorrect_output": "def mul(a,b): return a/b",
            },
            {
                "instruction": "Check even.",
                "input": "",
                "correct_output": "lambda n: n%2==0",
                "incorrect_output": "lambda n: n%2==1",
            },
        ]

        batch = collator(batch_samples)

        # Batch shape 검증
        assert batch["pos_input_ids"].shape == (3, 256)
        assert batch["neg_input_ids"].shape == (3, 256)

        # 각 샘플이 독립적으로 처리되었는지 확인
        for i in range(3):
            pos_labels = batch["pos_labels"][i]
            neg_labels = batch["neg_labels"][i]

            # Instruction 마스킹
            assert (pos_labels[:10] == -100).all()
            assert (neg_labels[:10] == -100).all()

            # Output 존재
            pos_attention = batch["pos_attention_mask"][i]
            neg_attention = batch["neg_attention_mask"][i]

            pos_non_padding = pos_labels[pos_attention == 1]
            neg_non_padding = neg_labels[neg_attention == 1]

            assert (pos_non_padding != -100).any()
            assert (neg_non_padding != -100).any()

    def test_truncation(self, tokenizer):
        """긴 텍스트 truncation 검증"""
        collator = PairwiseDataCollator(tokenizer, max_length=128)

        long_output = "Very long output. " * 100

        sample = {
            "instruction": "Task",
            "input": "",
            "correct_output": long_output,
            "incorrect_output": long_output,
        }

        batch = collator([sample])

        assert batch["pos_input_ids"].shape == (1, 128)
        assert batch["neg_input_ids"].shape == (1, 128)

    def test_deterministic_behavior(self, tokenizer):
        """동일 입력에 대해 항상 같은 결과"""
        collator = PairwiseDataCollator(tokenizer, max_length=256)

        sample = {
            "instruction": "Test",
            "input": "",
            "correct_output": "correct",
            "incorrect_output": "wrong",
        }

        batch1 = collator([sample])
        batch2 = collator([sample])

        assert torch.equal(batch1["pos_input_ids"], batch2["pos_input_ids"])
        assert torch.equal(batch1["neg_input_ids"], batch2["neg_input_ids"])
        assert torch.equal(batch1["pos_labels"], batch2["pos_labels"])
        assert torch.equal(batch1["neg_labels"], batch2["neg_labels"])


class TestPairwiseMaskingBoundaries:
    """Masking 경계 정확성 테스트"""

    def test_response_header_boundary(self, tokenizer):
        """### Response: 헤더 이후부터 학습 대상인지 확인"""
        collator = PairwiseDataCollator(tokenizer, max_length=512)

        instruction = "Write a function."
        correct_output = "def func(): pass"

        sample = {
            "instruction": instruction,
            "input": "",
            "correct_output": correct_output,
            "incorrect_output": "def wrong(): fail",
        }

        batch = collator([sample])

        pos_input_ids = batch["pos_input_ids"][0]
        pos_labels = batch["pos_labels"][0]
        pos_attention = batch["pos_attention_mask"][0]

        # Non-padding, non-masked 토큰 개수 계산
        valid_mask = pos_attention == 1
        non_masked_in_valid = (pos_labels[valid_mask] != -100).sum().item()

        # Output 토큰 개수 (대략적)
        output_tokens = tokenizer(correct_output, add_special_tokens=False)["input_ids"]
        expected_output_len = len(output_tokens)

        # 학습 대상 토큰이 output 길이와 비슷해야 함 (약간의 오차 허용)
        assert non_masked_in_valid >= expected_output_len * 0.8, \
            f"학습 대상 토큰({non_masked_in_valid})이 output 길이({expected_output_len})보다 너무 적음"

    def test_same_instruction_similar_masking(self, tokenizer):
        """동일 instruction의 pos/neg가 모두 올바르게 마스킹되는지 확인"""
        collator = PairwiseDataCollator(tokenizer, max_length=512)

        instruction = "Task description here."

        sample = {
            "instruction": instruction,
            "input": "",
            "correct_output": "correct output result",
            "incorrect_output": "wrong output result",
        }

        batch = collator([sample])

        pos_labels = batch["pos_labels"][0]
        neg_labels = batch["neg_labels"][0]
        pos_attention = batch["pos_attention_mask"][0]
        neg_attention = batch["neg_attention_mask"][0]

        # 둘 다 instruction 마스킹됨 (첫 10개 토큰)
        assert (pos_labels[:10] == -100).all(), "pos instruction not masked"
        assert (neg_labels[:10] == -100).all(), "neg instruction not masked"

        # 둘 다 학습 대상 토큰이 있음
        pos_non_masked = (pos_labels[pos_attention == 1] != -100).sum()
        neg_non_masked = (neg_labels[neg_attention == 1] != -100).sum()

        assert pos_non_masked > 0, "pos has no learning targets"
        assert neg_non_masked > 0, "neg has no learning targets"

        # 학습 대상 토큰 개수가 비슷함 (비슷한 output 길이)
        ratio = pos_non_masked / neg_non_masked
        assert 0.5 < ratio < 2.0, \
            f"pos/neg 학습 토큰 개수 차이가 너무 큼: {pos_non_masked} vs {neg_non_masked}"
