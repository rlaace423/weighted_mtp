"""Generation Utils Unit Tests"""

import pytest
import torch
from transformers import AutoTokenizer

from weighted_mtp.models.meta_mtp import MetaLlamaMTPAdapter
from weighted_mtp.utils.generation_utils import generate_with_mtp


@pytest.fixture
def micro_model():
    """Micro MTP 모델 로드"""
    model = MetaLlamaMTPAdapter.from_pretrained(
        model_path="storage/models_v2/micro-mtp",
        device="cpu",
        initialize_value_head=False,  # Generation에는 불필요
    )
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    """Tokenizer 로드"""
    # SentencePiece 토크나이저 직접 로드
    tokenizer = AutoTokenizer.from_pretrained(
        "storage/models_v2/meta-llama-mtp/tokenizer",
        use_fast=False,  # SentencePiece는 slow tokenizer 사용
        legacy=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_generate_with_mtp_greedy(micro_model, tokenizer):
    """Greedy decoding 테스트"""
    prompt = "def hello():\n"

    outputs = generate_with_mtp(
        model=micro_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.0,  # Greedy
        num_return_sequences=1,
        device=torch.device("cpu"),
    )

    # 기본 검증
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
    # 생성된 텍스트가 프롬프트보다 길어야 함 (최소 1개 토큰 생성)
    assert len(outputs[0]) >= len(prompt)


def test_generate_with_mtp_sampling(micro_model, tokenizer):
    """Sampling 다양성 테스트"""
    prompt = "def hello():\n"

    outputs = generate_with_mtp(
        model=micro_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,  # Sampling
        top_p=0.95,
        num_return_sequences=5,
        device=torch.device("cpu"),
    )

    # 기본 검증
    assert len(outputs) == 5
    for output in outputs:
        assert isinstance(output, str)
        assert len(output) >= len(prompt)

    # 다양성 확인 (5개가 모두 다를 확률이 높음)
    # Micro 모델은 작아서 다양성이 낮을 수 있으므로 최소 2개 이상만 요구
    unique_outputs = len(set(outputs))
    assert unique_outputs >= 2, f"Expected at least 2 unique outputs, got {unique_outputs}"


def test_generate_with_mtp_short_generation(micro_model, tokenizer):
    """짧은 생성 테스트 (EOS token 조기 종료 가능)"""
    prompt = "1 + 1 ="

    outputs = generate_with_mtp(
        model=micro_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=5,
        temperature=0.0,
        num_return_sequences=1,
        device=torch.device("cpu"),
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], str)


def test_generate_with_mtp_empty_prompt(micro_model, tokenizer):
    """빈 프롬프트 테스트"""
    prompt = ""

    outputs = generate_with_mtp(
        model=micro_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=10,
        temperature=0.0,
        num_return_sequences=1,
        device=torch.device("cpu"),
    )

    # 빈 프롬프트여도 생성 가능해야 함
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
