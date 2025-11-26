"""Data Collator for Alpaca-style SFT with Loss Masking

핵심 기능:
- Alpaca 템플릿 적용 (instruction + input + output)
- Instruction/Input 부분은 loss 계산에서 제외 (labels = -100)
- Output 부분만 학습 대상
- MTP 지원 (n_future_tokens=4)
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizer


def apply_alpaca_template(
    instruction: str,
    input_text: str,
    output: str = "",
    include_response_header: bool = True,
) -> str:
    """Alpaca 표준 템플릿 적용

    Args:
        instruction: 문제 설명
        input_text: 입력 예시 (빈 문자열 가능)
        output: 솔루션 코드 (기본값 빈 문자열)
        include_response_header: "### Response:" 헤더 포함 여부 (기본 True)

    Returns:
        템플릿 적용된 전체 텍스트
    """
    if input_text.strip():
        # Input이 있는 경우
        prompt = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
        )
    else:
        # Input이 없는 경우
        prompt = (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
        )

    if include_response_header:
        return prompt + f"### Response:\n{output}"
    else:
        return prompt.rstrip()


@dataclass
class AlpacaDataCollator:
    """Alpaca 형식 데이터를 토큰화하고 loss masking 적용

    Masking 전략:
    - Instruction 부분: labels = -100 (학습 제외)
    - Input 부분: labels = -100 (학습 제외)
    - Output 부분: labels = token_ids (학습 대상)
    - Padding: labels = -100 (학습 제외)

    Args:
        tokenizer: HuggingFace PreTrainedTokenizer
        max_length: 최대 시퀀스 길이 (기본 2048)
        n_future_tokens: MTP 미래 토큰 수 (기본 4)
        padding: Padding 전략 ("max_length" 또는 "longest")

    Examples:
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> collator = AlpacaDataCollator(tokenizer, max_length=2048)
        >>>
        >>> batch = [
        ...     {
        ...         "instruction": "Add two numbers.",
        ...         "input": "",
        ...         "output": "def add(a,b): return a+b"
        ...     }
        ... ]
        >>>
        >>> result = collator(batch)
        >>> # result = {
        >>> #     "input_ids": torch.Tensor,      # shape: (batch_size, seq_len)
        >>> #     "attention_mask": torch.Tensor, # shape: (batch_size, seq_len)
        >>> #     "labels": torch.Tensor,         # shape: (batch_size, seq_len)
        >>> # }
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    n_future_tokens: int = 4
    padding: str = "max_length"

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """배치를 토큰화하고 loss masking 적용

        Args:
            batch: 샘플 리스트, 각 샘플은 instruction, input, output, is_correct 키 포함

        Returns:
            토큰화된 배치 (input_ids, attention_mask, labels, is_correct)
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_is_correct = []

        for sample in batch:
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            output = sample["output"]
            is_correct = sample.get("is_correct", True)  # 기본값 True (backward compatibility)

            # 1. Instruction + Input + "### Response:\n" 부분 토큰화 (output 제외)
            # 이를 통해 prompt 길이를 정확히 계산
            prompt_text = apply_alpaca_template(
                instruction, input_text, output="", include_response_header=True
            )
            prompt_tokens = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )
            len_prompt = len(prompt_tokens["input_ids"])

            # 2. 전체 텍스트 토큰화 (instruction + input + output)
            full_text = apply_alpaca_template(instruction, input_text, output)
            tokenized = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokenized["input_ids"][0]  # shape: (seq_len,)
            attention_mask = tokenized["attention_mask"][0]  # shape: (seq_len,)

            # 3. Labels 생성 및 Masking
            labels = input_ids.clone()

            # 3-1. Instruction + Input 부분 마스킹 (BOS 포함)
            # prompt_text는 output=""이므로 "### Response:\n" 직전까지
            # len_prompt는 BOS를 포함한 길이
            labels[:len_prompt] = -100

            # 3-2. Padding 부분 마스킹
            labels[attention_mask == 0] = -100

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_is_correct.append(1.0 if is_correct else 0.0)

        # 4. 배치로 묶기
        return {
            "input_ids": torch.stack(batch_input_ids),  # (batch_size, seq_len)
            "attention_mask": torch.stack(batch_attention_mask),  # (batch_size, seq_len)
            "labels": torch.stack(batch_labels),  # (batch_size, seq_len)
            "is_correct": torch.tensor(batch_is_correct, dtype=torch.float32),  # (batch_size,)
        }


@dataclass
class PairwiseDataCollator:
    """Pairwise Ranking 학습용 Collator

    같은 instruction의 (correct, incorrect) 쌍을 배치로 구성.
    각 쌍에서 correct 코드의 value가 incorrect보다 높도록 학습.

    Args:
        tokenizer: HuggingFace PreTrainedTokenizer
        max_length: 최대 시퀀스 길이 (기본 2048)
        padding: Padding 전략 ("max_length" 또는 "longest")

    Examples:
        >>> collator = PairwiseDataCollator(tokenizer, max_length=2048)
        >>> batch = [
        ...     {
        ...         "instruction": "Add two numbers.",
        ...         "input": "",
        ...         "correct_output": "def add(a,b): return a+b",
        ...         "incorrect_output": "def add(a,b): return a-b",
        ...     }
        ... ]
        >>> result = collator(batch)
        >>> # result = {
        >>> #     "pos_input_ids": (batch_size, seq_len),
        >>> #     "pos_attention_mask": (batch_size, seq_len),
        >>> #     "pos_labels": (batch_size, seq_len),
        >>> #     "neg_input_ids": (batch_size, seq_len),
        >>> #     "neg_attention_mask": (batch_size, seq_len),
        >>> #     "neg_labels": (batch_size, seq_len),
        >>> # }
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    padding: str = "max_length"

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """배치를 토큰화

        Args:
            batch: 샘플 리스트, 각 샘플은 다음 키 포함:
                - instruction: 문제 설명
                - input: 입력 예시 (optional)
                - correct_output: 정답 코드
                - incorrect_output: 오답 코드

        Returns:
            pos_input_ids, pos_attention_mask, pos_labels: correct samples
            neg_input_ids, neg_attention_mask, neg_labels: incorrect samples
        """
        pos_input_ids = []
        pos_attention_mask = []
        pos_labels = []
        neg_input_ids = []
        neg_attention_mask = []
        neg_labels = []

        for sample in batch:
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            correct_output = sample["correct_output"]
            incorrect_output = sample["incorrect_output"]

            # Positive (correct) 토큰화
            pos_tokens = self._tokenize_sample(instruction, input_text, correct_output)
            pos_input_ids.append(pos_tokens["input_ids"])
            pos_attention_mask.append(pos_tokens["attention_mask"])
            pos_labels.append(pos_tokens["labels"])

            # Negative (incorrect) 토큰화
            neg_tokens = self._tokenize_sample(instruction, input_text, incorrect_output)
            neg_input_ids.append(neg_tokens["input_ids"])
            neg_attention_mask.append(neg_tokens["attention_mask"])
            neg_labels.append(neg_tokens["labels"])

        return {
            "pos_input_ids": torch.stack(pos_input_ids),
            "pos_attention_mask": torch.stack(pos_attention_mask),
            "pos_labels": torch.stack(pos_labels),
            "neg_input_ids": torch.stack(neg_input_ids),
            "neg_attention_mask": torch.stack(neg_attention_mask),
            "neg_labels": torch.stack(neg_labels),
        }

    def _tokenize_sample(
        self,
        instruction: str,
        input_text: str,
        output: str,
    ) -> dict[str, torch.Tensor]:
        """단일 샘플 토큰화 (AlpacaDataCollator 로직 재사용)

        Args:
            instruction: 문제 설명
            input_text: 입력 예시
            output: 솔루션 코드

        Returns:
            input_ids, attention_mask, labels (각각 1D tensor)
        """
        # 1. Prompt 부분 토큰화 (output 제외) → 길이 계산용
        prompt_text = apply_alpaca_template(
            instruction, input_text, output="", include_response_header=True
        )
        prompt_tokens = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        len_prompt = len(prompt_tokens["input_ids"])

        # 2. 전체 텍스트 토큰화
        full_text = apply_alpaca_template(instruction, input_text, output)
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # 3. Labels 생성 및 Masking (right-padding 기준)
        labels = input_ids.clone()
        labels[:len_prompt] = -100  # Instruction + Input 마스킹
        labels[attention_mask == 0] = -100  # Padding 마스킹

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
