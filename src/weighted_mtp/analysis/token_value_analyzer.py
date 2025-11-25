"""토큰 단위 Value 분석기

Critic head가 학습한 value를 토큰 단위로 추출하고 분석
오답 코드에서 value 급락 지점을 탐지하여 에러 인지 능력 검증
"""

import logging
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.data.collators import apply_alpaca_template

logger = logging.getLogger(__name__)


class TokenValueAnalyzer:
    """토큰 단위 value 분석기

    학습된 Critic 모델을 사용하여 각 토큰 위치에서의 value를 추출하고
    value 변화 패턴을 분석합니다.
    """

    def __init__(
        self,
        adapter: MetaLlamaMTPAdapter,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        """분석기 초기화

        Args:
            adapter: 학습된 Critic 모델 (value_head 포함)
            tokenizer: 토크나이저
            device: 디바이스
        """
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.device = device

        # Value head 존재 확인
        if adapter.value_head is None:
            raise ValueError("Adapter에 value_head가 없습니다. Critic checkpoint를 로드하세요.")

        # 평가 모드 설정
        self.adapter.eval()

    def analyze_sample(
        self,
        code_text: str,
        is_correct: bool,
        max_length: int = 2048,
        instruction: str = "",
        input_text: str = "",
    ) -> dict:
        """단일 코드 샘플의 토큰별 value 추출

        학습 시와 동일한 Alpaca 템플릿을 적용하여 분석합니다.
        Output 영역 (Response 이후)만 분석 대상으로 표시합니다.

        Args:
            code_text: 분석할 코드 텍스트 (output)
            is_correct: 정답 여부
            max_length: 최대 시퀀스 길이
            instruction: 문제 설명 (Alpaca 템플릿용)
            input_text: 입력 예시 (Alpaca 템플릿용)

        Returns:
            {
                "code": str,
                "tokens": List[str],
                "values": List[float],
                "is_correct": bool,
                "output_start_idx": int,  # output 시작 토큰 인덱스
                "output_values": List[float],  # output 영역의 value만
            }
        """
        # 1. Alpaca 템플릿 적용 (학습 시와 동일)
        # Prompt 부분 (output 제외)
        prompt_text = apply_alpaca_template(
            instruction=instruction,
            input_text=input_text,
            output="",
            include_response_header=True,
        )
        prompt_tokens = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )
        len_prompt = len(prompt_tokens)

        # 전체 텍스트 (instruction + input + output)
        full_text = apply_alpaca_template(
            instruction=instruction,
            input_text=input_text,
            output=code_text,
            include_response_header=True,
        )

        # 2. Tokenize (학습 시와 동일한 방식)
        tokenized = self.tokenizer(
            full_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # 3. Inference (Value head)
        with torch.no_grad():
            outputs = self.adapter(
                input_ids,
                attention_mask,
                return_value_logits=True,
            )
            value_logits = outputs["value_logits"]  # [1, seq_len, 1]

        # 4. Value 추출 (BFloat16 → Float32 변환)
        values = value_logits[0, :, 0].float().cpu().numpy()
        attn_mask_np = attention_mask[0].cpu().numpy()

        # 5. 토큰 디코딩 (유효 토큰만)
        seq_len = int(attn_mask_np.sum())
        tokens = input_ids[0, :seq_len].cpu().tolist()
        token_texts = [self.tokenizer.decode([t]) for t in tokens]

        # 6. Output 영역 추출 (prompt 이후 ~ 유효 토큰까지)
        output_start_idx = min(len_prompt, seq_len)
        output_values = values[output_start_idx:seq_len].tolist()

        return {
            "code": code_text,
            "full_text": full_text,
            "tokens": token_texts,
            "values": values[:seq_len].tolist(),
            "is_correct": is_correct,
            "output_start_idx": output_start_idx,
            "output_values": output_values,
            "seq_len": seq_len,
            "prompt_len": len_prompt,
        }

    def compute_value_changes(self, values: list[float]) -> dict:
        """Value 변화량 계산 및 급락 지점 탐지

        Args:
            values: 토큰별 value 리스트

        Returns:
            {
                "gradient": List[float],
                "drop_indices": List[int],
                "max_drop": {"position": int, "value": float},
                "mean_value": float,
                "std_value": float,
                "num_drops": int,
            }
        """
        values_arr = np.array(values)

        # Value gradient 계산 (dV/dt)
        gradient = np.diff(values_arr)

        # 급락 지점 탐지 (threshold 기반)
        threshold = -0.1
        drop_indices = np.where(gradient < threshold)[0].tolist()

        # 최대 급락 지점
        if len(gradient) > 0:
            max_drop_idx = int(np.argmin(gradient))
            max_drop_value = float(gradient[max_drop_idx])
        else:
            max_drop_idx = 0
            max_drop_value = 0.0

        return {
            "gradient": gradient.tolist(),
            "drop_indices": drop_indices,
            "max_drop": {
                "position": max_drop_idx,
                "value": max_drop_value,
            },
            "mean_value": float(values_arr.mean()),
            "std_value": float(values_arr.std()),
            "num_drops": len(drop_indices),
        }

    def analyze_sample_full(
        self,
        code_text: str,
        is_correct: bool,
        max_length: int = 2048,
        instruction: str = "",
        input_text: str = "",
    ) -> dict:
        """단일 샘플의 value 추출 및 변화 분석을 한번에 수행

        Args:
            code_text: 분석할 코드 텍스트 (output)
            is_correct: 정답 여부
            max_length: 최대 시퀀스 길이
            instruction: 문제 설명 (Alpaca 템플릿용)
            input_text: 입력 예시 (Alpaca 템플릿용)

        Returns:
            analyze_sample() + compute_value_changes() 결과 통합
        """
        # 기본 분석 (Alpaca 템플릿 적용)
        result = self.analyze_sample(
            code_text=code_text,
            is_correct=is_correct,
            max_length=max_length,
            instruction=instruction,
            input_text=input_text,
        )

        # Output 영역의 value 변화 분석 (핵심!)
        output_changes = self.compute_value_changes(result["output_values"])
        result["output_changes"] = output_changes

        # 전체 시퀀스 변화 분석 (참고용)
        full_changes = self.compute_value_changes(result["values"])
        result["full_changes"] = full_changes

        # Output 영역 통계
        if result["output_values"]:
            result["output_mean_value"] = float(np.mean(result["output_values"]))
            result["output_std_value"] = float(np.std(result["output_values"]))
        else:
            result["output_mean_value"] = 0.0
            result["output_std_value"] = 0.0

        return result
