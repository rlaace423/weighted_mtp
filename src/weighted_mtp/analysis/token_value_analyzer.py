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
    ) -> dict:
        """단일 코드 샘플의 토큰별 value 추출

        Args:
            code_text: 분석할 코드 텍스트
            is_correct: 정답 여부
            max_length: 최대 시퀀스 길이

        Returns:
            {
                "code": str,
                "tokens": List[str],
                "values": List[float],
                "is_correct": bool,
            }
        """
        # 1. Tokenize
        tokens = self.tokenizer.encode(
            code_text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )

        input_ids = torch.tensor([tokens]).to(self.device)

        # 2. Inference (Value head만)
        with torch.no_grad():
            outputs = self.adapter(input_ids, return_value_logits=True)
            value_logits = outputs["value_logits"]  # [1, seq_len, 1]

        # 3. Value 추출 (BFloat16 → Float32 변환 후 numpy)
        values = value_logits[0, :, 0].float().cpu().numpy()

        # 4. 토큰 디코딩
        token_texts = [self.tokenizer.decode([t]) for t in tokens]

        return {
            "code": code_text,
            "tokens": token_texts,
            "values": values.tolist(),
            "is_correct": is_correct,
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
    ) -> dict:
        """단일 샘플의 value 추출 및 변화 분석을 한번에 수행

        Args:
            code_text: 분석할 코드 텍스트
            is_correct: 정답 여부
            max_length: 최대 시퀀스 길이

        Returns:
            analyze_sample() + compute_value_changes() 결과 통합
        """
        # 기본 분석
        result = self.analyze_sample(code_text, is_correct, max_length)

        # Value 변화 분석 추가
        changes = self.compute_value_changes(result["values"])
        result.update(changes)

        return result
