"""MTP Generation 유틸리티

MTP 모델의 autoregressive text generation 지원
"""

from typing import Any

import torch
import torch.nn.functional as F

from weighted_mtp.models.meta_mtp import MetaLlamaMTPAdapter


def generate_with_mtp(
    model: MetaLlamaMTPAdapter,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """MTP 모델로 autoregressive generation

    MTP는 n_future_tokens개를 예측하지만, generation 시에는
    head 0 (다음 토큰)만 사용하여 표준 autoregressive 방식으로 생성

    Args:
        model: MetaLlamaMTPAdapter (eval mode 권장)
        tokenizer: HuggingFace AutoTokenizer
        prompt: 생성 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        temperature: Sampling temperature (0=greedy, >0=sampling)
        top_p: Nucleus sampling threshold
        num_return_sequences: 생성할 시퀀스 개수 (Pass@K용)
        device: 디바이스

    Returns:
        생성된 텍스트 리스트 (길이 num_return_sequences)

    Examples:
        >>> model = MetaLlamaMTPAdapter.from_pretrained(...)
        >>> model.eval()
        >>> tokenizer = AutoTokenizer.from_pretrained(...)
        >>> outputs = generate_with_mtp(
        ...     model, tokenizer, "def hello():\\n", max_new_tokens=50
        ... )
        >>> print(outputs[0])
    """
    model.eval()
    generated_texts = []

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(num_return_sequences):
        current_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass (MTP transformer 직접 호출)
            with torch.no_grad():
                # return_all_heads=True: [batch, seq, n_future, vocab]
                logits = model.transformer(current_ids, start_pos=0, return_all_heads=True)

            # MTP head 0만 사용 (다음 토큰 예측)
            next_token_logits = logits[:, -1, 0, :]  # [batch, vocab]

            # Temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Cumulative probability가 top_p를 초과하는 토큰 제거
                sorted_indices_to_remove = cumulative_probs > top_p
                # 첫 번째 토큰은 항상 유지 (cumulative_probs[0] <= top_p일 수 있음)
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                # 제거할 토큰의 logit을 -inf로 설정
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            # EOS token 감지 시 종료
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode
        generated_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts
