"""Value Head 방향성 검증 스크립트

Critic 체크포인트가 correct/incorrect를 올바른 방향으로 예측하는지 확인
"""
import torch
import json
from pathlib import Path

# 프로젝트 경로 설정
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.models.tokenizer_utils import load_tokenizer
from weighted_mtp.data.collators import AlpacaDataCollator


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. 모델 로드
    checkpoint_path = "storage/checkpoints/critic/critic-pretrain-linear-final/checkpoint_epoch_0.80.pt"
    print(f"Loading checkpoint: {checkpoint_path}")

    adapter = MetaLlamaMTPAdapter.from_pretrained(
        model_path=checkpoint_path,
        device=device,
        dtype="float32",  # CPU/MPS에서는 float32
    )
    adapter.eval()

    # 2. 토크나이저 로드
    tokenizer = load_tokenizer("storage/models/meta-llama-mtp/tokenizer")
    collator = AlpacaDataCollator(tokenizer=tokenizer, max_length=512)

    # 3. 테스트 데이터 로드 (correct/incorrect 샘플 각각)
    dataset_path = "storage/datasets/codecontests/processed/train.jsonl"

    correct_samples = []
    incorrect_samples = []

    with open(dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            if len(correct_samples) < 5 and sample.get("is_correct", False):
                correct_samples.append(sample)
            elif len(incorrect_samples) < 5 and not sample.get("is_correct", True):
                incorrect_samples.append(sample)
            if len(correct_samples) >= 5 and len(incorrect_samples) >= 5:
                break

    print(f"\nLoaded {len(correct_samples)} correct, {len(incorrect_samples)} incorrect samples")

    # 4. 예측값 비교
    print("\n" + "="*60)
    print("CORRECT 샘플 (is_correct=True, target=1.0)")
    print("="*60)

    correct_preds = []
    for i, sample in enumerate(correct_samples):
        batch = collator([sample])
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = adapter(input_ids, attention_mask, return_value_logits=True)
            value_logits = outputs["value_logits"]  # [1, seq, 1]

        # Response 토큰만 (labels != -100)
        labels = batch["labels"]
        response_mask = (labels != -100).squeeze(0)
        response_values = value_logits.squeeze()[response_mask]

        mean_pred = response_values.mean().item()
        correct_preds.append(mean_pred)
        print(f"  Sample {i}: mean_prediction = {mean_pred:.4f}")

    print("\n" + "="*60)
    print("INCORRECT 샘플 (is_correct=False, target=0.0)")
    print("="*60)

    incorrect_preds = []
    for i, sample in enumerate(incorrect_samples):
        batch = collator([sample])
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = adapter(input_ids, attention_mask, return_value_logits=True)
            value_logits = outputs["value_logits"]

        labels = batch["labels"]
        response_mask = (labels != -100).squeeze(0)
        response_values = value_logits.squeeze()[response_mask]

        mean_pred = response_values.mean().item()
        incorrect_preds.append(mean_pred)
        print(f"  Sample {i}: mean_prediction = {mean_pred:.4f}")

    # 5. 결과 요약
    avg_correct = sum(correct_preds) / len(correct_preds)
    avg_incorrect = sum(incorrect_preds) / len(incorrect_preds)

    print("\n" + "="*60)
    print("결과 요약")
    print("="*60)
    print(f"  Correct 샘플 평균 예측:   {avg_correct:.4f} (기대값: ~1.0)")
    print(f"  Incorrect 샘플 평균 예측: {avg_incorrect:.4f} (기대값: ~0.0)")
    print(f"  Pred Gap (correct - incorrect): {avg_correct - avg_incorrect:.4f}")

    if avg_correct > avg_incorrect:
        print("\n  ✓ Value head가 올바른 방향으로 학습됨")
    else:
        print("\n  ✗ Value head가 반대 방향으로 학습됨!")
        print("    → Correct 샘플에서 낮은 값, Incorrect 샘플에서 높은 값 예측")


if __name__ == "__main__":
    main()
