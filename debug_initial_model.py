"""초기 모델의 Train vs Val 출력 비교"""
import sys
sys.path.insert(0, 'src')

import torch
import json
from transformers import AutoTokenizer
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.data.collators import AlpacaDataCollator
from weighted_mtp.data.datasets import _compute_sampling_indices_from_metadata

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('storage/models/meta-llama-mtp/tokenizer')
collator = AlpacaDataCollator(tokenizer=tokenizer, max_length=512)

# 메타데이터 로드
with open('storage/datasets/codecontests/processed/train_metadata.json') as f:
    train_meta = json.load(f)['metadata']
with open('storage/datasets/codecontests/processed/validation_metadata.json') as f:
    val_meta = json.load(f)['metadata']

# sampling_config
sampling_config = {
    'sampling_method': 'difficulty',
    'difficulty': {
        'n_samples': 100,
        'auto_data_balancing': True,
        'correct_ratio': 0.5,
        'bins': {'diff_7': [7, 7], 'else': [8, 25]},
        'weights': {'diff_7': 0.35, 'else': 0.65},
    },
}

# 샘플링
train_indices = _compute_sampling_indices_from_metadata(train_meta, sampling_config, seed=42)
val_indices = _compute_sampling_indices_from_metadata(val_meta, sampling_config, seed=42)

# JSONL에서 샘플 로드
def load_samples_by_indices(jsonl_path, indices, max_samples=50):
    samples = []
    target_set = set(indices[:max_samples])
    idx_to_pos = {idx: pos for pos, idx in enumerate(indices[:max_samples])}
    result = [None] * len(target_set)

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i in target_set:
                sample = json.loads(line)
                result[idx_to_pos[i]] = sample
            if sum(1 for x in result if x is not None) >= len(target_set):
                break
    return [s for s in result if s is not None]

train_samples = load_samples_by_indices('storage/datasets/codecontests/processed/train.jsonl', train_indices, 20)
val_samples = load_samples_by_indices('storage/datasets/codecontests/processed/valid.jsonl', val_indices, 20)

print(f'Train samples loaded: {len(train_samples)}')
print(f'Val samples loaded: {len(val_samples)}')

# 모델 로드
print('Loading model...')
adapter = MetaLlamaMTPAdapter.from_pretrained(
    model_path='storage/models/meta-llama-mtp',
    device=device,
    dtype='bfloat16',
    value_head_type='linear',
)
adapter.eval()

# Value head 초기 가중치 확인
print('\n=== Value Head 초기 가중치 ===')
for name, param in adapter.value_head.named_parameters():
    print(f'{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}')


def compute_stats(adapter, batch, device, label):
    """배치에서 correct/incorrect별 value 평균 계산"""
    input_ids = batch['input_ids'].to(device)
    attn = batch['attention_mask'].to(device)
    is_correct = batch['is_correct']
    labels = batch['labels']

    with torch.no_grad():
        outputs = adapter(input_ids, attn, return_value_logits=True)
        values = outputs['value_logits'].squeeze(-1)

    # labels mask
    valid_mask = (labels != -100).to(device)

    correct_vals = []
    incorrect_vals = []

    for i in range(len(is_correct)):
        mask = valid_mask[i]
        vals = values[i][mask]
        mean_val = vals.mean().item()

        if is_correct[i].item() > 0.5:
            correct_vals.append(mean_val)
        else:
            incorrect_vals.append(mean_val)

    mean_correct = sum(correct_vals) / len(correct_vals) if correct_vals else 0
    mean_incorrect = sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else 0
    pred_gap = mean_correct - mean_incorrect

    print(f'\n=== {label} ===')
    print(f'Correct samples: {len(correct_vals)}, mean_value: {mean_correct:.4f}')
    print(f'Incorrect samples: {len(incorrect_vals)}, mean_value: {mean_incorrect:.4f}')
    print(f'Pred Gap: {pred_gap:.4f}')

    return mean_correct, mean_incorrect, pred_gap


# Train 데이터 예측
train_batch = collator(train_samples)
train_mc, train_mi, train_gap = compute_stats(adapter, train_batch, device, 'Train (Initial Model)')

# Val 데이터 예측
val_batch = collator(val_samples)
val_mc, val_mi, val_gap = compute_stats(adapter, val_batch, device, 'Validation (Initial Model)')

print('\n=== Summary ===')
print(f'Train pred_gap: {train_gap:.4f}')
print(f'Val pred_gap: {val_gap:.4f}')
print(f'Gap difference: {abs(train_gap - val_gap):.4f}')
