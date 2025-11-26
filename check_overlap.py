import json

# Train metadata
with open('storage/datasets/codecontests/processed/train_metadata.json', 'r') as f:
    train_meta = json.load(f).get('metadata', [])

# Valid metadata  
with open('storage/datasets/codecontests/processed/valid_metadata.json', 'r') as f:
    valid_meta = json.load(f).get('metadata', [])

train_problems = set(m['problem_id'] for m in train_meta)
valid_problems = set(m['problem_id'] for m in valid_meta)

overlap = train_problems & valid_problems

print(f'Train problems: {len(train_problems):,}')
print(f'Valid problems: {len(valid_problems):,}')
print(f'Overlap: {len(overlap):,}')
print(f'Overlap ratio: {len(overlap)/len(valid_problems)*100:.1f}% of valid problems')
print()
if len(overlap) > 0:
    print('⚠️ WARNING: Train과 Valid에 동일한 problem이 존재!')
    print('→ 모델이 problem-level 패턴을 암기할 수 있음')
else:
    print('✅ Train/Valid problem-level split 완료')

