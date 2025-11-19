# Baseline 2-GPU Training Hang 분석

**작성일**: 2025-11-19
**증상**: 23:08:25 이후 12분+ 추가 로그 없음, 첫 forward pass에서 hang 추정

---

## 1. 문제 요약

### 1.1 증상
- **마지막 로그**: `2025-11-19 14:08:25,042 [INFO] [BASELINE:R0] --- Training to epoch 0.50 ---`
- **Hang 위치**: 첫 번째 training batch의 forward pass 추정
- **환경**: VESSL, A100 80GB x2, FSDP FULL_SHARD, mixed precision (bfloat16)

### 1.2 로그 분석
```
Nov 19, 23:08:13  NCCL 초기화 완료 (rank 0, 1)
Nov 19, 23:08:14  FSDP wrapping 완료
Nov 19, 23:08:14  Tokenizer 로드 완료
Nov 19, 23:08:15  Dataset 로드 시작 (rank 0, 1)
Nov 19, 23:08:24  Dataset 로드 완료: 75,000 samples per rank
Nov 19, 23:08:25  Training loop 진입: "--- Training to epoch 0.50 ---"
Nov 19, 23:08:25  ❌ 이후 12분+ 로그 없음
```

---

## 2. 가능한 원인 분석

### 원인 1: FSDP 첫 Forward Pass에서 Parameter All-Gather Hang
**가능성**: ⭐⭐⭐⭐⭐ (가장 높음)

**설명**:
- FSDP FULL_SHARD에서 첫 forward는 모든 rank의 parameter를 all-gather
- 3.37B 파라미터를 all-gather하는 데 시간이 걸리며, NCCL 통신 문제 시 hang 발생
- 특히 mixed precision (bfloat16) 환경에서 parameter dtype 변환 이슈

**증거**:
- 로그가 "Training to epoch 0.50" 직후 멈춤
- NCCL 초기화는 정상 완료 (P2P CUMEM 연결 성공)
- Dataset 로드는 정상 (양쪽 rank 모두 완료)

**진단 방법**:
```python
# run_baseline.py Line 365 직전에 추가된 디버그 로그
if batch_count == 0:
    logger.info(f"[Rank {rank}] Starting first batch - moving data to device")
    # ... data move
    logger.info(f"[Rank {rank}] Data moved to device, starting forward pass")
logits = adapter(input_ids)  # ← 여기서 hang 예상
logger.info(f"[Rank {rank}] Forward pass completed")  # ← 이 로그가 안 나오면 확정
```

### 원인 2: DataLoader Iterator 생성 시 Hang
**가능성**: ⭐⭐

**설명**:
- `epoch_train_loader = iter(train_loader)` (Line 337)
- DistributedSampler가 양쪽 rank에서 다르게 동작

**증거**:
- 하지만 dataset 로드는 정상 완료
- 로그 순서상 이미 지나간 부분

### 원인 3: NCCL Timeout
**가능성**: ⭐⭐⭐

**설명**:
- NCCL 기본 timeout = 10분
- 현재 12분+ hang이므로 timeout 초과
- 하지만 timeout 시 보통 에러 발생해야 함

**진단**:
- 환경 변수 `NCCL_TIMEOUT_SECONDS` 확인
- 현재 설정되지 않았다면 기본값 600초 (10분)

### 원인 4: Mixed Precision (bfloat16) Initialization Issue
**가능성**: ⭐⭐⭐⭐

**설명**:
- Model dtype: bfloat16 (config에서 설정)
- FSDP mixed precision 설정됨
- 첫 forward에서 dtype casting hang 가능

**관련 코드**:
```python
# configs/baseline/baseline.yaml
models:
  policy:
    dtype: float16  # ← bfloat16으로 변환됨

# src/weighted_mtp/runtime/fsdp.py:73
mp_policy = MixedPrecision(
    param_dtype=model_dtype,  # bfloat16
    reduce_dtype=model_dtype,
    buffer_dtype=model_dtype,
)
```

---

## 3. 즉시 시도 가능한 해결책

### 해결책 1: NCCL Debug 로그 활성화
**Priority**: ⭐⭐⭐⭐⭐

```bash
# VESSL 환경에서 재실행
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=3600  # 1시간으로 연장

uv run torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    -m weighted_mtp.pipelines.run_baseline \
    --config configs/baseline/baseline.yaml \
    --override training.batch_size=4 \
    --override training.gradient_accumulation_steps=4
```

**기대 결과**:
- NCCL 통신 상세 로그 확인
- All-gather 진행 상황 확인
- Timeout 시 명확한 에러 메시지

### 해결책 2: 디버그 로그 추가된 코드로 재실행
**Priority**: ⭐⭐⭐⭐⭐

**현재 추가된 디버그 로그**:
1. Line 327-329: Training loop 진입 전 barrier + 로그
2. Line 353-369: 첫 배치 처리 상세 로그
   - Data move to device
   - Forward pass 시작
   - Forward pass 완료

**재실행 후 확인 사항**:
```
예상 로그 순서:
[Rank 0] Reaching barrier before training loop
[Rank 1] Reaching barrier before training loop
[Rank 0] Barrier passed, entering training loop
[Rank 1] Barrier passed, entering training loop
--- Training to epoch 0.50 ---
[Rank 0] Starting first batch - moving data to device
[Rank 1] Starting first batch - moving data to device
[Rank 0] Data moved to device, starting forward pass
[Rank 1] Data moved to device, starting forward pass
❓ 여기서 멈추면 → FSDP forward pass hang 확정
[Rank 0] Forward pass completed, logits shape: ...
[Rank 1] Forward pass completed, logits shape: ...
```

### 해결책 3: Batch Size 더 줄이기 (긴급 회피)
**Priority**: ⭐⭐

```bash
# batch_size=2, gradient_accumulation=8로 테스트
uv run torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    -m weighted_mtp.pipelines.run_baseline \
    --config configs/baseline/baseline.yaml \
    --override training.batch_size=2 \
    --override training.gradient_accumulation_steps=8
```

**이유**:
- 더 작은 batch로 all-gather 부담 감소
- 하지만 근본 원인 해결은 아님

### 해결책 4: Mixed Precision 비활성화 테스트
**Priority**: ⭐⭐⭐

```yaml
# configs/baseline/baseline.yaml 수정
models:
  policy:
    dtype: float32  # bfloat16 → float32

distributed:
  fsdp:
    mixed_precision: false  # true → false
```

**재실행**:
```bash
uv run torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    -m weighted_mtp.pipelines.run_baseline \
    --config configs/baseline/baseline.yaml \
    --override training.batch_size=4 \
    --override training.gradient_accumulation_steps=4
```

**주의**: 메모리 사용량 증가, OOM 가능성

---

## 4. 근본 원인 조사 계획

### Step 1: Hang 위치 정확히 파악
**실행**: 해결책 2 (디버그 로그)

**확인 사항**:
- [ ] Barrier 통과 여부
- [ ] Data move 완료 여부
- [ ] Forward pass 진입 여부
- [ ] Forward pass 완료 여부

### Step 2: FSDP Forward Hook 검사
**의심 코드**: `src/weighted_mtp/models/meta_mtp/adapter.py:219`

```python
# MetaLlamaMTPAdapter.forward()
logits = self.transformer(
    input_ids,
    start_pos=0,
    return_all_heads=True,
)
```

**검증**:
- Transformer forward가 FSDP all-gather를 트리거
- 3.37B params = ~13.5GB (bfloat16)
- 2 ranks all-gather = 27GB 통신

### Step 3: NCCL 통신 프로파일링
```bash
# NCCL 통신 trace 활성화
export NCCL_GRAPH_DUMP_FILE=nccl_graph.txt
export NCCL_ALGO=RING  # Algorithm 명시

# Pytorch profiler 추가
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    -m weighted_mtp.pipelines.run_baseline \
    ...
```

### Step 4: 더 작은 모델로 테스트
```bash
# Micro model (46M params)로 baseline 테스트
# 빠르게 FSDP 자체 문제인지 파악

uv run torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    -m weighted_mtp.pipelines.run_baseline \
    --config configs/baseline/baseline_local.yaml \
    --override training.batch_size=4 \
    --override data_sampling.n_samples=500
```

---

## 5. 유사 이슈 검색 결과

### PyTorch FSDP Known Issues

**Issue 1**: FSDP hang on first forward with large models
- https://github.com/pytorch/pytorch/issues/109173
- **증상**: 정확히 동일 - 첫 forward에서 hang
- **원인**: NCCL all-gather timeout
- **해결**: `NCCL_TIMEOUT` 증가 또는 `use_orig_params=True`

**Issue 2**: FSDP mixed precision hang
- https://github.com/pytorch/pytorch/issues/95118
- **증상**: bfloat16 mixed precision에서 hang
- **해결**: `MixedPrecision` 설정 조정

### Meta LLaMA FSDP Examples
- Meta의 레퍼런스 구현에서도 첫 forward에 시간 걸림 언급
- Recommendation: Warmup step with logging

---

## 6. 최우선 실행 계획

### Immediate Action (지금 즉시)

1. **NCCL Debug 활성화 + 디버그 로그로 재실행**
```bash
# VESSL에서 실행
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600

uv run torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    -m weighted_mtp.pipelines.run_baseline \
    --config configs/baseline/baseline.yaml \
    --override training.batch_size=4 \
    --override training.gradient_accumulation_steps=4
```

2. **로그 모니터링**
```bash
# 별도 터미널에서 실시간 모니터링
tail -f <vessl_log_file>

# 확인 포인트:
# - "Reaching barrier before training loop"
# - "Starting first batch"
# - "Data moved to device, starting forward pass"
# - ❓ "Forward pass completed" ← 이 로그가 나오는지 확인
```

3. **5분 후 GPU 상태 확인**
```bash
# 별도 터미널
watch -n 1 nvidia-smi

# 확인 사항:
# - GPU 0, 1 모두 활성화?
# - GPU Memory 사용량 증가?
# - GPU Utilization > 0%?
```

### If Still Hangs (5분 후에도 로그 없으면)

**Option A**: Mixed precision 비활성화
```yaml
# configs/baseline/baseline.yaml
models:
  policy:
    dtype: float32

distributed:
  fsdp:
    mixed_precision: false
```

**Option B**: use_orig_params=True 시도
```python
# src/weighted_mtp/runtime/fsdp.py:101
wrapped_model = FSDP(
    model,
    ...
    use_orig_params=True,  # False → True
)
```

---

## 7. 예상 결과

### 시나리오 1: FSDP All-Gather Timeout (확률 70%)
**로그 패턴**:
```
[Rank 0] Data moved to device, starting forward pass
[Rank 1] Data moved to device, starting forward pass
❌ 10분+ 대기
[NCCL] Timeout in AllGather operation
```

**해결**: `NCCL_TIMEOUT=3600` + 첫 forward에 시간 걸림 인정

### 시나리오 2: Mixed Precision Issue (확률 20%)
**로그 패턴**:
```
[Rank 0] Data moved to device, starting forward pass
❌ 즉시 hang, NCCL 로그 없음
```

**해결**: `mixed_precision=false` 또는 dtype 조정

### 시나리오 3: FSDP Parameter Initialization Bug (확률 10%)
**로그 패턴**:
```
[Rank 0] Reaching barrier before training loop
❌ Barrier에서 hang
```

**해결**: FSDP wrapping 순서 조정 또는 PyTorch 버전 확인

---

## 8. 코드 수정 사항 (현재 커밋)

### 추가된 디버그 로그

**파일**: `src/weighted_mtp/pipelines/run_baseline.py`

**Line 325-329**:
```python
# Debug: Training loop 진입 전 barrier
if is_distributed():
    logger.info(f"[Rank {rank}] Reaching barrier before training loop")
    barrier()
    logger.info(f"[Rank {rank}] Barrier passed, entering training loop")
```

**Line 352-369**:
```python
# Debug: 첫 배치 시작 로그
if batch_count == 0:
    logger.info(f"[Rank {rank}] Starting first batch - moving data to device")

input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
labels = batch["labels"].to(device)

# Debug: Forward 시작
if batch_count == 0:
    logger.info(f"[Rank {rank}] Data moved to device, starting forward pass")

# Forward (MTP만, Value head 없음)
logits = adapter(input_ids)

# Debug: Forward 완료
if batch_count == 0:
    logger.info(f"[Rank {rank}] Forward pass completed, logits shape: {logits.shape}")
```

**목적**: Hang 위치를 정확히 파악하여 근본 원인 진단

---

## 9. Next Steps

1. ✅ **디버그 로그 추가 완료** (현재 커밋)
2. ⏳ **사용자 VESSL 재실행** (NCCL_DEBUG=INFO)
3. ⏳ **로그 분석** (어느 단계에서 멈추는지)
4. ⏳ **근본 원인 파악** (FSDP vs NCCL vs Mixed Precision)
5. ⏳ **해결책 적용** (timeout 증가 또는 설정 변경)

---

**요약**: 현재 상황은 **FSDP 첫 forward pass에서 parameter all-gather hang**이 가장 유력합니다. 디버그 로그를 추가했으니 VESSL에서 재실행하여 정확한 hang 위치를 파악한 후 적절한 해결책을 적용하겠습니다.
