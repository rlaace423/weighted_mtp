# Weighted Multi-Token Prediction (WMTP)

Meta LLaMA MTP 네이티브 파이프라인을 사용하는 Critic-weighted WMTP 구현.

## 프로젝트 구조

```
weighted_mtp/
├── vendor/meta_llama/  # Meta 레퍼런스 코드
├── src/weighted_mtp/   # 프로젝트 소스
├── configs/            # 설정 파일
├── storage/            # 모델 및 데이터 자산
├── tests/              # 테스트
├── scripts/            # 유틸리티 스크립트
└── docs/               # 문서
```

## 빠른 시작

```bash
# 의존성 설치
uv pip install -e ".[dev]"

# 로컬 테스트 (Micro 모델)
uv run python -m weighted_mtp \
  --config configs/defaults.yaml \
  --recipe configs/recipe.baseline.yaml \
  --preset local-light \
  --dry-run

# 테스트 실행
uv run pytest tests/unit/
```

## 실험

- **Baseline MTP**: 표준 MTP (가중치 없음)
- **Verifiable Critic WMTP**: TD error 기반 토큰 가중치 + Critic Continual Learning
- **Rho-1 Weighted**: Reference 모델 loss 차이 기반 가중치

### Verifiable Critic WMTP 특징

PPO Best Practice를 따라 Stage2에서 Critic Continual Learning을 구현:

```yaml
# configs/recipe.verifiable.yaml
stage2:
  value_loss:
    enabled: true
    coefficient: 0.5      # Stable Baselines3 표준
    clip_range: 0.2       # Value loss clipping

  value_weighting:
    gae_gamma: 0.99
    gae_lambda: 0.95
    td_error_normalization: zscore
    weight_temperature: 1.0
    weight_clip_min: 0.1
    weight_clip_max: 5.0
```

## CLI 사용법

```bash
# Baseline 실험 (dry-run)
uv run python -m weighted_mtp \
  --recipe configs/recipe.baseline.yaml \
  --dry-run

# Verifiable Critic 실험 (dry-run)
uv run python -m weighted_mtp \
  --recipe configs/recipe.verifiable.yaml \
  --dry-run

# Rho-1 Weighted 실험 (dry-run)
uv run python -m weighted_mtp \
  --recipe configs/recipe.rho1_weighted.yaml \
  --dry-run

# 로컬 테스트 (Micro 모델 + small 데이터)
uv run python -m weighted_mtp \
  --recipe configs/recipe.baseline.yaml \
  --preset local-light \
  --use-micro-model \
  --dry-run
```

## 개발 현황

**Phase 2 완료** (2025-11-14):
- ✅ vendor/meta_llama/ 패키지 구성
- ✅ src/weighted_mtp/ 8개 모듈 스켈레톤
- ✅ pyproject.toml 및 의존성 설정
- ✅ configs/ 3개 recipe + preset
- ✅ CLI --dry-run 모드
- ✅ tests/unit/ 기본 테스트 (7 passed)

**다음 단계** (Phase 3+):
- Phase 3: 데이터 파이프라인 구현 (Loss masking collator)
- Phase 4: Meta Adapter 통합
- Phase 5: Value Weighting 모듈
- Phase 6: 학습 파이프라인 (Stage2 Value Loss 구현)

## 문서

- [00_ideal_structure.md](docs/00_ideal_structure.md): 이상적 구조
- [01_storage_preparation_plan.md](docs/01_storage_preparation_plan.md): Storage 재구성
- [02_implementation_plan.md](docs/02_implementation_plan.md): 전체 구현 계획
- [03_phase1_detailed_plan.md](docs/03_phase1_detailed_plan.md): Phase 1 상세
- [04_phase2_detailed_plan.md](docs/04_phase2_detailed_plan.md): Phase 2 상세

## 라이선스

MIT License