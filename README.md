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

**구현 완료** (2025-11-17):
- ✅ Pure PyTorch Transformer (Meta 독립, FSDP 호환)
- ✅ 4개 파이프라인 구현 (Baseline, Critic, Verifiable, Rho-1)
- ✅ 메타데이터 기반 데이터 로딩 (99% 메모리 절감)
- ✅ MLflow + S3 인프라 연동
- ✅ VESSL A100 4-GPU 분산학습 준비
- ✅ 통합 테스트 전체 통과

## 문서

- [ARCHITECTURE.md](docs/ARCHITECTURE.md): 코드베이스 아키텍처 및 핵심 구현
- [SETUP.md](docs/SETUP.md): 환경 설정 및 데이터 준비
- [VESSL.md](docs/VESSL.md): VESSL A100 4-GPU 실행 가이드
- [MLFLOW.md](docs/MLFLOW.md): MLflow 추적 및 S3 연동
- [RESEARCH.md](docs/RESEARCH.md): 연구 배경 및 이론
- [vessl_yaml_cheatsheet.md](docs/vessl_yaml_cheatsheet.md): VESSL YAML 참조

## 라이선스

MIT License