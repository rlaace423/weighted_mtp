# Value Model 체크포인트 최적화 계획

## 목표
Backbone이 Frozen 상태일 때 학습 가능한 MLP Head만 저장하여 Value Model(Critic)의 저장 효율성을 최적화합니다. 이는 HuggingFace에서 사전 학습된 Backbone을 로드하고 학습된 MLP Head를 부착하려는 사용자의 의도와 구현을 일치시킵니다.

## 현황 분석
- **저장 (`run_critic.py`)**: `save_value_model_checkpoint` 함수는 현재 Backbone의 Frozen 여부와 관계없이 전체 `backbone_state_dict`를 저장합니다. 이로 인해 기본 모델의 중복된 파라미터를 포함하는 불필요하게 큰 체크포인트 파일(약 5-6GB)이 생성됩니다.
- **로드 (`value_model.py`)**: `ValueModel.from_checkpoint` 메서드는 `from_pretrained`를 사용하여 모델을 초기화(기본 모델 로드)한 다음, 체크포인트에 `backbone_state_dict`가 존재하면 Backbone 가중치를 덮어씁니다.

## 개선 계획

### Phase 1: 저장 로직 최적화 (`run_critic.py`)
**목표**: `backbone_frozen`이 True일 때 `save_value_model_checkpoint`가 Backbone 가중치를 제외하도록 수정합니다.

- **조치**:
    - `save_value_model_checkpoint`가 `backbone_frozen` 플래그를 받도록(또는 config에서 유도하도록) 업데이트합니다.
    - `backbone_frozen`이 True이면 체크포인트 딕셔너리에서 `backbone_state_dict`를 `None`으로 설정하거나 키를 완전히 제외합니다.
    - 저장 프로세스를 로깅하여 Backbone이 저장되는지 또는 건너뛰는지 명확히 표시합니다.

### Phase 2: 로드 로직 개선 (`value_model.py`)
**목표**: `ValueModel.from_checkpoint`가 Head-only 체크포인트를 명시적이고 투명하게 처리하도록 보장합니다.

- **조치**:
    - `from_checkpoint`에서 `backbone_state_dict`의 존재 여부를 확인합니다.
    - 누락된 경우 명확한 메시지를 로깅합니다: "체크포인트에서 Backbone 가중치를 찾을 수 없습니다. {model_path}의 사전 학습된 가중치를 사용합니다."
    - 존재하는 경우 로드를 진행합니다(기존 전체 체크포인트에 대한 하위 호환성은 유지하되, 원칙 4에 따라 새로운 체크포인트에 대한 불필요한 Fallback 로직은 배제).
    - `value_head_state_dict`는 항상 엄격하게(Strict) 로드되도록 보장합니다.

## 검증
1. **파일 크기**: 새로운 체크포인트 파일 크기가 훨씬 작은지 확인합니다 (GB 단위 vs KB/MB 단위).
2. **로드**: `run_verifiable.py`가 Head-only 체크포인트를 성공적으로 로드하고 사전 학습된 Backbone과 함께 올바르게 작동하는지 검증합니다.
