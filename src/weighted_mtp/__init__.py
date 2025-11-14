"""
Weighted Multi-Token Prediction (WMTP) 프로젝트

Meta LLaMA MTP 네이티브 파이프라인을 사용하는 Critic-weighted WMTP 구현.
세 가지 핵심 실험을 지원한다:
- Baseline MTP
- Verifiable Critic WMTP
- Rho-1 Weighted

주요 모듈:
- cli: 사용자 진입점
- core: 설정, 로깅, 레지스트리
- data: 데이터 로딩 및 전처리
- models: Meta Adapter 및 체크포인트 관리
- value_weighting: TD error 기반 토큰 가중치 계산
- pipelines: 학습/평가 파이프라인
- runtime: 환경 초기화, MLflow 연동
- utils: 공통 유틸리티
"""

__version__ = "0.2.0"
