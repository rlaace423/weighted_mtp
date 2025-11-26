"""FSDP + LoRA 통합 테스트

FSDP 분산학습 환경에서 LoRA 정상 동작 검증
- LoRA 모델의 FSDP wrapping
- State dict에서 LoRA 가중치 포함 확인
- Checkpoint 저장/로드 검증
"""

import pytest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.models.lora import LoRALinear, get_lora_parameters
from weighted_mtp.runtime import wrap_model_fsdp, unwrap_model


# Fixture: micro-mtp 모델 경로
MICRO_MODEL_PATH = "storage/models/micro-mtp"


class TestLoRAFSDPIntegration:
    """FSDP + LoRA 통합 테스트"""

    @pytest.fixture
    def device(self):
        """테스트용 device"""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @pytest.fixture
    def lora_config(self):
        """기본 LoRA 설정"""
        return {
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "target_modules": ["wq", "wk", "wv", "wo"],
        }

    @pytest.fixture
    def adapter_with_lora(self, device, lora_config):
        """LoRA가 적용된 adapter"""
        adapter = MetaLlamaMTPAdapter.from_pretrained(
            model_path=MICRO_MODEL_PATH,
            device=device,
            dtype="float32",
            initialize_value_head=True,
            value_head_type="mlp",
            use_lora=True,
            lora_config=lora_config,
        )
        return adapter

    def test_lora_fsdp_wrapping(self, adapter_with_lora, device):
        """LoRA 모델 FSDP wrapping 테스트 (단일 장치 환경)

        단일 장치에서는 FSDP wrapping이 skip되어야 함
        """
        wrapped = wrap_model_fsdp(
            adapter_with_lora,
            device,
            sharding_strategy="FULL_SHARD",
            mixed_precision=False,
        )

        # 단일 장치에서는 원본 모델 반환
        assert wrapped is adapter_with_lora

    def test_lora_state_dict_keys(self, adapter_with_lora):
        """LoRA state_dict에 LoRA 가중치 포함 확인"""
        state_dict = adapter_with_lora.state_dict()

        # LoRA 관련 키 찾기
        lora_keys = [k for k in state_dict.keys() if "lora_" in k]

        # LoRA 키가 존재해야 함
        assert len(lora_keys) > 0, "State dict에 LoRA 가중치가 없습니다"

        # 각 target module마다 lora_A, lora_B가 있어야 함
        lora_a_keys = [k for k in lora_keys if "lora_A" in k]
        lora_b_keys = [k for k in lora_keys if "lora_B" in k]

        assert len(lora_a_keys) > 0, "lora_A 가중치가 없습니다"
        assert len(lora_b_keys) > 0, "lora_B 가중치가 없습니다"
        assert len(lora_a_keys) == len(lora_b_keys), "lora_A/lora_B 개수 불일치"

    def test_lora_state_dict_shapes(self, adapter_with_lora, lora_config):
        """LoRA state_dict 가중치 shape 검증"""
        state_dict = adapter_with_lora.state_dict()
        rank = lora_config["rank"]

        for key, tensor in state_dict.items():
            if "lora_A" in key:
                # lora_A: [rank, in_features]
                assert tensor.shape[0] == rank, f"{key}: rank 불일치"
            elif "lora_B" in key:
                # lora_B: [out_features, rank]
                assert tensor.shape[1] == rank, f"{key}: rank 불일치"

    def test_lora_checkpoint_save_load(self, adapter_with_lora, device, lora_config):
        """LoRA checkpoint 저장/로드 테스트"""
        # 원본 state_dict 저장
        original_state_dict = {k: v.clone() for k, v in adapter_with_lora.state_dict().items()}

        # LoRA 가중치 변경 (학습 시뮬레이션)
        with torch.no_grad():
            for name, param in adapter_with_lora.named_parameters():
                if "lora_" in name:
                    param.add_(torch.randn_like(param) * 0.1)

        # 변경된 state_dict
        modified_state_dict = adapter_with_lora.state_dict()

        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = Path(f.name)

        try:
            torch.save(modified_state_dict, checkpoint_path)

            # 새 adapter 생성 및 로드
            new_adapter = MetaLlamaMTPAdapter.from_pretrained(
                model_path=MICRO_MODEL_PATH,
                device=device,
                dtype="float32",
                initialize_value_head=True,
                value_head_type="mlp",
                use_lora=True,
                lora_config=lora_config,
            )

            # state_dict 로드
            loaded_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            new_adapter.load_state_dict(loaded_state_dict)

            # 로드된 가중치 검증
            for key in modified_state_dict.keys():
                if "lora_" in key:
                    original = modified_state_dict[key]
                    loaded = new_adapter.state_dict()[key]
                    assert torch.allclose(original, loaded), f"{key}: 가중치 불일치"

        finally:
            checkpoint_path.unlink(missing_ok=True)

    def test_lora_forward_after_load(self, adapter_with_lora, device, lora_config):
        """Checkpoint 로드 후 forward 동작 검증"""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

        # 원본 forward 결과
        adapter_with_lora.eval()
        with torch.no_grad():
            original_output = adapter_with_lora(input_ids)

        # state_dict 저장
        state_dict = adapter_with_lora.state_dict()

        # 새 adapter에 로드
        new_adapter = MetaLlamaMTPAdapter.from_pretrained(
            model_path=MICRO_MODEL_PATH,
            device=device,
            dtype="float32",
            initialize_value_head=True,
            value_head_type="mlp",
            use_lora=True,
            lora_config=lora_config,
        )
        new_adapter.load_state_dict(state_dict)
        new_adapter.eval()

        # 로드 후 forward 결과
        with torch.no_grad():
            loaded_output = new_adapter(input_ids)

        # 결과 비교
        assert torch.allclose(original_output, loaded_output, atol=1e-5), \
            "로드 후 forward 결과 불일치"

    def test_lora_gradient_flow_with_wrapped_model(self, adapter_with_lora, device):
        """FSDP wrapped 모델에서 gradient flow 테스트"""
        # 단일 장치에서는 wrapping이 skip되지만, gradient flow는 테스트 가능
        wrapped = wrap_model_fsdp(
            adapter_with_lora,
            device,
            sharding_strategy="FULL_SHARD",
            mixed_precision=False,
        )

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

        # Forward (value_logits 없이 호출하여 tensor 직접 반환)
        output = wrapped(input_ids)

        # Loss 계산
        loss = output.mean()

        # Backward
        loss.backward()

        # LoRA 파라미터에 gradient가 있어야 함
        unwrapped = unwrap_model(wrapped)
        lora_params = get_lora_parameters(unwrapped.transformer)

        grad_count = sum(1 for p in lora_params if p.grad is not None)
        assert grad_count > 0, "LoRA 파라미터에 gradient가 없습니다"

    def test_lora_only_state_dict(self, adapter_with_lora):
        """LoRA 전용 state_dict 추출 테스트"""
        full_state_dict = adapter_with_lora.state_dict()

        # LoRA 가중치만 추출
        lora_state_dict = {
            k: v for k, v in full_state_dict.items()
            if "lora_A" in k or "lora_B" in k
        }

        # 크기 비교
        full_size = sum(v.numel() for v in full_state_dict.values())
        lora_size = sum(v.numel() for v in lora_state_dict.values())

        # LoRA 가중치는 전체의 작은 비율이어야 함
        ratio = lora_size / full_size
        assert ratio < 0.1, f"LoRA 가중치 비율이 너무 높습니다: {ratio:.2%}"

        # LoRA 키가 존재해야 함
        assert len(lora_state_dict) > 0, "LoRA state_dict가 비어있습니다"


class TestLoRACheckpointUtils:
    """LoRA checkpoint 유틸리티 테스트"""

    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @pytest.fixture
    def lora_config(self):
        return {
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "target_modules": ["wq", "wk", "wv", "wo"],
        }

    def test_save_checkpoint_with_lora(self, device, lora_config):
        """save_checkpoint()에서 LoRA 가중치 저장 테스트"""
        from weighted_mtp.utils import save_checkpoint

        adapter = MetaLlamaMTPAdapter.from_pretrained(
            model_path=MICRO_MODEL_PATH,
            device=device,
            dtype="float32",
            initialize_value_head=True,
            value_head_type="mlp",
            use_lora=True,
            lora_config=lora_config,
        )

        # Dummy optimizer
        optimizer = torch.optim.AdamW(adapter.get_trainable_parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Checkpoint 저장
            save_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=1.0,
                train_metrics={"train_loss": 0.5},
                val_metrics={"val_loss": 0.4},
                checkpoint_path=checkpoint_path,
            )

            # Checkpoint 로드 및 검증
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # LoRA 키 확인
            adapter_state = checkpoint["adapter_state_dict"]
            lora_keys = [k for k in adapter_state.keys() if "lora_" in k]

            assert len(lora_keys) > 0, "Checkpoint에 LoRA 가중치가 없습니다"

    def test_load_checkpoint_with_lora(self, device, lora_config):
        """Checkpoint 로드 후 LoRA 동작 검증"""
        adapter = MetaLlamaMTPAdapter.from_pretrained(
            model_path=MICRO_MODEL_PATH,
            device=device,
            dtype="float32",
            initialize_value_head=True,
            value_head_type="mlp",
            use_lora=True,
            lora_config=lora_config,
        )

        # LoRA 가중치 수정
        with torch.no_grad():
            for name, param in adapter.named_parameters():
                if "lora_B" in name:
                    # lora_B를 non-zero로 설정 (학습 효과 시뮬레이션)
                    param.fill_(0.1)

        # Forward 결과 저장
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

        adapter.eval()
        with torch.no_grad():
            expected_output = adapter(input_ids)

        # state_dict 저장
        state_dict = adapter.state_dict()

        # 새 adapter 생성 (초기화된 LoRA)
        new_adapter = MetaLlamaMTPAdapter.from_pretrained(
            model_path=MICRO_MODEL_PATH,
            device=device,
            dtype="float32",
            initialize_value_head=True,
            value_head_type="mlp",
            use_lora=True,
            lora_config=lora_config,
        )

        # state_dict 로드
        new_adapter.load_state_dict(state_dict)
        new_adapter.eval()

        # Forward 결과 비교
        with torch.no_grad():
            actual_output = new_adapter(input_ids)

        assert torch.allclose(expected_output, actual_output, atol=1e-5), \
            "로드된 모델의 출력이 다릅니다"

    def test_save_lora_checkpoint_size(self, device, lora_config):
        """save_lora_checkpoint() 저장 크기 테스트"""
        from weighted_mtp.utils import save_checkpoint, save_lora_checkpoint

        adapter = MetaLlamaMTPAdapter.from_pretrained(
            model_path=MICRO_MODEL_PATH,
            device=device,
            dtype="float32",
            initialize_value_head=True,
            value_head_type="mlp",
            use_lora=True,
            lora_config=lora_config,
        )

        optimizer = torch.optim.AdamW(adapter.get_trainable_parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            full_path = Path(tmpdir) / "full_checkpoint.pt"
            lora_path = Path(tmpdir) / "lora_checkpoint.pt"

            # 전체 checkpoint 저장
            save_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=1.0,
                train_metrics={"train_loss": 0.5},
                val_metrics={"val_loss": 0.4},
                checkpoint_path=full_path,
            )

            # LoRA 전용 checkpoint 저장
            save_lora_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=1.0,
                train_metrics={"train_loss": 0.5},
                val_metrics={"val_loss": 0.4},
                checkpoint_path=lora_path,
                save_value_head=True,
            )

            # 크기 비교
            full_size = full_path.stat().st_size
            lora_size = lora_path.stat().st_size

            # LoRA checkpoint가 전체보다 작아야 함
            assert lora_size < full_size, \
                f"LoRA checkpoint({lora_size})가 전체({full_size})보다 작아야 합니다"

            # LoRA checkpoint가 전체의 50% 미만이어야 함
            ratio = lora_size / full_size
            assert ratio < 0.5, f"LoRA checkpoint 비율이 너무 높습니다: {ratio:.2%}"

    def test_load_lora_checkpoint(self, device, lora_config):
        """load_lora_checkpoint() 기능 테스트"""
        from weighted_mtp.utils import save_lora_checkpoint, load_lora_checkpoint

        adapter = MetaLlamaMTPAdapter.from_pretrained(
            model_path=MICRO_MODEL_PATH,
            device=device,
            dtype="float32",
            initialize_value_head=True,
            value_head_type="mlp",
            use_lora=True,
            lora_config=lora_config,
        )

        # LoRA 가중치 수정
        with torch.no_grad():
            for name, param in adapter.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.1)

        # Forward 결과 저장
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

        adapter.eval()
        with torch.no_grad():
            expected_output = adapter(input_ids)

        optimizer = torch.optim.AdamW(adapter.get_trainable_parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = Path(tmpdir) / "lora_checkpoint.pt"

            # LoRA checkpoint 저장
            save_lora_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=1.5,
                train_metrics={"train_loss": 0.3},
                val_metrics={"val_loss": 0.25},
                checkpoint_path=lora_path,
                save_value_head=True,
            )

            # 새 adapter 생성 (초기화된 LoRA)
            new_adapter = MetaLlamaMTPAdapter.from_pretrained(
                model_path=MICRO_MODEL_PATH,
                device=device,
                dtype="float32",
                initialize_value_head=True,
                value_head_type="mlp",
                use_lora=True,
                lora_config=lora_config,
            )

            # LoRA checkpoint 로드
            metadata = load_lora_checkpoint(
                adapter=new_adapter,
                checkpoint_path=lora_path,
                device=device,
            )

            # Metadata 검증
            assert metadata["epoch"] == 1.5
            assert metadata["val_metrics"]["val_loss"] == 0.25

            # Forward 결과 비교
            new_adapter.eval()
            with torch.no_grad():
                actual_output = new_adapter(input_ids)

            assert torch.allclose(expected_output, actual_output, atol=1e-5), \
                "LoRA checkpoint 로드 후 출력이 다릅니다"

