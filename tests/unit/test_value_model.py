"""ValueModel Unit Tests

독립 Value Model (HuggingFace 기반) 테스트
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from weighted_mtp.models.value_model import ValueModel
from weighted_mtp.models.value_head import (
    LinearValueHead,
    MLPValueHead,
    create_value_head,
)


class TestValueHead:
    """Value Head 테스트"""
    
    def test_linear_value_head_forward(self):
        """LinearValueHead forward 테스트"""
        value_head = LinearValueHead(hidden_size=512)
        hidden_states = torch.randn(2, 10, 512)
        
        output = value_head(hidden_states)
        
        assert output.shape == (2, 10, 1)
    
    def test_mlp_value_head_forward(self):
        """MLPValueHead forward 테스트"""
        value_head = MLPValueHead(hidden_size=512, dropout=0.1)
        hidden_states = torch.randn(2, 10, 512)
        
        output = value_head(hidden_states)
        
        assert output.shape == (2, 10, 1)
    
    def test_create_value_head_linear(self):
        """create_value_head factory - linear"""
        head = create_value_head(hidden_size=256, head_type="linear")
        
        assert isinstance(head, LinearValueHead)
        assert head.hidden_size == 256
    
    def test_create_value_head_mlp(self):
        """create_value_head factory - mlp"""
        head = create_value_head(hidden_size=512, head_type="mlp", dropout=0.2)
        
        assert isinstance(head, MLPValueHead)
        assert head.hidden_size == 512
    
    def test_create_value_head_invalid(self):
        """create_value_head - 잘못된 타입"""
        with pytest.raises(ValueError, match="Unknown value head type"):
            create_value_head(hidden_size=512, head_type="invalid")
    
    def test_mlp_value_head_structure(self):
        """MLPValueHead 구조 검증"""
        value_head = MLPValueHead(hidden_size=4096)
        
        # 4096 → 512 → 256 → 1
        assert value_head.mlp[0].in_features == 4096
        assert value_head.mlp[0].out_features == 512  # 4096 // 8
        assert value_head.mlp[3].in_features == 512
        assert value_head.mlp[3].out_features == 256  # 4096 // 16
        assert value_head.mlp[6].out_features == 1


class TestValueModel:
    """ValueModel 테스트"""
    
    @pytest.fixture
    def mock_backbone(self):
        """Mock HuggingFace LlamaModel"""
        backbone = MagicMock()
        backbone.return_value = MagicMock(
            last_hidden_state=torch.randn(2, 10, 512)
        )
        return backbone
    
    @pytest.fixture
    def mock_config(self):
        """Mock LlamaConfig"""
        config = MagicMock()
        config.hidden_size = 512
        config.num_hidden_layers = 4
        return config
    
    @pytest.fixture
    def value_model(self, mock_backbone, mock_config):
        """테스트용 ValueModel 인스턴스"""
        value_head = create_value_head(hidden_size=512, head_type="mlp")
        return ValueModel(mock_backbone, value_head, mock_config)
    
    def test_forward(self, value_model):
        """forward 테스트"""
        input_ids = torch.randint(0, 32000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        output = value_model(input_ids, attention_mask)
        
        assert output.shape == (2, 10, 1)
    
    def test_forward_without_attention_mask(self, value_model):
        """attention_mask 없이 forward 테스트"""
        input_ids = torch.randint(0, 32000, (2, 10))
        
        output = value_model(input_ids)
        
        assert output.shape == (2, 10, 1)
    
    def test_freeze_backbone(self, value_model):
        """freeze_backbone 테스트"""
        value_model.freeze_backbone()
        
        # Backbone 파라미터는 frozen
        for param in value_model.backbone.parameters():
            assert not param.requires_grad
        
        # Value head 파라미터는 여전히 학습 가능
        for param in value_model.value_head.parameters():
            assert param.requires_grad
    
    def test_unfreeze_backbone(self, value_model):
        """unfreeze_backbone 테스트"""
        value_model.freeze_backbone()
        value_model.unfreeze_backbone()
        
        for param in value_model.backbone.parameters():
            assert param.requires_grad
    
    def test_eval_mode(self, value_model):
        """eval_mode 테스트"""
        value_model.eval_mode()
        
        # 전체 파라미터 frozen
        for param in value_model.parameters():
            assert not param.requires_grad
        
        # eval 상태
        assert not value_model.training
    
    def test_get_trainable_parameters(self, value_model):
        """get_trainable_parameters 테스트"""
        trainable = value_model.get_trainable_parameters()
        
        # 모든 학습 가능 파라미터
        expected = [p for p in value_model.parameters() if p.requires_grad]
        assert len(trainable) == len(expected)
    
    def test_get_trainable_parameters_after_freeze(self, value_model):
        """freeze 후 get_trainable_parameters 테스트"""
        value_model.freeze_backbone()
        trainable = value_model.get_trainable_parameters()
        
        # Value head 파라미터만 학습 가능
        value_head_params = sum(1 for _ in value_model.value_head.parameters() if _.requires_grad)
        assert len(trainable) == value_head_params
    
    def test_hidden_size_property(self, value_model, mock_config):
        """hidden_size 프로퍼티 테스트"""
        assert value_model.hidden_size == 512
    
    def test_num_layers_property(self, value_model, mock_config):
        """num_layers 프로퍼티 테스트"""
        assert value_model.num_layers == 4


class TestValueModelCheckpoint:
    """ValueModel checkpoint 로드 테스트"""
    
    def test_from_checkpoint_missing_config(self, tmp_path):
        """config 누락 시 에러"""
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({"epoch": 1}, checkpoint_path)
        
        with pytest.raises(ValueError, match="모델 경로가 없습니다"):
            ValueModel.from_checkpoint(str(checkpoint_path))
    
    def test_from_checkpoint_with_state_dict(self, tmp_path):
        """state_dict가 있는 checkpoint 로드"""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        # Mock checkpoint
        checkpoint = {
            "epoch": 1,
            "config": {
                "models": {
                    "value_model": {
                        "path": "test/path",
                        "dtype": "bfloat16",
                    }
                },
                "training": {
                    "value_head_type": "mlp",
                    "dropout": 0.1,
                }
            },
            "backbone_state_dict": {},
            "value_head_state_dict": {},
        }
        torch.save(checkpoint, checkpoint_path)
        
        # from_pretrained를 mock
        with patch.object(ValueModel, 'from_pretrained') as mock_pretrained:
            mock_model = MagicMock()
            mock_model.backbone = MagicMock()
            mock_model.value_head = MagicMock()
            mock_pretrained.return_value = mock_model
            
            model = ValueModel.from_checkpoint(str(checkpoint_path))
            
            # from_pretrained 호출 검증
            mock_pretrained.assert_called_once_with(
                model_path="test/path",
                value_head_type="mlp",
                dropout=0.1,
                device="cuda",
                dtype="bfloat16",
            )
            
            # state_dict 로드 검증
            mock_model.backbone.load_state_dict.assert_called_once()
            mock_model.value_head.load_state_dict.assert_called_once()


class TestValueModelIntegration:
    """ValueModel 통합 테스트 (실제 모델 로드 필요)"""
    
    @pytest.fixture
    def ref_model_path(self):
        """Reference 모델 경로"""
        return Path("storage/models/ref-sheared-llama-2.7b/raw")
    
    @pytest.mark.skipif(
        not Path("storage/models/ref-sheared-llama-2.7b/raw").exists(),
        reason="Reference model not available"
    )
    def test_from_pretrained_real_model(self, ref_model_path):
        """실제 모델 로드 테스트"""
        model = ValueModel.from_pretrained(
            model_path=str(ref_model_path),
            value_head_type="mlp",
            dropout=0.1,
            device="cpu",
            dtype="float32",
        )
        
        assert model.hidden_size == 2560  # Sheared LLaMA 2.7B
        assert model.num_layers == 32
        
        # Forward 테스트
        input_ids = torch.randint(0, 32000, (1, 16))
        output = model(input_ids)
        assert output.shape == (1, 16, 1)

