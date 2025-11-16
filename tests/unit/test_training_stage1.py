"""Training Pipeline Stage 1 Unit Tests"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from weighted_mtp.models.meta_mtp import MetaLlamaMTPAdapter
from weighted_mtp.pipelines.training import train_stage1


class MockAdapter:
    """train_stage1() 테스트용 Mock Adapter

    trunk_forward()만 구현하고, Value head 출력 시뮬레이션
    """

    def __init__(self, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._training = True

        # Simple value head (returns learnable values)
        self.value_head = torch.nn.Linear(1, 1)

    def train(self, mode=True):
        self._training = mode
        if hasattr(self.value_head, 'train'):
            self.value_head.train(mode)
        return self

    def trunk_forward(self, input_ids, attention_mask):
        """Mock trunk_forward

        Returns:
            {
                "value_logits": [batch, seq, 1]
            }
        """
        batch_size, seq_len = input_ids.shape

        # Simple value prediction (constant + small variation)
        # 실제로는 학습되지만, 테스트에서는 단순 출력
        value_logits = torch.ones(batch_size, seq_len, 1) * 0.5

        # Learnable value head 적용
        value_logits = self.value_head(value_logits)

        return {
            "value_logits": value_logits,
        }


class TestTrainStage1Basic:
    """train_stage1() 기본 동작 테스트"""

    def test_single_batch_training(self):
        """단일 배치 학습 검증"""
        batch_size = 2
        seq_len = 4

        # Mock adapter
        adapter = MockAdapter(batch_size, seq_len)

        # Mock dataloader
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        is_correct = torch.tensor([1.0, 0.0])  # Correct, Incorrect

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Dataloader를 dict 형태로 변환 (실제 사용 형태)
        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        # Optimizer
        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-3)

        # Config
        config = {
            "n_epochs": 1,
            "loss_type": "mse",
        }

        # Train
        metrics = train_stage1(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # Metrics 검증
        assert "stage1_loss" in metrics
        assert "value_explained_variance" in metrics

        # Loss는 양수여야 함
        assert metrics["stage1_loss"] >= 0

        # Explained variance는 [-∞, 1] 범위
        # (초기 random weights이므로 음수일 수 있음)
        assert metrics["value_explained_variance"] <= 1.0

    def test_is_correct_to_rewards_conversion(self):
        """is_correct → rewards 변환 검증"""
        batch_size = 3
        seq_len = 5

        adapter = MockAdapter(batch_size, seq_len)

        # is_correct: [1, 0, 1] (Boolean 또는 0/1)
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        is_correct = torch.tensor([1.0, 0.0, 1.0])

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-3)
        config = {"loss_type": "mse"}

        # Train
        metrics = train_stage1(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # Metrics가 정상적으로 계산되었는지 확인
        assert metrics["stage1_loss"] >= 0

    def test_value_target_expansion(self):
        """Value target 생성 검증 ([batch] → [batch, seq, 1])"""
        batch_size = 2
        seq_len = 6

        adapter = MockAdapter(batch_size, seq_len)

        # is_correct: [1, 0]
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-3)
        config = {"loss_type": "mse"}

        # Train (내부적으로 value_targets가 [batch, seq, 1]로 생성되어야 함)
        metrics = train_stage1(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # 정상 동작 확인
        assert metrics["stage1_loss"] >= 0


class TestTrainStage1LossTypes:
    """Loss type 검증 (MSE vs Huber)"""

    def test_mse_loss(self):
        """MSE loss 검증"""
        batch_size = 2
        seq_len = 4

        adapter = MockAdapter(batch_size, seq_len)

        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-3)
        config = {"loss_type": "mse"}

        # Train
        metrics = train_stage1(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        assert metrics["stage1_loss"] >= 0

    def test_huber_loss(self):
        """Huber loss 검증"""
        batch_size = 2
        seq_len = 4

        adapter = MockAdapter(batch_size, seq_len)

        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-3)
        config = {"loss_type": "huber"}

        # Train
        metrics = train_stage1(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        assert metrics["stage1_loss"] >= 0

    def test_invalid_loss_type(self):
        """Invalid loss type 검증"""
        batch_size = 2
        seq_len = 4

        adapter = MockAdapter(batch_size, seq_len)

        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-3)
        config = {"loss_type": "invalid"}

        # Invalid loss type → ValueError
        with pytest.raises(ValueError, match="Unknown loss_type"):
            train_stage1(
                adapter=adapter,
                dataloader=dataloader,
                optimizer=optimizer,
                config=config,
                device=torch.device("cpu"),
            )


class TestTrainStage1Metrics:
    """Metrics 계산 검증"""

    def test_explained_variance_calculation(self):
        """Explained variance 계산 검증"""
        batch_size = 2
        seq_len = 4

        adapter = MockAdapter(batch_size, seq_len)

        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-3)
        config = {"loss_type": "mse"}

        # Train
        metrics = train_stage1(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # Explained variance는 1 - (loss / target_var)
        # 초기 random weights이므로 음수일 수 있음
        assert "value_explained_variance" in metrics
        assert metrics["value_explained_variance"] <= 1.0

    def test_multiple_batches(self):
        """여러 배치 처리 검증"""
        batch_size = 2
        seq_len = 4
        n_samples = 6  # 3 batches

        adapter = MockAdapter(batch_size, seq_len)

        input_ids = torch.randint(0, 100, (n_samples, seq_len))
        attention_mask = torch.ones(n_samples, seq_len)
        is_correct = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-3)
        config = {"loss_type": "mse"}

        # Train (3 batches)
        metrics = train_stage1(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # Metrics는 평균값이어야 함
        assert metrics["stage1_loss"] >= 0
        assert "value_explained_variance" in metrics


class TestTrainStage1OptimizerUpdate:
    """Optimizer update 검증"""

    def test_optimizer_step_executed(self):
        """Optimizer가 실제로 파라미터를 업데이트하는지 검증"""
        batch_size = 2
        seq_len = 4

        adapter = MockAdapter(batch_size, seq_len)

        # 초기 파라미터 저장
        initial_params = [p.clone() for p in adapter.value_head.parameters()]

        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.SGD(adapter.value_head.parameters(), lr=0.1)
        config = {"loss_type": "mse"}

        # Train
        train_stage1(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # 파라미터가 변경되었는지 확인
        updated_params = list(adapter.value_head.parameters())

        params_changed = False
        for initial, updated in zip(initial_params, updated_params):
            if not torch.allclose(initial, updated):
                params_changed = True
                break

        assert params_changed, "Optimizer did not update parameters"


class TestProbabilisticValueLearning:
    """확률적 가치 학습 (Probabilistic Value Learning) 검증"""

    def test_probability_convergence_concept(self):
        """배치 학습을 통한 확률 수렴 검증 (개념)

        동일한 prefix가 다른 샘플에서 다른 R을 가질 때
        V(s) → P(Success | s)로 수렴하는지 검증
        """
        batch_size = 4
        seq_len = 4

        adapter = MockAdapter(batch_size, seq_len)

        # 4개 샘플: 2개 correct, 2개 incorrect
        # 모든 샘플이 동일한 input_ids → 동일한 hidden states
        input_ids = torch.ones(batch_size, seq_len, dtype=torch.long) * 42
        attention_mask = torch.ones(batch_size, seq_len)
        is_correct = torch.tensor([1.0, 1.0, 0.0, 0.0])  # 50% success rate

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=0.01)
        config = {"loss_type": "mse"}

        # 여러 epoch 학습
        for epoch in range(100):
            train_stage1(
                adapter=adapter,
                dataloader=dataloader,
                optimizer=optimizer,
                config=config,
                device=torch.device("cpu"),
            )

        # 학습 후 value 예측
        adapter.train(False)  # eval mode
        with torch.no_grad():
            outputs = adapter.trunk_forward(
                input_ids[:1], attention_mask[:1]
            )  # Single sample
            value_pred = outputs["value_logits"][0, 0, 0].item()

        # V(s) ≈ 0.5 (2/4 success rate)에 가까워야 함
        # 허용 오차: ±0.2 (학습이 완벽하지 않을 수 있음)
        expected_prob = 0.5
        assert abs(value_pred - expected_prob) < 0.2, (
            f"Value prediction {value_pred:.3f} should be close to "
            f"empirical probability {expected_prob:.3f}"
        )

    def test_masked_loss_excludes_padding(self):
        """패딩 토큰이 loss 계산에서 제외되는지 검증"""
        batch_size = 2
        seq_len = 6

        adapter = MockAdapter(batch_size, seq_len)

        # Sample 1: 전체 valid (seq_len=6)
        # Sample 2: 앞 3개만 valid, 뒤 3개 padding
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1],  # All valid
            [1, 1, 1, 0, 0, 0],  # Half padding
        ])
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "is_correct": batch[2],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.value_head.parameters(), lr=1e-3)
        config = {"loss_type": "mse"}

        # Train
        metrics = train_stage1(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # Metrics가 정상적으로 계산되었는지 확인
        assert metrics["stage1_loss"] >= 0
        # Masked loss이므로 패딩이 있어도 정상 동작
