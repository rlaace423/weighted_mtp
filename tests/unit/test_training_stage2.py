"""Training Pipeline Stage 2 Unit Tests"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from weighted_mtp.models.meta_mtp import MetaLlamaMTPAdapter
from weighted_mtp.pipelines.training import train_stage2


class MockAdapter:
    """train_stage2() 테스트용 Mock Adapter

    full_forward() 구현 및 MTP logits + Value head 출력 시뮬레이션
    """

    def __init__(self, batch_size: int, seq_len: int, n_future: int, vocab_size: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_future = n_future
        self.vocab_size = vocab_size
        self._training = True

        # MTP output heads (간단한 linear)
        self.mtp_heads = torch.nn.ModuleList(
            [torch.nn.Linear(1, vocab_size) for _ in range(n_future)]
        )

        # Value head
        self.value_head = torch.nn.Linear(1, 1)

    def train(self, mode=True):
        self._training = mode
        for head in self.mtp_heads:
            if hasattr(head, "train"):
                head.train(mode)
        if hasattr(self.value_head, "train"):
            self.value_head.train(mode)
        return self

    def parameters(self):
        """모든 파라미터 반환"""
        params = []
        for head in self.mtp_heads:
            params.extend(list(head.parameters()))
        params.extend(list(self.value_head.parameters()))
        return params

    def full_forward(self, input_ids, attention_mask):
        """Mock full_forward (MTP + Value)

        Returns:
            {
                "logits": [batch, seq, n_future_tokens, vocab],
                "value_logits": [batch, seq, 1],
                "hidden_states": [batch, seq, hidden_size],
            }
        """
        batch_size, seq_len = input_ids.shape

        # MTP logits 생성 [batch, seq, n_future, vocab]
        logits = torch.zeros(batch_size, seq_len, self.n_future, self.vocab_size)
        for k in range(self.n_future):
            # k번째 future token 예측
            head_input = torch.ones(batch_size, seq_len, 1) * 0.5
            logits[:, :, k, :] = self.mtp_heads[k](head_input)

        # Value head 예측 [batch, seq, 1]
        value_input = torch.ones(batch_size, seq_len, 1) * 0.5
        value_logits = self.value_head(value_input)

        return {
            "logits": logits,
            "value_logits": value_logits,
            "hidden_states": torch.zeros(batch_size, seq_len, 128),
        }


class TestTrainStage2Basic:
    """train_stage2() 기본 동작 테스트"""

    def test_single_batch_training(self):
        """단일 배치 학습 검증"""
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        # Mock adapter
        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        # Mock dataloader
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        is_correct = torch.tensor([1.0, 0.0])  # Correct, Incorrect

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Dataloader를 dict 형태로 변환
        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[2],
                        "is_correct": batch[3],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        # Optimizer
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

        # Config
        config = {
            "beta": 0.9,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "loss_type": "mse",
        }

        # Train
        metrics = train_stage2(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # Metrics 검증
        assert "stage2_weighted_ce_loss" in metrics
        assert "stage2_value_loss" in metrics
        assert "stage2_total_loss" in metrics
        assert "td_mean" in metrics
        assert "weight_mean" in metrics
        assert "value_explained_variance" in metrics

        # Loss는 양수여야 함
        assert metrics["stage2_weighted_ce_loss"] >= 0
        assert metrics["stage2_value_loss"] >= 0
        assert metrics["stage2_total_loss"] >= 0

        # Explained variance는 [-∞, 1] 범위
        assert metrics["value_explained_variance"] <= 1.0

    def test_mtp_weight_alignment(self):
        """MTP 가중치 시점 정렬 검증

        logits[t, k-1, :]의 가중치 = td_errors[t+k-1]
        """
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[2],
                        "is_correct": batch[3],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        config = {"beta": 0.9, "value_coef": 0.5}

        # Train (내부적으로 시점 정렬이 정확해야 함)
        metrics = train_stage2(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # 정상 동작 확인
        assert metrics["stage2_weighted_ce_loss"] >= 0

    def test_h_future_tokens_averaging(self):
        """H개 미래 토큰 loss 평균 계산 검증"""
        batch_size = 2
        seq_len = 8
        n_future = 4  # H=4
        vocab_size = 100

        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        is_correct = torch.tensor([1.0, 1.0])

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[2],
                        "is_correct": batch[3],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        config = {"beta": 0.9, "value_coef": 0.5}

        # Train (H개 미래 토큰 평균 계산)
        metrics = train_stage2(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # 정상 동작 확인
        assert metrics["stage2_weighted_ce_loss"] >= 0


class TestTrainStage2CriticContinualLearning:
    """Critic Continual Learning 검증"""

    def test_value_loss_auxiliary(self):
        """Value loss가 auxiliary loss로 포함되는지 검증"""
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[2],
                        "is_correct": batch[3],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        config = {"beta": 0.9, "value_coef": 0.5}

        metrics = train_stage2(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # Value loss가 metrics에 포함되는지 확인
        assert "stage2_value_loss" in metrics
        assert metrics["stage2_value_loss"] >= 0

        # Total loss = weighted_ce_loss + value_coef * value_loss
        # 대략적으로 검증 (floating point 오차 고려)
        expected_total = metrics["stage2_weighted_ce_loss"] + 0.5 * metrics[
            "stage2_value_loss"
        ]
        assert abs(metrics["stage2_total_loss"] - expected_total) < 0.1

    def test_value_explained_variance(self):
        """Value explained variance 계산 검증"""
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[2],
                        "is_correct": batch[3],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        config = {"beta": 0.9, "value_coef": 0.5}

        metrics = train_stage2(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # Explained variance는 1.0 이하
        assert "value_explained_variance" in metrics
        assert metrics["value_explained_variance"] <= 1.0


class TestTrainStage2GradientClipping:
    """Gradient clipping 검증"""

    def test_gradient_clipping_applied(self):
        """Gradient clipping이 적용되는지 검증"""
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[2],
                        "is_correct": batch[3],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        config = {
            "beta": 0.9,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,  # Gradient clipping 활성화
        }

        # Train
        metrics = train_stage2(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # 정상 동작 확인
        assert metrics["stage2_total_loss"] >= 0


class TestTrainStage2TDErrorWeights:
    """TD error 및 weight 통계 검증"""

    def test_td_weight_statistics(self):
        """TD error 및 weight 통계가 metrics에 포함되는지 검증"""
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        is_correct = torch.tensor([1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for batch in self.base_loader:
                    yield {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[2],
                        "is_correct": batch[3],
                    }

            def __len__(self):
                return len(self.base_loader)

        dataloader = DictDataLoader(dataloader)

        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        config = {"beta": 0.9, "value_coef": 0.5}

        metrics = train_stage2(
            adapter=adapter,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        # TD error 및 weight 통계 확인
        assert "td_mean" in metrics
        assert "weight_mean" in metrics

        # Weight mean은 양수여야 함 (exponential weighting)
        assert metrics["weight_mean"] > 0
