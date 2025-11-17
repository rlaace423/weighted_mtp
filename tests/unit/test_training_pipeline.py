"""run_training_pipeline() Unit Tests"""


import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from weighted_mtp.pipelines.training import run_training_pipeline


class MockAdapter:
    """run_training_pipeline() 테스트용 Mock Adapter

    trunk_forward() + full_forward() 구현
    """

    def __init__(self, batch_size: int, seq_len: int, n_future: int, vocab_size: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_future = n_future
        self.vocab_size = vocab_size
        self._training = True

        # MTP output heads
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

    def state_dict(self):
        """State dict 반환 (checkpoint 저장용)"""
        state = {}
        for i, head in enumerate(self.mtp_heads):
            state[f"mtp_head_{i}"] = head.state_dict()
        state["value_head"] = self.value_head.state_dict()
        return state

    def trunk_forward(self, input_ids, attention_mask):
        """Mock trunk_forward (Value head만)

        Returns:
            {
                "hidden_states": [batch, seq, hidden_size],
                "value_logits": [batch, seq, 1],
            }
        """
        batch_size, seq_len = input_ids.shape

        hidden_states = torch.zeros(batch_size, seq_len, 128)
        value_input = torch.ones(batch_size, seq_len, 1) * 0.5
        value_logits = self.value_head(value_input)

        return {
            "hidden_states": hidden_states,
            "value_logits": value_logits,
        }

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

        # MTP logits 생성
        logits = torch.zeros(batch_size, seq_len, self.n_future, self.vocab_size)
        for k in range(self.n_future):
            head_input = torch.ones(batch_size, seq_len, 1) * 0.5
            logits[:, :, k, :] = self.mtp_heads[k](head_input)

        # Value head 예측
        value_input = torch.ones(batch_size, seq_len, 1) * 0.5
        value_logits = self.value_head(value_input)

        return {
            "logits": logits,
            "value_logits": value_logits,
            "hidden_states": torch.zeros(batch_size, seq_len, 128),
        }


class TestRunTrainingPipelineBasic:
    """run_training_pipeline() 기본 동작 테스트"""

    def test_stage1_to_stage2_flow(self, tmp_path):
        """Stage 1 → Stage 2 순차 실행 검증"""
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        # Mock adapter
        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        # Stage 1 DataLoader
        stage1_input_ids = torch.randint(0, vocab_size, (batch_size * 3, seq_len))
        stage1_attention_mask = torch.ones(batch_size * 3, seq_len)
        stage1_labels = torch.randint(0, vocab_size, (batch_size * 3, seq_len))
        stage1_is_correct = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        stage1_dataset = TensorDataset(
            stage1_input_ids, stage1_attention_mask, stage1_labels, stage1_is_correct
        )

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader
                self.batch_size = base_loader.batch_size
                self.dataset = base_loader.dataset
                self.collate_fn = None

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

        stage1_dataloader = DictDataLoader(
            DataLoader(stage1_dataset, batch_size=batch_size)
        )

        # Stage 2 DataLoader (동일한 구조)
        stage2_input_ids = torch.randint(0, vocab_size, (batch_size * 5, seq_len))
        stage2_attention_mask = torch.ones(batch_size * 5, seq_len)
        stage2_labels = torch.randint(0, vocab_size, (batch_size * 5, seq_len))
        stage2_is_correct = torch.tensor(
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        )

        stage2_dataset = TensorDataset(
            stage2_input_ids, stage2_attention_mask, stage2_labels, stage2_is_correct
        )
        stage2_dataloader = DictDataLoader(
            DataLoader(stage2_dataset, batch_size=batch_size)
        )

        # Config
        config = {
            "stage1": {"n_epochs": 0.5, "loss_type": "mse", "learning_rate": 1e-4},
            "stage2": {
                "n_epochs": 1.0,
                "beta": 0.9,
                "value_coef": 0.5,
                "learning_rate": 1e-5,
            },
        }

        # Run pipeline
        metrics = run_training_pipeline(
            adapter=adapter,
            stage1_dataloader=stage1_dataloader,
            stage2_dataloader=stage2_dataloader,
            config=config,
            device=torch.device("cpu"),
            save_dir=None,
        )

        # Metrics 검증
        assert "stage1" in metrics
        assert "stage2" in metrics

        # Stage 1 metrics
        assert "stage1_loss" in metrics["stage1"]
        assert "value_explained_variance" in metrics["stage1"]

        # Stage 2 metrics
        assert "stage2_weighted_ce_loss" in metrics["stage2"]
        assert "stage2_value_loss" in metrics["stage2"]
        assert "stage2_total_loss" in metrics["stage2"]

    def test_fractional_epochs(self):
        """Fractional epochs 처리 검증 (0.5, 2.5 epochs)"""
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        # DataLoader (총 4 배치)
        input_ids = torch.randint(0, vocab_size, (batch_size * 4, seq_len))
        attention_mask = torch.ones(batch_size * 4, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size * 4, seq_len))
        is_correct = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader
                self.batch_size = base_loader.batch_size
                self.dataset = base_loader.dataset
                self.collate_fn = None

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

        dataloader = DictDataLoader(DataLoader(dataset, batch_size=batch_size))

        # Config (0.5 epochs for stage1, 2.5 epochs for stage2)
        config = {
            "stage1": {"n_epochs": 0.5, "loss_type": "mse"},
            "stage2": {"n_epochs": 2.5, "beta": 0.9, "value_coef": 0.5},
        }

        # Run pipeline
        metrics = run_training_pipeline(
            adapter=adapter,
            stage1_dataloader=dataloader,
            stage2_dataloader=dataloader,
            config=config,
            device=torch.device("cpu"),
            save_dir=None,
        )

        # 정상 동작 확인
        assert metrics["stage1"]["stage1_loss"] >= 0
        assert metrics["stage2"]["stage2_total_loss"] >= 0

    def test_checkpoint_saving(self, tmp_path):
        """Checkpoint 저장 검증"""
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        # DataLoader
        input_ids = torch.randint(0, vocab_size, (batch_size * 4, seq_len))
        attention_mask = torch.ones(batch_size * 4, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size * 4, seq_len))
        is_correct = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader
                self.batch_size = base_loader.batch_size
                self.dataset = base_loader.dataset
                self.collate_fn = None

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

        dataloader = DictDataLoader(DataLoader(dataset, batch_size=batch_size))

        # Config
        config = {
            "stage1": {"n_epochs": 0.5, "loss_type": "mse"},
            "stage2": {"n_epochs": 1.0, "beta": 0.9, "value_coef": 0.5},
            "save_checkpoint_every": 1,
        }

        save_dir = tmp_path / "checkpoints"

        # Run pipeline
        _ = run_training_pipeline(
            adapter=adapter,
            stage1_dataloader=dataloader,
            stage2_dataloader=dataloader,
            config=config,
            device=torch.device("cpu"),
            save_dir=save_dir,
        )

        # Checkpoint 파일 존재 확인
        assert save_dir.exists()

        # Stage 1 checkpoint
        stage1_checkpoint = save_dir / "checkpoint_stage1_epoch_0.5.pt"
        assert stage1_checkpoint.exists()

        # Stage 2 final checkpoint
        stage2_checkpoint = save_dir / "checkpoint_stage2_epoch_1.0.pt"
        assert stage2_checkpoint.exists()

        # Checkpoint 로딩 가능 확인
        checkpoint = torch.load(stage1_checkpoint, weights_only=False)
        assert "stage" in checkpoint
        assert "epoch" in checkpoint
        assert "adapter_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "metrics" in checkpoint


class TestRunTrainingPipelineConfig:
    """Config 처리 검증"""

    def test_default_config_values(self):
        """기본값 처리 검증"""
        batch_size = 2
        seq_len = 8
        n_future = 4
        vocab_size = 100

        adapter = MockAdapter(batch_size, seq_len, n_future, vocab_size)

        # DataLoader
        input_ids = torch.randint(0, vocab_size, (batch_size * 2, seq_len))
        attention_mask = torch.ones(batch_size * 2, seq_len)
        labels = torch.randint(0, vocab_size, (batch_size * 2, seq_len))
        is_correct = torch.tensor([1.0, 0.0, 1.0, 0.0])

        dataset = TensorDataset(input_ids, attention_mask, labels, is_correct)

        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader
                self.batch_size = base_loader.batch_size
                self.dataset = base_loader.dataset
                self.collate_fn = None

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

        dataloader = DictDataLoader(DataLoader(dataset, batch_size=batch_size))

        # Minimal config (기본값 사용)
        config = {
            "stage1": {},
            "stage2": {},
        }

        # Run pipeline
        result_metrics = run_training_pipeline(
            adapter=adapter,
            stage1_dataloader=dataloader,
            stage2_dataloader=dataloader,
            config=config,
            device=torch.device("cpu"),
            save_dir=None,
        )

        # 정상 동작 확인 (기본값으로 실행됨)
        assert "stage1" in result_metrics
        assert "stage2" in result_metrics


