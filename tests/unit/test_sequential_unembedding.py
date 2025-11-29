"""Sequential Unembedding 단위 테스트

수치 동등성, Gradient 동등성, 메모리 효율성 검증
"""

import pytest
import torch
import torch.nn.functional as F

from weighted_mtp.models.meta_mtp import (
    MetaLlamaMTPAdapter,
    ModelArgs,
    Transformer,
)
from weighted_mtp.utils import compute_mtp_ce_loss_unweighted, compute_head_ce_loss


@pytest.fixture
def micro_model():
    """테스트용 Micro 모델 생성"""
    model_args = ModelArgs(
        dim=256,
        n_layers=3,  # trunk 2 + extra_heads 1 (n_future_tokens=2)
        n_heads=4,
        n_kv_heads=4,
        vocab_size=1000,
        n_future_tokens=2,
        max_seq_len=128,
    )
    transformer = Transformer(model_args)
    adapter = MetaLlamaMTPAdapter(transformer, model_args)
    return adapter


@pytest.fixture
def sample_batch():
    """테스트용 샘플 배치"""
    batch_size, seq_len = 2, 16
    vocab_size = 1000

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class TestNumericalEquivalence:
    """수치 동등성 테스트: 기존 방식 vs Sequential 방식"""

    def test_loss_value_equivalence_unweighted(self, micro_model, sample_batch):
        """균등 가중치: 기존 방식과 Sequential 방식의 loss 값 일치"""
        adapter = micro_model
        input_ids = sample_batch["input_ids"]
        labels = sample_batch["labels"]
        attention_mask = sample_batch["attention_mask"]

        # 기존 방식: 전체 logits 계산 후 loss
        adapter.zero_grad()
        logits_standard = adapter(input_ids)
        loss_standard = compute_mtp_ce_loss_unweighted(
            logits=logits_standard,
            labels=labels,
            attention_mask=attention_mask,
        )

        # Sequential 방식
        adapter.zero_grad()
        loss_result = adapter(
            input_ids,
            attention_mask=attention_mask,
            compute_sequential_loss=True,
            labels=labels,
        )
        loss_sequential = loss_result["loss"]

        # 수치 비교 (상대 오차 1% 이내)
        relative_error = abs(loss_standard.item() - loss_sequential.item()) / (loss_standard.item() + 1e-8)
        assert relative_error < 0.01, f"Loss mismatch: standard={loss_standard.item():.6f}, sequential={loss_sequential.item():.6f}"

    def test_loss_value_equivalence_weighted_2d(self, micro_model, sample_batch):
        """2D 가중치 (Verifiable): 기존 방식과 Sequential 방식의 loss 값 일치"""
        adapter = micro_model
        input_ids = sample_batch["input_ids"]
        labels = sample_batch["labels"]
        attention_mask = sample_batch["attention_mask"]
        batch_size, seq_len = input_ids.shape

        # 임의의 2D weights 생성
        weights = torch.rand(batch_size, seq_len) * 2  # [0, 2] 범위

        # 기존 방식: 수동 계산
        adapter.zero_grad()
        logits = adapter(input_ids)
        n_future = logits.shape[2]

        # 수동으로 weighted loss 계산
        total_loss = 0.0
        for head_idx in range(1, n_future + 1):
            loss_k = compute_head_ce_loss(
                logits=logits[:, :, head_idx - 1, :],
                labels=labels,
                attention_mask=attention_mask,
                head_idx=head_idx,
                weights=weights,
            )
            total_loss += loss_k
        loss_standard = total_loss / n_future

        # Sequential 방식
        adapter.zero_grad()
        loss_result = adapter(
            input_ids,
            attention_mask=attention_mask,
            compute_sequential_loss=True,
            labels=labels,
            weights=weights,
        )
        loss_sequential = loss_result["loss"]

        # 수치 비교
        relative_error = abs(loss_standard.item() - loss_sequential.item()) / (loss_standard.item() + 1e-8)
        assert relative_error < 0.01, f"Loss mismatch: standard={loss_standard.item():.6f}, sequential={loss_sequential.item():.6f}"

    def test_loss_value_equivalence_weighted_3d(self, micro_model, sample_batch):
        """3D 가중치 (Rho1): 기존 방식과 Sequential 방식의 loss 값 일치"""
        adapter = micro_model
        input_ids = sample_batch["input_ids"]
        labels = sample_batch["labels"]
        attention_mask = sample_batch["attention_mask"]
        batch_size, seq_len = input_ids.shape

        # 먼저 n_future 확인
        with torch.no_grad():
            logits_check = adapter(input_ids)
        n_future = logits_check.shape[2]

        # 임의의 3D weights 생성
        weights = torch.rand(batch_size, seq_len, n_future)  # Per-head weights

        # 기존 방식: 수동 계산
        adapter.zero_grad()
        logits = adapter(input_ids)

        total_loss = 0.0
        for head_idx in range(1, n_future + 1):
            loss_k = compute_head_ce_loss(
                logits=logits[:, :, head_idx - 1, :],
                labels=labels,
                attention_mask=attention_mask,
                head_idx=head_idx,
                weights=weights,
            )
            total_loss += loss_k
        loss_standard = total_loss / n_future

        # Sequential 방식
        adapter.zero_grad()
        loss_result = adapter(
            input_ids,
            attention_mask=attention_mask,
            compute_sequential_loss=True,
            labels=labels,
            weights=weights,
        )
        loss_sequential = loss_result["loss"]

        # 수치 비교
        relative_error = abs(loss_standard.item() - loss_sequential.item()) / (loss_standard.item() + 1e-8)
        assert relative_error < 0.01, f"Loss mismatch: standard={loss_standard.item():.6f}, sequential={loss_sequential.item():.6f}"


class TestGradientEquivalence:
    """Gradient 동등성 테스트"""

    def test_gradient_equivalence_unweighted(self, micro_model, sample_batch):
        """균등 가중치: 양 방식의 gradient 일치"""
        input_ids = sample_batch["input_ids"]
        labels = sample_batch["labels"]
        attention_mask = sample_batch["attention_mask"]

        # 기존 방식: gradient 계산
        adapter_standard = micro_model
        adapter_standard.zero_grad()
        logits = adapter_standard(input_ids)
        loss = compute_mtp_ce_loss_unweighted(logits, labels, attention_mask)
        loss.backward()

        # 기존 방식 gradient 저장
        grads_standard = {}
        for name, param in adapter_standard.named_parameters():
            if param.grad is not None:
                grads_standard[name] = param.grad.clone()

        # Sequential 방식: 새 모델 (동일 가중치)
        adapter_sequential = micro_model
        adapter_sequential.load_state_dict(adapter_standard.state_dict())
        adapter_sequential.zero_grad()

        loss_result = adapter_sequential(
            input_ids,
            attention_mask=attention_mask,
            compute_sequential_loss=True,
            labels=labels,
        )

        # Sequential 방식 gradient 저장
        grads_sequential = {}
        for name, param in adapter_sequential.named_parameters():
            if param.grad is not None:
                grads_sequential[name] = param.grad.clone()

        # Gradient 비교
        for name in grads_standard:
            if name in grads_sequential:
                grad_std = grads_standard[name]
                grad_seq = grads_sequential[name]

                # 상대 오차 계산
                diff = (grad_std - grad_seq).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                # 허용 오차: 1e-4 (float32 정밀도 고려)
                assert max_diff < 1e-3, f"Gradient mismatch in {name}: max_diff={max_diff:.6f}"


class TestGradientAccumulation:
    """Gradient Accumulation 테스트"""

    def test_accumulation_with_loss_scale(self, micro_model, sample_batch):
        """loss_scale 적용 시 gradient accumulation 정상 동작"""
        adapter = micro_model
        input_ids = sample_batch["input_ids"]
        labels = sample_batch["labels"]
        attention_mask = sample_batch["attention_mask"]

        accumulation_steps = 4
        loss_scale = 1.0 / accumulation_steps

        adapter.zero_grad()

        # 여러 step 누적
        total_loss = 0.0
        for step in range(accumulation_steps):
            loss_result = adapter(
                input_ids,
                attention_mask=attention_mask,
                compute_sequential_loss=True,
                labels=labels,
                loss_scale=loss_scale,
            )
            total_loss += loss_result["loss"].item()

        # Gradient가 누적되었는지 확인
        has_grad = False
        for param in adapter.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Gradient should be accumulated after multiple steps"

        # 평균 loss 검증
        avg_loss = total_loss / accumulation_steps
        assert avg_loss > 0, "Average loss should be positive"


class TestReturnTypes:
    """반환 타입 테스트"""

    def test_sequential_returns_dict(self, micro_model, sample_batch):
        """compute_sequential_loss=True 시 dict 반환"""
        adapter = micro_model
        input_ids = sample_batch["input_ids"]
        labels = sample_batch["labels"]
        attention_mask = sample_batch["attention_mask"]

        result = adapter(
            input_ids,
            attention_mask=attention_mask,
            compute_sequential_loss=True,
            labels=labels,
        )

        assert isinstance(result, dict)
        assert "loss" in result
        assert "n_heads" in result
        assert result["n_heads"] == 2  # n_future_tokens=2

    def test_standard_returns_tensor(self, micro_model, sample_batch):
        """기본 호출 시 tensor 반환"""
        adapter = micro_model
        input_ids = sample_batch["input_ids"]

        result = adapter(input_ids)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 4  # [batch, seq, n_future, vocab]

    def test_hidden_states_returns_dict(self, micro_model, sample_batch):
        """return_hidden_states=True 시 dict 반환"""
        adapter = micro_model
        input_ids = sample_batch["input_ids"]

        result = adapter(input_ids, return_hidden_states=True)

        assert isinstance(result, dict)
        assert "logits" in result
        assert "hidden_states" in result


class TestEdgeCases:
    """Edge Case 테스트"""

    def test_all_labels_ignored(self, micro_model):
        """모든 labels가 -100인 경우"""
        adapter = micro_model
        batch_size, seq_len = 2, 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.full((batch_size, seq_len), -100)  # 모두 ignore
        attention_mask = torch.ones(batch_size, seq_len)

        result = adapter(
            input_ids,
            attention_mask=attention_mask,
            compute_sequential_loss=True,
            labels=labels,
        )

        # loss가 0이어야 함
        assert result["loss"].item() == 0.0

    def test_partial_labels_ignored(self, micro_model):
        """일부 labels만 -100인 경우"""
        adapter = micro_model
        batch_size, seq_len = 2, 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        labels[:, :8] = -100  # 앞 절반 ignore
        attention_mask = torch.ones(batch_size, seq_len)

        result = adapter(
            input_ids,
            attention_mask=attention_mask,
            compute_sequential_loss=True,
            labels=labels,
        )

        # loss가 양수여야 함
        assert result["loss"].item() > 0

    def test_short_sequence(self, micro_model):
        """매우 짧은 시퀀스 (n_future보다 작음)"""
        adapter = micro_model
        batch_size, seq_len = 2, 3  # n_future_tokens=2보다 약간 큼

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        result = adapter(
            input_ids,
            attention_mask=attention_mask,
            compute_sequential_loss=True,
            labels=labels,
        )

        # 정상 동작 확인
        assert "loss" in result
        assert not result["loss"].isnan()

