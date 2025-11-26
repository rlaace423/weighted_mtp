"""dataloader.py get_difficulty_config 및 sampling 통합 테스트

핵심 검증 항목:
- get_difficulty_config 함수의 정적/동적 weights 추출
- get_curriculum_weights 함수의 epoch별 weights 추출
- 실제 샘플링 결과가 기대값과 일치하는지 검증
"""

import pytest
from omegaconf import OmegaConf

from weighted_mtp.data.dataloader import (
    get_curriculum_weights,
    get_difficulty_config,
)
from weighted_mtp.data import load_dataset


class TestGetCurriculumWeights:
    """get_curriculum_weights 함수 테스트"""

    def test_epoch_in_first_range(self):
        """첫 번째 epoch_range에 해당하는 경우"""
        schedule = [
            {"epoch_range": [0.0, 1.0], "difficulty_weights": {"low": 0.8, "high": 0.2}},
            {"epoch_range": [1.0, 3.0], "difficulty_weights": {"low": 0.3, "high": 0.7}},
        ]

        weights = get_curriculum_weights(0.5, schedule)
        assert weights == {"low": 0.8, "high": 0.2}

    def test_epoch_in_second_range(self):
        """두 번째 epoch_range에 해당하는 경우"""
        schedule = [
            {"epoch_range": [0.0, 1.0], "difficulty_weights": {"low": 0.8, "high": 0.2}},
            {"epoch_range": [1.0, 3.0], "difficulty_weights": {"low": 0.3, "high": 0.7}},
        ]

        weights = get_curriculum_weights(1.5, schedule)
        assert weights == {"low": 0.3, "high": 0.7}

    def test_epoch_at_boundary(self):
        """epoch이 경계값인 경우 (1.0은 두 번째 range)"""
        schedule = [
            {"epoch_range": [0.0, 1.0], "difficulty_weights": {"low": 0.8, "high": 0.2}},
            {"epoch_range": [1.0, 3.0], "difficulty_weights": {"low": 0.3, "high": 0.7}},
        ]

        weights = get_curriculum_weights(1.0, schedule)
        assert weights == {"low": 0.3, "high": 0.7}

    def test_epoch_beyond_range(self):
        """epoch이 모든 range를 초과한 경우 마지막 반환"""
        schedule = [
            {"epoch_range": [0.0, 1.0], "difficulty_weights": {"low": 0.8, "high": 0.2}},
            {"epoch_range": [1.0, 3.0], "difficulty_weights": {"low": 0.3, "high": 0.7}},
        ]

        weights = get_curriculum_weights(5.0, schedule)
        assert weights == {"low": 0.3, "high": 0.7}


class TestGetDifficultyConfig:
    """get_difficulty_config 함수 테스트"""

    def test_static_weights(self):
        """정적 difficulty_weights 사용"""
        config = OmegaConf.create({
            "data_sampling": {
                "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
                "difficulty_weights": {"diff_7": 0.35, "else": 0.65},
            }
        })

        weights, bins = get_difficulty_config(config)

        assert weights == {"diff_7": 0.35, "else": 0.65}
        assert bins == {"diff_7": [7, 7], "else": [8, 25]}

    def test_curriculum_schedule(self):
        """curriculum_schedule 사용 (동적 weights)"""
        config = OmegaConf.create({
            "data_sampling": {
                "difficulty_bins": {"low": [1, 3], "high": [4, 11]},
                "curriculum_learning": True,
                "curriculum_schedule": [
                    {"epoch_range": [0.0, 1.0], "difficulty_weights": {"low": 0.8, "high": 0.2}},
                    {"epoch_range": [1.0, 3.0], "difficulty_weights": {"low": 0.3, "high": 0.7}},
                ],
            }
        })

        # epoch 0.5
        weights, bins = get_difficulty_config(config, current_epoch=0.5)
        assert weights == {"low": 0.8, "high": 0.2}

        # epoch 1.5
        weights, bins = get_difficulty_config(config, current_epoch=1.5)
        assert weights == {"low": 0.3, "high": 0.7}

    def test_no_difficulty_config(self):
        """difficulty 설정이 없는 경우"""
        config = OmegaConf.create({
            "data_sampling": {
                "n_samples": 1000,
            }
        })

        weights, bins = get_difficulty_config(config)

        assert weights is None
        assert bins is None

    def test_curriculum_learning_false(self):
        """curriculum_learning=false면 정적 weights 사용"""
        config = OmegaConf.create({
            "data_sampling": {
                "difficulty_bins": {"low": [1, 3], "high": [4, 11]},
                "curriculum_learning": False,
                "curriculum_schedule": [
                    {"epoch_range": [0.0, 3.0], "difficulty_weights": {"low": 0.8, "high": 0.2}},
                ],
                "difficulty_weights": {"low": 0.5, "high": 0.5},
            }
        })

        weights, bins = get_difficulty_config(config)

        # curriculum_learning=false이므로 정적 weights 사용
        assert weights == {"low": 0.5, "high": 0.5}

    def test_bins_only_no_weights(self):
        """difficulty_bins만 있고 weights가 없는 경우"""
        config = OmegaConf.create({
            "data_sampling": {
                "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
            }
        })

        weights, bins = get_difficulty_config(config)

        assert weights is None
        assert bins == {"diff_7": [7, 7], "else": [8, 25]}


class TestSamplingWithDifficultyConfig:
    """실제 샘플링 결과 검증 (기대값 vs 실제값)"""

    def test_baseline_config_sampling(self):
        """Baseline config: diff_7=35%, else=65%, correct_only

        기대값:
        - 총 1000개 샘플
        - diff_7: 350개 (35%)
        - else: 650개 (65%)
        - 모두 correct
        """
        n_samples = 1000

        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": n_samples,
                "correct_ratio": 1.0,
                "difficulty_weights": {"diff_7": 0.35, "else": 0.65},
                "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
            },
            seed=42
        )

        assert len(dataset) == n_samples

        # difficulty 분포 확인
        diff_7_count = sum(
            1 for sample in dataset
            if sample["metadata"]["difficulty"] == 7
        )
        else_count = sum(
            1 for sample in dataset
            if 8 <= sample["metadata"]["difficulty"] <= 25
        )

        # 기대값 검증 (오차 ±5% 허용)
        expected_diff_7 = int(n_samples * 0.35)
        expected_else = int(n_samples * 0.65)

        assert abs(diff_7_count - expected_diff_7) <= n_samples * 0.05, \
            f"diff_7: expected ~{expected_diff_7}, got {diff_7_count}"
        assert abs(else_count - expected_else) <= n_samples * 0.05, \
            f"else: expected ~{expected_else}, got {else_count}"

        # 모두 correct 검증
        all_correct = all(sample["is_correct"] for sample in dataset)
        assert all_correct, "correct_ratio=1.0이면 모든 샘플이 correct여야 함"

    def test_critic_config_sampling(self):
        """Critic config: all difficulty, 50:50 balanced

        기대값:
        - 총 800개 샘플
        - correct: 400개 (50%)
        - incorrect: 400개 (50%)
        - 난이도 1-25 전체
        """
        n_samples = 800

        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": n_samples,
                "correct_ratio": 0.5,
                "difficulty_weights": {"all": 1.0},
                "difficulty_bins": {"all": [1, 25]},
            },
            seed=42
        )

        assert len(dataset) == n_samples

        # correct/incorrect 분포 확인
        correct_count = sum(1 for sample in dataset if sample["is_correct"])
        incorrect_count = len(dataset) - correct_count

        # 기대값 검증 (오차 ±10% 허용)
        expected_correct = int(n_samples * 0.5)

        assert abs(correct_count - expected_correct) <= n_samples * 0.1, \
            f"correct: expected ~{expected_correct}, got {correct_count}"

        # 난이도 범위 검증
        for sample in dataset:
            difficulty = sample["metadata"]["difficulty"]
            assert 1 <= difficulty <= 25, f"Expected difficulty 1-25, got {difficulty}"

    def test_mixed_difficulty_balanced_sampling(self):
        """Mixed difficulty + balanced: low=70%, medium=30%, 50:50

        기대값:
        - 총 500개 샘플
        - low (1-3): 350개 (70%)
        - medium (4-7): 150개 (30%)
        - 각 bin 내에서 correct/incorrect 50:50
        """
        n_samples = 500

        dataset = load_dataset(
            "codecontests",
            split="train",
            sampling_config={
                "n_samples": n_samples,
                "correct_ratio": 0.5,
                "difficulty_weights": {"low": 0.7, "medium": 0.3},
                "difficulty_bins": {"low": [1, 3], "medium": [4, 7]},
            },
            seed=42
        )

        assert len(dataset) == n_samples

        # difficulty 분포 확인
        low_count = sum(
            1 for sample in dataset
            if 1 <= sample["metadata"]["difficulty"] <= 3
        )
        medium_count = sum(
            1 for sample in dataset
            if 4 <= sample["metadata"]["difficulty"] <= 7
        )

        # 기대값 검증 (오차 ±10% 허용)
        expected_low = int(n_samples * 0.7)
        expected_medium = int(n_samples * 0.3)

        assert abs(low_count - expected_low) <= n_samples * 0.1, \
            f"low: expected ~{expected_low}, got {low_count}"
        assert abs(medium_count - expected_medium) <= n_samples * 0.1, \
            f"medium: expected ~{expected_medium}, got {medium_count}"

        # 전체 correct/incorrect 비율 검증
        correct_count = sum(1 for sample in dataset if sample["is_correct"])
        incorrect_count = len(dataset) - correct_count

        # correct_ratio=0.5이므로 correct/incorrect 모두 존재해야 함
        assert correct_count > 0, "correct 샘플이 있어야 함"
        assert incorrect_count > 0, "incorrect 샘플이 있어야 함"

    def test_reproducibility_with_difficulty(self):
        """동일 seed + difficulty 설정 → 동일 결과"""
        config = {
            "n_samples": 200,
            "correct_ratio": 1.0,
            "difficulty_weights": {"diff_7": 0.35, "else": 0.65},
            "difficulty_bins": {"diff_7": [7, 7], "else": [8, 25]},
        }

        dataset1 = load_dataset("codecontests", split="train", **config)
        dataset2 = load_dataset("codecontests", split="train", **config)

        ids1 = [sample["task_id"] for sample in dataset1]
        ids2 = [sample["task_id"] for sample in dataset2]

        assert ids1 == ids2, "동일 seed + config면 동일한 샘플이어야 함"
