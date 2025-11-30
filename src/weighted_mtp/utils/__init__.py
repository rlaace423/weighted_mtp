"""Weighted MTP 유틸리티 모듈

Checkpoint 관리, 성능 모니터링, S3 백업, Generation, Evaluation 기능 제공
"""

from weighted_mtp.utils.checkpoint_utils import (
    cleanup_old_checkpoints,
    load_checkpoint_for_evaluation,
    load_lora_checkpoint,
    save_checkpoint,
    save_hf_checkpoint,
    save_hf_lora_checkpoint,
    save_lora_checkpoint,
)
from weighted_mtp.utils.evaluation_utils import (
    compute_pass_at_k,
    evaluate_gsm8k_answer,
    evaluate_pass_at_k,
    execute_code_with_tests,
    execute_mbpp_tests,
    extract_gsm8k_answer,
)
from weighted_mtp.utils.generation_utils import (
    generate_with_mtp,
)
from weighted_mtp.utils.metrics_utils import (
    GPUMonitor,
    ThroughputTracker,
    compute_gradient_norm,
    get_model_size,
    get_system_info,
)
from weighted_mtp.utils.logging_utils import (
    compute_classification_metrics_from_counts,
    compute_critic_classification_counts,
    compute_gradient_clip_stats,
    compute_gradient_clip_stats_by_component,
    compute_gradient_norm_by_component,
    compute_value_function_stats,
    compute_weight_statistics,
)
from weighted_mtp.utils.s3_utils import (
    cleanup_s3_checkpoints,
    reset_s3_executor,
    s3_upload_executor,
    shutdown_s3_executor,
    upload_to_s3_async,
)
from weighted_mtp.utils.scheduler_utils import (
    create_param_groups,
    create_scheduler,
    get_scheduler_state,
    load_scheduler_state,
)
from weighted_mtp.utils.pairwise_utils import (
    compute_lambda_return,
    compute_lambda_value_loss,
    compute_mc_value_loss,
    compute_pairwise_accuracy,
    compute_token_variance,
    create_eos_only_mask,
    pairwise_ranking_loss,
)
from weighted_mtp.utils.loss_utils import (
    compute_mtp_ce_loss,
    compute_mtp_ce_loss_unweighted,
)

__all__ = [
    # Checkpoint utils
    "cleanup_old_checkpoints",
    "load_checkpoint_for_evaluation",
    "load_lora_checkpoint",
    "save_checkpoint",
    "save_hf_checkpoint",
    "save_hf_lora_checkpoint",
    "save_lora_checkpoint",
    # Evaluation utils
    "compute_pass_at_k",
    "evaluate_gsm8k_answer",
    "evaluate_pass_at_k",
    "execute_code_with_tests",
    "execute_mbpp_tests",
    "extract_gsm8k_answer",
    # Generation utils
    "generate_with_mtp",
    # Metrics utils
    "GPUMonitor",
    "ThroughputTracker",
    "compute_gradient_norm",
    "get_model_size",
    "get_system_info",
    # S3 utils
    "cleanup_s3_checkpoints",
    "reset_s3_executor",
    "s3_upload_executor",
    "shutdown_s3_executor",
    "upload_to_s3_async",
    # Logging utils
    "compute_classification_metrics_from_counts",
    "compute_critic_classification_counts",
    "compute_gradient_clip_stats",
    "compute_gradient_clip_stats_by_component",
    "compute_gradient_norm_by_component",
    "compute_value_function_stats",
    "compute_weight_statistics",
    # Scheduler utils
    "create_param_groups",
    "create_scheduler",
    "get_scheduler_state",
    "load_scheduler_state",
    # Loss utils
    "compute_lambda_return",
    "compute_lambda_value_loss",
    "compute_mc_value_loss",
    "compute_mtp_ce_loss",
    "compute_mtp_ce_loss_unweighted",
    "compute_pairwise_accuracy",
    "compute_token_variance",
    "create_eos_only_mask",
    "pairwise_ranking_loss",
]
