"""Metrics 및 모니터링 유틸리티

Training metrics, GPU monitoring, throughput tracking
"""

import time
from typing import Any

import psutil
import torch


class GPUMonitor:
    """GPU 사용량 모니터링

    CUDA 환경에서 GPU memory, utilization 추적
    """

    def __init__(self, device: torch.device):
        """초기화

        Args:
            device: torch.device
        """
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.device_index = device.index if self.is_cuda and device.index is not None else 0

    def get_metrics(self) -> dict[str, float]:
        """현재 GPU metrics 수집

        Returns:
            {
                "gpu_memory_allocated_gb": float,  # 현재 할당된 메모리 (GB)
                "gpu_memory_reserved_gb": float,   # 예약된 메모리 (GB)
                "gpu_memory_cached_gb": float,     # 캐시된 메모리 (GB)
                "gpu_utilization_pct": float,      # GPU 사용률 (%)
            }
        """
        if not self.is_cuda:
            return {
                "gpu_memory_allocated_gb": 0.0,
                "gpu_memory_reserved_gb": 0.0,
                "gpu_memory_cached_gb": 0.0,
                "gpu_utilization_pct": 0.0,
            }

        # PyTorch memory stats
        allocated = torch.cuda.memory_allocated(self.device_index) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device_index) / 1024**3
        cached = torch.cuda.memory_reserved(self.device_index) / 1024**3

        # GPU utilization (nvidia-smi 기반)
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(utilization.gpu)
            pynvml.nvmlShutdown()
        except (ImportError, Exception):
            # pynvml 없거나 실패 시 0
            gpu_util = 0.0

        return {
            "gpu_memory_allocated_gb": allocated,
            "gpu_memory_reserved_gb": reserved,
            "gpu_memory_cached_gb": cached,
            "gpu_utilization_pct": gpu_util,
        }


class ThroughputTracker:
    """학습 처리량 추적

    Samples/sec, tokens/sec, epoch time 계산
    """

    def __init__(self):
        """초기화"""
        self.epoch_start_time = None
        self.epoch_samples = 0
        self.epoch_tokens = 0

    def start_epoch(self) -> None:
        """Epoch 시작 시점 기록"""
        self.epoch_start_time = time.time()
        self.epoch_samples = 0
        self.epoch_tokens = 0

    def update(self, n_samples: int, n_tokens: int) -> None:
        """배치 처리 후 업데이트

        Args:
            n_samples: 배치 샘플 수
            n_tokens: 배치 토큰 수
        """
        self.epoch_samples += n_samples
        self.epoch_tokens += n_tokens

    def get_epoch_metrics(self) -> dict[str, float]:
        """Epoch 종료 시점 metrics 계산

        Returns:
            {
                "epoch_time_sec": float,
                "samples_per_sec": float,
                "tokens_per_sec": float,
            }
        """
        if self.epoch_start_time is None:
            return {
                "epoch_time_sec": 0.0,
                "samples_per_sec": 0.0,
                "tokens_per_sec": 0.0,
            }

        elapsed_time = time.time() - self.epoch_start_time

        if elapsed_time < 1e-6:
            return {
                "epoch_time_sec": elapsed_time,
                "samples_per_sec": 0.0,
                "tokens_per_sec": 0.0,
            }

        return {
            "epoch_time_sec": elapsed_time,
            "samples_per_sec": self.epoch_samples / elapsed_time,
            "tokens_per_sec": self.epoch_tokens / elapsed_time,
        }


def get_model_size(model: torch.nn.Module) -> dict[str, int]:
    """모델 파라미터 수 계산

    Args:
        model: PyTorch model

    Returns:
        {
            "total_params": int,
            "trainable_params": int,
            "non_trainable_params": int,
        }
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def get_system_info() -> dict[str, Any]:
    """시스템 정보 수집

    Returns:
        {
            "cpu_count": int,
            "cpu_percent": float,
            "ram_total_gb": float,
            "ram_available_gb": float,
            "ram_used_gb": float,
        }
    """
    cpu_count = psutil.cpu_count(logical=True)
    cpu_percent = psutil.cpu_percent(interval=0.1)

    mem = psutil.virtual_memory()
    ram_total_gb = mem.total / 1024**3
    ram_available_gb = mem.available / 1024**3
    ram_used_gb = mem.used / 1024**3

    return {
        "cpu_count": cpu_count,
        "cpu_percent": cpu_percent,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "ram_used_gb": ram_used_gb,
    }


def compute_gradient_norm(model: torch.nn.Module) -> dict[str, float]:
    """Gradient norm 계산

    Args:
        model: PyTorch model

    Returns:
        {
            "grad_norm": float,  # Total gradient norm
        }
    """
    total_norm = 0.0
    n_params = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            n_params += 1

    if n_params == 0:
        return {"grad_norm": 0.0}

    total_norm = total_norm**0.5

    return {"grad_norm": total_norm}
