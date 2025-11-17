#!/usr/bin/env python3
"""
Phase1 모델 설정 통합 스크립트

기능:
- HuggingFace에서 모델 다운로드
- PyTorch → SafeTensors 변환 (단일/Sharded 자동 감지)
- Micro 모델 생성 (Meta/HuggingFace 아키텍처 지원)
- MTP Config 동기화 (params.json → meta_adapter.yaml)
- 모델 무결성 검증

Usage:
    # Meta MTP 전체 설정
    python scripts/setup_models.py --model meta-llama-mtp --steps all --create-micro

    # Sheared-LLaMA 변환만
    python scripts/setup_models.py --model ref-sheared-llama --steps convert,verify

    # 검증만
    python scripts/setup_models.py --model meta-llama-mtp --steps verify
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file


# ============================================================================
# 모델 설정 정의
# ============================================================================

MODEL_CONFIGS = {
    "meta-llama-mtp": {
        "repo": "facebook/multi-token-prediction",
        "files": [
            "7B_1T_4/consolidated.pth",
            "7B_1T_4/params.json",
            "tokenizer.model",
        ],
        "format": "single_pth",
        "base_dir": "storage/models/meta-llama-mtp",
        "config_source": "raw/7B_1T_4/params.json",
        "config_type": "mtp",
    },
    "ref-sheared-llama": {
        "repo": "microsoft/rho-1",
        "files": [
            "reference_model/pytorch_model.bin.index.json",
            "reference_model/pytorch_model-00001-of-00002.bin",
            "reference_model/pytorch_model-00002-of-00002.bin",
            "reference_model/config.json",
            "tokenizer/tokenizer.model",
            "tokenizer/tokenizer.json",
        ],
        "format": "sharded",
        "base_dir": "storage/models/ref-sheared-llama-2.7b",
        "config_source": "raw/reference_model/config.json",
        "config_type": "hf",
    },
    "starling-rm": {
        "repo": "berkeley-nest/Starling-RM-7B-alpha",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.model"],
        "format": "single_bin",
        "base_dir": "storage/models/starling-rm-7b",
        "config_source": "raw/config.json",
        "config_type": "hf",
    },
}


# ============================================================================
# 유틸리티 함수
# ============================================================================

def calculate_sha256(file_path: Path) -> str:
    """파일의 SHA256 해시 계산"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_safetensors(
    safetensors_path: Path, original_state_dict: Optional[Dict] = None
) -> bool:
    """SafeTensors 파일 검증"""
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        safetensors_keys = set(f.keys())

    if original_state_dict:
        pytorch_keys = set(original_state_dict.keys())
        if safetensors_keys != pytorch_keys:
            print("❌ Key mismatch!")
            print(f"  Missing: {pytorch_keys - safetensors_keys}")
            print(f"  Extra: {safetensors_keys - pytorch_keys}")
            return False

    print(f"✓ All {len(safetensors_keys)} tensors verified")
    return True


# ============================================================================
# ModelSetup 클래스
# ============================================================================

class ModelSetup:
    def __init__(self, model_name: str):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.base_dir = Path(self.config["base_dir"])

    def download(self) -> None:
        """HuggingFace에서 모델 다운로드"""
        print("=" * 70)
        print(f"Downloading {self.model_name}")
        print("=" * 70)
        print()

        repo = self.config["repo"]
        files = self.config["files"]

        # 디렉터리 생성
        raw_dir = self.base_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # HuggingFace CLI로 다운로드
        for file in files:
            print(f"Downloading {file}...")
            cmd = [
                "huggingface-cli",
                "download",
                repo,
                file,
                "--local-dir",
                str(raw_dir),
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"✓ Downloaded {file}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download {file}")
                print(f"Error: {e.stderr}")
                raise

        print()
        print("✅ Download completed!")
        print()

    def convert(self) -> str:
        """PyTorch → SafeTensors 변환 (format 자동 감지)"""
        print("=" * 70)
        print(f"Converting {self.model_name} to SafeTensors")
        print("=" * 70)
        print()

        format_type = self.config["format"]

        if format_type == "single_pth":
            return self._convert_single_pth()
        elif format_type == "sharded":
            return self._convert_sharded()
        elif format_type == "single_bin":
            return self._convert_single_bin()
        else:
            raise ValueError(f"Unknown format: {format_type}")

    def _convert_single_pth(self) -> str:
        """단일 .pth 파일 변환 (Meta MTP)"""
        input_file = self.base_dir / "raw/7B_1T_4/consolidated.pth"
        output_dir = self.base_dir / "safetensors"
        output_file = output_dir / "model.safetensors"

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print()

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        input_size_gb = input_file.stat().st_size / (1024**3)
        print(f"Input size: {input_size_gb:.2f} GB")
        print()

        print("Loading PyTorch checkpoint...")
        state_dict = torch.load(input_file, map_location="cpu", weights_only=True)
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"✓ Loaded {len(state_dict)} tensors ({total_params/1e9:.2f}B params)")
        print()

        print("Converting to SafeTensors...")
        save_file(state_dict, str(output_file))
        output_size_gb = output_file.stat().st_size / (1024**3)
        print(f"✓ Saved to {output_file.name} ({output_size_gb:.2f} GB)")
        print()

        print("Calculating SHA256...")
        sha256_hash = calculate_sha256(output_file)
        sha256_file = output_dir / "SHA256SUMS"
        with open(sha256_file, "w") as f:
            f.write(f"{sha256_hash}  model.safetensors\n")
        print(f"✓ SHA256: {sha256_hash}")
        print()

        print("Verifying conversion...")
        verify_safetensors(output_file, state_dict)
        print()

        print("=" * 70)
        print("✅ Conversion completed!")
        print("=" * 70)
        print()

        return sha256_hash

    def _convert_sharded(self) -> str:
        """Sharded PyTorch 파일 병합 및 변환 (Sheared-LLaMA)"""
        raw_dir = self.base_dir / "raw/reference_model"
        output_dir = self.base_dir / "safetensors"
        output_file = output_dir / "model.safetensors"

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Input dir:  {raw_dir}")
        print(f"Output file: {output_file}")
        print()

        index_file = raw_dir / "pytorch_model.bin.index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        with open(index_file, "r") as f:
            index = json.load(f)

        shard_files = sorted(set(index["weight_map"].values()))
        print(f"Found {len(shard_files)} shard files:")
        for shard in shard_files:
            print(f"  - {shard}")
        print()

        print("Loading and merging shards...")
        merged_state_dict = {}
        for shard_file in shard_files:
            shard_path = raw_dir / shard_file
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not found: {shard_path}")

            print(f"  Loading {shard_file}...")
            shard_state = torch.load(shard_path, map_location="cpu", weights_only=True)
            merged_state_dict.update(shard_state)
            del shard_state

        total_params = sum(p.numel() for p in merged_state_dict.values())
        print(f"✓ Merged {len(merged_state_dict)} tensors ({total_params/1e9:.2f}B params)")
        print()

        print("Converting to SafeTensors...")
        save_file(merged_state_dict, str(output_file))
        output_size_gb = output_file.stat().st_size / (1024**3)
        print(f"✓ Saved to {output_file.name} ({output_size_gb:.2f} GB)")
        print()

        print("Calculating SHA256...")
        sha256_hash = calculate_sha256(output_file)
        sha256_file = output_dir / "SHA256SUMS"
        with open(sha256_file, "w") as f:
            f.write(f"{sha256_hash}  model.safetensors\n")
        print(f"✓ SHA256: {sha256_hash}")
        print()

        print("Verifying conversion...")
        verify_safetensors(output_file, merged_state_dict)
        print()

        print("=" * 70)
        print("✅ Conversion completed!")
        print("=" * 70)
        print()

        return sha256_hash

    def _convert_single_bin(self) -> str:
        """단일 .bin 파일 변환 (Starling-RM)"""
        input_file = self.base_dir / "raw/pytorch_model.bin"
        output_dir = self.base_dir / "safetensors"
        output_file = output_dir / "model.safetensors"

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print()

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        input_size_gb = input_file.stat().st_size / (1024**3)
        print(f"Input size: {input_size_gb:.2f} GB")
        print()

        print("Loading PyTorch checkpoint...")
        state_dict = torch.load(input_file, map_location="cpu", weights_only=True)
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"✓ Loaded {len(state_dict)} tensors ({total_params/1e9:.2f}B params)")
        print()

        print("Converting to SafeTensors...")
        save_file(state_dict, str(output_file))
        output_size_gb = output_file.stat().st_size / (1024**3)
        print(f"✓ Saved to {output_file.name} ({output_size_gb:.2f} GB)")
        print()

        print("Calculating SHA256...")
        sha256_hash = calculate_sha256(output_file)
        sha256_file = output_dir / "SHA256SUMS"
        with open(sha256_file, "w") as f:
            f.write(f"{sha256_hash}  model.safetensors\n")
        print(f"✓ SHA256: {sha256_hash}")
        print()

        print("Verifying conversion...")
        verify_safetensors(output_file, state_dict)
        print()

        print("=" * 70)
        print("✅ Conversion completed!")
        print("=" * 70)
        print()

        return sha256_hash

    def sync_config(self) -> None:
        """Config 동기화 (raw → configs/ 디렉토리로 복사 및 변환)"""
        print("=" * 70)
        print(f"Syncing Config for {self.model_name}")
        print("=" * 70)
        print()

        config_type = self.config.get("config_type")
        config_source_rel = self.config.get("config_source")

        if not config_source_rel:
            print(f"⚠️  No config source defined for {self.model_name}, skipping...")
            return

        config_source = self.base_dir / config_source_rel
        config_dir = self.base_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        if not config_source.exists():
            print(f"⚠️  Config source not found: {config_source}, skipping...")
            return

        # Config 타입에 따라 처리
        if config_type == "mtp":
            self._sync_mtp_config(config_source, config_dir)
        elif config_type == "hf":
            self._sync_hf_config(config_source, config_dir)
        else:
            print(f"⚠️  Unknown config type: {config_type}, skipping...")
            return

        print("=" * 70)
        print("✅ Config sync completed!")
        print("=" * 70)
        print()

    def _sync_mtp_config(self, params_file: Path, config_dir: Path) -> None:
        """Meta MTP Config 동기화 (params.json → configs/params.json + meta_adapter.yaml)"""
        # 1. params.json 복사
        params_dest = config_dir / "params.json"
        shutil.copy2(params_file, params_dest)
        print(f"✓ Copied params.json to configs/")

        # 2. params.json 로드
        with open(params_file, "r") as f:
            params = json.load(f)

        print("  Source params.json:")
        for key in ["n_future_tokens", "rope_theta", "dim", "n_layers"]:
            if key in params:
                print(f"    {key}: {params[key]}")
        print()

        # 3. intermediate_size 계산
        dim = params["dim"]
        ffn_dim_multiplier = params.get("ffn_dim_multiplier", 1.0)
        multiple_of = params.get("multiple_of", 256)

        hidden_dim = int(8 * dim / 3)
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        intermediate_size = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        print(f"  Calculated intermediate_size: {intermediate_size}")
        print()

        # 4. meta_adapter.yaml 생성
        adapter_config = {
            "model_name": "meta-llama-mtp",
            "n_future_tokens": params["n_future_tokens"],
            "hidden_size": params["dim"],
            "num_hidden_layers": params["n_layers"],
            "num_attention_heads": params["n_heads"],
            "num_key_value_heads": params["n_kv_heads"],
            "intermediate_size": intermediate_size,
            "rope_theta": params["rope_theta"],
            "max_position_embeddings": 2048,
            "rms_norm_eps": params["norm_eps"],
            "vocab_size": params["vocab_size"],
            "cache": {"freqs_cis": True, "causal_mask": True},
            "dtype": "float16",
        }

        adapter_file = config_dir / "meta_adapter.yaml"
        with open(adapter_file, "w") as f:
            yaml.dump(adapter_config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Created meta_adapter.yaml")
        print()

    def _sync_hf_config(self, config_source: Path, config_dir: Path) -> None:
        """HuggingFace Config 동기화 (config.json 복사)"""
        config_dest = config_dir / "config.json"
        shutil.copy2(config_source, config_dest)
        print(f"✓ Copied config.json to configs/")

        # Config 내용 출력
        with open(config_source, "r") as f:
            config = json.load(f)

        print("  Key parameters:")
        for key in ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"]:
            if key in config:
                print(f"    {key}: {config[key]}")
        print()
        print()

    def create_micro(self, model_type: Literal["mtp", "reference"]) -> None:
        """Micro 모델 생성 (4 layers, 512 hidden_size)"""
        print("=" * 70)
        print(f"Creating Micro {model_type.upper()} Model")
        print("=" * 70)
        print()

        source_file = self.base_dir / "safetensors/model.safetensors"
        if not source_file.exists():
            raise FileNotFoundError(
                f"Source safetensors not found: {source_file}\n"
                f"Please run conversion first."
            )

        target_dir = Path(f"storage/models/micro-{model_type}")
        (target_dir / "safetensors").mkdir(parents=True, exist_ok=True)
        (target_dir / "configs").mkdir(parents=True, exist_ok=True)

        print(f"Source: {source_file}")
        print(f"Target: {target_dir}")
        print()

        # 원본 모델 로드
        print("Loading source model...")
        state_dict = {}
        with safe_open(source_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        print(f"✓ Loaded {len(state_dict)} tensors")
        print()

        # Micro 모델 파라미터
        n_layers = 4
        hidden_size = 512
        intermediate_size = 1365  # int(512 * 8/3) rounded to multiple of 256

        print(f"Creating micro model: {n_layers} layers, {hidden_size} hidden_size")
        print()

        # 모델 타입별 슬라이싱
        if model_type == "mtp":
            micro_state_dict = self._slice_meta_model(
                state_dict, n_layers, hidden_size, intermediate_size
            )
        else:  # reference
            micro_state_dict = self._slice_hf_model(
                state_dict, n_layers, hidden_size, intermediate_size
            )

        total_params = sum(p.numel() for p in micro_state_dict.values())
        print(f"✓ Micro model: {total_params:,} params ({total_params/1e6:.1f}M)")
        print()

        # SafeTensors 저장
        output_file = target_dir / "safetensors/model.safetensors"
        print("Saving to SafeTensors...")
        save_file(micro_state_dict, str(output_file))

        output_size_mb = output_file.stat().st_size / (1024**2)
        print(f"✓ Saved to {output_file.name} ({output_size_mb:.1f} MB)")
        print()

        # SHA256 계산
        sha256_hash = calculate_sha256(output_file)
        sha256_file = target_dir / "safetensors/SHA256SUMS"
        with open(sha256_file, "w") as f:
            f.write(f"{sha256_hash}  model.safetensors\n")
        print(f"✓ SHA256: {sha256_hash}")
        print()

        # Config 및 Metadata 생성
        self._create_micro_config(
            target_dir, model_type, n_layers, hidden_size, intermediate_size, total_params, sha256_hash
        )

        print("=" * 70)
        print("✅ Micro model created!")
        print("=" * 70)
        print()

    def _slice_meta_model(
        self, state_dict: Dict, n_layers: int, hidden_size: int, intermediate_size: int
    ) -> Dict:
        """Meta 아키텍처 모델 슬라이싱"""
        micro_state_dict = {}

        for key, tensor in state_dict.items():
            # Embeddings
            if "tok_embeddings" in key or "output" in key:
                if len(tensor.shape) == 2:
                    micro_state_dict[key] = tensor[:, :hidden_size].contiguous()
                else:
                    micro_state_dict[key] = tensor[:hidden_size].contiguous()

            # Final norm
            elif "norm" in key and "layers" not in key:
                micro_state_dict[key] = tensor[:hidden_size].contiguous()

            # Layer-specific weights
            elif "layers." in key:
                layer_num = int(key.split(".")[1])
                if layer_num < n_layers:
                    # Attention
                    if "attention" in key:
                        if len(tensor.shape) == 2:
                            micro_state_dict[key] = tensor[:hidden_size, :hidden_size].contiguous()
                        else:
                            micro_state_dict[key] = tensor[:hidden_size].contiguous()

                    # FFN
                    elif "feed_forward" in key:
                        if "w1" in key or "w3" in key:
                            micro_state_dict[key] = tensor[:intermediate_size, :hidden_size].contiguous()
                        elif "w2" in key:
                            micro_state_dict[key] = tensor[:hidden_size, :intermediate_size].contiguous()

                    # Norms
                    elif "norm" in key:
                        micro_state_dict[key] = tensor[:hidden_size].contiguous()

        return micro_state_dict

    def _slice_hf_model(
        self, state_dict: Dict, n_layers: int, hidden_size: int, intermediate_size: int
    ) -> Dict:
        """HuggingFace 아키텍처 모델 슬라이싱"""
        micro_state_dict = {}

        for key, tensor in state_dict.items():
            # Embeddings
            if key == "lm_head.weight":
                micro_state_dict[key] = tensor[:, :hidden_size].contiguous()
            elif key == "model.embed_tokens.weight":
                micro_state_dict[key] = tensor[:, :hidden_size].contiguous()

            # Final norm
            elif key == "model.norm.weight":
                micro_state_dict[key] = tensor[:hidden_size].contiguous()

            # Layer-specific weights
            elif key.startswith("model.layers."):
                parts = key.split(".")
                layer_num = int(parts[2])

                if layer_num < n_layers:
                    # Self-attention
                    if "self_attn" in key:
                        if "rotary_emb.inv_freq" in key:
                            continue
                        elif any(proj in key for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                            micro_state_dict[key] = tensor[:hidden_size, :hidden_size].contiguous()

                    # MLP
                    elif "mlp" in key:
                        if "gate_proj" in key or "up_proj" in key:
                            micro_state_dict[key] = tensor[:intermediate_size, :hidden_size].contiguous()
                        elif "down_proj" in key:
                            micro_state_dict[key] = tensor[:hidden_size, :intermediate_size].contiguous()

                    # Layer norms
                    elif "layernorm" in key:
                        micro_state_dict[key] = tensor[:hidden_size].contiguous()

        return micro_state_dict

    def _create_micro_config(
        self,
        target_dir: Path,
        model_type: str,
        n_layers: int,
        hidden_size: int,
        intermediate_size: int,
        total_params: int,
        sha256_hash: str,
    ) -> None:
        """Micro 모델 Config 및 Metadata 생성"""
        # Config
        if model_type == "mtp":
            config = {
                "model_name": "micro-mtp",
                "n_future_tokens": 4,
                "hidden_size": hidden_size,
                "num_hidden_layers": n_layers,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": intermediate_size,
                "rope_theta": 10000.0,
                "max_position_embeddings": 2048,
                "rms_norm_eps": 1e-05,
                "vocab_size": 32000,
                "cache": {"freqs_cis": True, "causal_mask": True},
                "dtype": "float16",
                "target_device": "mps",
            }
        else:  # reference
            config = {
                "model_name": "micro-ref-sheared",
                "hidden_size": hidden_size,
                "num_hidden_layers": n_layers,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": intermediate_size,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-05,
                "vocab_size": 32000,
                "hidden_act": "silu",
                "torch_dtype": "float16",
                "target_device": "mps",
            }

        config_file = target_dir / "configs/config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved to {config_file.name}")

        # Metadata
        metadata = {
            "model_name": f"micro-{model_type}",
            "version": "v2.0.0",
            "n_params": total_params,
            "format": "safetensors",
            "dtype": "float16",
            "files": {
                "checkpoint": "safetensors/model.safetensors",
                "config": "configs/config.json",
            },
            "sha256": {"checkpoint": sha256_hash},
            "created_at": "2025-11-14",
            "target_device": "mps",
            "notes": f"Micro {model_type} model for local testing. {n_layers} layers, {hidden_size} hidden_size.",
        }

        if model_type == "reference":
            metadata["tokenizer_shared_with"] = "meta-llama-mtp"
            metadata["source"] = {
                "repo": "princeton-nlp/Sheared-LLaMA-2.7B",
                "revision": "micro-4layer-512dim",
            }

        metadata_file = target_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_file.name}")
        print()

    def verify(self) -> bool:
        """모델 무결성 검증"""
        print("=" * 70)
        print(f"Verifying {self.model_name}")
        print("=" * 70)
        print()

        errors = []

        # 1. 파일 존재 확인
        print("[1/3] Checking required files...")
        safetensors_file = self.base_dir / "safetensors/model.safetensors"
        sha256_file = self.base_dir / "safetensors/SHA256SUMS"

        for file in [safetensors_file, sha256_file]:
            if file.exists():
                size_mb = file.stat().st_size / (1024**2)
                print(f"  ✓ {file.name} ({size_mb:.1f} MB)")
            else:
                errors.append(f"Missing: {file}")
                print(f"  ✗ {file.name} NOT FOUND")
        print()

        if errors:
            print("❌ Verification failed!")
            for err in errors:
                print(f"  - {err}")
            return False

        # 2. SHA256 검증
        print("[2/3] Verifying SHA256...")
        with open(sha256_file, "r") as f:
            expected_hash = f.read().strip().split()[0]

        actual_hash = calculate_sha256(safetensors_file)

        print(f"  Expected: {expected_hash}")
        print(f"  Actual:   {actual_hash}")

        if expected_hash == actual_hash:
            print("  ✓ SHA256 match")
        else:
            errors.append("SHA256 mismatch")
            print("  ✗ SHA256 mismatch!")
        print()

        # 3. SafeTensors 구조 검증
        print("[3/3] Verifying SafeTensors structure...")
        try:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                print(f"  ✓ Loaded {len(keys)} tensors")

                # 샘플 텐서 확인
                sample_keys = keys[:3]
                for key in sample_keys:
                    tensor = f.get_tensor(key)
                    print(f"  ✓ {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        except Exception as e:
            errors.append(f"SafeTensors loading failed: {e}")
            print(f"  ✗ Error: {e}")
        print()

        # 결과
        print("=" * 70)
        if errors:
            print("❌ Verification failed!")
            print("=" * 70)
            for err in errors:
                print(f"  - {err}")
            return False
        else:
            print("✅ All checks passed!")
            print("=" * 70)
            print()
            print(f"Model: {self.model_name}")
            print(f"Location: {self.base_dir}")
            return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase1 모델 설정 통합 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Meta MTP 전체 설정
  python scripts/setup_models.py --model meta-llama-mtp --steps all --create-micro

  # Sheared-LLaMA 변환만
  python scripts/setup_models.py --model ref-sheared-llama --steps convert,verify

  # 검증만
  python scripts/setup_models.py --model meta-llama-mtp --steps verify
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to setup",
    )

    parser.add_argument(
        "--steps",
        default="all",
        help="Steps to run (comma-separated): download,convert,config,verify,all",
    )

    parser.add_argument(
        "--create-micro",
        action="store_true",
        help="Create micro model after conversion",
    )

    parser.add_argument(
        "--micro-type",
        choices=["mtp", "reference"],
        default="mtp",
        help="Type of micro model to create (only with --create-micro)",
    )

    args = parser.parse_args()

    # Steps 파싱
    if args.steps == "all":
        steps = ["download", "convert", "config", "verify"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    # ModelSetup 실행
    setup = ModelSetup(args.model)

    try:
        if "download" in steps:
            setup.download()

        if "convert" in steps:
            setup.convert()

        if "config" in steps:
            setup.sync_config()

        if args.create_micro:
            setup.create_micro(args.micro_type)

        if "verify" in steps:
            success = setup.verify()
            sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
