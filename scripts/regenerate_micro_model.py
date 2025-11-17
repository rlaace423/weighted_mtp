"""Micro 모델을 Pure PyTorch 구조로 재생성

Pure PyTorch Transformer 구조 (layers + extra_heads 분리)로
Micro 모델 checkpoint를 재생성합니다.

Usage:
    python scripts/regenerate_micro_model.py
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from weighted_mtp.models.meta_mtp import ModelArgs, Transformer


def regenerate_micro_model():
    """Micro 모델을 Pure PyTorch 구조로 재생성"""

    micro_dir = Path("storage/models/micro-mtp")

    print("=" * 60)
    print("Micro 모델 Pure PyTorch 구조 재생성")
    print("=" * 60)

    # 1. config.json 읽기
    config_path = micro_dir / "configs/config.json"
    print(f"\n[1/4] Config 읽기: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    print(f"  - hidden_size: {config['hidden_size']}")
    print(f"  - num_hidden_layers: {config['num_hidden_layers']}")
    print(f"  - n_future_tokens: {config['n_future_tokens']}")

    # 2. ModelArgs 생성
    print("\n[2/4] ModelArgs 생성")
    model_args = ModelArgs(
        dim=config["hidden_size"],
        n_layers=config["num_hidden_layers"],
        n_heads=config["num_attention_heads"],
        n_kv_heads=config["num_key_value_heads"],
        vocab_size=config["vocab_size"],
        n_future_tokens=config["n_future_tokens"],
        rope_theta=config["rope_theta"],
        max_seq_len=config["max_position_embeddings"],
        norm_eps=config["rms_norm_eps"],
    )

    print(f"  - n_layers: {model_args.n_layers}")
    print(f"  - n_future_tokens: {model_args.n_future_tokens}")
    print(f"  - trunk layers: {model_args.n_layers - model_args.n_future_tokens + 1}")
    print(f"  - extra_heads: {model_args.n_future_tokens - 1}")

    # 3. Pure PyTorch Transformer 생성 (랜덤 초기화)
    print("\n[3/4] Pure PyTorch Transformer 생성")
    transformer = Transformer(model_args)

    print(f"  - layers: {len(transformer.layers)}")
    print(f"  - extra_heads: {len(transformer.extra_heads)}")

    # State dict 키 확인
    state_dict = transformer.state_dict()
    print(f"  - Total parameters: {len(state_dict)} keys")

    # Trunk layers 키
    trunk_keys = [k for k in state_dict.keys() if k.startswith("layers.")]
    print(f"  - Trunk layers keys: {len(trunk_keys)}")

    # Extra heads 키
    extra_keys = [k for k in state_dict.keys() if k.startswith("extra_heads.")]
    print(f"  - Extra heads keys: {len(extra_keys)}")

    # freqs_cis는 일반 속성이므로 state_dict에 포함되지 않음
    print(f"  - freqs_cis: 일반 속성 (state_dict 미포함, runtime에 자동 계산)")

    # 4. Safetensors 저장
    output_path = micro_dir / "safetensors/model.safetensors"
    print(f"\n[4/4] Safetensors 저장: {output_path}")

    # Backup 기존 파일
    backup_path = micro_dir / "safetensors/model.safetensors.backup"
    if output_path.exists():
        print(f"  - Backup 기존 파일: {backup_path}")
        output_path.rename(backup_path)

    # State dict 저장 (freqs_cis는 자동으로 제외됨)
    save_file(state_dict, str(output_path))

    # 파일 크기 확인
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  - 저장 완료: {file_size_mb:.2f} MB")

    # SHA256SUMS 업데이트
    import hashlib

    sha256_hash = hashlib.sha256()
    with open(output_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    sha256sum = sha256_hash.hexdigest()

    sha256_path = micro_dir / "safetensors/SHA256SUMS"
    with open(sha256_path, "w") as f:
        f.write(f"{sha256sum}  model.safetensors\n")

    print(f"  - SHA256SUM 업데이트: {sha256_path}")

    print("\n" + "=" * 60)
    print("✅ Micro 모델 재생성 완료!")
    print("=" * 60)

    # 5. 검증: 로딩 테스트
    print("\n[검증] 재생성된 모델 로딩 테스트")
    from weighted_mtp.models.meta_mtp.checkpoints import load_meta_mtp_model

    try:
        loaded_transformer = load_meta_mtp_model(micro_dir, device="cpu")
        print(f"  ✅ 로딩 성공!")
        print(f"  - dim: {loaded_transformer.params.dim}")
        print(f"  - n_layers: {loaded_transformer.params.n_layers}")
        print(f"  - layers: {len(loaded_transformer.layers)}")
        print(f"  - extra_heads: {len(loaded_transformer.extra_heads)}")

        # Forward test
        input_ids = torch.randint(0, 32000, (2, 10))
        output = loaded_transformer(input_ids, start_pos=0, return_all_heads=True)
        print(f"  - Forward 테스트: {output.shape}")

        print("\n✅ 모든 검증 통과!")

    except Exception as e:
        print(f"  ❌ 로딩 실패: {e}")
        raise


if __name__ == "__main__":
    regenerate_micro_model()
