#!/usr/bin/env python3
"""Merge VERL FSDP sharded checkpoint into a single HuggingFace model directory.

After merging, you can serve the model with vLLM:
    python -m vllm.entrypoints.openai.api_server \
        --model <output_dir> --trust-remote-code

Usage:
    python scripts/merge_fsdp_to_hf.py \
        --ckpt checkpoints/verl_frontiercs_qwen35_9b/qwen35_9b_grpo/global_step_105 \
        --output models/qwen35_9b_grpo_step105
"""

import argparse
import gc
import json
import os
import shutil

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig


def get_auto_model_cls(config):
    arch = config.architectures[0]
    if "ForCausalLM" in arch:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
    elif "ForConditionalGeneration" in arch:
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture: {arch}")


def merge_fsdp_shards(ckpt_dir: str, output_dir: str, dtype: str = "bfloat16"):
    actor_dir = ckpt_dir
    if os.path.isdir(os.path.join(ckpt_dir, "actor")):
        actor_dir = os.path.join(ckpt_dir, "actor")

    fsdp_config_path = os.path.join(actor_dir, "fsdp_config.json")
    with open(fsdp_config_path) as f:
        fsdp_config = json.load(f)
    world_size = fsdp_config["world_size"]
    print(f"FSDP world_size: {world_size}")

    hf_dir = os.path.join(actor_dir, "huggingface")
    if not os.path.isdir(hf_dir):
        raise FileNotFoundError(f"huggingface config dir not found: {hf_dir}")

    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    print(f"Model architecture: {config.architectures}")

    torch_dtype = getattr(torch, dtype)
    auto_model_cls = get_auto_model_cls(config)

    # Get expected parameter shapes from the model architecture
    print("Building model skeleton to get expected parameter shapes...")
    with init_empty_weights():
        ref_model = auto_model_cls.from_config(
            config, torch_dtype=torch_dtype, trust_remote_code=True
        )
    expected_shapes = {name: tuple(p.shape) for name, p in ref_model.named_parameters()}
    del ref_model
    gc.collect()
    print(f"  Expected parameters: {len(expected_shapes)}")

    # Load all shards
    print(f"Loading {world_size} FSDP shards...")
    all_shards = []
    for rank in range(world_size):
        shard_path = os.path.join(actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        print(f"  Loading rank {rank}: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=False)
        all_shards.append(shard)

    shard_keys = list(all_shards[0].keys())
    print(f"  Keys in checkpoint: {len(shard_keys)}")
    print(f"  Sample key: {shard_keys[0]}")

    # Merge shards using expected shapes to determine sharding strategy
    print("Merging shards...")
    full_state_dict = {}
    replicated_count = 0
    sharded_count = 0
    unknown_count = 0

    for key in shard_keys:
        shard_tensors = [s[key] for s in all_shards]
        shard_shape = tuple(shard_tensors[0].shape)

        if key in expected_shapes:
            expected = expected_shapes[key]
            if shard_shape == expected:
                # Replicated parameter (e.g., layer norm, bias) - keep rank 0
                full_state_dict[key] = shard_tensors[0].to(torch_dtype).contiguous()
                replicated_count += 1
            elif len(shard_shape) > 0 and shard_shape[0] * world_size == expected[0] and shard_shape[1:] == expected[1:]:
                # Sharded along dim 0 - concatenate all ranks
                full_state_dict[key] = torch.cat(shard_tensors, dim=0).to(torch_dtype).contiguous()
                sharded_count += 1
            else:
                print(f"  WARNING: shape mismatch for {key}: shard={shard_shape}, expected={expected}")
                full_state_dict[key] = shard_tensors[0].to(torch_dtype).contiguous()
                unknown_count += 1
        else:
            # Key not in reference model - keep rank 0
            t = shard_tensors[0]
            full_state_dict[key] = (t.to(torch_dtype) if t.is_floating_point() else t).contiguous()
            unknown_count += 1

    del all_shards
    gc.collect()

    print(f"  Replicated: {replicated_count}, Sharded: {sharded_count}, Other: {unknown_count}")

    # Remap VERL keys to HuggingFace Qwen3_5 structure (model.language_model.X -> model.language_model.model.X)
    remapped = {}
    for key, value in full_state_dict.items():
        if key.startswith("model.language_model.") and not key.startswith("model.language_model.model."):
            new_key = "model.language_model.model." + key[len("model.language_model."):]
            remapped[new_key] = value
        else:
            remapped[key] = value
    full_state_dict = remapped
    print(f"  Remapped {sum(1 for k in full_state_dict if 'language_model.model' in k)} keys for HF compatibility")

    # Verify shapes match expected
    mismatches = []
    for key in full_state_dict:
        if key in expected_shapes:
            actual = tuple(full_state_dict[key].shape)
            if actual != expected_shapes[key]:
                mismatches.append((key, actual, expected_shapes[key]))
    if mismatches:
        print(f"\n  WARNING: {len(mismatches)} shape mismatches after merge:")
        for key, actual, expected in mismatches[:10]:
            print(f"    {key}: got {actual}, expected {expected}")
    else:
        print("  All merged shapes match expected shapes!")

    # Save as HuggingFace model
    os.makedirs(output_dir, exist_ok=True)

    # Copy tokenizer and config files
    for fname in os.listdir(hf_dir):
        src = os.path.join(hf_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

    # Save directly with torch.save (bypass save_pretrained and safetensors entirely)
    print("Saving model weights with torch.save...")
    shard_size = 5 * 1024 * 1024 * 1024  # 5GB per shard
    sorted_keys = sorted(full_state_dict.keys())
    current_shard = {}
    current_size = 0
    shard_idx = 0
    weight_map = {}

    for key in sorted_keys:
        tensor = full_state_dict[key]
        tensor_bytes = tensor.numel() * tensor.element_size()

        if current_size + tensor_bytes > shard_size and current_shard:
            shard_name = f"pytorch_model-{shard_idx + 1:05d}.bin"
            torch.save(current_shard, os.path.join(output_dir, shard_name))
            print(f"  Saved {shard_name} ({current_size / 1e9:.2f} GB, {len(current_shard)} params)")
            shard_idx += 1
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_bytes

    if current_shard:
        if shard_idx == 0:
            # Single file
            shard_name = "pytorch_model.bin"
        else:
            shard_name = f"pytorch_model-{shard_idx + 1:05d}.bin"
        torch.save(current_shard, os.path.join(output_dir, shard_name))
        print(f"  Saved {shard_name} ({current_size / 1e9:.2f} GB, {len(current_shard)} params)")
        shard_idx += 1

    # Write index file if sharded
    if shard_idx > 1:
        # Rebuild weight_map with correct shard names
        cur_shard = {}
        cur_size = 0
        s_idx = 0
        for key in sorted_keys:
            tensor = full_state_dict[key]
            tensor_bytes = tensor.numel() * tensor.element_size()
            if cur_size + tensor_bytes > shard_size and cur_shard:
                s_idx += 1
                cur_shard = {}
                cur_size = 0
            shard_file = f"pytorch_model-{s_idx + 1:05d}.bin"
            weight_map[key] = shard_file
            cur_shard[key] = True
            cur_size += tensor_bytes

        total_bytes = sum(t.numel() * t.element_size() for t in full_state_dict.values())
        index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
        with open(os.path.join(output_dir, "pytorch_model.bin.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    del full_state_dict
    gc.collect()

    model_files = [f for f in os.listdir(output_dir) if f.endswith('.bin')]
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in model_files)

    print(f"\nDone! HuggingFace model saved to: {output_dir}")
    print(f"  Files: {len(model_files)}, Total size: {total_size / 1e9:.2f} GB")
    print(f"\nTo serve with vLLM:")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"      --model {os.path.abspath(output_dir)} \\")
    print(f"      --trust-remote-code")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge VERL FSDP checkpoint to HuggingFace format")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint dir (e.g. global_step_105)")
    parser.add_argument("--output", required=True, help="Output directory for HuggingFace model")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()
    merge_fsdp_shards(args.ckpt, args.output, args.dtype)
