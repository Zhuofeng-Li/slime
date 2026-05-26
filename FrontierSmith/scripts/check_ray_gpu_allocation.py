#!/usr/bin/env python3
"""
Check Ray GPU allocation for VERL training.
Run before or during training to verify GPU assignment.

Usage:
  python scripts/check_ray_gpu_allocation.py
  python scripts/check_ray_gpu_allocation.py --n-gpus 4 --tp 1 --dp 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gpus", type=int, default=4, help="trainer.n_gpus_per_node")
    parser.add_argument("--tp", type=int, default=1, help="rollout tensor_model_parallel_size")
    parser.add_argument("--dp", type=int, default=4, help="rollout data_parallel_size")
    parser.add_argument("--nnodes", type=int, default=1)
    args = parser.parse_args()

    print("=" * 60)
    print("VERL GPU Allocation Check")
    print("=" * 60)

    # 1. Environment
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set (Ray uses all)")
    print(f"\n1. CUDA_VISIBLE_DEVICES: {cuda_devices}")
    if cuda_devices != "not set (Ray uses all)":
        n_visible = len(cuda_devices.split(","))
        print(f"   -> {n_visible} GPUs visible to this process")
        if n_visible < args.n_gpus:
            print(f"   WARNING: Only {n_visible} GPUs visible but n_gpus_per_node={args.n_gpus}")

    ray_noset = os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "0")
    print(f"   RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES: {ray_noset}")
    if ray_noset != "1":
        print("   WARNING: Should be 1 so Ray workers inherit CUDA_VISIBLE_DEVICES")

    # 2. Ray cluster (if running)
    try:
        import ray
        if ray.is_initialized():
            resources = ray.available_resources()
            gpu_total = resources.get("GPU", 0)
            print(f"\n2. Ray cluster (initialized):")
            print(f"   Available GPUs: {gpu_total}")
            node_resources = ray._private.state.available_resources_per_node()
            for node_id, res in node_resources.items():
                gpu = res.get("GPU", 0)
                print(f"   Node {node_id[:12]}...: {gpu} GPUs")
        else:
            print("\n2. Ray cluster: not initialized (run training to see allocation)")
    except ImportError:
        print("\n2. Ray: not imported")
    except Exception as e:
        print(f"\n2. Ray check failed: {e}")

    # 3. VERL architecture (colocated)
    world_size = args.tp * args.dp
    total_gpus = args.n_gpus * args.nnodes
    print(f"\n3. VERL resource pool (colocated mode):")
    print(f"   global_pool: [{args.n_gpus}] * {args.nnodes} = {total_gpus} GPUs")
    print(f"   All roles (Actor, Ref, vLLM rollout) share these {total_gpus} GPUs via time-switching")

    print(f"\n4. Rollout topology (TP={args.tp}, DP={args.dp}):")
    print(f"   world_size = TP * DP = {world_size}")
    if world_size != total_gpus:
        print(f"   WARNING: world_size ({world_size}) != total GPUs ({total_gpus})")
    else:
        print(f"   OK: {world_size} vLLM workers, 1 per GPU")

    # 5. Memory estimate
    print(f"\n5. Memory (rough estimate for Qwen3.5-27B bf16):")
    model_gb = 54  # 27B * 2 bytes
    if args.tp == 1:
        per_vllm_gpu = model_gb
        print(f"   TP=1: each vLLM instance = full model ~{model_gb}GB per GPU")
    else:
        per_vllm_gpu = model_gb / args.tp
        print(f"   TP={args.tp}: each vLLM shard ~{per_vllm_gpu:.0f}GB per GPU")
    print(f"   With gpu_memory_utilization=0.6, vLLM needs ~{per_vllm_gpu/0.6:.0f}GB physical per instance")
    print(f"   B200 has 192GB -> TP=1 (full model) fits; smaller GPUs may OOM")

    # 6. nvidia-smi if available
    print("\n6. Current GPU usage (nvidia-smi):")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            for line in out.stdout.strip().split("\n"):
                print(f"   {line}")
        else:
            print("   nvidia-smi failed")
    except FileNotFoundError:
        print("   nvidia-smi not found")
    except Exception as e:
        print(f"   {e}")

    print("\n" + "=" * 60)
    print("Summary: vLLM and FSDP do NOT overlap - they share the same GPUs")
    print("by design (colocated). Overlap would mean double allocation.")
    print("If OOM with TP=1/DP=4, try TP=2/DP=2 to reduce per-GPU memory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
