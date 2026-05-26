#!/usr/bin/env python3
"""Start a persistent vLLM-style inference server via VERL.

Exposes OpenAI-compatible HTTP API (same as `vllm serve`):
  - POST /v1/chat/completions
  - POST /v1/completions
  - etc.

Uses standalone mode: vLLM loads model directly from disk.
For Qwen3.5-VL with DTensor issues, use text-only export:
  python scripts/merge_fsdp_to_hf.py --ckpt <ckpt> --output models/xxx_text_only --text-only

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/run_verl_inference_server.py \
      actor_rollout_ref.model.path=models/qwen35_9b_grpo_step105 \
      trainer.n_gpus_per_node=4

  # Then connect:
  curl http://<server_address>/v1/chat/completions \
    -H "Authorization: Bearer token-abc123" \
    -H "Content-Type: application/json" \
    -d '{"model":"models/qwen35_9b_grpo_step105","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
"""

import asyncio
import os
import signal
import sys

import hydra
import ray
from omegaconf import OmegaConf

# Add verl to path if running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


async def run_server(config):
    from verl.workers.rollout.replica import get_rollout_replica_class

    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model

    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_addresses = [server._server_address for server in rollout_servers]
    return server_addresses, rollout_servers


@hydra.main(config_path="../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
            }
        }
    )
    OmegaConf.resolve(config)

    # Standalone mode: load_format dummy gets overridden to auto (no FSDP to sync)
    # Ensure we load from disk
    if config.actor_rollout_ref.rollout.load_format == "dummy":
        config.actor_rollout_ref.rollout.load_format = "auto"

    server_addresses, servers = asyncio.run(run_server(config))

    print("\n" + "=" * 60)
    print("VERL Inference Server is running (OpenAI-compatible API)")
    print("=" * 60)
    for i, addr in enumerate(server_addresses):
        print(f"  Replica {i}: http://{addr}")
    print("\nExample request:")
    base = f"http://{server_addresses[0]}"
    model_path = config.actor_rollout_ref.model.path
    model_name = model_path.split("/")[-1] if "/" in model_path else model_path
    print(f'  curl {base}/v1/chat/completions \\')
    print(f'    -H "Authorization: Bearer token-abc123" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model":"{model_name}","messages":[{{"role":"user","content":"Hello"}}],"max_tokens":64}}\'')
    print("\nPress Ctrl+C to stop.")
    print("=" * 60 + "\n")

    # Keep running until interrupted
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
