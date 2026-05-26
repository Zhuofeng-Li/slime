#!/usr/bin/env python3
"""Run inference with the merged GRPO checkpoint using HuggingFace Transformers.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/run_merged_model.py \
        --model models/qwen35_9b_grpo_step105 \
        --prompt "Write a function to add two numbers"

    # Interactive
    CUDA_VISIBLE_DEVICES=5 python scripts/run_merged_model.py \
        --model models/qwen35_9b_grpo_step105 --interactive
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to merged model dir")
    parser.add_argument("--prompt", default=None, help="Single prompt (optional)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    # AutoModel picks correct class from config (Qwen3_5ForConditionalGeneration)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(response)
        return

    if args.interactive:
        print("Interactive mode. Type your prompt, empty line to generate, 'quit' to exit.")
        buffer = []
        while True:
            try:
                line = input(">>> " if not buffer else "... ")
            except EOFError:
                break
            if line.strip().lower() == "quit":
                break
            if not line.strip():
                if buffer:
                    prompt = "\n".join(buffer)
                    messages = [{"role": "user", "content": prompt}]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            do_sample=args.do_sample,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    response = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    print(response)
                    print()
                buffer = []
            else:
                buffer.append(line)
        return

    print("Provide --prompt or --interactive")
    sys.exit(1)


if __name__ == "__main__":
    main()
