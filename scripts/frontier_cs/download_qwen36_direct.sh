#!/usr/bin/env bash
set -euo pipefail

repo="Qwen/Qwen3.6-35B-A3B"
out_dir="${1:-models/Qwen3.6-35B-A3B}"

mkdir -p "${out_dir}"

download_one() {
  local file="$1"
  local url="https://huggingface.co/${repo}/resolve/main/${file}"
  echo "[download] ${file}"
  wget -c -nv \
    --retry-connrefused \
    --waitretry=5 \
    --read-timeout=60 \
    --timeout=60 \
    -t 0 \
    -O "${out_dir}/${file}" \
    "${url}"
}

small_files=(
  ".gitattributes"
  "LICENSE"
  "README.md"
  "chat_template.jinja"
  "config.json"
  "configuration.json"
  "generation_config.json"
  "merges.txt"
  "model.safetensors.index.json"
  "preprocessor_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "video_preprocessor_config.json"
  "vocab.json"
)

for file in "${small_files[@]}"; do
  download_one "${file}"
done

for i in $(seq 1 26); do
  shard="$(printf "%05d" "${i}")"
  download_one "model-${shard}-of-00026.safetensors"
done

echo "[download] complete: ${out_dir}"
