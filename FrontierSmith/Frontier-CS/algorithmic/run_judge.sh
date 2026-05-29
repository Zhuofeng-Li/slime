#!/bin/bash
# Run judge container without docker compose.
# Usage: ./run_judge.sh
# Stop: docker stop Competitive-Programming

set -e
cd "$(dirname "$0")"

# Build image
# echo "Building image..."
# docker build -t frontiercs-judge .

# Stop existing container if any
sudo docker rm -f Competitive-Programming 2>/dev/null || true

# Run
echo "Starting container..."
sudo docker run -d \
  --name Competitive-Programming \
  --privileged \
  --shm-size=4g \
  -p 8082:8082 \
  -e PORT=8082 \
  -e GJ_ADDR=http://127.0.0.1:5050 \
  -e JUDGE_WORKERS=32 \
  -e GJ_PARALLELISM=32 \
  -e SAVE_OUTPUTS=false \
  -v "$(pwd)/judge/src:/app/src" \
  -v "$(pwd)/problems:/app/problems" \
  -v "$(pwd)/submissions:/app/submissions" \
  -v "$(pwd)/data:/app/data" \
  frontiercs-judge

echo "Judge running at http://localhost:8082"
echo "Check: curl http://localhost:8082/health"
