#!/bin/bash
CONTAINER_NAME=kvoicewalk
IMAGE_NAME=kvoicewalk:latest

# Always run from project root
cd "$(dirname "$0")/../.."  # Go from scripts/docker → root (~/kvoicewalk)

docker build -t $IMAGE_NAME .

docker run -it --rm \
  --gpus all \
  --name $CONTAINER_NAME \
  -v "$(pwd)":/app \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /app \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $IMAGE_NAME bash
