@echo off
set CONTAINER_NAME=kvoicewalk
set IMAGE_NAME=kvoicewalk:latest

REM Move to project root (.. .. from scripts\docker)
cd /d %~dp0\..\..

echo Building Docker image...
docker build -t %IMAGE_NAME% .

echo Starting container...
docker run -it --rm ^
  --gpus all ^
  --name %CONTAINER_NAME% ^
  -v %cd%:/app ^
  -v %USERPROFILE%\.cache\torch:/root/.cache/torch ^
  -v %USERPROFILE%\.cache\huggingface:/root/.cache/huggingface ^
  -w /app ^
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ^
  %IMAGE_NAME% ^
  bash
