#!/bin/bash

# Install uv if not present
if ! command -v uv &> /dev/null; then
  echo "uv not found — installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Add to PATH if installed in default location
  export PATH="$HOME/.cargo/bin:$PATH"

  if ! command -v uv &> /dev/null; then
    echo "uv installation failed or not in PATH"
    exit 1
  fi
fi

echo "uv is installed and ready"

uv run main.py --target_text "The old lighthouse keeper never imagined that one day he'd be guiding ships from the comfort of his living room, but with modern technology and an array of cameras, he did just that, sipping tea while the storm raged outside and gulls shrieked overhead." --target_audio ./example/target.wav
