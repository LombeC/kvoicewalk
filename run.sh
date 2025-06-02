#!/usr/bin/env bash

################################################################################
# run.sh
#
# Ensures `uv` is available by checking common install paths (e.g., ~/.local/bin).
# Runs KVoiceWalk using a specified target audio + prompt.
################################################################################

set -e

# === Step 1: Ensure uv is in PATH ===

# Add ~/.local/bin if it contains uv
if [[ -f "$HOME/.local/bin/uv" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

# Add ~/.cargo/bin if it contains uv (Rust install path)
if [[ -f "$HOME/.cargo/bin/uv" ]]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# === Step 2: Install if still missing ===
if ! command -v uv &> /dev/null; then
  echo "uv not found — installing via curl..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Add to PATH again (in case install just happened)
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

  if ! command -v uv &> /dev/null; then
    echo "uv installation failed or is not on PATH"
    exit 1
  fi
fi

echo "uv is ready: $(command -v uv)"

# === Step 3: Define target ===

TARGET_TEXT="The old lighthouse keeper never imagined that one day he'd be guiding ships from the comfort of his living room, but with modern technology and an array of cameras, he did just that, sipping tea while the storm raged outside and gulls shrieked overhead."
TARGET_AUDIO="example/target.wav"
STARTING_VOICE="example/94.79_0.94_31804.pt"
STEP_LIMIT=30000
MODE="anneal"
if [[ ! -f "$TARGET_AUDIO" ]]; then
  echo "Target audio not found at: $TARGET_AUDIO"
  exit 1
fi

# === Step 4: Run KVoiceWalk ===

echo "Running KVoiceWalk..."
uv run main.py --mode "$MODE" \
      --target_text "$TARGET_TEXT" \
      --target_audio "$TARGET_AUDIO" \
      --starting_voice "$STARTING_VOICE" \
      --step_limit "$STEP_LIMIT"
