#!/bin/bash

################################################################################
# play_all.sh
#
# Description:
# This script plays all `.wav` audio files in the `out/` directory, sorted by
# creation time (oldest to newest). It keeps track of the last played file
# across invocations using a `.last_played` file and resumes from the next file
# automatically. It is cross-platform and works on macOS and most Linux systems.
#
# Supported audio players:
#   - macOS: afplay
#   - Linux: aplay, paplay, or ffplay (auto-detected)
#
# Requirements:
# - `ls` must support the `-tU` option (standard on GNU and macOS)
# - `tail -r` (macOS) or `tac` (Linux)
#
# Usage:
#   ./play_all.sh
#
# Author: Lombe
################################################################################

cd "out" || {
  echo "Directory 'out/' does not exist. Exiting."
  exit 1
}

LAST_PLAYED_FILE=".last_played"  # Tracks the last played file to resume later

# === Detect audio playback command based on platform ===
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: use afplay, reverse list with tail -r
    PLAY_CMD="afplay"
    REVERSE_CMD="tail -r"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux: try to find a suitable audio player
    if command -v aplay &> /dev/null; then
        PLAY_CMD="aplay"
    elif command -v paplay &> /dev/null; then
        PLAY_CMD="paplay"
    elif command -v ffplay &> /dev/null; then
        PLAY_CMD="ffplay -autoexit -nodisp"
    else
        echo "No supported audio player found (aplay, paplay, or ffplay required)."
        exit 1
    fi
    REVERSE_CMD="tac"
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

# === Get sorted list of WAV files ===
# -tU sorts by creation time (oldest to newest); we reverse to get oldest first
FILES=($(ls -tU *.wav | $REVERSE_CMD))

# === Resume playback from last played file if available ===
START_INDEX=0
if [[ -f "$LAST_PLAYED_FILE" ]]; then
    LAST_PLAYED=$(<"$LAST_PLAYED_FILE")
    for i in "${!FILES[@]}"; do
        if [[ "${FILES[$i]}" == "$LAST_PLAYED" ]]; then
            START_INDEX=$((i + 1))  # Start from the next file
            break
        fi
    done
fi

# === Playback loop ===
for ((i=START_INDEX; i<${#FILES[@]}; i++)); do
    FILE="${FILES[$i]}"
    echo "Playing: $FILE"
    $PLAY_CMD "$FILE"

    # Save progress
    echo "$FILE" > "$LAST_PLAYED_FILE"
done
