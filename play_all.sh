#!/bin/bash

for f in *.wav; do
  echo "Playing: $f"
  afplay "$f"
done
