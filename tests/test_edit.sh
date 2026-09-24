#!/usr/bin/env bash
set -e

PORT=${1:-13005}
INPUT_FILE="test_flux_gen.png"
OUTPUT_FILE="test_flux_edit.png"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Creating base image first with test_generate.sh..."
    ./tests/test_generate.sh "$PORT"
fi

echo "Testing FLUX Image-to-Image editing on port $PORT..."
curl -s -X POST "http://127.0.0.1:$PORT/v1/images/edits" \
  -F "image=@$INPUT_FILE" \
  -F "prompt=A glowing mystical cyberpunk neon raven in a futuristic rainy Tokyo night" \
  -F "strength=0.65" \
  -F "steps=4" \
  --output "$OUTPUT_FILE"

if [ -s "$OUTPUT_FILE" ]; then
    echo "Success! Edited image saved to $OUTPUT_FILE ($(du -h $OUTPUT_FILE | cut -f1))"
else
    echo "Error: Output file is empty or edit failed."
    exit 1
fi
