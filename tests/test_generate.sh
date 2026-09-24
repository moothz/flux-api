#!/usr/bin/env bash
set -e

PORT=${1:-13005}
OUTPUT_FILE="test_flux_gen.png"

echo "Testing FLUX Text-to-Image generation on port $PORT..."
curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cinematic photograph of a majestic raven perched on a mossy ancient tree in a misty forest, highly detailed, 8k",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "seed": 42
  }' \
  --output "$OUTPUT_FILE"

if [ -s "$OUTPUT_FILE" ]; then
    echo "Success! Image saved to $OUTPUT_FILE ($(du -h $OUTPUT_FILE | cut -f1))"
else
    echo "Error: Output file is empty or generation failed."
    exit 1
fi
