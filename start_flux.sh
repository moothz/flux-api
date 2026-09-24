#!/usr/bin/env bash
set -e
cd /mnt/data/services/flux-api

cleanup() {
    echo "Stopping flux-api container..."
    docker compose stop -t 3
}
trap cleanup SIGINT SIGTERM EXIT

echo "Starting flux-api container..."
docker compose up
