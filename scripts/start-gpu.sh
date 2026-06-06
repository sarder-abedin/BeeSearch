#!/usr/bin/env bash
# scripts/start-gpu.sh
# Start the research assistant with GPU (NVIDIA CUDA) support.
#
# Usage:
#   ./scripts/start-gpu.sh           # standard start
#   ./scripts/start-gpu.sh --build   # force rebuild images
#
# On Ctrl-C or normal exit the script automatically runs:
#   docker compose -f docker-compose.gpu.yml down --remove-orphans
#
# Prerequisites:
#   NVIDIA Container Toolkit installed and nvidia-ctk configured.

set -euo pipefail

COMPOSE_FILE="docker-compose.gpu.yml"

_cleanup() {
    echo ""
    echo "Shutting down…"
    docker compose -f "$COMPOSE_FILE" down --remove-orphans
}

trap _cleanup EXIT INT TERM

docker compose -f "$COMPOSE_FILE" up --build "$@"
