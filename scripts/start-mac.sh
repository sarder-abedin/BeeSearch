#!/usr/bin/env bash
# scripts/start-mac.sh
# Start the research assistant on Apple Silicon (M1/M2/M3) using native Ollama.
#
# Usage:
#   ./scripts/start-mac.sh           # standard start
#   ./scripts/start-mac.sh --build   # force rebuild images
#
# On Ctrl-C or normal exit the script automatically runs:
#   docker compose -f docker-compose.mac.yml down --remove-orphans
#
# Prerequisites:
#   1. Install Ollama: https://ollama.com/download (macOS .dmg)
#   2. ollama pull llama3.2:3b   (or whichever model is configured)
#   Ollama starts automatically on login after installation.

set -euo pipefail

COMPOSE_FILE="docker-compose.mac.yml"

_cleanup() {
    echo ""
    echo "Shutting down…"
    docker compose -f "$COMPOSE_FILE" down --remove-orphans
}

trap _cleanup EXIT INT TERM

docker compose -f "$COMPOSE_FILE" up --build "$@"
