#!/usr/bin/env bash
# scripts/start.sh
# Start the research assistant (standard — Linux / Docker Desktop).
#
# Usage:
#   ./scripts/start.sh           # standard start
#   ./scripts/start.sh --build   # force rebuild images
#
# On Ctrl-C or normal exit the script automatically runs:
#   docker compose down --remove-orphans

set -euo pipefail

COMPOSE_FILE="docker-compose.yml"

_cleanup() {
    echo ""
    echo "Shutting down…"
    docker compose -f "$COMPOSE_FILE" down --remove-orphans
}

trap _cleanup EXIT INT TERM

docker compose -f "$COMPOSE_FILE" up --build "$@"
