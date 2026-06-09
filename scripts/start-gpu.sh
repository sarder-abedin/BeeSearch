#!/usr/bin/env bash
# scripts/start-gpu.sh
# Start BeeSearch with GPU (NVIDIA CUDA) support.
#
# Usage:
#   ./scripts/start-gpu.sh           # standard start
#   ./scripts/start-gpu.sh --build   # force rebuild images
#
# The browser opens automatically at http://localhost:8501 once the app
# passes its health-check.  On Ctrl-C the script shuts the containers down.
#
# Prerequisites:
#   NVIDIA Container Toolkit installed and nvidia-ctk configured.

set -euo pipefail

COMPOSE_FILE="docker-compose.gpu.yml"
APP_URL="http://localhost:${APP_PORT:-8501}"

_cleanup() {
    echo ""
    echo "Shutting down…"
    docker compose -f "$COMPOSE_FILE" down --remove-orphans
}

trap _cleanup EXIT INT TERM

# Poll the health endpoint in the background, then open the browser.
(
    echo "Waiting for BeeSearch to be ready at $APP_URL …"
    for i in $(seq 1 90); do
        if curl -sf "${APP_URL}/_stcore/health" >/dev/null 2>&1; then
            echo ""
            echo "BeeSearch is ready — opening $APP_URL"
            if command -v xdg-open >/dev/null 2>&1; then
                xdg-open "$APP_URL"
            elif command -v open >/dev/null 2>&1; then
                open "$APP_URL"
            else
                echo "Open your browser at: $APP_URL"
            fi
            exit 0
        fi
        sleep 2
    done
    echo "App did not become ready within 180 s — open $APP_URL manually."
) &

docker compose -f "$COMPOSE_FILE" up --build "$@"
