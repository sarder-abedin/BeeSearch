#!/usr/bin/env bash
# scripts/start-mac.sh
# Start BeeSearch on Apple Silicon (M1/M2/M3) using native Ollama.
#
# Usage:
#   ./scripts/start-mac.sh           # standard start
#   ./scripts/start-mac.sh --build   # force rebuild images
#
# The browser opens automatically at http://localhost:8501 once the app
# passes its health-check.  On Ctrl-C the script shuts the containers down.
#
# Prerequisites:
#   1. Install Ollama: https://ollama.com/download (macOS .dmg)
#   2. ollama pull llama3.2:3b   (or whichever model is configured)
#   Ollama starts automatically on login after installation.

set -euo pipefail

COMPOSE_FILE="docker-compose.mac.yml"
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
            open "$APP_URL"
            exit 0
        fi
        sleep 2
    done
    echo "App did not become ready within 180 s — open $APP_URL manually."
) &

docker compose -f "$COMPOSE_FILE" up --build "$@"
