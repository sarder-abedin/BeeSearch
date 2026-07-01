#!/usr/bin/env bash
# scripts/start-web.sh
# Start the BeeSearch React + FastAPI web app using Docker.
#
# Usage:
#   ./scripts/start-web.sh           # standard start (builds on first run)
#   ./scripts/start-web.sh --build   # force a full image rebuild
#
# The browser opens automatically at http://localhost:8000 once the app
# passes its health-check.  Press Ctrl-C to shut the containers down.
#
# All dependencies (npm install, pip install) are handled inside Docker —
# no manual setup required beyond having Docker installed.
#
# Apple Silicon Mac with native Ollama already running:
#   Add OLLAMA_BASE_URL=http://host.docker.internal:11434 to .env, then:
#   docker compose -f docker-compose.web.yml up web --build

set -euo pipefail

COMPOSE_FILE="docker-compose.web.yml"
APP_URL="http://localhost:${BACKEND_PORT:-8000}"

_cleanup() {
    echo ""
    echo "Shutting down…"
    docker compose -f "$COMPOSE_FILE" down --remove-orphans
}

trap _cleanup EXIT INT TERM

# Poll the health endpoint in the background, then open the browser.
(
    echo "Waiting for BeeSearch web app at $APP_URL …"
    for i in $(seq 1 90); do
        if curl -sf "${APP_URL}/api/health" >/dev/null 2>&1; then
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
