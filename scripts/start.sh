#!/usr/bin/env bash
# scripts/start.sh
# Start BeeSearch (Linux / Docker Desktop on Windows).
#
# Usage:
#   ./scripts/start.sh           # standard start
#   ./scripts/start.sh --build   # force rebuild images
#
# The browser opens automatically at http://localhost:8000 (React app)
# once the app passes its health-check.  On Ctrl-C the script shuts
# the containers down.
# Streamlit is also available at http://localhost:8501.

set -euo pipefail

COMPOSE_FILE="docker-compose.yml"
APP_URL="http://localhost:${REACT_PORT:-8000}"

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
