#!/usr/bin/env bash
# scripts/start-react.sh
# Launch the BeeSearch React + FastAPI web app (no Docker required).
#
# Usage:
#   ./scripts/start-react.sh               # normal start
#   ./scripts/start-react.sh --mock        # stub LLM + search (no Ollama needed)
#   ./scripts/start-react.sh --port 9000   # custom backend port (frontend auto-proxies to it)
#   ./scripts/start-react.sh --no-open     # skip opening the browser
#
# Prerequisites (must be on PATH):
#   - Python 3.10+  with requirements installed  (pip install -r requirements.txt)
#   - Node.js 20+   with npm bundled
#   - Ollama running with at least one model pulled (unless --mock is passed)
#
# Ctrl-C cleanly shuts down both the backend and frontend processes.

set -euo pipefail

# ── Resolve the repo root (the directory containing this script's parent) ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Augment PATH for common non-interactive-shell tool locations ─────────────
# Static well-known dirs: Homebrew (Apple Silicon + Intel), user-local bin
for _d in \
    /opt/homebrew/bin \
    /opt/homebrew/sbin \
    /usr/local/bin \
    /usr/local/sbin \
    "${HOME}/.local/bin" \
    "${HOME}/bin" \
    "${HOME}/.volta/bin" \
    "${HOME}/.asdf/shims"
do
    [[ -d "$_d" ]] && PATH="${_d}:${PATH}"
done

# Homebrew versioned node formulae: /opt/homebrew/opt/node*/bin or /usr/local/opt/node*/bin
for _d in /opt/homebrew/opt/node*/bin /usr/local/opt/node*/bin; do
    [[ -d "$_d" ]] && PATH="${_d}:${PATH}"
done

# nvm — source the nvm loader and activate the default version
if [[ -z "$(command -v node 2>/dev/null)" ]]; then
    NVM_DIR="${NVM_DIR:-${HOME}/.nvm}"
    if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
        # shellcheck source=/dev/null
        source "${NVM_DIR}/nvm.sh" --no-use
        nvm use default >/dev/null 2>&1 || nvm use node >/dev/null 2>&1 || true
    fi
fi

# fnm (Fast Node Manager) — eval its env setup
if [[ -z "$(command -v node 2>/dev/null)" ]]; then
    if command -v fnm >/dev/null 2>&1; then
        eval "$(fnm env)" 2>/dev/null || true
    fi
fi

# nvm glob fallback — pick the highest installed version bin dir
if [[ -z "$(command -v node 2>/dev/null)" ]]; then
    for _d in "${HOME}/.nvm/versions/node"/*/bin; do
        [[ -d "$_d" ]] && PATH="${_d}:${PATH}"
    done
fi

export PATH

BACKEND_PORT=8000
OPEN_BROWSER=true
MOCK_LLM=false

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock)       MOCK_LLM=true; shift ;;
        --no-open)    OPEN_BROWSER=false; shift ;;
        --port)       BACKEND_PORT="$2"; shift 2 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

FRONTEND_URL="http://localhost:5173"
BACKEND_URL="http://localhost:${BACKEND_PORT}"
BACKEND_HEALTH="${BACKEND_URL}/api/health"

# ── Prerequisite checks ──────────────────────────────────────────────────────
_check() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: '$1' not found on PATH."
        echo "       $2"
        exit 1
    fi
}
_check python3  "Install Python 3.10+ and run: pip install -r requirements.txt"
_check node     "Install Node.js 20+ from https://nodejs.org  (or check that your shell profile loads nvm / Homebrew)"
_check npm      "npm is bundled with Node.js — reinstall Node."

if [[ "$MOCK_LLM" == "false" ]]; then
    if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Warning: Ollama does not appear to be running on localhost:11434."
        echo "         Start Ollama, or re-run with --mock to use the stub LLM."
        echo "         Continuing anyway…"
    fi
fi

# ── Cleanup: kill both child processes on exit ───────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""

_cleanup() {
    echo ""
    echo "Shutting down…"
    [[ -n "$BACKEND_PID" ]]  && kill "$BACKEND_PID"  2>/dev/null || true
    [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Done."
}
trap _cleanup EXIT INT TERM

# ── Install frontend deps if node_modules is absent ──────────────────────────
cd "${REPO_ROOT}/frontend"
if [[ ! -d node_modules ]]; then
    echo "Installing frontend dependencies (npm install)…"
    npm install
fi

# ── Start backend ─────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
echo ""
echo "Starting FastAPI backend on port ${BACKEND_PORT}…"

BACKEND_CMD=(
    python3 -m uvicorn backend.app.main:app
    --reload
    --port "${BACKEND_PORT}"
    --log-level warning
)

if [[ "$MOCK_LLM" == "true" ]]; then
    BEESEARCH_MOCK_LLM=1 "${BACKEND_CMD[@]}" &
else
    "${BACKEND_CMD[@]}" &
fi
BACKEND_PID=$!

# ── Start frontend ────────────────────────────────────────────────────────────
echo "Starting React dev server (Vite)…"
cd "${REPO_ROOT}/frontend"
VITE_BACKEND_PORT="${BACKEND_PORT}" npm run dev -- --port 5173 &
FRONTEND_PID=$!

# ── Wait for backend to be ready, then open browser ──────────────────────────
(
    echo ""
    echo "Waiting for backend at ${BACKEND_URL} …"
    for i in $(seq 1 60); do
        if curl -sf "${BACKEND_HEALTH}" >/dev/null 2>&1; then
            echo "Backend ready."
            echo ""
            echo "  React app  →  ${FRONTEND_URL}"
            echo "  API docs   →  ${BACKEND_URL}/docs"
            echo ""
            if [[ "$OPEN_BROWSER" == "true" ]]; then
                if command -v xdg-open >/dev/null 2>&1; then
                    xdg-open "${FRONTEND_URL}"
                elif command -v open >/dev/null 2>&1; then
                    open "${FRONTEND_URL}"
                else
                    echo "Open your browser at: ${FRONTEND_URL}"
                fi
            fi
            exit 0
        fi
        sleep 2
    done
    echo "Backend did not become ready within 120 s — open ${FRONTEND_URL} manually."
) &

# ── Keep running until Ctrl-C ────────────────────────────────────────────────
echo "Press Ctrl-C to stop both servers."
wait "$BACKEND_PID" "$FRONTEND_PID"
