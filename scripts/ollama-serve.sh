#!/usr/bin/env bash
# scripts/ollama-serve.sh
# Drop-in replacement for `ollama serve`.
# If port 11434 is already occupied, kills the occupying process
# silently before starting Ollama — works on macOS and Linux.

set -euo pipefail

PORT=11434

_kill_port() {
  local pids
  if [[ "$(uname)" == "Darwin" ]]; then
    # macOS: lsof is always available
    pids=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
  else
    # Linux: try ss first, fall back to lsof
    pids=$(ss -tlnp 2>/dev/null \
      | awk -F'[,=]' "/\":$PORT\"/{for(i=1;i<=NF;i++) if(\$i==\"pid\") print \$(i+1)}" \
      || lsof -ti tcp:"$PORT" 2>/dev/null \
      || true)
  fi

  if [[ -n "$pids" ]]; then
    echo "[ollama-serve] Port $PORT in use (PID: $pids) — killing silently…" >&2
    echo "$pids" | xargs kill -9 2>/dev/null || true
    # Wait until the port is actually free (max 5 s)
    local waited=0
    while lsof -ti tcp:"$PORT" >/dev/null 2>&1 || \
          ss -tlnp 2>/dev/null | grep -q ":$PORT "; do
      sleep 0.5
      waited=$((waited + 1))
      [[ $waited -ge 10 ]] && { echo "[ollama-serve] Timed out waiting for port $PORT to free." >&2; exit 1; }
    done
    echo "[ollama-serve] Port $PORT free — starting Ollama." >&2
  fi
}

_kill_port
exec ollama serve "$@"
