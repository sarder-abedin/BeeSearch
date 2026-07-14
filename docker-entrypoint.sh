#!/bin/bash
# docker-entrypoint.sh
#
# Primary:   FastAPI backend + React SPA (port 8000) — built React assets
#            are served by FastAPI at "/" (see backend/app/main.py).
# Secondary: Streamlit UI (port 8501) — still available for power users.
# CLI:       Not started here — run ad hoc with
#            `docker compose exec app python main.py ...`.
#
# If either server process exits, this script exits too, so Docker's
# restart policy brings the whole container back up rather than leaving
# one surface silently dead while the other keeps running.

python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

streamlit run app.py &
STREAMLIT_PID=$!

_shutdown() {
    kill -TERM "$UVICORN_PID" "$STREAMLIT_PID" 2>/dev/null
    wait "$UVICORN_PID" "$STREAMLIT_PID" 2>/dev/null
    exit 0
}
trap _shutdown TERM INT

wait -n "$UVICORN_PID" "$STREAMLIT_PID"
EXIT_CODE=$?
kill -TERM "$UVICORN_PID" "$STREAMLIT_PID" 2>/dev/null
exit "$EXIT_CODE"
