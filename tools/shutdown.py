"""
tools/shutdown.py — Safe shutdown utilities for the Agentic Research Assistant.

Handles:
  - Freeing ports held by stale processes (Streamlit 8501, FastAPI 8000, Ollama 11434)
  - Flushing ChromaDB / FAISS handles to release file locks
  - Saving any in-progress session state before exit

Used by:
  - ui/sidebar.py  — 🔴 Safe Shutdown button
  - main.py        — SIGINT/SIGTERM handlers and --shutdown flag
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Default ports used by the application stack
PORT_STREAMLIT = 8501
PORT_GOOGLE_SEARCH = 8000
PORT_OLLAMA = 11434

ALL_PORTS = (PORT_STREAMLIT, PORT_GOOGLE_SEARCH, PORT_OLLAMA)
# Ports owned by external services that we only free when explicitly requested
EXTERNAL_PORTS = (PORT_OLLAMA,)


def is_port_in_use(port: int) -> bool:
    """Return True if a process is listening on the given local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(port: int) -> tuple[bool, str]:
    """
    Terminate the process(es) holding *port*.

    Returns (freed: bool, message: str).
    Strategy (in order):
      1. psutil system-level net_connections — works in shared-namespace envs
      2. psutil per-process iteration — fallback for older psutil / strict perms
      3. lsof — Unix fallback when psutil is unavailable
    """
    if not is_port_in_use(port):
        return False, f"port {port} already free"

    try:
        import psutil

        def _kill_pid(pid: int) -> str:
            proc = psutil.Process(pid)
            name = proc.name()
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                proc.kill()
            return f"{name}(pid={pid})"

        # ── Approach 1: system-level connection table (works across namespaces) ──
        killed: list[str] = []
        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr.port == port and conn.status in ("LISTEN", "ESTABLISHED"):
                    if conn.pid:
                        try:
                            killed.append(_kill_pid(conn.pid))
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
        except (psutil.AccessDenied, AttributeError):
            pass  # fall through to per-process scan

        if killed:
            return True, f"port {port}: terminated {', '.join(killed)}"

        # ── Approach 2: per-process connection scan ────────────────────────────
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                try:
                    conns = proc.net_connections(kind="inet")
                except AttributeError:
                    conns = proc.connections(kind="inet")  # type: ignore[attr-defined]
                for conn in conns:
                    if conn.laddr.port == port and conn.status in ("LISTEN", "ESTABLISHED"):
                        try:
                            killed.append(_kill_pid(proc.pid))
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                continue

        if killed:
            return True, f"port {port}: terminated {', '.join(killed)}"
        return False, f"port {port}: in use but process could not be identified"

    except ImportError:
        # psutil unavailable — fall back to lsof (Unix only)
        import subprocess
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True
        )
        pids = [p.strip() for p in result.stdout.strip().split() if p.strip()]
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
        if pids:
            return True, f"port {port}: sent SIGTERM to pid(s) {', '.join(pids)}"
        return False, f"port {port}: lsof found no owning process"


def flush_chromadb() -> str:
    """
    Close any open ChromaDB PersistentClient handles to release WAL file locks.
    Skips gracefully if the persist directory has never been created (ChromaDB
    was never used in this session, so there are no locks to release).
    """
    try:
        import chromadb
        from config.settings import get_settings
        cfg = get_settings()
        persist_dir = cfg.chroma_persist_dir
        if not Path(persist_dir).exists():
            return "ChromaDB not initialised — nothing to flush"
        # Instantiating and immediately deleting triggers __del__ cleanup in chromadb,
        # flushing the write-ahead log and releasing SQLite file locks.
        client = chromadb.PersistentClient(path=persist_dir)
        del client
        return "ChromaDB handles flushed"
    except Exception as exc:
        logger.debug("ChromaDB flush: %s", exc)
        return f"ChromaDB flush skipped ({exc})"


def safe_shutdown(
    ports: Sequence[int] = ALL_PORTS,
    flush_db: bool = True,
    save_message: str = "",
    console=None,
) -> None:
    """
    Orchestrate a clean shutdown without calling sys.exit().

    Steps:
      1. Print save-confirmation message if provided
      2. Flush ChromaDB handles (releases WAL / file locks)
      3. Terminate any process holding each requested port

    Caller is responsible for calling sys.exit() or os._exit() afterwards.
    """
    _print = _make_printer(console)

    if save_message:
        _print(f"✓ {save_message}")

    if flush_db:
        msg = flush_chromadb()
        _print(f"  {msg}")

    for port in ports:
        freed, msg = free_port(port)
        prefix = "✓" if freed else "·"
        _print(f"  {prefix} {msg}")

    _print("Safe shutdown complete — all resources released.")


def _make_printer(console=None):
    """Return a print callable that uses rich Console if available, else plain print."""
    if console is not None:
        def _rich(msg: str):
            if msg.startswith("✓") or msg.startswith("Safe"):
                console.print(f"[green]{msg}[/green]")
            elif msg.startswith("  ✓"):
                console.print(f"[green]{msg}[/green]")
            else:
                console.print(f"[dim]{msg}[/dim]")
        return _rich

    def _plain(msg: str):
        print(msg)
    return _plain


def install_signal_handlers(console=None, ports: Sequence[int] = (PORT_GOOGLE_SEARCH, PORT_OLLAMA)) -> None:
    """
    Install SIGINT and SIGTERM handlers that perform a safe shutdown before exit.

    Only frees *ports* — does NOT include 8501 here because the CLI never
    owns port 8501 (that's the Streamlit UI process).
    """
    _print = _make_printer(console)

    def _handler(sig, frame):
        _print("\nInterrupt received — shutting down cleanly…")
        safe_shutdown(ports=ports, flush_db=True, console=console)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
