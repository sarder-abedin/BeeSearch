"""
config/hardware.py
──────────────────
Hardware detection and model recommendation.

Detects: OS, CPU, RAM, GPU type (Apple Silicon / NVIDIA / CPU-only).
Queries Ollama for pulled models and recommends the best model + num_ctx
that fits the available memory.

Docker note
───────────
When running inside Docker on macOS the container sees the Docker VM's
memory allocation (often 6–8 GB), not the host's full unified memory.
`detect_hardware()` sets `in_docker=True` and `is_docker_on_apple_silicon`
so the UI can prompt the user to supply their actual RAM.
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Known models ordered best → smallest. Used for recommendations.
# Performance tiers keyed by RAM range.  Ordered from highest to lowest so
# get_recommended_tier() can do a first-match scan.
TIER_CONFIGS: List[Dict] = [
    {
        "tier": "maximum",
        "label": "Maximum",
        "ram_min_gb": 32,
        "description": "32 GB+ RAM — maximum context, highest quality outputs",
        "num_ctx": 65536,
        "hybrid_top_k": 15,
        "chunk_size": 1200,
        "chunk_overlap": 250,
        "max_results": 12,
        "large_doc_page_threshold": 200,
    },
    {
        "tier": "high",
        "label": "High",
        "ram_min_gb": 16,
        "description": "16–32 GB RAM — large context, high quality outputs",
        "num_ctx": 32768,
        "hybrid_top_k": 12,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_results": 10,
        "large_doc_page_threshold": 80,
    },
    {
        "tier": "standard",
        "label": "Standard",
        "ram_min_gb": 8,
        "description": "8–16 GB RAM — balanced performance",
        "num_ctx": 16384,
        "hybrid_top_k": 8,
        "chunk_size": 800,
        "chunk_overlap": 150,
        "max_results": 6,
        "large_doc_page_threshold": 50,
    },
    {
        "tier": "low",
        "label": "Low",
        "ram_min_gb": 0,
        "description": "Less than 8 GB RAM — optimised for speed and low memory",
        "num_ctx": 8192,
        "hybrid_top_k": 5,
        "chunk_size": 600,
        "chunk_overlap": 100,
        "max_results": 4,
        "large_doc_page_threshold": 20,
    },
]


def get_recommended_tier(hw: Dict) -> Dict:
    """Return the TIER_CONFIGS entry that matches the detected (or overridden) RAM.

    Args:
        hw: A hardware profile dict as returned by `detect_hardware()` (or a
            copy with `ram_gb` overridden, e.g. for the sidebar's manual RAM
            override when running in Docker).
    """
    usable = _usable_ram(hw)
    for tier in TIER_CONFIGS:  # ordered highest → lowest
        if usable >= tier["ram_min_gb"]:
            return tier
    return TIER_CONFIGS[-1]  # always return low as fallback


KNOWN_MODELS: List[Dict] = [
    # ── Quality 5 — large / reasoning ────────────────────────────────────────
    {
        "name": "qwq:32b",
        "ram_gb": 20,
        "num_ctx": 32768,
        "quality": 5,
        "label": "Alibaba QwQ 32B",
        "note": "Deep reasoning model — best for complex analysis",
    },
    {
        "name": "phi4:14b",
        "ram_gb": 14,
        "num_ctx": 16384,
        "quality": 5,
        "label": "Microsoft Phi-4 14B",
        "note": "Highest reasoning quality",
    },
    {
        "name": "qwen3:32b",
        "ram_gb": 20,
        "num_ctx": 32768,
        "quality": 5,
        "label": "Alibaba Qwen 3 32B",
        "note": "Top-tier quality with hybrid thinking mode",
    },
    # ── Quality 4 — high quality ──────────────────────────────────────────────
    {
        "name": "qwen3:14b",
        "ram_gb": 9,
        "num_ctx": 32768,
        "quality": 4,
        "label": "Alibaba Qwen 3 14B",
        "note": "Excellent quality, fits in 16 GB RAM",
    },
    {
        "name": "mistral-nemo:12b",
        "ram_gb": 12,
        "num_ctx": 131072,
        "quality": 4,
        "label": "Mistral NeMo 12B",
        "note": "Best context window (128k tokens)",
    },
    {
        "name": "qwen2.5:14b",
        "ram_gb": 9,
        "num_ctx": 32768,
        "quality": 4,
        "label": "Alibaba Qwen 2.5 14B",
        "note": "Strong multilingual, solid reasoning",
    },
    # ── Quality 3 — standard ──────────────────────────────────────────────────
    {
        "name": "qwen3:8b",
        "ram_gb": 6,
        "num_ctx": 32768,
        "quality": 3,
        "label": "Alibaba Qwen 3 8B",
        "note": "Strong all-rounder with thinking mode",
    },
    {
        "name": "gemma2:9b",
        "ram_gb": 9,
        "num_ctx": 32768,
        "quality": 3,
        "label": "Google Gemma 2 9B",
        "note": "Strong general reasoning",
    },
    {
        "name": "llama3.1:8b",
        "ram_gb": 8,
        "num_ctx": 32768,
        "quality": 3,
        "label": "Meta Llama 3.1 8B",
        "note": "Reliable all-rounder",
    },
    # ── Quality 2 — efficient ─────────────────────────────────────────────────
    {
        "name": "qwen3:4b",
        "ram_gb": 3,
        "num_ctx": 32768,
        "quality": 2,
        "label": "Alibaba Qwen 3 4B",
        "note": "Surprisingly capable for its size",
    },
    {
        "name": "qwen2.5:7b",
        "ram_gb": 7,
        "num_ctx": 32768,
        "quality": 2,
        "label": "Alibaba Qwen 2.5 7B",
        "note": "Efficient, good multilingual support",
    },
    # ── Quality 1 — minimal ───────────────────────────────────────────────────
    {
        "name": "qwen3:1.7b",
        "ram_gb": 1.5,
        "num_ctx": 32768,
        "quality": 1,
        "label": "Alibaba Qwen 3 1.7B",
        "note": "Tiny footprint, usable on very low RAM",
    },
    {
        "name": "llama3.2:3b",
        "ram_gb": 3,
        "num_ctx": 32768,
        "quality": 1,
        "label": "Meta Llama 3.2 3B",
        "note": "Fastest, lowest memory use",
    },
]


def detect_hardware() -> Dict:
    """Return a hardware profile dict for the current machine.

    Returns:
        Dict with keys: `os`, `arch`, `cpu`, `ram_gb`, `gpu_type`
        (`"apple_silicon"` | `"nvidia"` | `"amd"` | `"cpu"`),
        `is_apple_silicon`, `in_docker`, `is_docker_on_apple_silicon`.
        Consumed by `recommend_config()`, `get_recommended_tier()`, and
        `ui/sidebar.py::render_sidebar()`'s Hardware panel.

    GPU detection note: inside a Docker container the GPU devices are not
    accessible to the web service, so auto-detection always returns "cpu".
    Set the ``GPU_TYPE`` environment variable (``nvidia`` / ``amd``) in
    your compose file's web service to override this.
    """
    import os as _os
    in_docker = _is_docker()
    os_name = platform.system()
    arch = platform.machine()

    hw = {
        "os": os_name,
        "arch": arch,
        "cpu": _get_cpu_name(in_docker=in_docker, arch=arch),
        "ram_gb": _get_ram_gb(),
        "gpu_type": None,
        "is_apple_silicon": False,
        "in_docker": in_docker,
        "is_docker_on_apple_silicon": False,
    }

    # Native macOS on Apple Silicon
    native_apple = os_name == "Darwin" and arch == "arm64"
    # Docker container on Apple Silicon Mac: Linux + arm64/aarch64 + in Docker.
    # (Could also be a cloud ARM VM, but those expose their full RAM correctly
    # so the override prompt is harmless there.)
    docker_apple = in_docker and os_name == "Linux" and arch in ("arm64", "aarch64")

    hw["is_apple_silicon"] = native_apple or docker_apple
    hw["is_docker_on_apple_silicon"] = docker_apple

    # Allow Docker web containers (which can't access GPU devices) to report
    # the correct accelerator via the GPU_TYPE environment variable.
    gpu_override = _os.environ.get("GPU_TYPE", "").lower().strip()
    if gpu_override in ("nvidia", "amd", "apple_silicon", "cpu"):
        hw["gpu_type"] = gpu_override
        if gpu_override == "apple_silicon":
            hw["is_apple_silicon"] = True
    else:
        hw["gpu_type"] = _get_gpu_type(hw["is_apple_silicon"])
    return hw


_EMBED_PREFIXES = (
    "nomic-embed", "mxbai-embed", "bge-m3", "all-minilm",
    "text-embedding", "embed-", "snowflake-arctic-embed",
    "qwen3-embed",          # qwen3-embedding:0.6b / 4b / 8b
)


def get_available_models(ollama_base_url: str) -> List[str]:
    """Query Ollama for pulled chat model names, excluding embedding-only models."""
    try:
        resp = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            return [
                m["name"]
                for m in resp.json().get("models", [])
                if not any(m["name"].lower().startswith(p) for p in _EMBED_PREFIXES)
            ]
    except Exception:
        pass
    return []


# Canonical embedding models supported by the Hybrid RAG stack.
# Listed in recommended order (quality / popularity).
KNOWN_EMBED_MODELS: List[Dict] = [
    {
        "name": "nomic-embed-text",
        "dim": 768,
        "size_gb": 0.27,
        "note": "Recommended — fast, high quality (MIT licence)",
    },
    {
        "name": "mxbai-embed-large",
        "dim": 1024,
        "size_gb": 0.67,
        "note": "Highest accuracy, larger footprint",
    },
    {
        "name": "bge-m3",
        "dim": 1024,
        "size_gb": 1.2,
        "note": "Multilingual, best for non-English documents",
    },
    {
        "name": "qwen3-embedding:0.6b",
        "dim": 1024,
        "size_gb": 0.5,
        "note": "Qwen3 embedding — strong multilingual, 32k context",
    },
    {
        "name": "qwen3-embedding:4b",
        "dim": 2560,
        "size_gb": 2.6,
        "note": "Qwen3 embedding — higher accuracy, 32k context",
    },
    {
        "name": "qwen3-embedding:8b",
        "dim": 4096,
        "size_gb": 5.2,
        "note": "Qwen3 embedding — best accuracy, 32k context",
    },
    {
        "name": "all-minilm",
        "dim": 384,
        "size_gb": 0.046,
        "note": "Smallest and fastest — lower quality",
    },
]


def get_available_embed_models(ollama_base_url: str) -> List[str]:
    """Return known embedding model names that are already pulled in Ollama.

    For models with an explicit tag (e.g. qwen3-embedding:8b) requires an
    exact name match. For tag-less entries (e.g. nomic-embed-text) matches
    any pulled variant (e.g. nomic-embed-text:latest).
    """
    try:
        resp = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            pulled_full = {m["name"] for m in resp.json().get("models", [])}
            result = []
            for m in KNOWN_EMBED_MODELS:
                n = m["name"]
                if ":" in n:
                    if n in pulled_full:
                        result.append(n)
                else:
                    if any(p.split(":")[0] == n for p in pulled_full):
                        result.append(n)
            return result
    except Exception:
        pass
    return []


def get_all_pulled_embed_models(ollama_base_url: str) -> List[str]:
    """Return all pulled embedding model names, including models not in KNOWN_EMBED_MODELS."""
    try:
        resp = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            return [
                m["name"]
                for m in resp.json().get("models", [])
                if any(m["name"].lower().startswith(p) for p in _EMBED_PREFIXES)
            ]
    except Exception:
        pass
    return []


def get_model_suggestion(model_name: str) -> Optional[Dict]:
    """Return suggested settings for a known model, or None if unrecognised."""
    for m in KNOWN_MODELS:
        if m["name"] == model_name or m["name"].split(":")[0] == model_name.split(":")[0]:
            chunk_size, chunk_overlap = _chunk_settings_for_num_ctx(m["num_ctx"])
            return {
                "name": m["name"],
                "num_ctx": m["num_ctx"],
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "ram_gb": m["ram_gb"],
                "label": m["label"],
                "note": m["note"],
                "quality": m["quality"],
            }
    return None


def _chunk_settings_for_num_ctx(num_ctx: int):
    """Return (chunk_size, chunk_overlap) scaled to the model's context window."""
    if num_ctx >= 65536:
        return 1200, 250
    if num_ctx >= 32768:
        return 1000, 200
    if num_ctx >= 16384:
        return 800, 150
    return 600, 100


def recommend_config(hw: Dict, available_models: List[str]) -> Dict:
    """
    Return the best model + num_ctx for this hardware.

    Keys returned:
        model (str|None)  — recommended model name, None if nothing is pulled
        num_ctx (int)     — recommended context window
        reasoning (str)   — human-readable explanation
        hardware_note (str)
        pull_command (str|None)  — set when no compatible model is pulled
        can_run (bool)
    """
    usable = _usable_ram(hw)

    if hw["is_apple_silicon"]:
        hardware_note = (
            "Apple Silicon (Metal GPU) — Ollama uses Metal and the Neural Engine "
            "automatically. Unified memory is shared between CPU and GPU."
        )
    elif hw["gpu_type"] == "nvidia":
        hardware_note = "NVIDIA GPU (CUDA) — model weights will run on the GPU."
    elif hw["gpu_type"] == "amd":
        hardware_note = (
            "AMD Radeon GPU (ROCm) — Ollama will offload model layers to the GPU. "
            "Ensure ROCm drivers are installed and Ollama was started with ROCm support."
        )
    else:
        hardware_note = (
            "CPU-only — inference will be slower. "
            "A smaller model (3B–7B) is strongly recommended."
        )

    avail_set = set(available_models)

    best_available: Optional[Dict] = None
    safe_available: Optional[Dict] = None   # best pulled model with < 85% RAM usage
    best_pullable: Optional[Dict] = None    # fits RAM but not yet pulled

    for m in KNOWN_MODELS:
        fits = m["ram_gb"] <= usable
        pulled = m["name"] in avail_set or any(
            a.split(":")[0] == m["name"].split(":")[0] for a in avail_set
        )
        not_tight = fits and (usable == 0 or (m["ram_gb"] / usable) < 0.85)
        if fits:
            if pulled and best_available is None:
                best_available = m
            if pulled and safe_available is None and not_tight:
                safe_available = m
            if best_pullable is None:
                best_pullable = m

    if best_available:
        tight = usable > 0 and (best_available["ram_gb"] / usable) >= 0.85
        # safe_alternative is only meaningful when best model is tight and a safer
        # pulled model exists that is different from best_available
        safe_alt = (
            safe_available
            if tight and safe_available and safe_available["name"] != best_available["name"]
            else None
        )
        reasoning = (
            f"{best_available['label']} — {best_available['note']}. "
            f"Fits in {usable:.0f} GB usable memory."
        )
        if tight:
            reasoning += (
                f" ⚠ Tight fit: {best_available['ram_gb']} GB model in "
                f"{usable:.0f} GB usable RAM."
            )
            if safe_alt:
                reasoning += (
                    f" Safer alternative: {safe_alt['name']} "
                    f"({safe_alt['ram_gb']} GB)."
                )
        return {
            "model": best_available["name"],
            "num_ctx": best_available["num_ctx"],
            "reasoning": reasoning,
            "hardware_note": hardware_note,
            "pull_command": None,
            "can_run": True,
            "tight_fit": tight,
            "safe_alternative": safe_alt,
        }

    pull_target = best_pullable or KNOWN_MODELS[-1]
    return {
        "model": None,
        "num_ctx": pull_target["num_ctx"],
        "reasoning": "No compatible models are currently pulled in Ollama.",
        "hardware_note": hardware_note,
        "pull_command": f"ollama pull {pull_target['name']}",
        "can_run": False,
        "tight_fit": False,
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_cpu_name(in_docker: bool = False, arch: str = "") -> str:
    """Best-effort CPU model name, with an Apple-Silicon-in-Docker special case.

    Tries `sysctl` (macOS), then `/proc/cpuinfo` (Linux/Docker), then
    `platform.processor()`, falling back to `"Unknown CPU"`.
    """
    # Native macOS: sysctl gives the exact chip name (e.g. "Apple M3 Max")
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            name = r.stdout.strip()
            if name:
                return name
    except Exception:
        pass

    # Linux (including Docker): /proc/cpuinfo carries the actual model name
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        if name:
                            # Docker on Apple Silicon reports a generic ARM name;
                            # replace it with a more informative label.
                            _generic_arm = {"armv8", "arm", "cortex"}
                            if in_docker and arch in ("arm64", "aarch64") and any(
                                g in name.lower() for g in _generic_arm
                            ):
                                return "Apple Silicon (Docker)"
                            return name
    except Exception:
        pass

    # If in Docker on aarch64 and nothing better was found, label it meaningfully
    if in_docker and arch in ("arm64", "aarch64"):
        return "Apple Silicon (Docker)"

    name = platform.processor()
    _arch_strings = {"x86_64", "AMD64", "aarch64", "arm64", "i686", "i386"}
    if name and name not in _arch_strings:
        return name
    return "Unknown CPU"


def _is_docker() -> bool:
    """Return True when the process is running inside a Docker container."""
    # Docker creates this sentinel file in every container
    try:
        if Path("/.dockerenv").exists():
            return True
    except Exception:
        pass
    # Fallback: cgroup membership (cgroup v1)
    try:
        with open("/proc/1/cgroup") as f:
            content = f.read()
            if "docker" in content or "containerd" in content:
                return True
    except Exception:
        pass
    return False


def _get_ram_gb() -> float:
    """Total system RAM in GB, preferring `psutil` with OS-specific fallbacks.

    Returns 0.0 if every detection method fails (rather than raising), so
    callers degrade to the "low" tier instead of crashing.
    """
    # psutil (cross-platform, most accurate)
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    # macOS fallback
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            )
            return round(int(r.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        pass
    # Linux fallback
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except Exception:
        pass
    return 0.0


def _get_gpu_type(is_apple_silicon: bool) -> str:
    """Classify the accelerator as ``"apple_silicon"``, ``"nvidia"``, ``"amd"``, or ``"cpu"``.

    Detection order: Apple Silicon → NVIDIA (nvidia-smi) → AMD (rocm-smi,
    then /sys/class/drm vendor IDs, then lspci).  Any unrecognised GPU
    falls back to ``"cpu"`` so callers always get a valid string.
    """
    if is_apple_silicon:
        return "apple_silicon"

    # ── NVIDIA ────────────────────────────────────────────────────────────────
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return "nvidia"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ── AMD ───────────────────────────────────────────────────────────────────
    # 1. rocm-smi — only present when ROCm drivers are installed
    try:
        r = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return "amd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. /sys/class/drm — AMD vendor ID 0x1002; works with stock amdgpu driver
    try:
        import glob
        for vendor_path in glob.glob("/sys/class/drm/card*/device/vendor"):
            with open(vendor_path) as f:
                if f.read().strip().lower() == "0x1002":
                    return "amd"
    except Exception:
        pass

    # 3. lspci fallback (covers systems where DRM sysfs isn't mounted)
    try:
        r = subprocess.run(
            ["lspci", "-mm"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                lower = line.lower()
                if "display" in lower or "vga" in lower or "3d" in lower:
                    if "amd" in lower or "ati" in lower or "radeon" in lower:
                        return "amd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "cpu"


def _usable_ram(hw: Dict) -> float:
    """Estimate RAM budget available for a model."""
    ram = hw["ram_gb"]
    if hw["is_apple_silicon"]:
        return ram * 0.75   # leave 25% for macOS + active apps
    if hw["gpu_type"] in ("nvidia", "amd"):
        return ram * 0.80   # GPU offload — less RAM pressure from OS
    return max(0.0, ram - 4.0)  # CPU-only: reserve ~4 GB for OS
