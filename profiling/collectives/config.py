"""Resolve paths to upstream repos (read-only invocation targets).

Path resolution order (first match wins):
  1. Environment variables: PYPTO_ROOT, SIMPLER_ROOT, PYPTO_NOTES_ROOT
  2. Sibling directories (dev workspace layout): ../pypto, ../simpler
  3. Docker layout (hw-native-sys.cann9.0 image): /opt/pypto, /opt/pypto/runtime
"""

from __future__ import annotations

import os
from pathlib import Path

_PROFILING_ROOT = Path(__file__).resolve().parents[1]
_TOOLING_ROOT = _PROFILING_ROOT.parent

# ── Docker-standard paths (hw-native-sys.cann9.0 image) ──────────────────
_DOCKER_PYPTO = Path("/opt/pypto")
_DOCKER_SIMPLER = _DOCKER_PYPTO / "runtime"
_DOCKER_PTO_ISA = Path("/opt/pto-isa")


def _default_hw_native_sys() -> Path:
    return _TOOLING_ROOT.parent


def _first_existing(*candidates: Path) -> Path | None:
    """Return the first path that exists on disk, or None."""
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    return None


def pypto_root() -> Path:
    if "PYPTO_ROOT" in os.environ:
        return Path(os.environ["PYPTO_ROOT"]).resolve()
    # Dev workspace: ../pypto ; Docker: /opt/pypto
    found = _first_existing(
        _default_hw_native_sys() / "pypto",
        _DOCKER_PYPTO,
    )
    if found is not None:
        return found
    raise FileNotFoundError(
        "Cannot find pypto. Set PYPTO_ROOT=/path/to/pypto or ensure the repository "
        f"is at one of: {_default_hw_native_sys() / 'pypto'}, {_DOCKER_PYPTO}"
    )


def simpler_root() -> Path:
    if "SIMPLER_ROOT" in os.environ:
        return Path(os.environ["SIMPLER_ROOT"]).resolve()
    # Dev workspace: ../simpler ; Docker: /opt/pypto/runtime
    found = _first_existing(
        _default_hw_native_sys() / "simpler",
        _DOCKER_SIMPLER,
        _default_hw_native_sys() / "pypto" / "runtime",
    )
    if found is not None:
        return found
    raise FileNotFoundError(
        "Cannot find simpler. Set SIMPLER_ROOT=/path/to/simpler or ensure it is "
        f"at one of: {_default_hw_native_sys() / 'simpler'}, {_DOCKER_SIMPLER}"
    )


def pto_isa_root() -> Path:
    """PTO-ISA installation root (read-only, for reference)."""
    if "PTO_ISA_ROOT" in os.environ:
        return Path(os.environ["PTO_ISA_ROOT"]).resolve()
    found = _first_existing(
        _default_hw_native_sys() / "pto-isa",
        _DOCKER_PTO_ISA,
    )
    if found is not None:
        return found
    # Not critical for benchmarking — HCCL baseline doesn't need pto-isa
    return _DOCKER_PTO_ISA  # return the expected path even if missing


def pypto_notes_root() -> Path:
    return Path(os.environ.get(
        "PYPTO_NOTES_ROOT",
        str(_TOOLING_ROOT.parent / "pypto-3.0-notes"),
    )).resolve()


def profiling_root() -> Path:
    return _PROFILING_ROOT



def results_root() -> Path:
    return _PROFILING_ROOT / "results"
