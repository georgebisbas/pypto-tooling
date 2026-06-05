"""Resolve paths to upstream repos (read-only invocation targets)."""

from __future__ import annotations

import os
from pathlib import Path

_PROFILING_ROOT = Path(__file__).resolve().parents[1]
_TOOLING_ROOT = _PROFILING_ROOT.parent


def _default_hw_native_sys() -> Path:
    return _TOOLING_ROOT.parent


def pypto_root() -> Path:
    return Path(os.environ.get("PYPTO_ROOT", _default_hw_native_sys() / "pypto")).resolve()


def simpler_root() -> Path:
    default = _default_hw_native_sys() / "simpler"
    # Docker: simpler is the runtime submodule inside pypto
    docker_path = _default_hw_native_sys() / "pypto" / "runtime"
    if not default.is_dir() and docker_path.is_dir():
        default = docker_path
    return Path(os.environ.get("SIMPLER_ROOT", default)).resolve()


def pypto_notes_root() -> Path:
    return Path(os.environ.get("PYPTO_NOTES_ROOT", _TOOLING_ROOT.parent / "pypto-3.0-notes")).resolve()


def profiling_root() -> Path:
    return _PROFILING_ROOT


def results_root() -> Path:
    return _PROFILING_ROOT / "results"
