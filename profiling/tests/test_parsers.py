"""Unit tests for the shared timing-metric parsers (collectives.metrics)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROFILING = Path(__file__).resolve().parents[1]
if str(_PROFILING) not in sys.path:
    sys.path.insert(0, str(_PROFILING))

from collectives.metrics import parse_device_wall_s, parse_hccl_per_rank  # noqa: E402


def test_device_wall_single_span():
    text = "[2026-08-11 07:00:00][TIMING] [STRACE] name=simpler_run.runner_run.device_wall dur=123456 clk=dev"
    assert parse_device_wall_s(text) == pytest.approx(123456 / 1e9)


def test_device_wall_slowest_rank_wins():
    text = (
        "[STRACE] name=simpler_run.runner_run.device_wall dur=1000 clk=dev\n"
        "[STRACE] name=simpler_run.runner_run.device_wall dur=5000 clk=dev"
    )
    assert parse_device_wall_s(text) == pytest.approx(5000 / 1e9)


def test_device_wall_none_when_absent():
    assert parse_device_wall_s("no spans here") is None
    assert parse_device_wall_s("") is None


def test_device_wall_from_subprocess_blob():
    """A _run_with_phases-style blob (stdout + '--- STDERR ---' suffix)."""
    blob = (
        "[allreduce] running 2-chip allreduce DAG...\n"
        "--- STDERR ---\n"
        "[2026-08-11][TIMING] [STRACE] v=1 pid=1 tid=1 inv=1 hid=0 depth=2 "
        "name=simpler_run.runner_run.device_wall ts=0 dur=7500000 clk=dev\n"
    )
    assert parse_device_wall_s(blob) == pytest.approx(7.5e-3)


def test_hccl_per_rank_timed():
    line = "HCCL_TIMED round=1 per_rank=[0.000105, 9.9e-05]"
    assert parse_hccl_per_rank(line) == pytest.approx([0.000105, 9.9e-05])


def test_hccl_per_rank_warmup_parses_too():
    line = "HCCL_WARMUP round=1 per_rank=[0.032622, 0.032635]"
    assert parse_hccl_per_rank(line) == pytest.approx([0.032622, 0.032635])


def test_hccl_per_rank_none():
    assert parse_hccl_per_rank("HCCL_COMM_SETUP_OK setup_s=1.28") is None
    assert parse_hccl_per_rank("HCCL_TIMED round=1 per_rank=") is None
    assert parse_hccl_per_rank("garbage") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
