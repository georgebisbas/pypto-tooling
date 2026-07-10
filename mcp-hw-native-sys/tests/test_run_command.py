"""Tests for subprocess handling (mcp_hwnative_sys.server._run_command)."""

from __future__ import annotations

import pytest

from mcp_hwnative_sys import server


def test_normal_command_succeeds():
    cp = server._run_command(["printf", "hello"])
    assert cp.returncode == 0
    assert cp.stdout == "hello"


def test_timeout_returns_124_without_raising():
    cp = server._run_command(["sleep", "5"], timeout_seconds=1)
    assert cp.returncode == 124
    assert "timed out" in cp.stderr.lower()


def test_missing_binary_raises_runtime_error():
    with pytest.raises(RuntimeError, match="not found"):
        server._run_command(["definitely_not_a_real_binary_xyz_123"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
