"""Tests for mtime-based JSON caching (mcp_hwnative_sys.paths.load_json_cached)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mcp_hwnative_sys.paths import load_json_cached


def test_returns_same_object_when_unchanged(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"a": 1}))
    first = load_json_cached(p)
    second = load_json_cached(p)
    assert first is second  # served from cache, not re-parsed


def test_reloads_when_mtime_changes(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"v": 1}))
    assert load_json_cached(p)["v"] == 1

    p.write_text(json.dumps({"v": 2}))
    # Force a distinct mtime in case the writes land in the same tick.
    stat = p.stat()
    os.utime(p, (stat.st_atime, stat.st_mtime + 10))
    assert load_json_cached(p)["v"] == 2


def test_missing_file_raises_and_evicts(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"a": 1}))
    load_json_cached(p)
    p.unlink()
    with pytest.raises(OSError):
        load_json_cached(p)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
