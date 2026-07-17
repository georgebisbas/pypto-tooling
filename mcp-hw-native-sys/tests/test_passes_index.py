"""Tests for pass-index building (mcp_hwnative_sys.passes_index)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_hwnative_sys import passes_index


def test_build_warns_when_pass_manager_missing(monkeypatch, tmp_path: Path):
    # workspace without pypto/python/pypto/ir/pass_manager.py -> warning, no passes.
    monkeypatch.setattr(passes_index, "workspace_root", lambda: tmp_path)
    result = passes_index.build_passes_index()
    assert result["passes"] == []
    assert "warning" in result and "not found" in result["warning"]


def test_build_warns_when_regex_matches_nothing(monkeypatch, tmp_path: Path):
    pm = tmp_path / "pypto" / "python" / "pypto" / "ir" / "pass_manager.py"
    pm.parent.mkdir(parents=True)
    pm.write_text("# refactored file with no PassSpec tuples\n")
    monkeypatch.setattr(passes_index, "workspace_root", lambda: tmp_path)
    result = passes_index.build_passes_index()
    assert result["passes"] == []
    assert "warning" in result and "0 passes" in result["warning"]


def test_healthy_index_has_no_warning():
    # The real workspace pass_manager.py yields passes and no warning.
    result = passes_index.build_passes_index()
    assert result["passes"], "expected passes from the real pass_manager.py"
    assert "warning" not in result


def test_pass_count_key_is_pypto_scoped():
    # pass_manager.py is the only pipeline ever scraped (no PTOAS/pto-isa/simpler
    # equivalent exists), so the count must be labeled as pypto-specific rather
    # than implying it's a generic cross-repo figure.
    result = passes_index.build_passes_index()
    assert "pypto_pass_count" in result
    assert result["pypto_pass_count"] == len(result["passes"])
    assert "pass_count" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
