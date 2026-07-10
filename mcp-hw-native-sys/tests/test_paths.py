"""Tests for path resolution and containment guards (mcp_hwnative_sys.paths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_hwnative_sys.paths import is_within, safe_relpath


def test_is_within_nested_path():
    assert is_within(Path("/ws/pypto/src/ir/x.cpp"), Path("/ws/pypto"))


def test_is_within_root_itself():
    assert is_within(Path("/ws/pypto"), Path("/ws/pypto"))


def test_is_within_rejects_sibling_prefix():
    # Regression for the str.startswith bypass: /ws/pypto-lib shares the
    # textual prefix "/ws/pypto" but is a sibling, not a child.
    assert not is_within(Path("/ws/pypto-lib/secret"), Path("/ws/pypto"))


def test_is_within_rejects_parent_escape():
    assert not is_within(Path("/ws/other/file"), Path("/ws/pypto"))


def test_is_within_real_traversal(tmp_path: Path):
    # Simulate read_file(repo="pypto", path="../pypto-lib/secret").
    root = tmp_path / "pypto"
    sibling = tmp_path / "pypto-lib"
    root.mkdir()
    sibling.mkdir()
    (sibling / "secret").write_text("nope")
    escaped = (root / "../pypto-lib/secret").resolve()
    assert not is_within(escaped, root.resolve())
    assert is_within((root / "src.py").resolve(), root.resolve())


def test_safe_relpath_inside_root(tmp_path: Path):
    root = tmp_path
    child = tmp_path / "a" / "b.txt"
    assert safe_relpath(child, root) == str(Path("a") / "b.txt")


def test_safe_relpath_outside_root_returns_absolute(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "x.txt"
    assert safe_relpath(outside, root) == str(outside.resolve())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
