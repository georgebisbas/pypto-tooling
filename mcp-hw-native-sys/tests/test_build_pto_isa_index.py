"""Tests for pto-isa abstraction-card generation (tools/build_pto_isa_index.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_pto_isa_index import build_pto_isa_cards  # noqa: E402

_MANIFEST = json.dumps(
    {
        "instructions": [
            {
                "instruction": "TADD",
                "category": "Elementwise (Tile-Tile)",
                "summary_en": "Elementwise add of two tiles.",
            },
            {
                "instruction": "TSYNC",
                "category": "Synchronization",
                "summary_en": "Synchronize PTO execution.",
            },
        ]
    }
)

_COMM_README = """# Communication ISA

| | Instruction | PTO Name | Description |
|-|-----------|---------|-------------|
| | [TPUT](./TPUT.md) | `pto.tput` | Put data to a remote NPU |
| | [TGET](./TGET.md) | `pto.tget` | Get data from a remote NPU |
| | [TADD](./TADD.md) | `pto.tadd_collective` | Reduce-add across ranks |
"""


def _make_fixture(tmp_path: Path) -> Path:
    isa_dir = tmp_path / "pto-isa" / "docs" / "isa"
    comm_dir = isa_dir / "comm"
    comm_dir.mkdir(parents=True)
    (isa_dir / "manifest.yaml").write_text(_MANIFEST, encoding="utf-8")
    (comm_dir / "README.md").write_text(_COMM_README, encoding="utf-8")
    (isa_dir / "TADD.md").write_text("# TADD\n", encoding="utf-8")
    # TSYNC.md deliberately absent to exercise the has_doc=False path.
    (comm_dir / "TPUT.md").write_text("# TPUT\n", encoding="utf-8")
    return tmp_path


def test_generates_cards_for_manifest_and_comm_instructions(tmp_path: Path):
    root = _make_fixture(tmp_path)
    cards = build_pto_isa_cards(root)
    assert set(cards) == {"TADD", "TSYNC", "TPUT", "TGET"}


def test_name_collision_across_sources_merges_instead_of_overwriting(tmp_path: Path):
    # pto-isa reuses TADD for both a local elementwise op (manifest) and a
    # distributed collective (comm/README.md) in this fixture -- neither
    # side's information should be silently dropped.
    root = _make_fixture(tmp_path)
    cards = build_pto_isa_cards(root)
    tadd = cards["TADD"]
    assert "Elementwise (Tile-Tile)" in tadd["tags"]
    assert "Communication" in tadd["tags"]
    assert "pto-isa/docs/isa/TADD.md" in tadd["docs_canonical"]
    assert "Elementwise add of two tiles." in tadd["one_liner"]
    assert "Reduce-add across ranks" in tadd["one_liner"]
    assert set(tadd["generated_from"]) == {
        "pto-isa/docs/isa/manifest.yaml",
        "pto-isa/docs/isa/comm/README.md",
    }


def test_cards_are_tagged_as_generated_with_provenance(tmp_path: Path):
    root = _make_fixture(tmp_path)
    cards = build_pto_isa_cards(root)
    assert cards["TSYNC"]["source"] == "generated"
    assert cards["TSYNC"]["generated_from"] == "pto-isa/docs/isa/manifest.yaml"
    assert cards["TSYNC"]["one_liner"] == "Synchronize PTO execution."
    assert cards["TPUT"]["generated_from"] == "pto-isa/docs/isa/comm/README.md"
    assert cards["TPUT"]["tags"] == ["Communication"]


def test_missing_doc_page_yields_empty_paths(tmp_path: Path):
    root = _make_fixture(tmp_path)
    cards = build_pto_isa_cards(root)
    assert cards["TSYNC"]["docs_canonical"] == []
    assert cards["TADD"]["docs_canonical"] == ["pto-isa/docs/isa/TADD.md"]


def test_real_workspace_smoke():
    # Real pto-isa checkout should yield well over 130 instructions
    # (131 manifest entries + ~11 comm ops).
    from mcp_hwnative_sys.paths import workspace_root

    cards = build_pto_isa_cards(workspace_root())
    assert len(cards) >= 130


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
