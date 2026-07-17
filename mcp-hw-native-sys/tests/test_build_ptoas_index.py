"""Tests for PTOAS abstraction-card generation (tools/build_ptoas_index.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_ptoas_index import build_ptoas_cards  # noqa: E402

_FAKE_TD = """//===----------------------------------------------------------------------===//
// Matmul Ops
//===----------------------------------------------------------------------===//

def TMatmulOp : PTO_TOp<"tmatmul", [SomeTrait]> {
  let summary = "PTO matrix multiplication operation, destination-style in tile world.";
  let description = [{
    Performs a tile-level matmul with an optional bias accumulation, lowered
    to the AICore cube pipeline.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
}

//===----------------------------------------------------------------------===//
// Synchronization Ops
//===----------------------------------------------------------------------===//

def BarrierSyncOp : PTO_Op<"barrier_sync"> {
  let summary = "High-level barrier mapped from SyncOpType to PIPE";
}

def SectionCubeOp : PTO_SectionOp<"section.cube">;
"""


def _make_fixture(tmp_path: Path) -> Path:
    ir_dir = tmp_path / "PTOAS" / "include" / "PTO" / "IR"
    ir_dir.mkdir(parents=True)
    (ir_dir / "PTOOps.td").write_text(_FAKE_TD, encoding="utf-8")
    (ir_dir / "VPTOOps.td").write_text("", encoding="utf-8")
    return tmp_path


def test_extracts_summary_description_and_category(tmp_path: Path):
    root = _make_fixture(tmp_path)
    cards = build_ptoas_cards(root)
    assert set(cards) == {"tmatmul", "barrier_sync", "section.cube"}
    tmatmul = cards["tmatmul"]
    assert tmatmul["tags"] == ["Matmul Ops"]
    assert "matrix multiplication" in tmatmul["one_liner"]
    assert "cube pipeline" in tmatmul["description"]
    assert tmatmul["source"] == "generated"
    assert tmatmul["generated_from"] == "PTOAS/include/PTO/IR/PTOOps.td"


def test_summary_only_op_has_empty_description(tmp_path: Path):
    root = _make_fixture(tmp_path)
    cards = build_ptoas_cards(root)
    barrier = cards["barrier_sync"]
    assert barrier["tags"] == ["Synchronization Ops"]
    assert barrier["one_liner"] == "High-level barrier mapped from SyncOpType to PIPE"
    assert barrier["description"] == ""


def test_terse_def_with_no_body_still_captured(tmp_path: Path):
    root = _make_fixture(tmp_path)
    cards = build_ptoas_cards(root)
    section = cards["section.cube"]
    assert section["one_liner"] == ""
    assert section["tags"] == ["Synchronization Ops"]


def test_real_workspace_smoke():
    # Real PTOAS checkout defines several hundred op mnemonics across both files.
    from mcp_hwnative_sys.paths import workspace_root

    cards = build_ptoas_cards(workspace_root())
    assert len(cards) >= 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
