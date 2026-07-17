"""Tests for merging generated + hand-curated abstraction cards
(mcp_hwnative_sys.knowledge.load_abstractions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_hwnative_sys import knowledge


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _wire_paths(monkeypatch, tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(knowledge, "project_root", lambda: tmp_path)
    monkeypatch.setattr(knowledge, "abstractions_config_path", lambda: config_dir / "abstractions.json")
    monkeypatch.setattr(
        knowledge, "ascend_abstractions_config_path", lambda: config_dir / "ascend_abstractions.json"
    )
    monkeypatch.setattr(knowledge, "_abstractions_cache", None)
    return config_dir


def test_generated_cards_fill_in_uncurated_names(monkeypatch, tmp_path: Path):
    config_dir = _wire_paths(monkeypatch, tmp_path)
    _write_json(config_dir / "abstractions.json", {"Submit": {"layer": "pypto/ir"}})
    _write_json(config_dir / "pto_isa_generated.json", {"TADD": {"source": "generated", "one_liner": "add"}})

    merged = knowledge.load_abstractions()
    assert "Submit" in merged
    assert merged["TADD"]["source"] == "generated"


def test_hand_curated_card_wins_on_name_collision(monkeypatch, tmp_path: Path):
    config_dir = _wire_paths(monkeypatch, tmp_path)
    _write_json(
        config_dir / "abstractions.json",
        {"TADD": {"layer": "curated/layer", "one_liner": "hand-written, authoritative"}},
    )
    _write_json(
        config_dir / "ptoas_generated.json",
        {"TADD": {"source": "generated", "one_liner": "auto-scraped, should lose"}},
    )

    merged = knowledge.load_abstractions()
    assert merged["TADD"]["one_liner"] == "hand-written, authoritative"
    assert merged["TADD"].get("source") is None


def test_explain_abstraction_reports_source_provenance(monkeypatch, tmp_path: Path):
    config_dir = _wire_paths(monkeypatch, tmp_path)
    _write_json(config_dir / "abstractions.json", {"Submit": {"layer": "pypto/ir"}})
    _write_json(config_dir / "pto_isa_generated.json", {"TADD": {"source": "generated", "layer": "pto-isa/instr"}})

    curated = knowledge.explain_abstraction_impl("Submit")
    assert curated["source"] == "curated"
    generated = knowledge.explain_abstraction_impl("TADD")
    assert generated["source"] == "generated"


def test_missing_generated_files_do_not_break_loading(monkeypatch, tmp_path: Path):
    config_dir = _wire_paths(monkeypatch, tmp_path)
    _write_json(config_dir / "abstractions.json", {"Submit": {"layer": "pypto/ir"}})
    # Neither generated file nor ascend_abstractions.json exists on disk.
    merged = knowledge.load_abstractions()
    assert merged == {"Submit": {"layer": "pypto/ir"}}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
