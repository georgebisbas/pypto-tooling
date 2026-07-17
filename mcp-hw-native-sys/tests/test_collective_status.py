"""Tests for the collective-status parity-matrix parser
(mcp_hwnative_sys.collective_status)."""

from __future__ import annotations

import pytest

from mcp_hwnative_sys.collective_status import (
    collective_status_impl,
    parse_current_status_markdown,
)

_CURRENT_STATUS_MD = """# Distributed communication — current status

*last_verified: 2026-07-16 (PR sync). Lots of prose here with [#1997](url) links.*

## Stack placement

Some unrelated table:

| Context | Rank count |
| ------- | ---------- |
| L3 | P=2 |

## Collective feature parity matrix

*last_verified: 2026-07-16*

| Feature | AllReduce (ring) | Barrier | All-to-All |
|---------|:---:|:---:|:---:|
| **Composite intrinsic** (`pld.tensor.*`) | ✅ #1942 | ✅ #1782 | ✅ #1937 |
| **Dynamic NR** | ⚠️ Monolithic only | N/A (no data) | ❌ |
| **Auto-select algorithm** | 🔵 Plan 42 | N/A | — |

### Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Merged to `main` |
| 🟡 | In review (PR open) |
| 🔵 | Planned (not started) |
| ❌ | Gap — plan exists but not implemented |
| N/A | Not applicable |

### Notes

- Some prose that should never be parsed as table rows.
"""


def test_parses_last_verified_from_matrix_section_not_doc_top():
    data = parse_current_status_markdown(_CURRENT_STATUS_MD)
    assert data["last_verified"] == "2026-07-16"


def test_parses_all_axes_and_ops():
    data = parse_current_status_markdown(_CURRENT_STATUS_MD)
    assert set(data["matrix"]) == {
        "Composite intrinsic (`pld.tensor.*`)",
        "Dynamic NR",
        "Auto-select algorithm",
    }
    assert set(data["matrix"]["Dynamic NR"]) == {"AllReduce (ring)", "Barrier", "All-to-All"}


def test_known_symbol_resolves_via_legend():
    data = parse_current_status_markdown(_CURRENT_STATUS_MD)
    cell = data["matrix"]["Composite intrinsic (`pld.tensor.*`)"]["AllReduce (ring)"]
    assert cell["symbol"] == "✅"
    assert cell["ref"] == "#1942"
    assert cell["meaning"] == "Merged to `main`"


def test_unmapped_symbol_passes_through_raw_instead_of_dropping():
    data = parse_current_status_markdown(_CURRENT_STATUS_MD)
    # "⚠️" never appears in the legend table -- must not be silently discarded.
    warn_cell = data["matrix"]["Dynamic NR"]["AllReduce (ring)"]
    assert warn_cell["symbol"] == "⚠️"
    assert warn_cell["ref"] == "Monolithic only"
    assert warn_cell["meaning"] == "⚠️"

    # bare em-dash also never appears in the legend table.
    dash_cell = data["matrix"]["Auto-select algorithm"]["All-to-All"]
    assert dash_cell["symbol"] == "—"
    assert dash_cell["meaning"] == "—"


def test_na_with_trailing_parenthetical_is_recognized():
    data = parse_current_status_markdown(_CURRENT_STATUS_MD)
    cell = data["matrix"]["Dynamic NR"]["Barrier"]
    assert cell["symbol"] == "N/A"
    assert cell["ref"] == "(no data)"
    assert cell["meaning"] == "Not applicable"


def test_unrelated_table_before_matrix_is_ignored():
    data = parse_current_status_markdown(_CURRENT_STATUS_MD)
    assert "Context" not in data["matrix"]


def test_collective_status_impl_filters_by_op_and_axis(monkeypatch):
    import mcp_hwnative_sys.collective_status as cs

    monkeypatch.setattr(cs, "load_collective_status", lambda: parse_current_status_markdown(_CURRENT_STATUS_MD))

    only_barrier = collective_status_impl(op="Barrier")
    for row in only_barrier["matrix"].values():
        assert set(row) == {"Barrier"}

    only_dynamic_nr = collective_status_impl(axis="Dynamic NR")
    assert set(only_dynamic_nr["matrix"]) == {"Dynamic NR"}

    single_cell = collective_status_impl(op="Barrier", axis="Dynamic NR")
    assert single_cell["matrix"]["Dynamic NR"]["Barrier"]["symbol"] == "N/A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
