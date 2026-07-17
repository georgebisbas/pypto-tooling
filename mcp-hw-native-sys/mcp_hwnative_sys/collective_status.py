"""Structured access to the collective-comm feature parity matrix in
pypto-3.0-notes/distributed/current_status.md.

Read-only with respect to that document: it is parsed, never written to.
Scoped narrowly to the "Collective feature parity matrix" table + its
Legend table + the matrix section's own last_verified line -- the rest of
current_status.md (Notes prose, Changelog, other tables) has too many
inline PR/SHA/date grammar variants to normalize reliably in one pass, so
it is deliberately left out of this first version.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mcp_hwnative_sys.paths import project_root, workspace_root

_MATRIX_HEADING = "## Collective feature parity matrix"
_LEGEND_HEADING = "### Legend"

# Symbols with a recognized meaning in the legend table always sort first in
# a cell; anything else (e.g. "⚠️", "—") is passed through raw rather than
# dropped, since the matrix itself uses a couple of symbols the legend never
# defines.
_KNOWN_SYMBOL_RE = re.compile(r"^(✅|🟡|🔵|❌|⚠️|N/A|—)")


def collective_status_path() -> Path:
    return project_root() / "config" / "collective_status.json"


def _split_table_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    return [c.strip() for c in stripped.strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c)


def _clean_label(label: str) -> str:
    return label.replace("**", "").strip()


def _split_cell(cell: str) -> tuple[str, str]:
    match = _KNOWN_SYMBOL_RE.match(cell)
    if not match:
        return "", cell
    symbol = match.group(1)
    rest = cell[match.end() :].strip()
    return symbol, rest


def _find_section(text: str, heading: str, until_headings: tuple[str, ...] = ()) -> str:
    start = text.find(heading)
    if start == -1:
        return ""
    start += len(heading)
    end = len(text)
    for other in until_headings:
        pos = text.find(other, start)
        if pos != -1:
            end = min(end, pos)
    return text[start:end]


def _parse_legend(text: str) -> dict[str, str]:
    section = _find_section(text, _LEGEND_HEADING, ("### Notes", "## "))
    legend: dict[str, str] = {}
    for line in section.splitlines():
        cells = _split_table_row(line)
        if not cells or len(cells) < 2 or _is_separator_row(cells):
            continue
        symbol, meaning = cells[0], cells[1]
        if symbol in ("Symbol",):
            continue
        legend[symbol] = meaning
    return legend


def _parse_matrix(text: str) -> tuple[str | None, dict[str, dict[str, dict[str, Any]]]]:
    section = _find_section(text, _MATRIX_HEADING, (_LEGEND_HEADING,))

    last_verified = None
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("*last_verified:"):
            last_verified = stripped.split(":", 1)[1].strip().rstrip("*").strip()
            break

    lines = section.splitlines()
    header_idx: int | None = None
    header_cells: list[str] = []
    for i, line in enumerate(lines):
        row_cells = _split_table_row(line)
        if row_cells and row_cells[0] == "Feature":
            header_idx, header_cells = i, row_cells
            break
    if header_idx is None:
        return last_verified, {}

    op_names = header_cells[1:]

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for line in lines[header_idx + 2 :]:
        cells = _split_table_row(line)
        if not cells:
            break
        if len(cells) != len(header_cells):
            continue
        axis = _clean_label(cells[0])
        row: dict[str, dict[str, Any]] = {}
        for op_name, raw_cell in zip(op_names, cells[1:]):
            symbol, ref = _split_cell(raw_cell)
            row[op_name] = {"raw": raw_cell, "symbol": symbol, "ref": ref}
        matrix[axis] = row

    return last_verified, matrix


def parse_current_status_markdown(text: str) -> dict[str, Any]:
    last_verified, matrix = _parse_matrix(text)
    legend = _parse_legend(text)
    for row in matrix.values():
        for cell in row.values():
            cell["meaning"] = legend.get(cell["symbol"], cell["symbol"] or None)
    return {"last_verified": last_verified, "legend": legend, "matrix": matrix}


def sync_collective_status_from_workspace() -> dict[str, Any]:
    doc = workspace_root() / "pypto-3.0-notes/distributed/current_status.md"
    if not doc.exists():
        return {"error": f"Missing {doc}"}
    parsed = parse_current_status_markdown(doc.read_text(encoding="utf-8"))
    # Best-effort cache write: a read-only config/ must not make the
    # collective_status read-path tool fail -- the parsed data is returned
    # regardless. Never writes to the source doc itself.
    try:
        collective_status_path().write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return parsed


def load_collective_status() -> dict[str, Any]:
    path = collective_status_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return sync_collective_status_from_workspace()


def collective_status_impl(op: str = "", axis: str = "") -> dict[str, Any]:
    data = load_collective_status()
    if "error" in data:
        return data

    matrix = data.get("matrix", {})
    if not op and not axis:
        return data

    op_lower = op.strip().lower()
    axis_lower = axis.strip().lower()

    filtered: dict[str, dict[str, Any]] = {}
    for axis_name, row in matrix.items():
        if axis_lower and axis_lower not in axis_name.lower():
            continue
        matched_row = {
            op_name: cell
            for op_name, cell in row.items()
            if not op_lower or op_lower in op_name.lower()
        }
        if matched_row:
            filtered[axis_name] = matched_row

    return {
        "last_verified": data.get("last_verified"),
        "legend": data.get("legend", {}),
        "matrix": filtered,
    }
