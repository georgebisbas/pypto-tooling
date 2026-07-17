#!/usr/bin/env python3
"""Generate abstraction cards for pto-isa instructions.

Reads pto-isa's own structured instruction sources (docs/isa/manifest.yaml for
the ~131 tile-local instructions, docs/isa/comm/README.md's table for the
~11 comm/collective instructions not in the manifest) and writes them to
config/pto_isa_generated.json. Never touches config/abstractions.json — the
generated file is merged in at load time by mcp_hwnative_sys.knowledge with
hand-curated cards always taking precedence on key collisions.

Read-only with respect to pto-isa itself: source material only, never written to.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"

sys.path.insert(0, str(PROJECT_ROOT))

from mcp_hwnative_sys.paths import workspace_root  # noqa: E402

_COMM_ROW_RE = re.compile(
    r"\|\s*\|\s*\[(\w+)\]\(\./\1\.md\)\s*\|\s*`([\w.]+)`\s*\|\s*([^|]+?)\s*\|"
)


def _load_manifest_instructions(root: Path) -> list[dict]:
    manifest = root / "pto-isa/docs/isa/manifest.yaml"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return data.get("instructions", [])


def _scan_comm_readme(root: Path) -> list[dict]:
    readme = root / "pto-isa/docs/isa/comm/README.md"
    if not readme.exists():
        return []
    text = readme.read_text(encoding="utf-8")
    rows = []
    for match in _COMM_ROW_RE.finditer(text):
        rows.append(
            {
                "instruction": match.group(1),
                "pto_name": match.group(2),
                "summary_en": match.group(3).strip(),
            }
        )
    return rows


def _merge_colliding_card(existing: dict, incoming: dict) -> dict:
    """Combine two same-mnemonic cards from different sources rather than
    dropping one. pto-isa reuses a handful of mnemonics (e.g. TSCATTER,
    TGATHER) for genuinely distinct local-tile vs. distributed-collective
    instructions, so a plain overwrite silently loses real information."""
    merged = dict(existing)
    merged["tags"] = sorted(set(existing.get("tags", [])) | set(incoming.get("tags", [])))
    merged["paths"] = sorted(set(existing.get("paths", [])) | set(incoming.get("paths", [])))
    merged["docs_canonical"] = sorted(
        set(existing.get("docs_canonical", [])) | set(incoming.get("docs_canonical", []))
    )
    one_liners = [existing.get("one_liner", ""), incoming.get("one_liner", "")]
    merged["one_liner"] = " / ".join(dict.fromkeys(x for x in one_liners if x))
    merged["generated_from"] = sorted(
        {existing.get("generated_from", ""), incoming.get("generated_from", "")} - {""}
    )
    return merged


def build_pto_isa_cards(root: Path) -> dict[str, dict]:
    cards: dict[str, dict] = {}

    for entry in _load_manifest_instructions(root):
        name = entry.get("instruction")
        if not name:
            continue
        doc_rel = f"pto-isa/docs/isa/{name}.md"
        has_doc = (root / doc_rel).exists()
        cards[name] = {
            "layer": "pto-isa/instr",
            "kind": "isa_instruction",
            "tags": [entry.get("category", "")] if entry.get("category") else [],
            "repos": ["pto-isa"],
            "paths": [doc_rel] if has_doc else [],
            "docs_canonical": [doc_rel] if has_doc else [],
            "one_liner": entry.get("summary_en", ""),
            "source": "generated",
            "generated_from": "pto-isa/docs/isa/manifest.yaml",
        }

    for row in _scan_comm_readme(root):
        name = row["instruction"]
        doc_rel = f"pto-isa/docs/isa/comm/{name}.md"
        has_doc = (root / doc_rel).exists()
        comm_card = {
            "layer": "pto-isa/instr",
            "kind": "isa_instruction",
            "tags": ["Communication"],
            "repos": ["pto-isa"],
            "paths": [doc_rel] if has_doc else [],
            "docs_canonical": [doc_rel] if has_doc else [],
            "one_liner": row["summary_en"],
            "source": "generated",
            "generated_from": "pto-isa/docs/isa/comm/README.md",
        }
        if name in cards:
            cards[name] = _merge_colliding_card(cards[name], comm_card)
        else:
            cards[name] = comm_card

    return cards


def main() -> int:
    root = workspace_root()
    cards = build_pto_isa_cards(root)
    out_path = CONFIG_DIR / "pto_isa_generated.json"
    out_path.write_text(json.dumps(cards, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(cards)} pto-isa cards to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
