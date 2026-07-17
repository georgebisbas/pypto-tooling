#!/usr/bin/env python3
"""Generate abstraction cards for PTOAS IR ops.

Regex-scans PTOAS's TableGen op definitions (PTOOps.td, VPTOOps.td) for
`def Foo : SomeBaseOp<"mnemonic">` records and their `let summary`/
`let description` doc fields, writing the result to
config/ptoas_generated.json. No mlir-tblgen doc-gen target exists in PTOAS
to lean on instead, so this is a best-effort regex extraction -- category
comes from the nearest preceding banner comment and degrades to
"uncategorized" once banners stop appearing late in PTOOps.td.

Never touches config/abstractions.json -- the generated file is merged in
at load time by mcp_hwnative_sys.knowledge with hand-curated cards always
taking precedence on key collisions. Read-only with respect to PTOAS.
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

_OP_RE = re.compile(r'def\s+(\w+)\s*:\s*\w+<"([\w.]+)"')
_SUMMARY_RE = re.compile(r'let\s+summary\s*=\s*"(.*?)";')
_DESCRIPTION_RE = re.compile(r"let\s+description\s*=\s*\[\{(.*?)\}\];", re.DOTALL)
_BANNER_RE = re.compile(r"^//===.*===//\n//\s*(.*?)\s*\n//===.*===//", re.MULTILINE)

_TD_SOURCES = (
    "PTOAS/include/PTO/IR/PTOOps.td",
    "PTOAS/include/PTO/IR/VPTOOps.td",
)


def _banner_category(banners: list[tuple[int, str]], pos: int) -> str:
    preceding = [title for start, title in banners if start < pos]
    return preceding[-1] if preceding else "uncategorized"


def _block_for(text: str, start: int, next_start: int | None) -> str:
    return text[start : next_start if next_start is not None else len(text)]


def build_ptoas_cards(root: Path) -> dict[str, dict]:
    cards: dict[str, dict] = {}

    for rel in _TD_SOURCES:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        banners = [(m.start(), m.group(1)) for m in _BANNER_RE.finditer(text)]
        matches = list(_OP_RE.finditer(text))
        for idx, match in enumerate(matches):
            mnemonic = match.group(2)
            if mnemonic in cards:
                continue
            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else None
            block = _block_for(text, match.start(), next_start)
            summary_matches = _SUMMARY_RE.findall(block)
            summary = summary_matches[0] if summary_matches else ""
            description_matches = _DESCRIPTION_RE.findall(block)
            description = " ".join(description_matches[0].split()) if description_matches else ""
            category = _banner_category(banners, match.start())
            cards[mnemonic] = {
                "layer": "ptoas/ir",
                "kind": "ir_op",
                "tags": [category],
                "repos": ["PTOAS"],
                "paths": [rel],
                "docs_canonical": [],
                "one_liner": summary or description,
                "description": description,
                "source": "generated",
                "generated_from": rel,
            }

    return cards


def main() -> int:
    root = workspace_root()
    cards = build_ptoas_cards(root)
    out_path = CONFIG_DIR / "ptoas_generated.json"
    out_path.write_text(json.dumps(cards, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(cards)} PTOAS op cards to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
