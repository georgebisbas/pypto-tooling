#!/usr/bin/env python3
"""Suggest abstractions index entries from codebase scans."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"

sys.path.insert(0, str(PROJECT_ROOT))

from mcp_hwnative_sys.paths import workspace_root  # noqa: E402


def _scan_pass_manager(root: Path) -> list[str]:
    path = root / "pypto/python/pypto/ir/pass_manager.py"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'"([A-Z][A-Za-z0-9]+)"', text)))


def _scan_ir_headers(root: Path) -> list[str]:
    ir_dir = root / "pypto/include/pypto/ir"
    if not ir_dir.exists():
        return []
    names: set[str] = set()
    for header in ir_dir.rglob("*.h"):
        content = header.read_text(encoding="utf-8", errors="replace")
        names.update(re.findall(r"class\s+(\w+)", content))
    return sorted(names)


def _scan_pto_isa_manifest(root: Path) -> list[str]:
    manifest = root / "pto-isa/docs/isa/manifest.yaml"
    if not manifest.exists():
        return []
    text = manifest.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"^\s*-\s*([A-Z][A-Z0-9_]+)\s*$", text, re.MULTILINE)))


def main() -> int:
    root = workspace_root()
    existing_path = CONFIG_DIR / "abstractions.json"
    existing: dict = {}
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))

    suggestions = {
        "pass_manager_passes": _scan_pass_manager(root),
        "ir_header_classes": _scan_ir_headers(root)[:50],
        "pto_isa_instructions": _scan_pto_isa_manifest(root)[:50],
    }

    missing_passes = [p for p in suggestions["pass_manager_passes"] if p not in existing]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "existing_abstractions": len(existing),
        "suggested_missing_passes": missing_passes[:30],
        "sample_ir_classes": suggestions["ir_header_classes"][:20],
        "sample_pto_isa_ops": suggestions["pto_isa_instructions"][:20],
    }

    print(json.dumps(report, indent=2))

    marker = CONFIG_DIR / ".index_build_time"
    marker.write_text(report["generated_at"], encoding="utf-8")
    print(f"\nWrote build marker: {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
