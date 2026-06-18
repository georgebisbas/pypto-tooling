#!/usr/bin/env python3
"""Verify knowledge config paths and tier policy."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"

sys.path.insert(0, str(PROJECT_ROOT))

from mcp_hwnative_sys.knowledge import (  # noqa: E402
    EPHEMERAL_PREFIXES,
    _parse_notes_freshness,
    load_abstractions,
    load_knowledge_config,
    resolve_doc_tier,
)
from mcp_hwnative_sys.paths import workspace_root  # noqa: E402


def _check_path(root: Path, path: str, errors: list[str], warnings: list[str]) -> None:
    if not (root / path).exists():
        errors.append(f"Missing path: {path}")


def main() -> int:
    root = workspace_root()
    config = load_knowledge_config()
    abstractions = load_abstractions()
    errors: list[str] = []
    warnings: list[str] = []
    freshness = _parse_notes_freshness()
    today = date.today()

    for task_type, route in config.get("routes", {}).items():
        for path in route.get("read_first_canonical", []):
            tier = resolve_doc_tier(path)
            if tier == "ephemeral":
                errors.append(f"Ephemeral doc in read_first_canonical for {task_type}: {path}")
            _check_path(root, path, errors, warnings)

        for path in route.get("read_first_enriched", []):
            _check_path(root, path, errors, warnings)
            verified = freshness.get(path)
            if verified:
                age = (today - datetime.strptime(verified, "%Y-%m-%d").date()).days
                if age > 30:
                    warnings.append(f"Stale enriched doc ({age}d): {path}")

    for uri, resource in config.get("resources", {}).items():
        paths = resource.get("paths", [])
        single = resource.get("path")
        if single:
            paths = [*paths, single]
        for path in paths:
            if path:
                _check_path(root, path, errors, warnings)

    for path in config.get("notes_topics", {}).values():
        _check_path(root, path, errors, warnings)

    for name, card in abstractions.items():
        for path in card.get("paths", []):
            if path:
                _check_path(root, path, errors, warnings)
        for path in card.get("docs_canonical", []):
            _check_path(root, path, errors, warnings)

    for prefix in EPHEMERAL_PREFIXES:
        for route in config.get("routes", {}).values():
            for path in route.get("read_first_canonical", []) + route.get("read_first_enriched", []):
                if path.startswith(prefix):
                    errors.append(f"Ephemeral path referenced in route: {path}")

    print(f"Workspace: {root}")
    print(f"Routes: {len(config.get('routes', {}))}")
    print(f"Abstractions: {len(abstractions)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    for item in errors[:30]:
        print(f"  ERROR: {item}")
    for item in warnings[:20]:
        print(f"  WARN: {item}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
