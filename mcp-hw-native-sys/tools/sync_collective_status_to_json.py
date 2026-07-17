#!/usr/bin/env python3
"""Sync current_status.md's parity matrix into config/collective_status.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_hwnative_sys.collective_status import (  # noqa: E402
    collective_status_path,
    sync_collective_status_from_workspace,
)


def main() -> int:
    data = sync_collective_status_from_workspace()
    if "error" in data:
        print(data["error"], file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "written": str(collective_status_path()),
                "axes": len(data.get("matrix", {})),
                "last_verified": data.get("last_verified"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
