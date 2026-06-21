#!/usr/bin/env python3
"""Sync status_prs.md into config/program_status.json for MCP agents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_hwnative_sys.program_status import (  # noqa: E402
    program_status_path,
    sync_program_status_from_workspace,
)


def main() -> int:
    data = sync_program_status_from_workspace()
    if "error" in data:
        print(data["error"], file=sys.stderr)
        return 1
    print(json.dumps({"written": str(program_status_path()), "open_prs": len(data.get("open_prs", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
