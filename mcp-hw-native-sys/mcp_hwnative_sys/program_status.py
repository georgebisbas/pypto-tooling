from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_hwnative_sys.paths import load_repos_config, project_root, workspace_root


def program_status_path() -> Path:
    return project_root() / "config" / "program_status.json"


def _parse_pr_row(cells: list[str]) -> dict[str, Any] | None:
    cleaned = [c.strip() for c in cells if c.strip()]
    if len(cleaned) < 4:
        return None
    pr_cell = cleaned[0]
    if not pr_cell.startswith("[#") and not pr_cell.startswith("#"):
        return None
    pr_num = pr_cell.split("#")[1].split("]")[0].split(")")[0]
    return {
        "pr": f"#{pr_num}",
        "title": cleaned[1] if len(cleaned) > 1 else "",
        "branch": cleaned[2].strip("`") if len(cleaned) > 2 else "",
        "ci": cleaned[4] if len(cleaned) > 4 else "",
        "notes": cleaned[-1] if len(cleaned) > 5 else "",
    }


def parse_status_prs_markdown(text: str) -> dict[str, Any]:
    """Parse status_prs.md SYNC sections into structured JSON."""
    dashboard = {"open": 0, "merged_all": 0, "merged_90d": 0, "closed": 0}
    dash_match = re_search_dashboard(text)
    if dash_match:
        dashboard = dash_match

    open_prs: list[dict[str, Any]] = []
    wip: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []

    current_repo = ""
    section = ""
    for line in text.splitlines():
        if "<!-- SYNC:ACTIVE_START -->" in line:
            section = "active"
            current_repo = ""
            continue
        if "<!-- SYNC:ACTIVE_END -->" in line:
            section = ""
            current_repo = ""
            continue
        if "<!-- SYNC:MERGED_START -->" in line:
            section = "merged"
            current_repo = ""
            continue
        if "<!-- SYNC:MERGED_END -->" in line:
            section = ""
            continue
        if line.startswith("### ") and section == "active":
            current_repo = line.removeprefix("### ").strip()
            continue
        if section == "active" and (line.startswith("| [#") or line.startswith("| #")):
            if line.startswith("|----") or "Title" in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            row = _parse_pr_row(cells)
            if row:
                row["repo"] = current_repo
                open_prs.append(row)

        if "## 4. Pre-PR / fork-only WIP" in line:
            section = "wip"
            continue
        if section == "wip" and (
            line.startswith("| pypto |") or line.startswith("| simpler |") or line.startswith("| pypto-lib |")
        ):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 6:
                wip.append(
                    {
                        "repo": cells[0],
                        "branch": cells[1].strip("`"),
                        "sha": cells[2].strip("`"),
                        "plan": cells[3],
                        "npu": cells[4],
                        "blocker": cells[5],
                    }
                )
                if cells[5] and "blocked" in cells[5].lower():
                    blockers.append({"branch": cells[1].strip("`"), "blocker": cells[5]})

    cross_index: list[dict[str, str]] = []
    in_cross = False
    for line in text.splitlines():
        if "<!-- SYNC:CROSSINDEX_START -->" in line:
            in_cross = True
            continue
        if "<!-- SYNC:CROSSINDEX_END -->" in line:
            in_cross = False
            continue
        if in_cross and line.startswith("| ") and not line.startswith("| Plan"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 4 and cells[0].isdigit():
                cross_index.append(
                    {
                        "plan": cells[0],
                        "repo": cells[1],
                        "prs": cells[2],
                        "title": cells[3],
                    }
                )

    last_verified = None
    for line in text.splitlines():
        if line.startswith("*last_verified:"):
            last_verified = line.split(":", 1)[1].strip().rstrip("*")
            break

    return {
        "last_verified": last_verified,
        "dashboard": dashboard,
        "open_prs": open_prs,
        "fork_wip": wip,
        "blockers": blockers,
        "plan_cross_index": cross_index,
    }


def re_search_dashboard(text: str) -> dict[str, int] | None:
    import re

    match = re.search(
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
        text,
    )
    if not match:
        return None
    return {
        "open": int(match.group(1)),
        "merged_all": int(match.group(2)),
        "merged_90d": int(match.group(3)),
        "closed": int(match.group(4)),
    }


def sync_program_status_from_workspace() -> dict[str, Any]:
    status_md = workspace_root() / "pypto-3.0-notes/pr_plans/status_prs.md"
    if not status_md.exists():
        return {"error": f"Missing {status_md}"}
    parsed = parse_status_prs_markdown(status_md.read_text(encoding="utf-8"))
    program_status_path().write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
    return parsed


def load_program_status() -> dict[str, Any]:
    path = program_status_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return sync_program_status_from_workspace()


def program_status_impl() -> dict[str, Any]:
    data = load_program_status()
    # Highlight plan 33 / #1782 dependency for agents
    highlights: list[str] = []
    for item in data.get("fork_wip", []):
        blocker = item.get("blocker", "")
        if "#1782" in blocker or "1782" in blocker:
            highlights.append(
                f"Plan 33 ({item.get('branch')}) blocked on #1782 composite collectives merge"
            )
    for pr in data.get("open_prs", []):
        if pr.get("pr") == "#1782":
            highlights.append(f"Blocking PR open: pypto #1782 ({pr.get('branch')})")
    data["highlights"] = highlights
    data["source"] = "pypto-3.0-notes/pr_plans/status_prs.md (structured)"
    return data
