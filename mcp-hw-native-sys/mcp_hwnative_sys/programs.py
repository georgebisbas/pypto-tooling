from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from mcp_hwnative_sys.paths import load_json_cached, project_root


def programs_config_path() -> Path:
    return project_root() / "config" / "programs.json"


def load_programs_config() -> dict[str, Any]:
    path = programs_config_path()
    if not path.exists():
        return {"programs": []}
    return load_json_cached(path)


def match_program_hints(repo_name: str, branch: str) -> list[dict[str, Any]]:
    """Return active program hints for a repo branch."""
    hints: list[dict[str, Any]] = []
    for program in load_programs_config().get("programs", []):
        if program.get("repo") != repo_name:
            continue
        pattern = str(program.get("branch_pattern", ""))
        if not pattern or not fnmatch.fnmatch(branch, pattern):
            continue
        hints.append(
            {
                "program": program.get("id"),
                "plan": program.get("plan"),
                "memory": program.get("memory"),
                "agent_verify": program.get("agent_verify"),
                "developer_verify": program.get("developer_verify"),
                "route_task": program.get("route_task"),
                "blockers": program.get("blockers", []),
            }
        )
    return hints
