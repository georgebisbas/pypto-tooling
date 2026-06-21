from __future__ import annotations

from typing import Any, Callable

from mcp_hwnative_sys.knowledge import (
    route_task_impl,
    search_abstractions_impl,
)
from mcp_hwnative_sys.programs import load_programs_config, match_program_hints
from mcp_hwnative_sys.program_status import program_status_impl


def _build_read_plan(route: dict[str, Any]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for item in route.get("read_first_canonical", []):
        plan.append(
            {
                "path": item["path"],
                "tier": item.get("tier", "canonical"),
                "max_chars": 8000,
                "priority": len(plan) + 1,
            }
        )
    for item in route.get("read_first_enriched", []):
        plan.append(
            {
                "path": item["path"],
                "tier": "enriched",
                "max_chars": 6000,
                "priority": len(plan) + 1,
                "last_verified": item.get("last_verified"),
                "hint": "Use read_doc(path, section=...) for specific phase",
            }
        )
    for item in route.get("rules", []):
        plan.append(
            {
                "path": item["path"],
                "tier": item.get("tier", "canonical"),
                "max_chars": 4000,
                "priority": len(plan) + 1,
                "kind": "rule",
            }
        )
    return plan[:12]


def _abstraction_seeds(task_type: str, detail: str) -> list[str]:
    query = detail.strip() or task_type.replace("_", " ")
    try:
        matches = search_abstractions_impl(query, max_results=5)
        return [m["name"] for m in matches.get("matches", [])]
    except ValueError:
        return []


def bootstrap_session_impl(
    task_type: str,
    detail: str = "",
    include_health: bool = True,
    health_fetcher: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    route = route_task_impl(task_type, detail)
    read_plan = _build_read_plan(route)
    seeds = _abstraction_seeds(task_type, detail)

    health_summary: dict[str, Any] | None = None
    program_hints: list[dict[str, Any]] = []
    if include_health and health_fetcher is not None:
        health = health_fetcher()
        repos = health.get("repositories", [])
        health_summary = {
            "dirty_repos": [
                {
                    "repo": r.get("repo"),
                    "branch": r.get("branch"),
                    "staged": r.get("staged", 0),
                    "unstaged": r.get("unstaged", 0),
                    "untracked": r.get("untracked", 0),
                }
                for r in repos
                if not r.get("clean", True)
            ],
            "all_branches": {r.get("repo"): r.get("branch") for r in repos},
        }
        for r in repos:
            hints = r.get("active_program_hints", [])
            if hints:
                program_hints.extend(hints)
            branch = r.get("branch", "")
            if branch and r.get("repo"):
                program_hints.extend(match_program_hints(r["repo"], branch))

    # Dedupe program hints
    seen: set[str] = set()
    unique_hints: list[dict[str, Any]] = []
    for hint in program_hints:
        key = str(hint.get("program", hint))
        if key in seen:
            continue
        seen.add(key)
        unique_hints.append(hint)

    status_snippet = None
    try:
        status = program_status_impl()
        status_snippet = {
            "highlights": status.get("highlights", [])[:5],
            "open_pr_count": len(status.get("open_prs", [])),
        }
    except Exception:
        status_snippet = None

    return {
        "task_type": task_type,
        "route": route,
        "read_plan": read_plan,
        "abstraction_seeds": seeds,
        "program_hints": unique_hints,
        "health_summary": health_summary,
        "program_status": status_snippet,
        "session_hint": "Do not re-read docs from read_plan unless editing that area.",
        "bootstrap_prompt": route.get("bootstrap_prompt"),
        "programs_configured": len(load_programs_config().get("programs", [])),
    }
