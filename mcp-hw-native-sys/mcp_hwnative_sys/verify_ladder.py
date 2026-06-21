from __future__ import annotations

from typing import Any

from mcp_hwnative_sys.paths import load_repos_config


# Longer prefixes first for greedy matching.
_VERIFY_RULES: list[tuple[str, str, list[str]]] = [
    ("pypto/src/codegen/orchestration/", "pypto", ["pypto:codegen_tests", "pypto:unit_tests_fast"]),
    ("pypto/src/codegen/distributed/", "pypto", ["pypto:system_tests_sim", "pypto:codegen_tests"]),
    ("pypto/src/codegen/pto/", "pypto", ["pypto:codegen_tests"]),
    ("pypto/src/codegen/", "pypto", ["pypto:codegen_tests", "pypto:unit_tests_fast"]),
    ("pypto/src/ir/transforms/", "pypto", ["pypto:unit_tests_fast"]),
    ("pypto/src/ir/op/distributed/", "pypto", ["pypto:system_tests_sim"]),
    ("pypto/python/pypto/ir/", "pypto", ["pypto:unit_tests_fast"]),
    ("host_orch", "pypto", ["pypto-tooling:host_collectives_ut_sim"]),
    ("LowerHostTensorCollectives", "pypto", ["pypto-tooling:host_collectives_ut_sim"]),
    ("simpler/src/common/comm/", "simpler", ["simpler:system_tests"]),
    ("simpler/examples/l3/", "simpler", ["simpler:system_tests"]),
    ("simpler/src/", "simpler", ["simpler:unit_tests"]),
    ("PTOAS/", "PTOAS", ["PTOAS:unit_tests"]),
    ("pto-isa/", "pto-isa", ["pto-isa:cpu_sim_tests"]),
    ("pypto-lib/", "pypto-lib", ["pypto-lib:golden_tests"]),
    ("pypto-tooling/profiling/", "pypto-tooling", []),
]


def verify_ladder_impl(changed_paths: list[str]) -> dict[str, Any]:
    if not changed_paths:
        raise ValueError("changed_paths cannot be empty")

    normalized = [p.replace("\\", "/").strip() for p in changed_paths if p.strip()]
    matched_rules: list[dict[str, Any]] = []
    task_keys: list[str] = []

    for path in normalized:
        for prefix, repo, tasks in _VERIFY_RULES:
            if prefix in path or path.startswith(prefix):
                matched_rules.append({"path": path, "prefix": prefix, "repo": repo, "tasks": tasks})
                task_keys.extend(tasks)
                break

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered_tasks: list[str] = []
    for key in task_keys:
        if key in seen:
            continue
        seen.add(key)
        ordered_tasks.append(key)

    raw_config = load_repos_config()
    task_config = raw_config.get("tasks", {})

    ladder: list[dict[str, Any]] = []
    for task_key in ordered_tasks:
        if ":" not in task_key:
            continue
        repo, task_name = task_key.split(":", 1)
        spec = task_config.get(repo, {}).get(task_name) or task_config.get("default", {}).get(task_name)
        if spec is None:
            ladder.append({"task": task_key, "found": False})
            continue
        if isinstance(spec, str):
            entry = {"task": task_key, "found": True, "category": "misc", "risk": "unknown"}
        else:
            entry = {
                "task": task_key,
                "found": True,
                "category": spec.get("category", "misc"),
                "risk": spec.get("risk", "unknown"),
                "duration_hint": spec.get("duration_hint"),
                "developer_only": spec.get("developer_only", False),
                "long_running": spec.get("long_running", False),
            }
        ladder.append(entry)

    return {
        "changed_paths": normalized,
        "matched_rules": matched_rules,
        "suggested_tasks": ordered_tasks,
        "ladder": ladder,
        "note": "Run agent_verify_tasks only; skip developer_only tasks unless explicitly asked.",
    }
