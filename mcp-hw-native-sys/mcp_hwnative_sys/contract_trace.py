from __future__ import annotations

import json
from typing import Any

from mcp_hwnative_sys.knowledge import load_abstractions
from mcp_hwnative_sys.paths import contract_artifacts_config_path
from mcp_hwnative_sys.program_status import program_status_impl

# Path-prefix heuristics (moved from knowledge.py to avoid import cycles)
_STACK_TRACE_RULES: list[tuple[str, str, str, str, list[str], list[str], list[str]]] = [
    (
        "pypto/src/ir/transforms/",
        "pypto",
        "passes",
        "IR transformation passes (before codegen)",
        ["pypto", "PTOAS", "pto-isa", "simpler"],
        ["codegen"],
        [],
    ),
    (
        "pypto/src/codegen/pto/",
        "pypto",
        "codegen_pto",
        "InCore PTO codegen → .pto MLIR",
        ["pypto"],
        ["PTOAS", "pto-isa", "AICore"],
        ["A2A3 mixed kernels may inject GM pipe buffer (RequiresGMPipeBuffer); A5 uses on-chip fractal path"],
    ),
    (
        "pypto/src/codegen/orchestration/",
        "pypto",
        "codegen_orch",
        "Orchestration codegen → PTO2 runtime C++",
        ["pypto"],
        ["simpler", "AICPU"],
        [],
    ),
    (
        "pypto/src/codegen/distributed/",
        "pypto",
        "codegen_dist",
        "Distributed codegen → multi-rank orchestration",
        ["pypto"],
        ["simpler", "comm-domain"],
        [],
    ),
    (
        "pypto/src/ir/op/distributed/",
        "pypto",
        "distributed_ops",
        "Distributed IR ops and collectives",
        ["pypto"],
        ["distributed_codegen", "pto-isa comm", "simpler"],
        ["HCCL windows; signal [NR,1]; notify/wait lowered to TNOTIFY/TWAIT"],
    ),
    (
        "PTOAS/",
        "PTOAS",
        "assembler",
        ".pto MLIR assembler and optimizer",
        ["pypto"],
        ["pto-isa", "AICore binaries"],
        [],
    ),
    (
        "pto-isa/include/",
        "pto-isa",
        "isa",
        "Virtual tile ISA C++ implementations",
        ["PTOAS", "pypto"],
        ["AICore execution"],
        ["MTE pipes bridge GM/L1/L0; cube vs vector instruction families"],
    ),
    (
        "simpler/src/common/comm/",
        "simpler",
        "runtime_comm",
        "Comm-domain and distributed runtime",
        ["pypto distributed codegen"],
        ["multi-chip execution"],
        ["HCCL window layout; CommRemotePtr peer addressing; LD_PRELOAD for comm_init"],
    ),
]


def _trace_stack_base(symbol_or_path: str) -> dict[str, Any]:
    if not symbol_or_path.strip():
        raise ValueError("symbol_or_path cannot be empty")

    token = symbol_or_path.strip().replace("\\", "/")
    abstractions = load_abstractions()
    abs_key = next((k for k in abstractions if k.lower() == token.lower()), None)
    if abs_key:
        card = abstractions[abs_key]
        return {
            "input": symbol_or_path,
            "matched_as": "abstraction",
            "name": abs_key,
            "layer": card.get("layer"),
            "repos": card.get("repos", []),
            "paths": card.get("paths", []),
            "upstream": card.get("related", []),
            "downstream": card.get("downstream", []),
            "verify_tasks": card.get("verify_tasks", []),
        }

    for prefix, repo, stage, description, upstream, downstream, arch_notes in _STACK_TRACE_RULES:
        if token.startswith(prefix) or prefix.rstrip("/") in token:
            result: dict[str, Any] = {
                "input": symbol_or_path,
                "matched_as": "path_prefix",
                "repo": repo,
                "pipeline_stage": stage,
                "description": description,
                "upstream": upstream,
                "downstream": downstream,
            }
            if arch_notes:
                result["arch_implications"] = arch_notes
            return result

    return {
        "input": symbol_or_path,
        "matched_as": "none",
        "hint": "Use explain_abstraction or search_abstractions for concepts; pass a repo-relative path for heuristics.",
    }

_contract_artifacts_cache: dict[str, dict[str, Any]] | None = None


def _load_contract_artifacts() -> dict[str, dict[str, Any]]:
    global _contract_artifacts_cache
    if _contract_artifacts_cache is None:
        path = contract_artifacts_config_path()
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                _contract_artifacts_cache = json.load(fh)
        else:
            _contract_artifacts_cache = {}
    return _contract_artifacts_cache


def trace_contract_impl(symbol_or_path: str) -> dict[str, Any]:
    """Trace symbol through dependency triangle with contract artifacts."""
    base = _trace_stack_base(symbol_or_path)
    token = symbol_or_path.strip()
    contract_artifacts = _load_contract_artifacts()

    # Match contract artifacts by exact or case-insensitive key
    contract_key = None
    for key in contract_artifacts:
        if key.lower() == token.lower() or key in token:
            contract_key = key
            break

    artifacts = contract_artifacts.get(contract_key, {}) if contract_key else {}

    # Enrich from abstraction card if present
    abstractions = load_abstractions()
    abs_key = next((k for k in abstractions if k.lower() == token.lower()), None)
    if abs_key and not artifacts:
        card = abstractions[abs_key]
        artifacts = {
            "ir_layer": card.get("layer"),
            "paths": card.get("paths", []),
            "downstream": card.get("downstream", []),
            "verify_tasks": card.get("verify_tasks", []),
        }

    # Link to program status for active PR blockers
    status = program_status_impl()
    pr_links: list[str] = []
    for pr in status.get("open_prs", []):
        title = pr.get("title", "").lower()
        if token.lower() in title or (contract_key and contract_key.lower() in title):
            pr_links.append(f"{pr.get('repo')} {pr.get('pr')}: {pr.get('title')}")

    triangle_raw = {
        "pypto_ir": artifacts.get("ir_layer") or base.get("layer"),
        "pto_mlir": artifacts.get("pto_mlir"),
        "pto_isa": artifacts.get("pto_isa"),
        "orch_abi": artifacts.get("orch_abi"),
        "runtime": artifacts.get("runtime"),
    }
    contract_triangle = {k: v for k, v in triangle_raw.items() if v is not None}

    result: dict[str, Any] = {
        **base,
        "cross_layer_verify": artifacts.get("verify_tasks", base.get("verify_tasks", [])),
        "active_pr_links": pr_links[:5],
        "program_highlights": status.get("highlights", [])[:5],
    }
    if contract_triangle:
        result["contract_triangle"] = contract_triangle
    if contract_key:
        result["matched_contract"] = contract_key
    return result
