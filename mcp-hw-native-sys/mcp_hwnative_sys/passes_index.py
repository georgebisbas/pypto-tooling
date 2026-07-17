from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mcp_hwnative_sys.paths import project_root, workspace_root


def passes_index_path() -> Path:
    return project_root() / "config" / "passes_index.json"


def _phase_for_order(order: int) -> str:
    if order <= 7:
        return "1_normalization_ssa"
    if order <= 14:
        return "2_scope_tensor_lowering"
    if order <= 21:
        return "3_tile_layout"
    if order <= 26:
        return "4_cross_core_scheduling"
    if order <= 32:
        return "5_memory"
    return "6_codegen_finalization"


def _default_verify_for_pass(name: str) -> list[str]:
    distributed = {
        "MaterializeCommDomainScopes",
        "LowerHostTensorCollectives",
        "LowerCompositeOps",
    }
    codegen = {
        "DeriveCallDirections",
        "AutoDeriveTaskDependencies",
        "MaterializeRuntimeScopes",
        "NormalizeReturnOrder",
    }
    if name in distributed:
        return ["pypto:system_tests_sim"]
    if name in codegen:
        return ["pypto:codegen_tests"]
    return ["pypto:unit_tests_fast"]


def build_passes_index() -> dict[str, Any]:
    root = workspace_root()
    path = root / "pypto/python/pypto/ir/pass_manager.py"
    if not path.exists():
        return {
            "version": "1.0.0",
            "strategy": "Default",
            "passes": [],
            "warning": f"pass_manager.py not found at {path}",
        }

    text = path.read_text(encoding="utf-8")
    # Extract pass names from PassSpec tuples: ("PassName", lambda: passes.foo())
    names = re.findall(r'\(\s*"([A-Z][A-Za-z0-9]+)"\s*,\s*lambda:\s*passes\.', text)
    # Deduplicate preserving first occurrence (Simplify appears twice in Default pipeline)
    seen: set[str] = set()
    ordered_names: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered_names.append(name)

    passes: list[dict[str, Any]] = []
    for order, name in enumerate(ordered_names, start=1):
        passes.append(
            {
                "name": name,
                "order": order,
                "phase": _phase_for_order(order),
                "strategy": "Default",
                "verify_tasks": _default_verify_for_pass(name),
            }
        )

    result: dict[str, Any] = {
        "version": "1.0.0",
        "generated_from": "pypto/python/pypto/ir/pass_manager.py",
        "strategy": "Default",
        "pypto_pass_count": len(passes),
        "passes": passes,
    }
    if not passes:
        # The scrape regex is tied to the ("Name", lambda: passes.…) shape of
        # pass_manager.py; a refactor there yields zero matches. Flag it so
        # knowledge_health surfaces the failure instead of reporting 0 passes.
        result["warning"] = (
            "Extracted 0 passes from pass_manager.py — the extraction regex "
            "may be stale (pass_manager.py refactored)."
        )
    return result


def load_passes_index() -> dict[str, Any]:
    path = passes_index_path()
    if not path.exists():
        data = build_passes_index()
        # Best-effort cache write: a read-only config/ (e.g. installed package
        # or container) must not make this read-path tool fail.
        try:
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
        return data
    return json.loads(path.read_text(encoding="utf-8"))


def explain_pass_impl(name: str) -> dict[str, Any]:
    index = load_passes_index()
    passes = index.get("passes", [])
    key = name.strip()
    match = next((p for p in passes if p["name"] == key), None)
    if match is None:
        # case-insensitive fallback
        lowered = key.lower()
        match = next((p for p in passes if p["name"].lower() == lowered), None)
    if match is None:
        available = ", ".join(p["name"] for p in passes[:15])
        raise ValueError(f"Unknown pass '{name}'. Examples: {available}")

    order = match["order"]
    neighbors = {
        "previous": passes[order - 2]["name"] if order > 1 else None,
        "next": passes[order]["name"] if order < len(passes) else None,
    }

    from mcp_hwnative_sys.knowledge import load_abstractions

    abstractions = load_abstractions()
    related = [k for k, card in abstractions.items() if key in card.get("related", []) or key == k]

    return {
        "name": match["name"],
        "order": order,
        "phase": match["phase"],
        "strategy": match.get("strategy", "Default"),
        "neighbors": neighbors,
        "verify_tasks": match.get("verify_tasks", []),
        "related_abstractions": related[:10],
        "entrypoint_hint": f"pypto/src/ir/transforms/ (search for {match['name']})",
    }
