from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_hwnative_sys.paths import (
    abstractions_config_path,
    entrypoints_config_path,
    knowledge_config_path,
    load_repos_config,
    project_root,
    resolve_workspace_path,
    safe_relpath,
    workspace_root,
)

EPHEMERAL_PREFIXES = ("pypto-3.0-notes/pr_plans/", "pypto-3.0-notes/pull_requests/")
NOTES_FRESHNESS_PATH = "pypto-3.0-notes/NOTES_FRESHNESS.md"

_STACK_TRACE_RULES: list[tuple[str, str, str, list[str], list[str]]] = [
    (
        "pypto/src/ir/transforms/",
        "pypto",
        "passes",
        "IR transformation passes (before codegen)",
        ["pypto", "PTOAS", "pto-isa", "simpler"],
        ["codegen"],
    ),
    (
        "pypto/src/codegen/pto/",
        "pypto",
        "codegen_pto",
        "InCore PTO codegen → .pto MLIR",
        ["pypto"],
        ["PTOAS", "pto-isa", "AICore"],
    ),
    (
        "pypto/src/codegen/orchestration/",
        "pypto",
        "codegen_orch",
        "Orchestration codegen → PTO2 runtime C++",
        ["pypto"],
        ["simpler", "AICPU"],
    ),
    (
        "pypto/src/codegen/distributed/",
        "pypto",
        "codegen_dist",
        "Distributed codegen → multi-rank orchestration",
        ["pypto"],
        ["simpler", "comm-domain"],
    ),
    (
        "pypto/src/ir/op/distributed/",
        "pypto",
        "distributed_ops",
        "Distributed IR ops and collectives",
        ["pypto"],
        ["distributed_codegen", "pto-isa comm", "simpler"],
    ),
    (
        "PTOAS/",
        "PTOAS",
        "assembler",
        ".pto MLIR assembler and optimizer",
        ["pypto"],
        ["pto-isa", "AICore binaries"],
    ),
    (
        "pto-isa/include/",
        "pto-isa",
        "isa",
        "Virtual tile ISA C++ implementations",
        ["PTOAS", "pypto"],
        ["AICore execution"],
    ),
    (
        "simpler/src/common/worker/",
        "simpler",
        "runtime",
        "Device worker execution",
        ["pypto orchestration codegen"],
        ["Ascend AICPU/AICore"],
    ),
    (
        "simpler/src/common/comm/",
        "simpler",
        "runtime_comm",
        "Comm-domain and distributed runtime",
        ["pypto distributed codegen"],
        ["multi-chip execution"],
    ),
    (
        "pypto-lib/golden/",
        "pypto-lib",
        "validation",
        "Golden test harness",
        ["pypto"],
        ["device validation"],
    ),
    (
        "pypto-lib/models/",
        "pypto-lib",
        "models",
        "End-to-end LLM model kernels",
        ["pypto", "pypto-lib"],
        ["training/inference workloads"],
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_knowledge_config() -> dict[str, Any]:
    return _load_json(knowledge_config_path())


def load_entrypoints() -> dict[str, Any]:
    return _load_json(entrypoints_config_path())


def load_abstractions() -> dict[str, Any]:
    return _load_json(abstractions_config_path())


def resolve_doc_tier(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/")
    for prefix in EPHEMERAL_PREFIXES:
        if normalized.startswith(prefix):
            return "ephemeral"
    if normalized.startswith("pypto-3.0-notes/"):
        return "enriched"
    return "canonical"


def _parse_notes_freshness() -> dict[str, str]:
    root = workspace_root()
    freshness_path = root / NOTES_FRESHNESS_PATH
    if not freshness_path.exists():
        return {}

    text = freshness_path.read_text(encoding="utf-8")
    mapping: dict[str, str] = {}
    for match in re.finditer(r"\[([^\]]+\.md)\]\(([^)]+)\)[^\n]*\|\s*(\d{4}-\d{2}-\d{2})", text):
        link_path = match.group(2)
        verified = match.group(3)
        if link_path.startswith("../"):
            rel = link_path.removeprefix("../")
        else:
            rel = f"pypto-3.0-notes/{link_path}"
        mapping[rel] = verified
    return mapping


def _doc_front_matter(relative_path: str) -> str:
    tier = resolve_doc_tier(relative_path)
    lines = [f"tier: {tier}", f"path: {relative_path}"]
    if tier == "enriched":
        verified = _parse_notes_freshness().get(relative_path)
        if verified:
            lines.append(f"last_verified: {verified}")
    return "---\n" + "\n".join(lines) + "\n---\n\n"


def _truncate_slice(text: str, max_chars: int, relative_path: str) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    return f"{truncated}\n\n... (truncated — read full file at {relative_path})"


def read_doc_slice(relative_path: str, max_chars: int = 8000) -> str:
    resolved = resolve_workspace_path(relative_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {relative_path}")

    content = resolved.read_text(encoding="utf-8", errors="replace")
    body = _truncate_slice(content, max_chars, relative_path)
    return _doc_front_matter(relative_path) + body


def read_multiple_docs(paths: list[str], max_chars: int = 8000) -> str:
    if not paths:
        raise ValueError("paths cannot be empty")

    parts: list[str] = []
    remaining = max_chars
    for index, path in enumerate(paths):
        if remaining <= 0:
            parts.append(f"\n... (additional docs omitted: {', '.join(paths[index:])})")
            break
        slice_text = read_doc_slice(path, max_chars=remaining)
        parts.append(slice_text)
        remaining = max_chars - sum(len(part) for part in parts)

    return "\n\n".join(parts)


def read_doc_payload(path: str, max_chars: int = 12000) -> dict[str, Any]:
    if max_chars < 500 or max_chars > 50000:
        raise ValueError("max_chars must be between 500 and 50000")

    tier = resolve_doc_tier(path)
    if tier == "ephemeral":
        raise ValueError(f"Refusing to serve ephemeral-tier doc via read_doc: {path}")

    exists = resolve_workspace_path(path).exists()
    content = read_doc_slice(path, max_chars=max_chars) if exists else ""
    return {
        "path": path,
        "tier": tier,
        "exists": exists,
        "content": content,
    }


def _resolve_entrypoints(entrypoint_areas: list[str]) -> dict[str, list[str]]:
    entrypoints = load_entrypoints()
    output: dict[str, list[str]] = {}
    for area in entrypoint_areas:
        if ":" in area:
            repo, key = area.split(":", 1)
            paths = entrypoints.get(repo, {}).get(key, [])
            if paths:
                output[f"{repo}/{key}"] = paths
        else:
            repo_paths = entrypoints.get(area, {})
            if repo_paths:
                output[area] = [path for paths in repo_paths.values() for path in paths]
    return output


def route_task_impl(task_type: str, detail: str = "") -> dict[str, Any]:
    config = load_knowledge_config()
    routes = config.get("routes", {})
    route = routes.get(task_type)
    if route is None:
        available = ", ".join(sorted(routes))
        raise ValueError(f"Unknown task_type '{task_type}'. Available: {available}")

    canonical_docs = [
        {
            "path": path,
            "tier": resolve_doc_tier(path),
            "exists": resolve_workspace_path(path).exists(),
        }
        for path in route.get("read_first_canonical", [])
    ]
    enriched_docs = [
        {
            "path": path,
            "tier": "enriched",
            "exists": resolve_workspace_path(path).exists(),
            "last_verified": _parse_notes_freshness().get(path),
        }
        for path in route.get("read_first_enriched", [])
    ]
    rules = [
        {
            "path": path,
            "tier": resolve_doc_tier(path),
            "exists": resolve_workspace_path(path).exists(),
        }
        for path in route.get("rules", [])
    ]

    return {
        "task_type": task_type,
        "description": route.get("description", ""),
        "detail": detail.strip() or None,
        "read_first_canonical": canonical_docs,
        "read_first_enriched": enriched_docs,
        "rules": rules,
        "entrypoints": _resolve_entrypoints(route.get("entrypoint_areas", [])),
        "verify_tasks": route.get("verify_tasks", []),
        "agent_verify_tasks": route.get("agent_verify_tasks", route.get("verify_tasks", [])),
        "developer_verify_tasks": route.get("developer_verify_tasks", []),
        "resources": [f"hw-native-sys://{uri}" for uri in route.get("resources", [])],
        "bootstrap_prompt": "start_compiler_work" if not task_type.startswith("distributed") and task_type != "host_collectives_program" else "start_distributed_work",
    }


def list_knowledge_topics_impl() -> dict[str, Any]:
    config = load_knowledge_config()
    routes = config.get("routes", {})
    resources = config.get("resources", {})
    notes_topics = config.get("notes_topics", {})

    return {
        "task_types": [
            {
                "task_type": key,
                "description": value.get("description", ""),
                "resources": value.get("resources", []),
            }
            for key, value in sorted(routes.items())
        ],
        "resources": [
            {
                "uri": f"hw-native-sys://{key}",
                "tier": value.get("tier", "canonical"),
            }
            for key, value in sorted(resources.items())
        ],
        "notes_topics": [
            {"topic": key, "uri": f"hw-native-sys://notes/{key}"}
            for key in sorted(notes_topics)
        ],
        "prompts": ["start_compiler_work", "start_distributed_work"],
    }


def explain_abstraction_impl(name: str) -> dict[str, Any]:
    abstractions = load_abstractions()
    # Case-insensitive lookup
    key = next((k for k in abstractions if k.lower() == name.lower()), None)
    if key is None:
        available = ", ".join(sorted(abstractions)[:20])
        raise ValueError(f"Unknown abstraction '{name}'. Examples: {available}")

    card = abstractions[key]
    return {
        "name": key,
        "layer": card.get("layer"),
        "kind": card.get("kind"),
        "repos": card.get("repos", []),
        "paths": card.get("paths", []),
        "docs_canonical": [
            {"path": p, "exists": resolve_workspace_path(p).exists()}
            for p in card.get("docs_canonical", [])
        ],
        "docs_enriched": [
            {
                "path": p,
                "exists": resolve_workspace_path(p).exists(),
                "last_verified": _parse_notes_freshness().get(p),
            }
            for p in card.get("docs_enriched", [])
        ],
        "rules": card.get("rules", []),
        "related": card.get("related", []),
        "downstream": card.get("downstream", []),
        "verify_tasks": card.get("verify_tasks", []),
        "agent_verify_tasks": card.get("agent_verify_tasks", card.get("verify_tasks", [])),
        "developer_verify_tasks": card.get("developer_verify_tasks", []),
        "agent_policy": card.get("agent_policy", []),
    }


def search_abstractions_impl(query: str, max_results: int = 20) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query cannot be empty")
    if max_results < 1 or max_results > 100:
        raise ValueError("max_results must be between 1 and 100")

    needle = query.strip().lower()
    abstractions = load_abstractions()
    matches: list[dict[str, Any]] = []

    for name, card in abstractions.items():
        haystack = " ".join(
            [
                name,
                str(card.get("layer", "")),
                str(card.get("kind", "")),
                " ".join(card.get("repos", [])),
                " ".join(card.get("related", [])),
                " ".join(card.get("downstream", [])),
            ]
        ).lower()
        if needle in haystack or needle in name.lower():
            matches.append(
                {
                    "name": name,
                    "layer": card.get("layer"),
                    "kind": card.get("kind"),
                    "repos": card.get("repos", []),
                }
            )
        if len(matches) >= max_results:
            break

    return {"query": query, "match_count": len(matches), "matches": matches}


def find_entrypoints_impl(repo: str, area: str = "") -> dict[str, Any]:
    entrypoints = load_entrypoints()
    repo_map = entrypoints.get(repo)
    if repo_map is None:
        available = ", ".join(sorted(entrypoints))
        raise ValueError(f"Unknown repo '{repo}'. Available: {available}")

    if area:
        paths = repo_map.get(area)
        if paths is None:
            available_areas = ", ".join(sorted(repo_map))
            raise ValueError(f"Unknown area '{area}' for repo '{repo}'. Available: {available_areas}")
        return {"repo": repo, "area": area, "paths": paths}

    return {"repo": repo, "areas": repo_map}


def trace_in_stack_impl(symbol_or_path: str) -> dict[str, Any]:
    if not symbol_or_path.strip():
        raise ValueError("symbol_or_path cannot be empty")

    token = symbol_or_path.strip().replace("\\", "/")

    # Try abstraction index first
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

    # Path-prefix heuristics
    for prefix, repo, stage, description, upstream, downstream in _STACK_TRACE_RULES:
        if token.startswith(prefix) or prefix.rstrip("/") in token:
            return {
                "input": symbol_or_path,
                "matched_as": "path_prefix",
                "repo": repo,
                "pipeline_stage": stage,
                "description": description,
                "upstream": upstream,
                "downstream": downstream,
            }

    return {
        "input": symbol_or_path,
        "matched_as": "none",
        "hint": "Use explain_abstraction or search_abstractions for concepts; pass a repo-relative path for heuristics.",
    }


def knowledge_health_impl() -> dict[str, Any]:
    config = load_knowledge_config()
    root = workspace_root()
    missing: list[str] = []
    stale_enriched: list[dict[str, str]] = []
    freshness = _parse_notes_freshness()
    today = date.today()

    def check_path(path: str) -> None:
        if not (root / path).exists():
            missing.append(path)

    for route in config.get("routes", {}).values():
        for path in route.get("read_first_canonical", []):
            check_path(path)
        for path in route.get("read_first_enriched", []):
            check_path(path)
            verified = freshness.get(path)
            if verified:
                verified_date = datetime.strptime(verified, "%Y-%m-%d").date()
                age_days = (today - verified_date).days
                if age_days > 30:
                    stale_enriched.append({"path": path, "last_verified": verified, "age_days": str(age_days)})

    for resource in config.get("resources", {}).values():
        for path in resource.get("paths", [resource.get("path")]):
            if path:
                check_path(path)

    for path in config.get("notes_topics", {}).values():
        check_path(path)

    abstractions = load_abstractions()
    index_build_marker = project_root() / "config" / ".index_build_time"
    last_index_build = None
    if index_build_marker.exists():
        last_index_build = index_build_marker.read_text(encoding="utf-8").strip()

    return {
        "config_version": config.get("version", "unknown"),
        "workspace_root": str(root),
        "abstraction_count": len(abstractions),
        "missing_paths_count": len(missing),
        "missing_paths": missing[:50],
        "stale_enriched_count": len(stale_enriched),
        "stale_enriched": stale_enriched[:20],
        "last_index_build": last_index_build,
    }


def render_resource(uri_suffix: str) -> str:
    config = load_knowledge_config()
    resources = config.get("resources", {})
    notes_topics = config.get("notes_topics", {})

    if uri_suffix == "agent/routing":
        topics = list_knowledge_topics_impl()
        lines = ["# Task routing index", ""]
        for item in topics["task_types"]:
            lines.append(f"- **{item['task_type']}**: {item['description']}")
        return _doc_front_matter("config/knowledge.json") + "\n".join(lines)

    if uri_suffix.startswith("notes/"):
        topic = uri_suffix.removeprefix("notes/")
        path = notes_topics.get(topic)
        if path is None:
            available = ", ".join(sorted(notes_topics))
            raise ValueError(f"Unknown notes topic '{topic}'. Available: {available}")
        return read_doc_slice(path, max_chars=8000)

    resource = resources.get(uri_suffix)
    if resource is None:
        available = ", ".join(sorted(resources.keys()))
        raise ValueError(f"Unknown resource '{uri_suffix}'. Available: {available}")

    paths = resource.get("paths")
    if paths:
        return read_multiple_docs(paths, max_chars=resource.get("max_chars", 8000))

    path = resource.get("path")
    if not path:
        raise ValueError(f"Resource '{uri_suffix}' has no path configured")
    return read_doc_slice(path, max_chars=resource.get("max_chars", 8000))


def get_repository_meta() -> dict[str, Any]:
    return load_repos_config().get("repository_meta", {})


def _register_resource(mcp: FastMCP, uri_suffix: str) -> None:
    def _handler() -> str:
        return render_resource(uri_suffix)

    _handler.__name__ = f"resource_{uri_suffix.replace('/', '_')}"
    mcp.resource(f"hw-native-sys://{uri_suffix}")(_handler)


def register_knowledge(mcp: FastMCP) -> None:
    """Register knowledge-layer resources, tools, and prompts on the MCP server."""

    config = load_knowledge_config()

    for uri_suffix in config.get("resources", {}):
        _register_resource(mcp, uri_suffix)

    def agent_routing_resource() -> str:
        return render_resource("agent/routing")

    mcp.resource("hw-native-sys://agent/routing")(agent_routing_resource)

    for topic in config.get("notes_topics", {}):
        def _make_notes_handler(note_topic: str):
            def _handler() -> str:
                return render_resource(f"notes/{note_topic}")

            _handler.__name__ = f"notes_{note_topic}"
            return _handler

        mcp.resource(f"hw-native-sys://notes/{topic}")(_make_notes_handler(topic))

    @mcp.tool()
    def route_task(task_type: str, detail: str = "") -> dict[str, Any]:
        """Return read-first docs, rules, entrypoints, and verify tasks for a compiler workflow."""
        return route_task_impl(task_type, detail)

    @mcp.tool()
    def list_knowledge_topics() -> dict[str, Any]:
        """List available task routes, MCP resources, notes topics, and bootstrap prompts."""
        return list_knowledge_topics_impl()

    @mcp.tool()
    def read_doc(path: str, max_chars: int = 12000) -> dict[str, Any]:
        """Read a workspace document with tier labeling (canonical or enriched only)."""
        return read_doc_payload(path, max_chars)

    @mcp.tool()
    def explain_abstraction(name: str) -> dict[str, Any]:
        """Explain a compiler/runtime abstraction by name (IR nodes, passes, codegen, ISA, runtime)."""
        return explain_abstraction_impl(name)

    @mcp.tool()
    def search_abstractions(query: str, max_results: int = 20) -> dict[str, Any]:
        """Search the abstraction index by keyword."""
        return search_abstractions_impl(query, max_results)

    @mcp.tool()
    def find_entrypoints(repo: str, area: str = "") -> dict[str, Any]:
        """Find code entrypoints for a repo and optional area (e.g. pypto, codegen_orch)."""
        return find_entrypoints_impl(repo, area)

    @mcp.tool()
    def trace_in_stack(symbol_or_path: str) -> dict[str, Any]:
        """Trace where a symbol or path sits in the pypto→PTOAS→pto-isa→simpler stack."""
        return trace_in_stack_impl(symbol_or_path)

    @mcp.tool()
    def knowledge_health() -> dict[str, Any]:
        """Check knowledge config health: missing paths, stale enriched docs, index build time."""
        return knowledge_health_impl()

    @mcp.prompt(title="Start full-stack compiler work")
    def start_compiler_work(area: str = "stack_overview") -> str:
        return f"""You are working on the hw-native-sys compiler stack for Ascend NPUs.

Before editing any code:
1. Read MCP resources: hw-native-sys://overview/ecosystem and hw-native-sys://agent/invariants
2. Call route_task with task_type="{area}"
3. Call repository_health with include_clean=false
4. For specific concepts, use explain_abstraction or search_abstractions
5. Run the verify_tasks from route_task before claiming work is done

Stack: pypto (compiler) → PTOAS (assembler) → pto-isa (tile ISA) → simpler (runtime). pypto-lib is the model/harness layer on top.

Canonical docs are authoritative. Enriched notes (pypto-3.0-notes) are secondary — check tier labels."""

    @mcp.prompt(title="Start distributed / large-scale work")
    def start_distributed_work(focus: str = "collectives") -> str:
        focus_route = {
            "collectives": "distributed_collectives",
            "host_collectives": "host_collectives_program",
            "codegen": "distributed_codegen",
            "runtime": "distributed_runtime",
            "inference": "large_model_inference",
        }.get(focus, "distributed")

        return f"""You are working on distributed / large-scale training or inference on Ascend NPUs.

Before editing any code:
1. Read MCP resources: hw-native-sys://pypto/distributed, hw-native-sys://agent/distributed_work_policy, hw-native-sys://flows/distributed_allreduce
2. Call route_task with task_type="{focus_route}" (use host_collectives_program for plan 33 host builtins)
3. Call explain_abstraction for relevant concepts (e.g. host_collectives_program, LowerHostTensorCollectives)
4. Call repository_health with include_clean=false — check active_program_hints on pypto
5. Read notes topic host_collectives when resuming fork work (hw-native-sys://notes/host_collectives)

Verification split (George / gbisbas workflow):
- **Agent gate:** run agent_verify_tasks from route_task via run_task — for host collectives use pypto-tooling:host_collectives_ut_sim (sim Docker, not bare-metal pytest).
- **Developer gate:** pypto:host_collectives_st_npu on NPU — agents must NOT run this or open upstream PRs unless asked.
- **Do not commit** pypto/runtime submodule pointer changes unless the plan explicitly scopes runtime.
- Push only to fork-gbisbas; record git rev-parse HEAD in pypto-3.0-notes/memories/ when handing off.

Distinguish compiler layer (pypto distributed ops/codegen) from runtime layer (simpler comm-domain, L3 worker)."""
