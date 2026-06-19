from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_hwnative_sys.paths import (
    abstractions_config_path,
    ascend_abstractions_config_path,
    entrypoints_config_path,
    knowledge_config_path,
    load_repos_config,
    project_root,
    resolve_doc_path,
    resolve_workspace_path,
    safe_relpath,
    workspace_root,
)

EPHEMERAL_PREFIXES = ("pypto-3.0-notes/pr_plans/", "pypto-3.0-notes/pull_requests/")
NOTES_FRESHNESS_PATH = "pypto-3.0-notes/NOTES_FRESHNESS.md"

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
        "pypto/include/pypto/backend/910B/",
        "pypto",
        "backend_a2a3",
        "Ascend910B backend handler — A2/A3 alignment and mixed-kernel policy",
        ["pypto codegen"],
        ["AICore launch"],
        ["512B GM granularity; dual-AIV for unsplit mixed kernels; L0C 128KiB"],
    ),
    (
        "pypto/include/pypto/backend/950/",
        "pypto",
        "backend_a5",
        "Ascend950 backend handler — A5 fractal and alignment policy",
        ["pypto codegen"],
        ["AICore launch"],
        ["128B min GM granularity; V2C fractal adapter; L0C 256KiB"],
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
        "simpler/src/common/worker/",
        "simpler",
        "runtime",
        "Device worker execution",
        ["pypto orchestration codegen"],
        ["Ascend AICPU/AICore"],
        [],
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
    (
        "pypto-lib/golden/",
        "pypto-lib",
        "validation",
        "Golden test harness",
        ["pypto"],
        ["device validation"],
        [],
    ),
    (
        "pypto-lib/models/",
        "pypto-lib",
        "models",
        "End-to-end LLM model kernels",
        ["pypto", "pypto-lib"],
        ["training/inference workloads"],
        [],
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
    merged = _load_json(abstractions_config_path())
    ascend_path = ascend_abstractions_config_path()
    if ascend_path.exists():
        ascend_cards = _load_json(ascend_path)
        merged.update(ascend_cards)
    return merged


_ABSTRACTION_ALIASES: dict[str, str] = {
    "aicore-cube": "AIC",
    "cube": "AIC",
    "aicore-vector": "AIV",
    "vector": "AIV",
    "hccl": "HCCLWindow",
    "hcclwindow": "HCCLWindow",
    "commremoteptr": "CommRemotePtr",
    "910b": "Ascend910B",
    "910c": "Ascend910B",
    "950": "Ascend950",
    "arch35": "Ascend950",
    "notifyop": "NotifyOp",
    "waitcmp": "WaitCmp",
}


def _resolve_abstraction_name(name: str) -> str | None:
    abstractions = load_abstractions()
    if name in abstractions:
        return name
    lowered = name.lower().replace(" ", "").replace("_", "")
    key = next((k for k in abstractions if k.lower() == lowered), None)
    if key:
        return key
    alias = _ABSTRACTION_ALIASES.get(lowered)
    if alias and alias in abstractions:
        return alias
    return None


def _bootstrap_prompt_for_task(task_type: str) -> str:
    if task_type in ("ascend_arch", "ascend_runtime", "npu_tuning", "npu_verify_handoff"):
        return "start_ascend_work"
    if task_type.startswith("distributed") or task_type == "host_collectives_program":
        return "start_distributed_work"
    return "start_compiler_work"


def resolve_doc_tier(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/")
    if normalized.startswith("content/"):
        return "mcp-owned"
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
    resolved = resolve_doc_path(relative_path)
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

    exists = resolve_doc_path(path).exists()
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
        "bootstrap_prompt": _bootstrap_prompt_for_task(task_type),
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
        "prompts": ["start_compiler_work", "start_distributed_work", "start_ascend_work", "start_npu_verify"],
    }


def explain_abstraction_impl(name: str) -> dict[str, Any]:
    abstractions = load_abstractions()
    key = _resolve_abstraction_name(name)
    if key is None:
        available = ", ".join(sorted(abstractions)[:25])
        raise ValueError(f"Unknown abstraction '{name}'. Examples: {available}")

    card = abstractions[key]
    return {
        "name": key,
        "layer": card.get("layer"),
        "kind": card.get("kind"),
        "tags": card.get("tags", []),
        "arch_families": card.get("arch_families", []),
        "repos": card.get("repos", []),
        "paths": card.get("paths", []),
        "docs_canonical": [
            {"path": p, "exists": _path_exists(p)}
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
                " ".join(card.get("tags", [])),
                " ".join(card.get("arch_families", [])),
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
                    "tags": card.get("tags", []),
                    "arch_families": card.get("arch_families", []),
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


def _path_exists(relative_path: str) -> bool:
    if relative_path.replace("\\", "/").startswith("content/"):
        return resolve_doc_path(relative_path).exists()
    return resolve_workspace_path(relative_path).exists()


def knowledge_health_impl() -> dict[str, Any]:
    config = load_knowledge_config()
    root = workspace_root()
    missing: list[str] = []
    stale_enriched: list[dict[str, str]] = []
    ascend_issues: list[str] = []
    freshness = _parse_notes_freshness()
    today = date.today()

    def check_path(path: str) -> None:
        if path and not _path_exists(path):
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

    ascend_arch_path = "pypto-3.0-notes/performance_tuning.md/ascend-architectures.md"
    if not _path_exists(ascend_arch_path):
        ascend_issues.append(f"Missing ascend arch reference: {ascend_arch_path}")
    else:
        verified = freshness.get(ascend_arch_path)
        if verified:
            age_days = (today - datetime.strptime(verified, "%Y-%m-%d").date()).days
            if age_days > 30:
                ascend_issues.append(f"Stale ascend-architectures.md ({age_days}d since last_verified)")

    for name in ("which_platform.md", "alignment_rules.md", "hccl_container_checklist.md"):
        content_path = f"content/ascend/{name}"
        if not resolve_doc_path(content_path).exists():
            ascend_issues.append(f"Missing MCP content: {content_path}")

    for resource in config.get("resources", {}).values():
        paths = resource.get("paths", [])
        single = resource.get("path")
        if single:
            paths = [*paths, single]
        for path in paths:
            check_path(path)

    for path in config.get("notes_topics", {}).values():
        check_path(path)

    abstractions = load_abstractions()
    for card in abstractions.values():
        for path in card.get("paths", []):
            check_path(path)
        for path in card.get("docs_canonical", []):
            check_path(path)

    index_build_marker = project_root() / "config" / ".index_build_time"
    last_index_build = None
    if index_build_marker.exists():
        last_index_build = index_build_marker.read_text(encoding="utf-8").strip()

    ascend_route_count = sum(1 for k in config.get("routes", {}) if k.startswith(("ascend_", "npu_")))

    return {
        "config_version": config.get("version", "unknown"),
        "workspace_root": str(root),
        "abstraction_count": len(abstractions),
        "ascend_route_count": ascend_route_count,
        "missing_paths_count": len(missing),
        "missing_paths": list(dict.fromkeys(missing))[:50],
        "stale_enriched_count": len(stale_enriched),
        "stale_enriched": stale_enriched[:20],
        "ascend_issues_count": len(ascend_issues),
        "ascend_issues": ascend_issues[:20],
        "last_index_build": last_index_build,
    }


def render_resource(uri_suffix: str) -> str:
    config = load_knowledge_config()
    resources = config.get("resources", {})
    notes_topics = config.get("notes_topics", {})

    if uri_suffix == "agent/routing":
        topics = list_knowledge_topics_impl()
        lines = ["# Task routing index", ""]
        lines.append("## Compiler / stack")
        for item in topics["task_types"]:
            if item["task_type"].startswith(("ascend_", "npu_")):
                continue
            lines.append(f"- **{item['task_type']}**: {item['description']}")
        lines.append("")
        lines.append("## Ascend architecture / NPU")
        for item in topics["task_types"]:
            if item["task_type"].startswith(("ascend_", "npu_")):
                lines.append(f"- **{item['task_type']}**: {item['description']}")
        lines.append("")
        lines.append("Bootstrap: `start_ascend_work` (focus: arch | tuning | hccl | verify)")
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
        """Explain a stack abstraction: IR/passes/codegen/ISA/runtime or Ascend hardware (AIC, HCCL, etc.)."""
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
        """Check knowledge config health: missing paths, stale enriched docs, Ascend corpus, index build time."""
        return knowledge_health_impl()

    @mcp.tool()
    def ascend_env_check() -> dict[str, Any]:
        """Read-only Ascend/CANN environment check: devices, HCCL preload, Docker hints."""
        from mcp_hwnative_sys.ascend_env import ascend_env_check_impl

        return ascend_env_check_impl()

    @mcp.tool()
    def generate_verify_handoff(
        repo: str,
        branch: str,
        sha: str = "",
        task_type: str = "npu_verify_handoff",
        device_ids: str = "0,1",
        platform: str = "a2a3",
        fork_remote: str = "fork-gbisbas",
    ) -> dict[str, Any]:
        """Generate markdown handoff for developer NPU verification in a container."""
        from mcp_hwnative_sys.handoff import generate_verify_handoff_impl

        return generate_verify_handoff_impl(
            repo=repo,
            branch=branch,
            sha=sha,
            task_type=task_type,
            device_ids=device_ids,
            platform=platform,
            fork_remote=fork_remote,
        )

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

    @mcp.prompt(title="Start Ascend architecture / NPU work")
    def start_ascend_work(focus: str = "arch") -> str:
        focus_route = {
            "arch": "ascend_arch",
            "tuning": "npu_tuning",
            "hccl": "ascend_runtime",
            "runtime": "ascend_runtime",
            "verify": "npu_verify_handoff",
        }.get(focus, "ascend_arch")

        return f"""You are working on Huawei Ascend NPU architecture, tuning, or distributed runtime topics.

Before editing any code:
1. Read MCP resources: hw-native-sys://ascend/hardware, hw-native-sys://ascend/arch_families
2. Call route_task with task_type="{focus_route}"
3. Call explain_abstraction for hardware concepts (AIC, AIV, MTE, HCCLWindow, BackendHandler910B, etc.)
4. Call ascend_env_check when on or handing off to an NPU host/container
5. For developer NPU verify: generate_verify_handoff — agents must NOT open upstream PRs

Ascend expertise layers:
- **Hardware:** AIC (cube), AIV (vector), AICPU scheduler, GM/L1/L0/UB (see ascend/memory_hierarchy)
- **Arch families:** A2A3 (910B/910C) vs A5 (950) — hw-native-sys://ascend/arch_families
- **Distributed:** HCCL windows, signal buffers [NR,1], container checklist — ascend/hccl_container_checklist

Canonical docs are authoritative. Enriched notes (pypto-3.0-notes) are secondary — check tier labels."""

    @mcp.prompt(title="Start NPU container verification (developer gate)")
    def start_npu_verify() -> str:
        return """You are verifying pypto/simpler changes on real Ascend NPUs (developer gate).

Workflow:
1. Call ascend_env_check — confirm devices, CANN_HOME, HCCL LD_PRELOAD path
2. Read hw-native-sys://ascend/hccl_container_checklist
3. Call generate_verify_handoff with repo, branch, platform, device_ids
4. Checkout branch, pip install --no-build-isolation -e ".[dev]"
5. export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so  (test shell only)
6. Run developer_verify_tasks from route_task(npu_verify_handoff) — NOT agent sim tasks
7. Record git rev-parse HEAD; do not open upstream PR unless explicitly asked"""
