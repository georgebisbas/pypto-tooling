from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from mcp_hwnative_sys.doc_sections import build_section_toc, extract_section
from mcp_hwnative_sys.paths import (
    abstractions_config_path,
    ascend_abstractions_config_path,
    entrypoints_config_path,
    knowledge_config_path,
    load_json_cached,
    load_repos_config,
    project_root,
    resolve_doc_path,
    resolve_workspace_path,
    workspace_root,
)

EPHEMERAL_PREFIXES = ("pypto-3.0-notes/pr_plans/", "pypto-3.0-notes/pull_requests/")
NOTES_FRESHNESS_PATH = "pypto-3.0-notes/NOTES_FRESHNESS.md"


def load_knowledge_config() -> dict[str, Any]:
    return load_json_cached(knowledge_config_path())


def load_entrypoints() -> dict[str, Any]:
    return load_json_cached(entrypoints_config_path())


# Merged abstractions are cached against the mtimes of both source files so an
# edit to either abstractions.json or ascend_abstractions.json is picked up.
_abstractions_cache: tuple[tuple[float, float], dict[str, Any]] | None = None


def load_abstractions() -> dict[str, Any]:
    global _abstractions_cache
    base_path = abstractions_config_path()
    ascend_path = ascend_abstractions_config_path()
    base_mtime = base_path.stat().st_mtime
    ascend_mtime = ascend_path.stat().st_mtime if ascend_path.exists() else 0.0
    key = (base_mtime, ascend_mtime)
    if _abstractions_cache is not None and _abstractions_cache[0] == key:
        return _abstractions_cache[1]
    merged = dict(load_json_cached(base_path))
    if ascend_path.exists():
        merged.update(load_json_cached(ascend_path))
    _abstractions_cache = (key, merged)
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
    if normalized.startswith("pypto_top_level_documents/"):
        return "design"
    return "canonical"


# Cache the parsed freshness table against the source file's mtime — it is
# read once per enriched path in knowledge_health_impl's loops.
_notes_freshness_cache: tuple[float, dict[str, str]] | None = None


def _parse_notes_freshness() -> dict[str, str]:
    global _notes_freshness_cache
    root = workspace_root()
    freshness_path = root / NOTES_FRESHNESS_PATH
    if not freshness_path.exists():
        _notes_freshness_cache = None
        return {}

    mtime = freshness_path.stat().st_mtime
    if _notes_freshness_cache is not None and _notes_freshness_cache[0] == mtime:
        return _notes_freshness_cache[1]

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
    _notes_freshness_cache = (mtime, mapping)
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


def read_doc_slice(
    relative_path: str,
    max_chars: int = 8000,
    section: str | None = None,
) -> str:
    resolved = resolve_doc_path(relative_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {relative_path}")

    content = resolved.read_text(encoding="utf-8", errors="replace")
    if section:
        extracted = extract_section(content, section)
        if extracted is None:
            toc = build_section_toc(content, max_entries=20)
            body = (
                f"Section '{section}' not found in {relative_path}.\n\n"
                f"{toc}\n\n"
                f"... (falling back to document head)\n\n"
                f"{_truncate_slice(content, max_chars, relative_path)}"
            )
        else:
            body = _truncate_slice(extracted, max_chars, relative_path)
    else:
        body = _truncate_slice(content, max_chars, relative_path)
    return _doc_front_matter(relative_path) + body


def read_multiple_docs(
    paths: list[str],
    max_chars: int = 8000,
    sections: list[str | None] | None = None,
) -> str:
    if not paths:
        raise ValueError("paths cannot be empty")

    # max_chars is a global budget across all docs. Track usage incrementally
    # so the loop stays O(n) rather than re-summing every part each iteration.
    parts: list[str] = []
    used = 0
    for index, path in enumerate(paths):
        remaining = max_chars - used
        if remaining <= 0:
            parts.append(f"\n... (additional docs omitted: {', '.join(paths[index:])})")
            break
        section = None
        if sections and index < len(sections):
            section = sections[index]
        slice_text = read_doc_slice(path, max_chars=remaining, section=section)
        parts.append(slice_text)
        used += len(slice_text)

    return "\n\n".join(parts)


def read_doc_payload(path: str, max_chars: int = 12000, section: str = "") -> dict[str, Any]:
    if max_chars < 500 or max_chars > 50000:
        raise ValueError("max_chars must be between 500 and 50000")

    tier = resolve_doc_tier(path)
    if tier == "ephemeral":
        raise ValueError(f"Refusing to serve ephemeral-tier doc via read_doc: {path}")

    exists = resolve_doc_path(path).exists()
    section_arg = section.strip() or None
    content = read_doc_slice(path, max_chars=max_chars, section=section_arg) if exists else ""
    return {
        "path": path,
        "tier": tier,
        "exists": exists,
        "section": section_arg,
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


def _abstraction_relevance(name: str, card: dict[str, Any], needle: str) -> int:
    name_lower = name.lower()
    if name_lower == needle:
        return 4
    if needle in name_lower:
        return 3
    tags = " ".join(card.get("tags", [])).lower()
    if needle in tags:
        return 2
    layer = str(card.get("layer", "")).lower()
    kind = str(card.get("kind", "")).lower()
    if needle in layer or needle in kind:
        return 1
    return 0


def search_abstractions_impl(
    query: str,
    max_results: int = 20,
    fields: str = "summary",
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query cannot be empty")
    if max_results < 1 or max_results > 100:
        raise ValueError("max_results must be between 1 and 100")

    needle = query.strip().lower()
    abstractions = load_abstractions()
    scored: list[tuple[int, str, dict[str, Any]]] = []

    for name, card in abstractions.items():
        haystack = " ".join(
            [
                name,
                str(card.get("layer", "")),
                str(card.get("kind", "")),
                str(card.get("one_liner", "")),
                " ".join(card.get("tags", [])),
                " ".join(card.get("arch_families", [])),
                " ".join(card.get("repos", [])),
                " ".join(card.get("related", [])),
                " ".join(card.get("downstream", [])),
            ]
        ).lower()
        if needle in haystack or needle in name.lower():
            scored.append((_abstraction_relevance(name, card, needle), name, card))

    scored.sort(key=lambda t: -t[0])
    matches: list[dict[str, Any]] = []
    for _, name, card in scored[:max_results]:
        if fields == "full":
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
        else:
            matches.append(
                {
                    "name": name,
                    "layer": card.get("layer"),
                    "one_liner": card.get("one_liner") or card.get("kind", ""),
                }
            )

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
    from mcp_hwnative_sys.contract_trace import trace_contract_impl

    return trace_contract_impl(symbol_or_path)


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

    ascend_arch_path = "pypto-3.0-notes/performance_tuning/ascend-architectures.md"
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

    # Surface a passes-index scrape failure (e.g. pass_manager.py refactored so
    # the extraction regex no longer matches) instead of silently reporting 0.
    from mcp_hwnative_sys.passes_index import load_passes_index

    passes_index = load_passes_index()
    passes_index_warning = passes_index.get("warning")

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
        "pass_count": len(passes_index.get("passes", [])),
        "passes_index_warning": passes_index_warning,
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
    section_hint = resource.get("section_hint")
    if paths:
        sections = [section_hint if index == 0 else None for index in range(len(paths))]
        return read_multiple_docs(paths, max_chars=resource.get("max_chars", 8000), sections=sections)

    path = resource.get("path")
    if not path:
        raise ValueError(f"Resource '{uri_suffix}' has no path configured")
    return read_doc_slice(
        path,
        max_chars=resource.get("max_chars", 8000),
        section=section_hint,
    )


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
    def list_task_types() -> list[dict[str, str]]:
        """Return all valid task_type values with one-line descriptions for use with bootstrap_session and route_task."""
        config = load_knowledge_config()
        routes = config.get("routes", {})
        return [
            {"task_type": key, "description": value.get("description", "")}
            for key, value in sorted(routes.items())
        ]

    @mcp.tool()
    def route_task(
        task_type: Annotated[str, Field(description='Task type key — use list_task_types() to enumerate valid values, e.g. "distributed_codegen", "ascend_arch"')],
        detail: Annotated[str, Field(description="Optional free-text context (e.g. symbol or feature name) passed through to the routing output")] = "",
    ) -> dict[str, Any]:
        """Return read-first docs, rules, entrypoints, and verify tasks for a compiler workflow."""
        return route_task_impl(task_type, detail)

    @mcp.tool()
    def list_knowledge_topics() -> dict[str, Any]:
        """List available task routes, MCP resources, notes topics, and bootstrap prompts."""
        return list_knowledge_topics_impl()

    @mcp.tool()
    def read_doc(
        path: Annotated[str, Field(description='Document path. Paths starting with "content/" are MCP-owned (project-relative). All others are workspace-relative (e.g. "pypto-3.0-notes/arch.md"). Use list_knowledge_topics() to discover registered paths.')],
        max_chars: Annotated[int, Field(description="Maximum characters to return (500–50000)", ge=500, le=50000)] = 12000,
        section: Annotated[str, Field(description='Extract a specific markdown section by exact heading text (case-sensitive). Leave empty to read from the top. Use read_doc with a bad section name to see the TOC.')] = "",
    ) -> dict[str, Any]:
        """Read a workspace document with tier labeling. Optional section extracts a markdown heading."""
        return read_doc_payload(path, max_chars, section)

    @mcp.tool()
    def explain_abstraction(
        name: Annotated[str, Field(description='Abstraction name or alias, e.g. "AIC", "HCCLWindow", "Ascend910B", "cube". Use search_abstractions() to discover names.')],
    ) -> dict[str, Any]:
        """Explain a stack abstraction: IR/passes/codegen/ISA/runtime or Ascend hardware (AIC, HCCL, etc.)."""
        return explain_abstraction_impl(name)

    @mcp.tool()
    def search_abstractions(
        query: Annotated[str, Field(description="Keyword to search across abstraction names, layers, kinds, tags, and related fields")],
        max_results: Annotated[int, Field(description="Maximum results to return (1–100)", ge=1, le=100)] = 20,
        fields: Annotated[str, Field(description='"summary" returns name+layer+one_liner; "full" adds tags, arch_families, repos')] = "summary",
    ) -> dict[str, Any]:
        """Search the abstraction index by keyword. Results are ranked by relevance (exact name > name-contains > tag > layer/kind)."""
        return search_abstractions_impl(query, max_results, fields)

    @mcp.tool()
    def explain_pass(
        name: Annotated[str, Field(description="Pass name from the Default pipeline, e.g. LowerCompositeOps. Case-insensitive fallback is applied.")],
    ) -> dict[str, Any]:
        """Explain a pass in the Default pipeline: order, phase, neighbors, verify tasks."""
        from mcp_hwnative_sys.passes_index import explain_pass_impl

        return explain_pass_impl(name)

    @mcp.tool()
    def program_status() -> dict[str, Any]:
        """Structured PR/plan status from status_prs.md (open PRs, blockers, plan cross-index)."""
        from mcp_hwnative_sys.program_status import program_status_impl

        return program_status_impl()

    @mcp.tool()
    def verify_ladder(
        changed_paths: Annotated[list[str], Field(description='List of changed file paths (workspace-relative or repo-prefixed), e.g. ["pypto/src/codegen/pto/foo.cc", "simpler/src/common/comm/bar.cc"]. Used to derive minimal verify task set.')],
    ) -> dict[str, Any]:
        """Suggest minimal verify tasks for a set of changed file paths."""
        from mcp_hwnative_sys.verify_ladder import verify_ladder_impl

        return verify_ladder_impl(changed_paths)

    @mcp.tool()
    def summarize_profile(
        run_dir: Annotated[str, Field(description="Path to a profiling campaign directory containing results.json. Accepts workspace-relative or absolute paths.")],
    ) -> dict[str, Any]:
        """Summarize a pypto-tooling profiling campaign directory (results.json, anomalies)."""
        from mcp_hwnative_sys.profiling_summarize import summarize_profile_impl

        return summarize_profile_impl(run_dir)

    @mcp.tool()
    def trace_contract(
        symbol_or_path: Annotated[str, Field(description='Symbol name or path to trace through the stack (e.g. "LowerHostTensorCollectives", "pypto/src/codegen/distributed/foo.cc"). Matched against abstraction cards, path-prefix rules, and contract artifacts.')],
    ) -> dict[str, Any]:
        """Trace symbol through dependency triangle with contract artifacts and cross-layer verify."""
        from mcp_hwnative_sys.contract_trace import trace_contract_impl

        return trace_contract_impl(symbol_or_path)

    @mcp.tool()
    def find_entrypoints(
        repo: Annotated[str, Field(description='Repository name from list_repositories(), e.g. "pypto", "simpler"')],
        area: Annotated[str, Field(description='Optional sub-area key within the repo, e.g. "codegen_orch". Leave empty to list all areas for the repo.')] = "",
    ) -> dict[str, Any]:
        """Find code entrypoints for a repo and optional area (e.g. pypto, codegen_orch)."""
        return find_entrypoints_impl(repo, area)

    @mcp.tool()
    def trace_in_stack(
        symbol_or_path: Annotated[str, Field(description='Symbol name or file path to locate in the pypto→PTOAS→pto-isa→simpler stack. Path prefix matching is used for file paths; abstraction card matching for concept names.')],
    ) -> dict[str, Any]:
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
        repo: Annotated[str, Field(description='Repository name to verify, e.g. "pypto"')],
        branch: Annotated[str, Field(description="Branch name to check out on the NPU host")],
        sha: Annotated[str, Field(description='Git SHA to record in the handoff. Leave empty to use a placeholder (fill after checkout with git rev-parse HEAD).')] = "",
        task_type: Annotated[str, Field(description='Route key for developer_verify_tasks. Default "npu_verify_handoff" covers the standard NPU gate.')] = "npu_verify_handoff",
        device_ids: Annotated[str, Field(description='Comma-separated NPU device IDs to pass to pytest (e.g. "0,1")')] = "0,1",
        platform: Annotated[str, Field(description='Target Ascend platform family for test flags, e.g. "a2a3" (Ascend910B) or "a3" (Ascend910C)')] = "a2a3",
        fork_remote: Annotated[str, Field(description="Git remote name on the NPU host that has the branch to verify")] = "fork-gbisbas",
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
1. Call bootstrap_session with task_type="{area}" (single-call bootstrap)
2. Follow read_plan from bootstrap_session; use read_doc(path, section=...) for large enriched notes
3. Use explain_pass / explain_abstraction for specific concepts
4. Run agent_verify_tasks from route before claiming work is done

Stack: pypto → PTOAS → pto-isa → simpler. pypto-lib is the model/harness layer."""

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
1. Call bootstrap_session with task_type="{focus_route}"
2. Check program_status for open PRs and blockers (e.g. plan 33 → #1782)
3. Use trace_contract for collectives/pass symbols (LowerHostTensorCollectives, pld.tensor.*)
4. Run agent_verify_tasks only — not developer_verify_tasks

Agent gate: sim Docker UT. Developer gate: NPU ST. Push to fork-gbisbas only."""

    @mcp.prompt(title="Start Ascend architecture / NPU work")
    def start_ascend_work(focus: str = "arch") -> str:
        focus_route = {
            "arch": "ascend_arch",
            "tuning": "npu_tuning",
            "hccl": "ascend_runtime",
            "runtime": "ascend_runtime",
            "verify": "npu_verify_handoff",
        }.get(focus, "ascend_arch")

        return f"""You are working on Huawei Ascend NPU architecture, tuning, or distributed runtime.

Before editing any code:
1. Call bootstrap_session with task_type="{focus_route}"
2. Use explain_abstraction for hardware concepts (AIC, AIV, HCCLWindow, etc.)
3. Call ascend_env_check on NPU hosts; generate_verify_handoff for developer verify

Canonical docs are authoritative. Enriched notes are secondary."""

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
