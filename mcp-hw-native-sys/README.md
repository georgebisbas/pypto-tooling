# hw-native-sys MCP server

A local Model Context Protocol (MCP) server for full-stack compiler development across the hw-native-sys workspace. It combines **operations** (git health, code search, running named tasks) with a **knowledge layer** (architecture docs, task routing, an abstraction index, pass pipeline info, cross-repo status) so an agent — or you — can get oriented on `pypto → PTOAS → pto-isa → simpler → pypto-lib` in one or two calls instead of grepping five repos by hand.

This doc is the full reference: setup, every tool/resource/prompt, the config files behind them, how the knowledge index is built and kept honest, and how to extend the server yourself.

## Repositories it operates over

| Repo | Role |
|------|------|
| `pypto` | Compiler framework: Python DSL → IR → passes → codegen |
| `PTOAS` | PTO assembler/optimizer: `.pto` MLIR → AICore/AIV kernel C++ |
| `pto-isa` | Virtual tile ISA: C++ headers, CPU/NPU backends |
| `simpler` | PTO2 runtime: task graph execution on AICore/AICPU |
| `pypto-lib` | Model zoo and golden validation harness |
| `pypto-3.0-notes` | Enriched planning notes, retrospectives, cross-repo status (secondary tier — not canonical) |
| `pypto_top_level_documents` | Top-level design/architecture proposals (design tier — non-canonical, forward-looking) |
| `pytorch-hccl-tests` | OSU-style PyTorch/HCCL bandwidth micro-benchmarks (NPU) |
| `pypto-tooling` | Docker images, profiling campaigns, and this MCP server |

## Setup

```bash
cd /home/georgios/workspace/hw-native-sys/pypto-tooling/mcp-hw-native-sys
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python ≥3.10, the `mcp` package (installed via the above), and `rg` (ripgrep) on `PATH` for `search_code`.

### Workspace root resolution

The server needs to know where the sibling repos live. In order of precedence:

1. `HW_NATIVE_SYS_ROOT` env var, if set.
2. `config/repos.json`'s `"workspace_root"` field (checked in as `"../.."`, i.e. two directories up from `mcp-hw-native-sys/` — this is what makes the server work out of the box for the standard checkout layout).
3. Fallback: `project_root().parents[1]`.

You generally don't need to set `HW_NATIVE_SYS_ROOT` unless you're running the server from a copy that isn't in its usual place relative to the sibling repos.

### Quick local run (stdio, manual)

```bash
source .venv/bin/activate
export HW_NATIVE_SYS_ROOT=/home/georgios/workspace/hw-native-sys   # optional, see above
hw-native-sys-mcp
```

### Claude Code integration

`pypto-tooling/.mcp.json` already registers this server under the name `hw-native-sys`:

```json
{
  "mcpServers": {
    "hw-native-sys": {
      "command": "/home/georgios/workspace/hw-native-sys/pypto-tooling/mcp-hw-native-sys/.venv/bin/hw-native-sys-mcp"
    }
  }
}
```

Any Claude Code session started with `pypto-tooling` (or a parent directory) as the working directory picks this up automatically — tools appear as `mcp__hw-native-sys__<tool_name>`. No env var needed since `config/repos.json`'s relative `workspace_root` resolves correctly from the checked-in `.venv` location.

### Cursor / VS Code MCP integration

Register a stdio MCP server manually:

- **command:** `/home/georgios/workspace/hw-native-sys/pypto-tooling/mcp-hw-native-sys/.venv/bin/hw-native-sys-mcp`
- **env:** `HW_NATIVE_SYS_ROOT=/home/georgios/workspace/hw-native-sys` (optional, see workspace root resolution above)

## Recommended daily workflow

1. Call the **`start_compiler_work`** prompt (or `start_distributed_work` / `start_ascend_work` / `start_npu_verify` depending on the task) — this gives you the exact next steps.
2. Call **`bootstrap_session(task_type=...)`** — one call returns route metadata (`read_plan`), repo health, and active-program hints together.
3. Follow `read_plan`: read canonical docs first, enriched docs second. Use **`read_doc(path, section=...)`** to pull a single markdown section out of a large note instead of the whole file.
4. Use **`explain_pass`** / **`explain_abstraction`** / **`search_abstractions`** / **`trace_contract`** / **`trace_in_stack`** to pin down stack concepts before writing code.
5. Call **`program_status`** for open PRs/blockers, and **`collective_status`** if the work touches collective communication ops.
6. Implement.
7. Run **`verify_ladder(changed_paths)`** to get the minimal test set, then run `agent_verify_tasks` via **`run_task`**. Never run `developer_verify_tasks` (NPU/hardware-gated) yourself — those are for the human developer.

## Tools

### Operations (`mcp_hwnative_sys/server.py`)

| Tool | Purpose |
|------|---------|
| `list_repositories` | Repos, paths, architecture metadata, disk availability |
| `repository_health` | Branch, dirty state, ahead/behind upstream, last commit, `active_program_hints` per repo |
| `search_code` | Ripgrep across one/many/all repos. `mode=locations` (default, file+line only) or `mode=context` (+ matched text and surrounding lines); `use_regex`, `file_glob`, `group_by_file` |
| `list_tasks` | Named tasks configured for a repo (from `config/repos.json`), with risk/warning metadata |
| `run_task` | Run a named task in a repo, with `extra_args` and a timeout |
| `run_command` | Ad-hoc shell command in a repo's root; destructive patterns (`git reset --hard`, `rm -rf /`, …) are blocked |
| `explain_task` | Show the exact command + metadata for one named task |
| `git_log` | Structured commit list (sha, author, date, message) for a repo |
| `git_diff` | `git diff` for a repo, `stat_only` for orientation or full patch text |
| `read_file` | Read an arbitrary source file from a repo (paginated via `offset`/`max_lines`) without shelling out |
| `bootstrap_session` | Single-call session bootstrap: route + `read_plan` + health + program hints |

### Knowledge (`mcp_hwnative_sys/knowledge.py` and friends)

| Tool | Purpose |
|------|---------|
| `list_task_types` | All valid `task_type` values (for `route_task`/`bootstrap_session`) with descriptions |
| `route_task` | Read-first docs (canonical + enriched), rules, entrypoints, and verify tasks for a `task_type` |
| `list_knowledge_topics` | Enumerate all task routes, MCP resources, notes topics, and bootstrap prompts in one call |
| `read_doc` | Read a workspace doc with tier labeling (`canonical`/`enriched`/`design`/`mcp-owned`); optional `section` extracts one markdown heading |
| `explain_abstraction` | Concept card for an IR node, pass, codegen stage, ISA instruction, PTOAS op, or Ascend hardware concept. Reports `source: curated` or `source: generated` (see Provenance below) |
| `search_abstractions` | Keyword search across the full abstraction index (name, layer, kind, tags, `one_liner`, related/downstream); ranked by relevance |
| `explain_pass` | Pass-pipeline card: order, phase, neighbors, verify tasks (from the `Default` pypto pipeline) |
| `program_status` | Structured open PRs, blockers, and plan cross-index from `pypto-3.0-notes/pr_plans/status_prs.md` |
| `collective_status` | Collective-comm feature parity status (merged/planned/gap) from the parity matrix in `pypto-3.0-notes/distributed/current_status.md`, with optional `op`/`axis` substring filters. Read-only — never writes to the source doc |
| `verify_ladder` | Minimal suggested verify tasks for a list of changed file paths (longest-matching-prefix rules) |
| `find_entrypoints` | Code entrypoints for a repo and optional sub-area |
| `trace_in_stack` / `trace_contract` | Locate a symbol or path in the `pypto → PTOAS → pto-isa → simpler` stack, with dependency-triangle and contract-artifact enrichment |
| `knowledge_health` | Self-audit: missing paths, stale enriched docs (>30 days since `last_verified`), Ascend corpus checks, pto-isa/PTOAS index **coverage**, pass-index build status |
| `ascend_env_check` | Read-only NPU/CANN/HCCL environment diagnosis (devices, `LD_PRELOAD`, Docker hints) |
| `generate_verify_handoff` | Generate a markdown handoff for a human developer to run NPU/hardware verification in a container |
| `summarize_profile` | Summarize a `pypto-tooling/profiling/` campaign directory (`results.json`, anomalies) |

## MCP resources

Fixed URIs, read via an MCP resource client (or by finding the matching path via `read_doc`/`list_knowledge_topics`):

| Prefix | Example URIs | Content |
|--------|--------------|---------|
| `overview/*` | `overview/ecosystem`, `overview/pipeline` | Multi-repo roles, compilation pipeline |
| `pypto/*` | `pypto/ir`, `pypto/passes`, `pypto/codegen`, `pypto/distributed` | pypto subsystem overviews |
| `ptoas/*`, `pto-isa/*`, `simpler/*` | `ptoas/overview`, `pto-isa/overview`, `simpler/overview`, `simpler/l3_distributed_collectives` | Sibling-repo overviews |
| `pypto-lib/*` | `pypto-lib/overview`, `pypto-lib/status`, `pypto-lib/building_blocks`, `pypto-lib/attention`, `pypto-lib/models`, `pypto-lib/distributed_support`, `pypto-lib/moe` | Model zoo / harness layer |
| `pytorch-hccl-tests/*` | `pytorch-hccl-tests/overview`, `pytorch-hccl-tests/bandwidth-runbook` | HCCL bandwidth benchmarking |
| `agent/*` | `agent/invariants`, `agent/distributed_work_policy`, `agent/routing` | Agent-facing rules and the task-routing index |
| `ascend/*` | `ascend/hardware`, `ascend/arch_families`, `ascend/memory_hierarchy`, `ascend/cann_mapping`, `ascend/hccl_runtime`, `ascend/platform_decisions`, `ascend/alignment_rules`, `ascend/hccl_container_checklist` | Ascend hardware/platform reference |
| `flows/*` | `flows/compile_to_device`, `flows/matmul_demo`, `flows/distributed_allreduce`, `flows/dependency_triangle`, `flows/performance` | End-to-end worked examples |
| `notes/*` | see notes topics below | Enriched notes (secondary tier) |

**Doc tiers** (returned by `read_doc`/`route_task`): `canonical` (sibling repo docs — authoritative) > `enriched` (`pypto-3.0-notes` — secondary, check `last_verified`) > `design` (`pypto_top_level_documents` — forward-looking proposals, non-canonical) > `mcp-owned` (`content/` — this server's own decision-tree docs) > `ephemeral` (`pr_plans/`, `pull_requests/` — living/scratch, refused by `read_doc`, use `program_status`/`collective_status` instead).

### Notes topics (`notes/{topic}`, resource or `read_doc`)

`abstractions_master`, `codegen_infrastructure`, `dependency_triangle`, `distributed_work_policy`, `host_collectives`, `kernel_orchestration`, `machine_hierarchy`, `moe`, `multi_level_runtime_ring`, `notes_simpler`, `pass_infrastructure`, `ptoas_abstractions`, `ptoisa_abstractions`, `pypto_abstractions`, `pypto_lib_attention`, `pypto_lib_building_blocks`, `pypto_lib_distributed_support`, `pypto_lib_models`, `pypto_lib_status`, `runtime_arch_index`, `runtime_async`, `runtime_design`, `serving_implementation_plan`, `sharded_tensor`, `simpler_abstractions`, `simpler_distributed_runtime_design`, `stack_availability`, `tensor_layout`, `tensor_valid_shape`, `tpush_tpop_isa_design`

## MCP prompts

| Prompt | Params | Use when |
|--------|--------|----------|
| `start_compiler_work` | `area` (= task_type, default `stack_overview`) | General compiler work — any new session should start here or with one of the below |
| `start_distributed_work` | `focus`: `collectives` / `host_collectives` / `codegen` / `runtime` / `inference` | Collectives, L3 runtime, distributed codegen, large-scale inference |
| `start_ascend_work` | `focus`: `arch` / `tuning` / `hccl` / `runtime` / `verify` | Ascend hardware architecture, performance tuning, HCCL |
| `start_npu_verify` | — | Developer-only: hand off to real-NPU container verification (agent must not run this itself — see the prompt body for the exact gate) |

Each prompt returns a short markdown playbook naming the exact tool-call sequence for that kind of work.

## Task types (`route_task` / `bootstrap_session` / `list_task_types`)

| task_type | Covers |
|-----------|--------|
| `stack_overview` | Any new session — multi-repo roles and compilation pipeline |
| `ir_change` | New IR nodes, types, or structural changes |
| `pass_change` | Pass pipeline additions or modifications |
| `codegen_pto` | InCore codegen to `.pto` MLIR (AICore path) |
| `codegen_orch` | Orchestration codegen to PTO2 runtime C++ (AICPU path) |
| `distributed` | Distributed ops, collectives, multi-rank |
| `distributed_collectives` | Composite collectives, ring vs. mesh algorithms |
| `host_collectives_program` | Host builtin collectives program (barrier, broadcast, reduce_scatter, allgather) |
| `distributed_codegen` | Distributed codegen backend |
| `distributed_runtime` | `simpler` comm-domain, L3 worker, remote execution |
| `large_model_inference` | pypto-lib models, golden harness, inference paths |
| `pypto_lib_building_blocks` | What building blocks/ops/models exist in pypto-lib and their distributed support |
| `ptoas` | PTO assembler and optimizer (`.pto` → C++) |
| `pto_isa` | Virtual tile ISA headers and backends |
| `runtime` | `simpler` task runtime execution on Ascend |
| `pypto_lib` | Model zoo, kernels, golden validation harness |
| `performance` | Compile/runtime profiling for training/inference tuning |
| `ascend_arch` | Ascend chip architecture: AIC/AIV, memory hierarchy, A2A3 vs. A5 |
| `ascend_runtime` | HCCL, comm windows, CANN container verify, distributed execution |
| `npu_tuning` | Performance tuning: block_dim, swimlanes, PMU, arch-specific backend handlers |
| `npu_verify_handoff` | Developer NPU verification handoff — container checkout, HCCL STs, record SHA |
| `hccl_bandwidth` | PyTorch/HCCL collective + p2p bandwidth benchmarking via `pytorch-hccl-tests` |

`route_task` returns **`agent_verify_tasks`** (safe for an agent to run, e.g. sim-Docker UTs) separately from **`developer_verify_tasks`** (NPU/hardware-gated — an agent must never run these; they're for the human developer, typically via `generate_verify_handoff`).

### Host collectives (plan 33)

| Agent | Task |
|-------|------|
| Sim UT gate | `pypto-tooling:host_collectives_ut_sim` |
| NPU ST (developer) | `pypto:host_collectives_st_npu` |

Read `hw-native-sys://agent/distributed_work_policy` and `hw-native-sys://notes/host_collectives` before resuming fork work in this area.

## Configuration files

| File | Purpose | Curation |
|------|---------|----------|
| `config/repos.json` | Workspace root, repo paths, named tasks, `repository_meta` | Hand-maintained |
| `config/knowledge.json` | Task routes, resources, notes topics | Hand-maintained |
| `config/entrypoints.json` | Per-repo code entrypoints, by area | Hand-maintained |
| `config/abstractions.json` | Hand-curated compiler/stack concept cards | Hand-maintained — **always wins** over generated cards on name collision |
| `config/ascend_abstractions.json` | Ascend hardware, arch, HCCL concept cards | Hand-maintained, merged into the same abstraction index as `abstractions.json` |
| `config/pto_isa_generated.json` | ~140 pto-isa instruction cards (tile-local + comm) | **Generated** by `tools/build_pto_isa_index.py` from `pto-isa/docs/isa/manifest.yaml` + `docs/isa/comm/README.md` |
| `config/ptoas_generated.json` | ~500 PTOAS IR op cards | **Generated** by `tools/build_ptoas_index.py`, regex-scraped from `PTOOps.td`/`VPTOOps.td`'s `let summary`/`let description` fields |
| `config/passes_index.json` | Default pipeline pass order, phase, verify tasks | **Generated** by `tools/build_knowledge_index.py` from `pypto/python/pypto/ir/pass_manager.py` — see caveat below |
| `config/programs.json` | Branch → active program hints (route, verify, blockers) | Hand-maintained |
| `config/program_status.json` | Structured PR status | **Generated** by `tools/sync_status_to_json.py` from `pypto-3.0-notes/pr_plans/status_prs.md` |
| `config/collective_status.json` | Structured collective-comm parity matrix | **Generated** by `tools/sync_collective_status_to_json.py` from `pypto-3.0-notes/distributed/current_status.md` |
| `content/ascend/*.md` | MCP-owned decision trees (platform, alignment, HCCL) | Hand-maintained |

All generated files are checked into git (so a fresh checkout works without a build step) but are meant to be periodically regenerated — see below. None of the generator scripts ever write to the sibling repos or to `pypto-3.0-notes`; they only read from them.

### Provenance: curated vs. generated abstraction cards

`load_abstractions()` merges four sources: `pto_isa_generated.json` and `ptoas_generated.json` first (broad, mechanical coverage), then `abstractions.json` and `ascend_abstractions.json` last — so **any hand-curated card always wins outright** on a name collision. `explain_abstraction` reports which one you got via its `source` field (`curated` or `generated`). Generated cards additionally carry `generated_from` (the exact source file scraped) so you can tell where a summary came from.

Why this split exists: pto-isa and PTOAS have far more instructions/ops (~140 and ~500 respectively) than anyone has hand-written cards for (~15 combined, as of writing). Rather than leave the long tail undocumented, the generators mechanically extract what pto-isa/PTOAS already document about themselves (structured `manifest.yaml` entries, TableGen `let summary` fields) — lower-quality than hand curation, but far better than nothing, and it never silently overrides a hand-written card.

### Maintaining the knowledge config

Run these after upstream changes to the scraped sources (pass pipeline, pto-isa manifest, PTOAS `.td` files, PR/plan status, or the collective status matrix):

```bash
# Verify every path referenced by knowledge.json/abstractions/entrypoints actually exists
python tools/verify_knowledge_config.py

# Rebuild passes_index.json (from pypto's pass_manager.py) + suggest new abstraction
# candidates (printed to stdout only -- never auto-merged into abstractions.json)
python tools/build_knowledge_index.py

# Rebuild pto_isa_generated.json from pto-isa/docs/isa/manifest.yaml + comm/README.md
python tools/build_pto_isa_index.py

# Rebuild ptoas_generated.json from PTOAS's PTOOps.td / VPTOOps.td
python tools/build_ptoas_index.py

# Sync status_prs.md -> program_status.json for agents
python tools/sync_status_to_json.py

# Sync current_status.md's parity matrix -> collective_status.json
python tools/sync_collective_status_to_json.py
```

**Caveat on `build_knowledge_index.py`**: it only rebuilds `passes_index.json` when re-run explicitly — `load_passes_index()` does not invalidate the on-disk cache on its own (unlike `load_abstractions()`, which is mtime-keyed). If you rebuild it and `pypto_pass_count` comes back as `0` with a warning, that means `pypto/python/pypto/ir/pass_manager.py` upstream no longer matches the scraper's expected `("Name", lambda: passes.foo())` shape — check whether the pass pipeline has since moved to a different registration mechanism before assuming the scraper is simply stale. **Don't blindly overwrite a healthy checked-in cache with a broken re-scrape** — diff it first; if the rebuild produces materially less data than what's committed, something upstream changed and needs a matching fix in `passes_index.py`, not a cache overwrite.

### Self-auditing: `knowledge_health`

Call `knowledge_health` any time you want a health check on the knowledge layer itself, without a manual audit:

- `missing_paths` — any route/resource/abstraction path that no longer exists on disk.
- `stale_enriched` — enriched docs whose `last_verified` (from `pypto-3.0-notes/NOTES_FRESHNESS.md`) is more than 30 days old.
- `coverage.pto_isa_indexed` / `coverage.ptoas_indexed` — how many generated cards currently exist, so index drift (e.g. after a pto-isa/PTOAS refactor) is visible without re-running the multi-agent audit that originally found this gap.
- `pypto_pass_count` / `pypto_passes_index_warning` — whether the pass-pipeline scrape is currently healthy (see caveat above). Explicitly scoped to pypto — no other repo's pass pipeline is scraped, so don't read this as a cross-repo figure.
- `ascend_issues`, `last_index_build`, `ascend_route_count` — misc corpus checks.

## Task profile (operations)

Balanced profile: fast daily tasks (git, lint) plus heavier tasks (docker, profiling, hardware tests). Warnings are surfaced by `list_tasks`, `explain_task`, and `run_task`. Destructive patterns (`git reset --hard`, `git clean -fdx`, `rm -rf /`, `rm -rf ~`) are blocked at the `run_command`/`run_task` layer regardless of which repo task config requests them.

## Example agent prompts

- "Invoke `start_compiler_work` with area=`codegen_orch` and follow the bootstrap."
- "`route_task` for `host_collectives_program` — sim Docker UT vs NPU ST split."
- "`explain_abstraction` for `host_collectives_program`."
- "`explain_abstraction` for `BackendHandler910B` — when is GM pipe buffer required?"
- "`explain_abstraction` for `TSCATTER` — note it covers both the local-tile and collective-comm meaning, merged from two sources."
- "`route_task` `ascend_runtime` — HCCL windows and container flags."
- "`ascend_env_check` then `generate_verify_handoff` for branch feat/foo."
- "`search_abstractions` hccl window."
- "`explain_abstraction` for `IterArgCarryAnalyzer`."
- "`trace_in_stack` for `pypto/src/codegen/pto/pto_codegen.cpp`."
- "`search_abstractions` for allreduce."
- "`collective_status` with axis=`Dynamic NR` — what's the parity gap across ops?"
- "`knowledge_health` — any stale or missing docs, or coverage gaps?"

## Prerequisite notes

- Simulator/CPU tests: Python deps and build toolchain.
- Hardware tasks: Ascend runtime/device environment.
- Docker tasks: daemon available; can be heavy on disk/network.
- Profiling: start with `profiling_smoke` before `profiling_full`.

## Extending this server

Every tool follows the same shape: a plain, unit-testable `_impl(...)` function (in `mcp_hwnative_sys/<module>.py`) plus a thin `@mcp.tool()`-decorated wrapper that calls it.

1. Put the real logic in a module-level `def foo_impl(...) -> dict[str, Any]` — no MCP/pydantic types inside, so it can be imported and called directly from tests.
2. Register it in `register_knowledge(mcp)` (in `knowledge.py`) or directly in `server.py`, using `Annotated[T, Field(description=...)]` for every parameter — the description is what the calling agent sees, so make it concrete (include example values). Prefer a **local import inside the tool function body** for the impl module (e.g. `from mcp_hwnative_sys.foo import foo_impl`) to avoid import cycles, matching the existing convention for `explain_pass`, `trace_contract`, `verify_ladder`, `collective_status`, etc.
3. Raise plain `ValueError`/`RuntimeError`/`FileNotFoundError` for user-facing errors — there's no custom exception hierarchy.
4. Add a test in `tests/test_<module>.py`: plain `pytest` functions (no classes), `from __future__ import annotations`, `monkeypatch.setattr(<module>, "workspace_root", lambda: tmp_path)` (or the relevant path function) to sandbox filesystem-touching code, plus one smoke test against the real workspace. If the tool lives in `tools/` (a maintenance script, not part of the installed package), import it in tests via a `sys.path.insert(0, str(TOOLS_DIR))` at the top of the test file, matching `test_build_pto_isa_index.py`/`test_build_ptoas_index.py`.
5. Run `pytest tests/` from `mcp-hw-native-sys/` (use the project's own `.venv`: `.venv/bin/python -m pytest tests/`).
6. If your tool scrapes a source that could drift (like the pto-isa/PTOAS generators or the pass-pipeline scraper), prefer writing to a **new** generated JSON file that gets layered in at load time, rather than writing into a hand-curated config — that way hand edits are never at risk of being silently overwritten by a bad scrape, and regenerating is always safe to re-run.

## Known caveats

- `pypto/python/pypto/ir/pass_manager.py` has moved to building its pipeline via a runtime C++ `PassPipeline` object; the static regex-based pass scraper in `passes_index.py` can no longer recover pass names by re-scraping live (the checked-in `passes_index.json` cache still has real, valid data — only a fresh `build_passes_index()` call is affected). Fixing this properly means dynamically instantiating pypto's pass manager instead of regex-scraping — not yet done.
- A "Resource already exists" warning may print at server startup for a handful of `notes/*` resource URIs — harmless (the server still initializes correctly), but indicates some resource registration path runs more than once somewhere; not yet root-caused.
