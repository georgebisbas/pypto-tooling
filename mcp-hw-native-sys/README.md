# hw-native-sys MCP server

A local Model Context Protocol (MCP) server for full-stack compiler development across the hw-native-sys workspace. It combines **operations** (git health, search, run tasks) with a **knowledge layer** (architecture docs, task routing, abstraction index) so agents start each session oriented on pypto → PTOAS → pto-isa → simpler → pypto-lib.

## Repositories

- `PTOAS` — assembler/optimizer
- `pto-isa` — virtual tile ISA
- `pypto` — compiler framework
- `pypto-3.0-notes` — enriched planning notes (secondary tier)
- `pypto-lib` — model zoo and golden harness
- `pypto-tooling` — Docker, profiling, this MCP server
- `pypto_top_level_documents` — top-level design & architecture docs (serving, runtime, sharded tensor, ISA proposals; design tier)
- `pytorch-hccl-tests` — OSU-style PyTorch/HCCL bandwidth micro-benchmarks (NPU)
- `simpler` — PTO2 runtime

## Daily workflow (recommended)

1. Start MCP server (see Setup below).
2. Invoke MCP prompt **`start_compiler_work`** (or **`start_distributed_work`** / **`start_ascend_work`**).
3. Call **`bootstrap_session`** with your task type — returns route metadata, `read_plan`, health, and program hints in one call.
4. Follow `read_plan`; use **`read_doc(path, section=...)`** for large enriched notes.
5. Use **`explain_pass`** / **`explain_abstraction`** / **`trace_contract`** for stack concepts.
6. Call **`program_status`** for open PRs and blockers.
7. Implement, then run **`verify_ladder(changed_paths)`** or **`agent_verify_tasks`** via **`run_task`**.

A workspace Cursor rule at `.cursor/rules/hw-native-sys-mcp.md` reminds agents to bootstrap via MCP before editing.

## Tools

### Operations

| Tool | Purpose |
|------|---------|
| `list_repositories` | Repos, paths, architecture metadata |
| `repository_health` | Branch, dirty state, ahead/behind, `active_program_hints` from `config/programs.json` |
| `search_code` | Ripgrep (`mode=locations` default; `mode=context` opt-in; `group_by_file`) |
| `bootstrap_session` | Single-call bootstrap: route + read_plan + health + program hints |
| `list_tasks` | Named tasks for a repo |
| `run_task` | Run configured task |
| `run_command` | Ad-hoc shell in a repo |
| `explain_task` | Task command and metadata |

### Knowledge

| Tool | Purpose |
|------|---------|
| `route_task` | Read-first docs, rules, entrypoints, verify tasks by workflow |
| `read_doc` | Read workspace doc with tier label; optional `section` for markdown heading extraction |
| `explain_pass` | Pass pipeline card: order, phase, neighbors, verify tasks |
| `explain_abstraction` | Concept card (IR, passes, codegen, ISA, runtime) |
| `search_abstractions` | Keyword search (`fields=summary` default) |
| `trace_in_stack` / `trace_contract` | Stack position + dependency triangle contracts |
| `verify_ladder` | Minimal verify tasks for changed file paths |
| `program_status` | Structured open PRs, blockers, plan cross-index |
| `summarize_profile` | Summarize `pypto-tooling/profiling/` campaign directories |
| `knowledge_health` | Missing paths, stale enriched docs, Ascend corpus checks |
| `ascend_env_check` | Read-only NPU/CANN/HCCL environment diagnosis |
| `generate_verify_handoff` | Markdown handoff for developer NPU container verify |

## MCP resources

Fixed URIs (read via MCP resource client):

| URI | Content |
|-----|---------|
| `hw-native-sys://overview/ecosystem` | Multi-repo roles and pipeline |
| `hw-native-sys://overview/pipeline` | Compilation pipeline excerpt |
| `hw-native-sys://pypto/ir` | IR overview |
| `hw-native-sys://pypto/passes` | Pass manager framework |
| `hw-native-sys://pypto/codegen` | PTO + orchestration codegen |
| `hw-native-sys://pypto/distributed` | Distributed ops |
| `hw-native-sys://ptoas/overview` | PTOAS assembler |
| `hw-native-sys://pto-isa/overview` | Tile ISA |
| `hw-native-sys://simpler/overview` | Runtime |
| `hw-native-sys://pypto-lib/overview` | Model/harness layer |
| `hw-native-sys://agent/invariants` | Key `.claude/rules` |
| `hw-native-sys://agent/distributed_work_policy` | Agent git/verify policy for distributed & host collectives (George workflow) |
| `hw-native-sys://agent/routing` | Task routing index |
| `hw-native-sys://notes/{topic}` | Enriched notes (pass_infrastructure, host_collectives, …) |
| `hw-native-sys://flows/*` | End-to-end flows (compile_to_device, distributed_allreduce, …) |
| `hw-native-sys://ascend/*` | Ascend hardware, arch families, HCCL/container checklists |

**Doc tiers:** `canonical` (sibling repo docs) is authoritative. `enriched` (pypto-3.0-notes) is secondary — check `last_verified` in front matter. `design` (pypto_top_level_documents) holds forward-looking design/architecture proposals — non-canonical.

## MCP prompts

| Prompt | Use when |
|--------|----------|
| `start_compiler_work` | General compiler work (`area` = task_type) |
| `start_distributed_work` | Collectives, L3, large-scale inference (`focus` = collectives / **host_collectives** / codegen / runtime / inference) |
| `start_ascend_work` | Ascend architecture, tuning, HCCL (`focus` = arch / tuning / hccl / verify) |
| `start_npu_verify` | Developer NPU container verification handoff |

## Task types (`route_task`)

`stack_overview`, `ir_change`, `pass_change`, `codegen_pto`, `codegen_orch`, `distributed`, `distributed_collectives`, **`host_collectives_program`**, `distributed_codegen`, `distributed_runtime`, `large_model_inference`, **`pypto_lib_building_blocks`**, `ptoas`, `pto_isa`, `runtime`, `pypto_lib`, `performance`, **`ascend_arch`**, **`ascend_runtime`**, **`npu_tuning`**, **`npu_verify_handoff`**, **`hccl_bandwidth`**

`route_task` returns **`agent_verify_tasks`** (agent gate) and **`developer_verify_tasks`** (NPU/hardware — agent must not run) when configured.

### Host collectives (plan 33)

| Agent | Task |
|-------|------|
| Sim UT gate | `pypto-tooling:host_collectives_ut_sim` |
| NPU ST (developer) | `pypto:host_collectives_st_npu` |

Read `hw-native-sys://agent/distributed_work_policy` and `hw-native-sys://notes/host_collectives` before resuming fork work.

## Setup

```bash
cd /home/gb4018/workspace/hw-native-sys/pypto-tooling/mcp-hw-native-sys
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick local run

```bash
source .venv/bin/activate
export HW_NATIVE_SYS_ROOT=/home/gb4018/workspace/hw-native-sys
hw-native-sys-mcp
```

## Cursor / VS Code MCP integration

Register stdio MCP server:

- **command:** `/home/gb4018/workspace/hw-native-sys/pypto-tooling/mcp-hw-native-sys/.venv/bin/hw-native-sys-mcp`
- **env:** `HW_NATIVE_SYS_ROOT=/home/gb4018/workspace/hw-native-sys`

## Configuration

| File | Purpose |
|------|---------|
| `config/repos.json` | Workspace root, repos, tasks, `repository_meta` |
| `config/knowledge.json` | Task routes, resources, notes topics |
| `config/entrypoints.json` | Per-repo code entrypoints |
| `config/abstractions.json` | Compiler/stack concept cards (~40) |
| `config/ascend_abstractions.json` | Ascend hardware, arch, HCCL concept cards |
| `config/passes_index.json` | Default pipeline pass order (from `build_knowledge_index.py`) |
| `config/programs.json` | Branch → active program hints (route, verify, blockers) |
| `config/program_status.json` | Structured PR status (from `sync_status_to_json.py`) |
| `content/ascend/*.md` | MCP-owned decision trees (platform, alignment, HCCL) |

### Maintain knowledge config

```bash
# Verify all referenced paths exist
python tools/verify_knowledge_config.py

# Scan codebase for index suggestions (passes_index.json + .index_build_time)
python tools/build_knowledge_index.py

# Sync status_prs.md → program_status.json for agents
python tools/sync_status_to_json.py
```

## Task profile (operations)

Balanced profile: fast daily tasks (git, lint) plus heavier tasks (docker, profiling, hardware tests). Warnings surfaced by `list_tasks`, `explain_task`, `run_task`.

Destructive patterns (`git reset --hard`, etc.) are blocked in task config.

## Example agent prompts

- "Invoke `start_compiler_work` with area=`codegen_orch` and follow the bootstrap."
- "`route_task` for `host_collectives_program` — sim Docker UT vs NPU ST split."
- "`explain_abstraction` for `host_collectives_program`."
- "`explain_abstraction` for `BackendHandler910B` — when is GM pipe buffer required?"
- "`route_task` `ascend_runtime` — HCCL windows and container flags"
- "`ascend_env_check` then `generate_verify_handoff` for branch feat/foo"
- "`search_abstractions` hccl window"
- "`explain_abstraction` for `IterArgCarryAnalyzer`."
- "`trace_in_stack` for `pypto/src/codegen/pto/pto_codegen.cpp`."
- "`search_abstractions` for allreduce."
- "`knowledge_health` — any stale or missing docs?"

## Prerequisite notes

- Simulator/CPU tests: Python deps and build toolchain.
- Hardware tasks: Ascend runtime/device environment.
- Docker tasks: daemon available; can be heavy on disk/network.
- Profiling: start with `profiling_smoke` before `profiling_full`.
