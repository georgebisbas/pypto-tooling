# hw-native-sys MCP server

A local Model Context Protocol (MCP) server for full-stack compiler development across the hw-native-sys workspace. It combines **operations** (git health, search, run tasks) with a **knowledge layer** (architecture docs, task routing, abstraction index) so agents start each session oriented on pypto → PTOAS → pto-isa → simpler → pypto-lib.

## Repositories

- `PTOAS` — assembler/optimizer
- `pto-isa` — virtual tile ISA
- `pypto` — compiler framework
- `pypto-3.0-notes` — enriched planning notes (secondary tier)
- `pypto-lib` — model zoo and golden harness
- `pypto-tooling` — Docker, profiling, this MCP server
- `simpler` — PTO2 runtime

## Daily workflow (recommended)

1. Start MCP server (see Setup below).
2. Invoke MCP prompt **`start_compiler_work`** (or **`start_distributed_work`** for collectives/L3).
3. Read resources `hw-native-sys://overview/ecosystem` and `hw-native-sys://agent/invariants`.
4. Call **`route_task`** with your task type (e.g. `pass_change`, `codegen_orch`, `distributed`).
5. Call **`repository_health`** with `include_clean=false`.
6. Use **`explain_abstraction`** / **`search_abstractions`** for stack concepts.
7. Implement, then run **`verify_tasks`** from `route_task` via **`run_task`**.

A workspace Cursor rule at `.cursor/rules/hw-native-sys-mcp.md` reminds agents to bootstrap via MCP before editing.

## Tools

### Operations

| Tool | Purpose |
|------|---------|
| `list_repositories` | Repos, paths, architecture metadata |
| `repository_health` | Branch, dirty state, ahead/behind |
| `search_code` | Ripgrep across repos |
| `list_tasks` | Named tasks for a repo |
| `run_task` | Run configured task |
| `run_command` | Ad-hoc shell in a repo |
| `explain_task` | Task command and metadata |

### Knowledge

| Tool | Purpose |
|------|---------|
| `route_task` | Read-first docs, rules, entrypoints, verify tasks by workflow |
| `list_knowledge_topics` | Discover task types, resources, prompts |
| `read_doc` | Read workspace doc with tier label (canonical/enriched) |
| `explain_abstraction` | Concept card (IR, passes, codegen, ISA, runtime) |
| `search_abstractions` | Keyword search over abstraction index |
| `find_entrypoints` | Code paths per repo/area |
| `trace_in_stack` | Where a symbol/path sits in the pipeline |
| `knowledge_health` | Missing paths, stale enriched docs |

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
| `hw-native-sys://agent/routing` | Task routing index |
| `hw-native-sys://notes/{topic}` | Enriched notes (pass_infrastructure, codegen_infrastructure, …) |
| `hw-native-sys://flows/*` | End-to-end flows (compile_to_device, distributed_allreduce, …) |

**Doc tiers:** `canonical` (sibling repo docs) is authoritative. `enriched` (pypto-3.0-notes) is secondary — check `last_verified` in front matter.

## MCP prompts

| Prompt | Use when |
|--------|----------|
| `start_compiler_work` | General compiler work (`area` = task_type) |
| `start_distributed_work` | Collectives, L3, large-scale inference (`focus` = collectives/codegen/runtime/inference) |

## Task types (`route_task`)

`stack_overview`, `ir_change`, `pass_change`, `codegen_pto`, `codegen_orch`, `distributed`, `distributed_collectives`, `distributed_codegen`, `distributed_runtime`, `large_model_inference`, `ptoas`, `pto_isa`, `runtime`, `pypto_lib`, `performance`

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
| `config/abstractions.json` | ~40 stack concept cards |

### Maintain knowledge config

```bash
# Verify all referenced paths exist
python tools/verify_knowledge_config.py

# Scan codebase for index suggestions (updates .index_build_time marker)
python tools/build_knowledge_index.py
```

## Task profile (operations)

Balanced profile: fast daily tasks (git, lint) plus heavier tasks (docker, profiling, hardware tests). Warnings surfaced by `list_tasks`, `explain_task`, `run_task`.

Destructive patterns (`git reset --hard`, etc.) are blocked in task config.

## Example agent prompts

- "Invoke `start_compiler_work` with area=`codegen_orch` and follow the bootstrap."
- "`route_task` for `distributed_collectives` — what should I read first?"
- "`explain_abstraction` for `IterArgCarryAnalyzer`."
- "`trace_in_stack` for `pypto/src/codegen/pto/pto_codegen.cpp`."
- "`search_abstractions` for allreduce."
- "`knowledge_health` — any stale or missing docs?"

## Prerequisite notes

- Simulator/CPU tests: Python deps and build toolchain.
- Hardware tasks: Ascend runtime/device environment.
- Docker tasks: daemon available; can be heavy on disk/network.
- Profiling: start with `profiling_smoke` before `profiling_full`.
