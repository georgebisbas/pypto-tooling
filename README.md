# pypto-tooling

Personal tooling umbrella: agent skills, runbooks, and glue for PyPTO /
simpler development on Ascend 910B. Most of the heavy machinery that used to
live in this repo moved to dedicated sibling repos — see below.

## Repo map

This repo is now a thin umbrella. The three former sub-projects are standalone
repos, siblings under `hw-native-sys/`:

| Repo | Contents |
| ---- | -------- |
| [`../pypto-docker`](../pypto-docker) | Dockerfiles (CANN + sim images for pypto / simpler / pypto-lib / pytorch-hccl-tests), image build scripts, `docker-entrypoint-cann.sh`, and the full image build/runtime README |
| [`../mcp-hw-native-sys`](../mcp-hw-native-sys) | MCP server (pip package) operating over the hw-native-sys workspace — repos, tasks, knowledge index, verify ladder |
| [`../pypto-profiling`](../pypto-profiling) | Personal collective benchmark harness (pypto vs simpler vs HCCL L3 collectives) |

## What stays here

- `debugging_skills/` — NPU / docker debugging skills + error-code reference
- `task-submit/` — reference doc for the `task-submit` device-locking CLI
- `diagnose_npu.py` — 10-point NPU hardware health check (copy into a container and run)
- `bz910b-reproduce.md` — Ascend 910B server image repro runbook
- `personal_setup.md` — personal git / env setup notes
- `.mcp.json` — Claude Code MCP registration for `mcp-hw-native-sys`

The docker-specific skills moved to `pypto-docker`:
- `pypto-docker/build_skills/` — in-container PyPTO (`pypto_core`) rebuild workflow
- `pypto-docker/dockerfile_skills/` — Dockerfile construction skill + incident logs

## MCP registration

`.mcp.json` registers the `hw-native-sys` MCP server from the sibling
`mcp-hw-native-sys` repo. Any Claude Code session started in `pypto-tooling`
(or a parent) picks it up; tools appear as `mcp__hw-native-sys__<tool_name>`.

## Sim Dev Iteration Workflow (local code changes)

The full sim image workflow (build, mount, `pip install -e`, targeted vs full
test suites) moved with the images to
[`pypto-docker/README.md`](../pypto-docker/README.md#sim-dev-iteration-workflow-local-code-changes).
The short form:

```bash
# One-time: build the sim image (~15-30 min)
docker build -t pypto3-hw-native-sys:sim -f Dockerfile.hw-native-sys.sim.ubuntu22.04 .

# Every code change: mount workspace + pip install -e (~2-5 min)
docker run --rm \
  -v /path/to/hw-native-sys/pypto:/opt/pypto \
  pypto3-hw-native-sys:sim \
  bash -c "pip install --no-build-isolation -e '/opt/pypto[dev]' 2>&1 | tail -1 && \
           pytest tests/ut -n auto --maxprocesses 8 -v"
```

Key rules:
- **Never rebuild the image for code changes** — `-v` mount + `pip install -e` is the iteration loop.
- **Always run ruff inside Docker** — the host `.ruff_cache` gets root-owned from previous Docker runs.
- **Test targeted suites first, then full suite before pushing.**
