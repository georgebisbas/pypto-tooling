# Sim Docker iteration workflow

*MCP-owned — canonical source for sim Docker test loops.*

## NPU-or-sim policy (MCP auto-redirect)

The MCP server never builds/tests directly on a local repo unless NPUs are
reachable. `run_task` checks `npu-smi` on every heavy (build/test/package)
task:

| NPU reachable? | Behaviour |
|----------------|-----------|
| Yes | Host build/test runs as configured — allowed by policy. |
| No | The task is re-routed into the sim Docker image for the repo (mounted worktree + in-container install), or refused with guidance when the repo has no sim image (e.g. PTOAS). |
| No + image missing | Refused with the exact `docker build` command to create the sim image first. |

`run_command` refuses ad-hoc build/test commands (`cmake`, `make`, `ninja`,
`pip install`, `pytest`, `build_runtimes`, `run_cpu.py`, `run_st.py`,
`./build.sh`) when no NPU is reachable — use `run_task` (which auto-redirects)
or run the command inside the container yourself. Commands that already run
inside a container (`docker run …`, `docker exec …`) are always allowed.

Sim image tags (built from `pypto-tooling/Dockerfile.*sim.ubuntu22.04`):

| Repo | Image |
|------|-------|
| pypto / pto-isa | `pypto3-hw-native-sys:sim` |
| simpler | `simpler-hw-native-sys:sim` |
| pypto-lib | `pypto-lib-hw-native-sys:sim` |

```bash
# Build the sim images (pypto-tooling repo root)
docker build -t pypto3-hw-native-sys:sim -f Dockerfile.hw-native-sys.sim.ubuntu22.04 .
docker build -t simpler-hw-native-sys:sim -f Dockerfile.simpler.sim.ubuntu22.04 .
docker build -t pypto-lib-hw-native-sys:sim -f Dockerfile.pypto-lib.sim.ubuntu22.04 .
```

## The golden rule

**Never run `cmake --build build --parallel` inside Docker.** `--parallel` without `-jN` spawns unlimited concurrent compiler jobs that compete with the host kernel scheduler, saturating all CPU cores and freezing the machine.

## When this applies

- `pytest tests/st/distributed/` with `--forked --platform=a2a3sim --device=...`
- Any sim Docker verification after C++ or kernel-template changes
- **Not** required for NPU/CANN container runs (use `hccl_container_checklist`)

## Correct workflow (for every code change)

Build the image **once** (`docker build`), then iterate with `-v` mount + `pip install -e`:

```bash
# Build the image ONCE (~15-30 min — see pypto-tooling Dockerfile)
docker build -t pypto3-hw-native-sys:sim -f Dockerfile.hw-native-sys.sim.ubuntu22.04 .

# Every code change: mount + pip install (scikit-build-core, ~2-5 min)
docker run --rm --shm-size=4g \
  -v $(pwd):/mount_home/pypto \
  pypto3-hw-native-sys:sim bash -c "
    cd /mount_home/pypto
    rm -rf build _skbuild
    pip install --no-build-isolation -e '.[dev]'
    PYTHONPATH=/mount_home/pypto/python:\$PYTHONPATH \
      pytest tests/st/distributed/<test_file>.py \
        -v --forked --platform=a2a3sim --device=0,1,2,3
  "
```

## Key rules

1. **Never rebuild the image for code changes** — `-v` mount + `pip install -e` is the iteration loop.
2. **Always `rm -rf build _skbuild`** before `pip install` in the container — stale `CMakeCache.txt` from host or previous Docker runs will break the build.
3. **Always `--shm-size=4g`** for distributed sim tests — forked workers + torch shared memory exhaust default 64 MB.
4. **Always run ruff inside Docker** — the host `.ruff_cache` gets root-owned from previous Docker runs.
5. **`cmake --build build --parallel` inside Docker is banned** — it spawns unlimited parallel compiler jobs that freeze the host. `pip install -e` uses scikit-build-core + Ninja which caps parallelism to available cores.

## Why `cmake --build --parallel` freezes

| Mechanism | Behaviour |
|-----------|-----------|
| GNU Make `--parallel` (no `-jN`) | Spawns unlimited concurrent jobs |
| Docker cgroup CPU accounting | Jobs compete with host kernel scheduler |
| Host CPU saturation | All cores at 100% → system unresponsive |
| `pip install -e` (scikit-build-core) | Ninja caps parallelism to available cores |

## Targeted test examples

```bash
# Ring allreduce ST (P=2 and P=4)
pytest tests/st/distributed/test_l3_host_tensor_allreduce_ring.py \
  -v --forked --platform=a2a3sim --device=0,1,2,3

# Ring lowering + codegen UTs
pytest tests/ut/ir/transforms/test_lower_host_tensor_collectives.py \
      tests/ut/codegen/distributed/test_host_orch_distributed.py -v

# All host collectives UTs
pytest tests/ut/ir/transforms/test_lower_host_tensor_collectives.py \
      tests/ut/codegen/distributed/test_host_orch_distributed.py -v \
      -k "host_barrier or host_broadcast or host_reduce_scatter or \
          host_allgather or host_allreduce or materializes or template_package"
```

## Reference

- `pypto-tooling/README.md` § Sim Dev Iteration Workflow
- `pypto-3.0-notes/memories/sim_docker_workflow.md` — lessons learned
- `pypto-3.0-notes/pr_plans/00-branch-and-pr-standards.md` § Sim Docker iteration loop
