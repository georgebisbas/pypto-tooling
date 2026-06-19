# HCCL / multi-NPU container checklist

*MCP-owned — full workflow in `pypto-3.0-notes/tools/docker-cann-pr-test-workflow.md`.*

## When this applies

- `pytest tests/st/distributed/` with **2+ ranks** on real NPUs
- Any test using HCCL comm init (`comm_domain`, window buffers, collectives STs)
- **Not** required for `a2a3sim` / `a5sim` single-device runs

## Docker run flags (multi-NPU / HCCL)

```bash
docker run --rm -it --privileged --ipc=host --pid=host \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /dev:/dev \
  pypto3-hw-native-sys:cann9
```

| Flag | Why |
|------|-----|
| `--privileged` | NPU device access |
| `--ipc=host` | Ascend IPC |
| `--pid=host` | HCCL validates **host** PIDs, not container PIDs |
| `/dev` bind | Device nodes |

## LD_PRELOAD (HCCL) — shell only, not image-wide

```bash
export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so
# or ASCEND_HOME_PATH equivalent path to libhccl.so
pytest tests/st/distributed/... --platform=a2a3 --device="0,1"
```

**Do not** bake `LD_PRELOAD` into the Docker image — it injects into every process (including IDE servers) and can hang attach.

Weak HCCL symbols in `host_runtime.so` stay NULL without preload → **SIGSEGV on comm_init**.

## Inside container: branch + install

```bash
cd /opt/pypto   # or mounted checkout
git fetch <fork> <branch> && git checkout <branch>
pip install --no-build-isolation -e ".[dev]"   # C++ extension rebuild
```

Use `--no-build-isolation` so pip sees the venv's cmake/nanobind.

## Signal / window discipline (distributed composites)

- INT32 signal buffer: shape **`[NR, 1]`** — one slot per rank
- Data windows: per-rank HCCL partition; `CommRemotePtr` for peer access
- Single-shot signal buffers per collective call (Set 1 / wait Ge 1; allreduce uses second barrier Ge 2)

See `hw-native-sys://pypto/distributed` and `explain_abstraction HCCLWindow`.

## Agent vs developer gate

| Who | Runs |
|-----|------|
| **Agent** | Sim UT, Docker `host_collectives_ut_sim`, codegen tests |
| **Developer** | `pypto:host_collectives_st_npu`, distributed ST on hardware |

Agents: use `generate_verify_handoff` + `ascend_env_check`, then hand off markdown to NPU container. **Do not** open upstream PRs unless asked.

## pytest device examples

```bash
# P=2
pytest tests/st/distributed/collectives/ -v --platform=a2a3 --device="0,1"

# P=4
pytest tests/st/distributed/collectives/ -v --platform=a2a3 --device="0,1,2,3"
```

## Troubleshooting

| Symptom | Check |
|---------|-------|
| SIGSEGV at comm_init | `LD_PRELOAD` set? `--pid=host`? |
| Rank 1 signal OOB | Signal shape `[1,1]` instead of `[NR,1]` |
| Device not found | `npu-smi info`, `/dev` mounts |
| Wrong arch binary | Rebuild for `a2a3` vs `a5` platform |

Use MCP tool `ascend_env_check` for automated diagnosis.
