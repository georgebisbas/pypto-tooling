# profiling — personal collective benchmark harness

**Personal project.** Benchmark drivers for PyPTO vs simpler L3 collectives live here — **not** in `hw-native-sys/pypto` or `hw-native-sys/simpler`.

## Spec (methodology)

Authoritative plan: [pypto-3.0-notes/allreduce_benchmark_variants/collectives_performance_benchmark_plan.md](../../pypto-3.0-notes/allreduce_benchmark_variants/collectives_performance_benchmark_plan.md)

Profiling playbook: [pypto-3.0-notes/performance_tuning.md/profiling.md](../../pypto-3.0-notes/performance_tuning.md/profiling.md)

## Principles

1. **EquivalenceCase** — one case object drives both `simpler` and `pypto` (same P, count, dtype, devices, window, golden, orchestration profile).
2. **Same orchestration** — `orch_profile: mesh_l3_host_domain_v1` (1 domain, P chip submits, 0 sub-workers).
3. **Artifact bundles** — each run stores `run.log`, `timing.json`, `manifest.json`, and optional `profiling/` under `results/campaigns/...`.
4. **Figures** — `plot_figures.py` builds PNGs from `results.json` for reports.

## Layout

```text
profiling/
  collectives/          # Python package (equivalence, golden, runners, plots)
  results/              # gitignored campaign outputs
  requirements.txt      # matplotlib, pandas
```

## Environment

| Variable | Dev workspace default | Docker (cann9.0) default |
|----------|----------------------|---------------------------|
| `PYPTO_ROOT` | `../pypto` | `/opt/pypto` (auto-detected) |
| `SIMPLER_ROOT` | `../simpler` | `/opt/pypto/runtime` (auto-detected) |
| `PYPTO_NOTES_ROOT` | `../pypto-3.0-notes` | must be set or mounted |
| `PTO_ISA_ROOT` | `../pto-isa` | `/opt/pto-isa` |

Auto-detection checks sibling directories first, then falls back to Docker-standard
paths (`/opt/pypto`, `/opt/pypto/runtime`). Set the env var to override.

## Benchmark methodology

All stacks in `run_sweep.py` report a shared metric schema in `results.json`:

| Field | Meaning |
|-------|---------|
| `setup_s` | One-time compile + init + comm setup (first warmup round only) |
| `execute_s` | **Primary timed metric** — collective execution only |
| `wall_s` | Total round wall (kept for debugging / subprocess stacks) |
| `bw_execute_mb_s` | `n_bytes / execute_s` |
| `per_rank_execute_s` | HCCL only: per-rank times from `HCCL_TIMED` lines |

**Per-stack `execute_s` definition:**

| Stack | Source |
|-------|--------|
| **hccl** | `max(per_rank)` from `HCCL_WARMUP` / `HCCL_TIMED` (slowest rank = collective completion) |
| **simpler-own** | `worker.run()` wall time via in-process session reuse (persistent HCCL window via `Worker.allocate_persistent_domain`) |
| **simpler / pypto / pto-isa** | `phases["execute"]` when available, else subprocess wall (includes framework overhead) |

HCCL and simpler-own run warmup + timed rounds in **one process** (campaign mode), so
`setup_s` is amortized once and excluded from timed means. simpler-own allocates its
comm scratch window once via `Worker.allocate_persistent_domain()` (simpler runtime API)
instead of `orch.allocate_domain()` per execute. Subprocess stacks still pay full init
per round; their `execute_s` is the best available phase marker until session wrappers
land.

`wall_s_mean` in aggregate rows is retained for backward compatibility but **deprecated**
for cross-stack comparison — use `execute_s_mean` and `bw_execute_mb_s` instead.

## Status

| Component | Status |
|-----------|--------|
| `equivalence.py`, `golden.py`, `artifacts.py` | ✅ Working |
| `run_sweep.py` (validate-case, pair-mesh, cross-variant) | ✅ Implemented (E1) |
| `run_campaign.sh` (strong-scaling, cross-variant, full-sweep modes) | ✅ Implemented |
| `cases/generate.py` (case generator for sweeps) | ✅ Implemented (72 cases generated) |
| `summarize.py` (aggregation, paired comparison, reports) | ✅ Implemented (E2) |
| `plot_figures.py` (total-time + phase/compile breakdown figures) | 🟡 Basic (E3) |
| `hccl_bench.py` / `hccl_bench.cc` (HCCL baseline microbenchmark) | ✅ Implemented |

Current figure outputs from a full campaign include:

- `figures/strong_scaling_t_total.png` — total wall time vs `P`
- `figures/paired_stack_ratio.png` — `pypto / simpler` ratio per case
- `figures/phase_breakdown.png` — stacked `startup/compile/init/execute` phase means per stack
- `figures/compile_breakdown.png` — PyPTO compile sub-stages (`passes` / `codegen` / residual other)

## Quick start

### Dev workspace (sibling directories)

```bash
cd pypto-tooling/profiling

# Validate a case file
PYTHONPATH=. python -m collectives.run_sweep validate-case \
  --case-file collectives/cases/mesh_p2_n256_fp32.json

# Run a paired comparison (simpler + pypto, on hardware)
PYTHONPATH=. python -m collectives.run_sweep pair-mesh \
  --case-file collectives/cases/mesh_p2_n256_fp32.json \
  --stacks hccl,simpler,pypto \
  --timed-rounds 5 --warmup-rounds 2 \
  --campaign demo \
  --out results/campaigns/demo/run_001/results.json

# Strong scaling campaign: mesh P=2,4,8
bash run_campaign.sh --variant mesh --p-values 2,4,8 --count 65536

# Cross-variant: mesh vs ring at P=4
bash run_campaign.sh --mode cross-variant --variants mesh,ring \
  --p-values 4 --count 65536 --stacks hccl,simpler

# Generate cases (after adding new variants/sizes)
PYTHONPATH=. python collectives/cases/generate.py --dry-run
PYTHONPATH=. python collectives/cases/generate.py
```

### Docker (hw-native-sys.cann9.0 image)

The Docker image has pypto at `/opt/pypto` and simpler at `/opt/pypto/runtime`.
Paths are auto-detected — no env vars needed. Mount the profiling directory:

```bash
# Build the image (from pypto-tooling/)
docker build -t pypto3-hw-native-sys:cann9 \
  -f Dockerfile.hw-native-sys.cann9.0 .

# Run with HCCL support (multi-device)
docker run --rm -it --privileged --ipc=host --pid=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /dev:/dev \
  -v $(pwd):/pypto-tooling \
  pypto3-hw-native-sys:cann9

# Inside the container
cd /pypto-tooling/profiling
export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so

# Validate paths (should auto-detect /opt/pypto and /opt/pypto/runtime)
PYTHONPATH=. python -c "from collectives.config import pypto_root, simpler_root; print(pypto_root(), simpler_root())"

# Run a campaign
bash run_campaign.sh --variant mesh --p-values 2,4 --count 65536
```

Manual (for debugging a single stack):

```bash
# Dev workspace
export PYPTO_ROOT=../pypto SIMPLER_ROOT=../simpler
# Docker
export PYPTO_ROOT=/opt/pypto SIMPLER_ROOT=/opt/pypto/runtime

cd "$PYPTO_ROOT"
pytest tests/st/distributed/test_l3_allreduce.py -v --platform a2a3 -d 0-1

cd "$SIMPLER_ROOT"
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
python examples/workers/l3/allreduce_ring_distributed/main.py -p a2a3 -d 0-3
```
