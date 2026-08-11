# profiling — personal collective benchmark harness

**Personal project.** Benchmark drivers for PyPTO vs simpler L3 collectives live here — **not** in `hw-native-sys/pypto` or `hw-native-sys/simpler`.

## Spec (methodology)

Authoritative plan: [pypto-3.0-notes/allreduce_benchmark_variants/collectives_performance_benchmark_plan.md](../../pypto-3.0-notes/allreduce_benchmark_variants/collectives_performance_benchmark_plan.md)

Profiling playbook: [pypto-3.0-notes/performance_tuning.md/profiling.md](../../pypto-3.0-notes/performance_tuning.md/profiling.md)

## Principles

1. **EquivalenceCase** — one case object drives every stack (same P, count, dtype, devices, window, golden, orchestration profile).
2. **Same orchestration** — `orch_profile: mesh_l3_host_domain_v1` (1 domain, P chip submits, 0 sub-workers).
3. **Artifact bundles** — each run stores `run.log`, `timing.json`, `manifest.json`, and optional `profiling/` under `results/campaigns/...`.
4. **Figures** — `plot_figures.py` builds PNGs from `results.json` for reports.

## Layout

```text
profiling/
  collectives/          # Python package (stacks, equivalence, golden, runners, plots)
  collectives/stacks.py # stack capability registry (single source of truth)
  collectives/runners/  # per-stack in-process session runners
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
| `execute_s` | **Primary timed metric** — `rt.run()` wall (host dispatch + collective) |
| `device_wall_s_mean` | Pure on-device collective time (slowest-rank `[STRACE] device_wall` span); `None` when the runtime did not emit it |
| `wall_s` | Total round wall (kept for debugging / subprocess stacks) |
| `bw_execute_mb_s` | `n_bytes / execute_s` |
| `per_rank_execute_s` | HCCL only: per-rank times from `HCCL_TIMED` lines |

**`execute_s` vs `device_wall_s`:** for the pypto stacks `execute_s` includes
per-dispatch host overhead (on the 2026-08-10 NPU run that was ~16 ms at P=2,
scaling to ~240 ms at P=8 — larger than the collective itself).
`device_wall_s_mean` is the true apples-to-apples number vs HCCL's 200 µs.
Pass `--batch N` to run N back-to-back `rt.run()` per timed round and divide,
amortising that dispatch overhead for a second view (each round still reports
both metrics).

**Per-stack `execute_s` definition:**

| Stack | Source |
|-------|--------|
| **hccl** | `max(per_rank)` from `HCCL_WARMUP` / `HCCL_TIMED` (slowest rank = collective completion) |
| **simpler-own** | `worker.run()` wall time via in-process session reuse (persistent HCCL window via `Worker.allocate_persistent_domain`) |
| **pypto-composite / pypto-host** | `rt.run()` wall time via in-process session reuse (compile + `prepare()` once; each timed round is a fresh `rt.run`) |
| **simpler / pto-isa** | `phases["execute"]` when available, else subprocess wall (includes framework overhead) |

HCCL, simpler-own, pypto-composite and pypto-host run warmup + timed rounds in **one
process** (campaign mode), so `setup_s` is amortized once and excluded from timed means.
simpler-own allocates its comm scratch window once via `Worker.allocate_persistent_domain()`
(simpler runtime API) instead of `orch.allocate_domain()` per execute; the pypto runners
compile + `prepare()` their distributed worker once and reuse it across rounds. Subprocess
stacks still pay full init per round; their `execute_s` is the best available phase marker
until session wrappers land.

`wall_s_mean` in aggregate rows is retained for backward compatibility but **deprecated**
for cross-stack comparison — use `execute_s_mean` and `bw_execute_mb_s` instead.

## Status

| Component | Status |
|-----------|--------|
| `stacks.py` (stack capability registry) | ✅ Working |
| `equivalence.py`, `golden.py`, `artifacts.py` | ✅ Working |
| `runners/simpler_own.py`, `runners/pypto_own.py` (in-process sessions) | ✅ Working |
| `run_sweep.py` (validate-case, pair-mesh, cross-variant) | ✅ Implemented (E1) |
| `run_campaign.sh` (strong-scaling, cross-variant, full-sweep modes) | ✅ Implemented |
| `cases/generate.py` (case generator for sweeps) | ✅ Implemented (72 cases generated) |
| `summarize.py` (aggregation, paired comparison, reports) | ✅ Implemented (E2) |
| `plot_figures.py` (8 figures: scaling, efficiency, bw-crossover, ratios, phase/setup/compile breakdown, PMU) | ✅ Working |
| `hccl_bench.py` / `hccl_bench.cc` (HCCL baseline microbenchmark) | ✅ Implemented |

## Stack capability matrix

| Stack | kind | Variants | Count constraint | Notes |
|-------|------|----------|------------------|-------|
| **hccl** | campaign | mesh, ring, twophase | unbounded | CANN `HcclAllReduce` baseline; algorithm internal to HCCL |
| **simpler** | subprocess | mesh, ring, twophase | `[256]` only | hand-written L3 C++ allreduce (`--mode mesh\|ring\|twophase`) |
| **simpler-own** | campaign | mesh | unbounded | our dynamic-count AIV kernel via simpler `KernelCompiler` |
| **pypto-composite** | campaign | mesh, ring | unbounded | InCore `pld.tensor.allreduce` composite via `@pl.jit.host` |
| **pypto-host** | campaign | mesh, ring | unbounded | HOST builtin `pld.tensor.allreduce` via `@pl.jit.host`; ring = Sum + FP32 only |
| **pto-isa** | subprocess | mesh | unbounded | PTO-ISA path |

Default apples-to-apples set (`DEFAULT_STACKS`): `hccl,simpler,pypto-composite,pypto-host`.
`simpler-own` and `pto-isa` are opt-in via `--stacks`.

Current figure outputs from a full campaign include:

- `figures/strong_scaling_t_total.png` — execute time vs `P`, one line per stack
- `figures/strong_scaling_efficiency.png` — parallel efficiency `E(P)=T(Pmin)·Pmin/(T(P)·P)` per stack
- `figures/message_size_bw_eff.png` — bandwidth (MB/s) vs payload count (log-x crossover)
- `figures/paired_stack_ratio.png` — per-case `stack / simpler` ratio (all stacks except simpler/hccl)
- `figures/phase_breakdown.png` — stacked `startup/compile/init/execute` phase means per stack
- `figures/setup_breakdown.png` — grouped `compile` / `init` / `execute` bars per stack
- `figures/compile_breakdown.png` — PyPTO compile sub-stages (`passes` / `codegen` / residual other), captured via the thread-local `CompileProfiler` around `host_orch.compile`
- `figures/pmu_utilization.png` — pipe utilisation ratios from `pmu.csv` (needs `--profile pmu`; pypto stacks route DFX artifacts into the bundle under `cases/<case>/<stack>/dfx/`)
- `figures/wall_vs_device.png` — `execute_s` vs `device_wall_s` per pypto (case, stack), exposing the per-dispatch overhead directly

Without matplotlib (e.g. the minimal sim container) every figure falls back to a
`.txt` sibling with the same data — `plot_figures.py` never fails on a missing
plotting dependency.

**Template pictures / figure previews:** `collectives/make_demo_results.py`
writes a synthetic demo campaign (81 runs across mesh/ring × P=2/4/8 × counts
4K/64K/1M × 5 stacks) so every figure renders without a real benchmark:

```bash
PYTHONPATH=. python -m collectives.make_demo_results
python -m collectives.plot_figures --run-dir results/campaigns/demo_figures/run_001
# → results/campaigns/demo_figures/run_001/figures/*.png
```

**Profiling perturbs the numbers.** `--profile pmu` (and DFX generally) adds
per-dispatch instrumentation: on a2a3sim a pypto execute went from ~0.02s to
~0.12s (composite) / ~0.05s to ~0.25s (host). Never mix profiled and unprofiled
rounds in one comparison — run a campaign either fully profiled or not at all,
and collect PMU data in a dedicated campaign.

## Quick start

### Dev workspace (sibling directories)

```bash
cd pypto-tooling/profiling

# Validate a case file
PYTHONPATH=. python -m collectives.run_sweep validate-case \
  --case-file collectives/cases/mesh_p2_n256_fp32.json

# Run a paired comparison (4-stack apples-to-apples, on hardware)
PYTHONPATH=. python -m collectives.run_sweep pair-mesh \
  --case-file collectives/cases/mesh_p2_count65536_fp32_a2a3_d0-1.json \
  --stacks hccl,simpler,pypto-composite,pypto-host \
  --timed-rounds 5 --warmup-rounds 2 \
  --campaign demo \
  --out results/campaigns/demo/run_001/results.json

# Same comparison on the simulator (pypto stacks + simpler-own only; hccl/simpler need CANN)
PYTHONPATH=. python -m collectives.run_sweep pair-mesh \
  --case-file collectives/cases/mesh_p2_count65536_fp32_a2a3_d0-1.json \
  --stacks pypto-composite,pypto-host \
  --platform a2a3sim \
  --timed-rounds 2 --warmup-rounds 1 \
  --campaign smoke \
  --out results/campaigns/smoke/run_001/results.json

# Strong scaling campaign: mesh P=2,4,8
bash run_campaign.sh --variant mesh --p-values 2,4,8 --count 65536

# Message-size sweep: mesh P × count — finds the latency→bandwidth crossover
# (feeds figures/message_size_bw_eff.png). Bigger counts amortize launch/sync
# overhead so the collective's asymptotic bandwidth shows up.
bash run_campaign.sh --variant mesh --p-values 2,4,8 \
  --counts 4096,16384,65536,262144,1048576 \
  --stacks hccl,pypto-composite,pypto-host \
  --platform a2a3 --campaign strong_mesh_sizes

# Cross-variant: mesh vs ring at P=4
bash run_campaign.sh --mode cross-variant --variants mesh,ring \
  --p-values 4 --count 65536 --stacks hccl,simpler,pypto-composite,pypto-host

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
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-3 --mode ring
```
