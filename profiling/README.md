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

| Variable | Default (sibling checkout) |
|----------|----------------------------|
| `PYPTO_ROOT` | `../pypto` |
| `SIMPLER_ROOT` | `../simpler` |
| `PYPTO_NOTES_ROOT` | `../pypto-3.0-notes` |

## Status

| Component | Status |
|-----------|--------|
| `equivalence.py`, `golden.py`, `artifacts.py` | ✅ Working |
| `run_sweep.py` (validate-case, pair-mesh) | ✅ Implemented (E1) |
| `summarize.py` (aggregation, paired comparison, reports) | ✅ Implemented (E2) |
| `plot_figures.py` (total-time + phase/compile breakdown figures) | 🟡 Basic (E3) |

Current figure outputs from a full campaign include:

- `figures/strong_scaling_t_total.png` — total wall time vs `P`
- `figures/paired_stack_ratio.png` — `pypto / simpler` ratio per case
- `figures/phase_breakdown.png` — stacked `startup/compile/init/execute` phase means per stack
- `figures/compile_breakdown.png` — PyPTO compile sub-stages (`passes` / `codegen` / residual other)

## Quick start

```bash
cd pypto-tooling/profiling

# Validate a case file
python -m collectives.run_sweep validate-case \
  --case-file collectives/cases/mesh_p2_n256_fp32.json

# Run a paired comparison (simpler + pypto, on hardware)
python -m collectives.run_sweep pair-mesh \
  --case-file collectives/cases/mesh_p2_n256_fp32.json \
  --stacks simpler,pypto \
  --campaign demo \
  --out results/campaigns/demo/run_001/results.json
```

Manual (for debugging a single stack):

```bash
export PYPTO_ROOT=../pypto
export SIMPLER_ROOT=../simpler

cd "$PYPTO_ROOT"
pytest tests/st/distributed/test_l3_allreduce.py -v --platform a2a3 -d 0-1

cd "$SIMPLER_ROOT"
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
```
