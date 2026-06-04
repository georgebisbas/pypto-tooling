# collectives

Python package for Phase E collective benchmarks. See [../README.md](../README.md) and the [notes spec](../../../pypto-3.0-notes/allreduce_benchmark_variants/collectives_performance_benchmark_plan.md).

| Module | Role |
|--------|------|
| `equivalence.py` | `EquivalenceCase`, `ORCH_PROFILES`, validation, `equivalence_hash` |
| `golden.py` | Shared input fill + expected output verifier |
| `artifacts.py` | Per-run directory layout + `manifest.json` |
| `config.py` | `PYPTO_ROOT`, `SIMPLER_ROOT` resolution |
| `run_sweep.py` | `pair-mesh` and single-stack runners (E1) |
| `summarize.py` | Tables, `vs_paired_stack`, `--emit-report` (E2–E4) |
| `plot_figures.py` | Figure catalog from `results.json` (E4) |
| `cases/*.json` | Checked-in `EquivalenceCase` fixtures |
