#!/usr/bin/env python3
"""Generate a synthetic demo campaign so every figure in plot_figures.py lights up.

Writes ``results/campaigns/demo_figures/run_001/results.json`` with realistic
(but synthetic) NPU-scale numbers — mesh + ring across P=2/4/8 and payload
counts 4K/64K/1M — for hccl / simpler / simpler-own / pypto-composite /
pypto-host. Use it as template-picture generator and as living documentation of
the results.json schema that run_sweep.py produces.

Usage:
    PYTHONPATH=. python -m collectives.make_demo_results
    python -m collectives.plot_figures --run-dir results/campaigns/demo_figures/run_001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_COUNTS = (4096, 65536, 1048576)
_P_VALUES = (2, 4, 8)
_BYTES_PER_ELT = 4  # fp32

# (P, count) -> execute seconds per stack. Ballpark NPU allreduce numbers:
# HCCL is the fastest baseline; hand-written simpler in between; pypto stacks
# pay a small framework tax; host builtin is the slowest of the pypto pair.
_BASE_EXEC: dict[tuple[int, int], dict[str, float]] = {
    (2, 4096):    {"hccl": 8e-6, "simpler": 20e-6, "simpler-own": 18e-6,
                   "pypto-composite": 26e-6, "pypto-host": 42e-6},
    (2, 65536):   {"hccl": 32e-6, "simpler": 62e-6, "simpler-own": 55e-6,
                   "pypto-composite": 82e-6, "pypto-host": 125e-6},
    (2, 1048576): {"hccl": 420e-6, "simpler": 720e-6, "simpler-own": 680e-6,
                   "pypto-composite": 920e-6, "pypto-host": 1350e-6},
    (4, 4096):    {"hccl": 14e-6, "simpler": 34e-6, "simpler-own": 30e-6,
                   "pypto-composite": 44e-6, "pypto-host": 70e-6},
    (4, 65536):   {"hccl": 55e-6, "simpler": 105e-6, "simpler-own": 95e-6,
                   "pypto-composite": 140e-6, "pypto-host": 210e-6},
    (4, 1048576): {"hccl": 720e-6, "simpler": 1250e-6, "simpler-own": 1150e-6,
                   "pypto-composite": 1580e-6, "pypto-host": 2350e-6},
    (8, 4096):    {"hccl": 26e-6, "simpler": 60e-6, "simpler-own": 52e-6,
                   "pypto-composite": 78e-6, "pypto-host": 125e-6},
    (8, 65536):   {"hccl": 100e-6, "simpler": 190e-6, "simpler-own": 170e-6,
                   "pypto-composite": 250e-6, "pypto-host": 380e-6},
    (8, 1048576): {"hccl": 1300e-6, "simpler": 2300e-6, "simpler-own": 2100e-6,
                   "pypto-composite": 2900e-6, "pypto-host": 4300e-6},
}

# One-time setup per stack (compile + init), seconds.
_SETUP_S = {
    "hccl": 0.05,
    "simpler": 0.0,          # subprocess — setup attributed per round
    "simpler-own": 0.8,
    "pypto-composite": 2.9,
    "pypto-host": 6.4,
}
# Fraction of execute_s that is the pure on-device collective (device_wall).
# HCCL reports device time directly; the pypto stacks carry framework overhead
# that execute_s includes but device_wall excludes.
_DEVICE_FRAC = {
    "hccl": 1.0,
    "simpler": 0.90,
    "simpler-own": 0.85,
    "pypto-composite": 0.80,
    "pypto-host": 0.70,
}
_COMPILE_S = {"pypto-composite": 0.8, "pypto-host": 1.4}
_INIT_S = {"pypto-composite": 2.1, "pypto-host": 5.0}
_COMPILE_PROFILE = {
    "pypto-composite": {"total": 0.80, "passes": 0.010, "codegen": 0.76, "other": 0.03},
    "pypto-host": {"total": 1.40, "passes": 0.005, "codegen": 1.36, "other": 0.035},
}

_STACKS = ("hccl", "simpler", "simpler-own", "pypto-composite", "pypto-host")


def _case_id(variant: str, p: int, count: int) -> str:
    devices = ",".join(str(i) for i in range(p))
    return f"{variant}_p{p}_count{count}_fp32_a2a3_d{devices}"


def _write_demo_pmu(run_dir: Path) -> None:
    """Write synthetic pmu.csv files so the pmu_utilization figure has data."""
    header = (
        "thread_id,core_id,task_id,func_id,core_type,pmu_total_cycles,"
        "vec_busy_cycles,cube_busy_cycles,scalar_busy_cycles,mte1_busy_cycles,"
        "mte2_busy_cycles,mte3_busy_cycles,icache_miss,icache_req,event_type"
    )
    for stack, vec, mte2 in (("pypto-composite", 0.42, 0.58),
                             ("pypto-host", 0.12, 0.78)):
        for p in _P_VALUES:
            cid = _case_id("mesh", p, 65536)
            for rank in range(p):
                pmu_dir = run_dir / "cases" / cid / stack / "dfx" / "dfx_outputs" / f"rank{rank}" / "d0"
                pmu_dir.mkdir(parents=True, exist_ok=True)
                total = 1_000_000
                rows = [
                    header,
                    f"1,{10 + rank},0x1,0,1,{total},{int(total * vec)},0,"
                    f"{int(total * 0.3)},{int(total * 0.1)},{int(total * mte2)},"
                    f"{int(total * 0.05)},{int(total * 0.02)},{int(total * 0.5)},2",
                ]
                (pmu_dir / "pmu.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def generate(run_dir: Path) -> Path:
    """Write a synthetic results.json. Returns its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    for variant in ("mesh", "ring"):
        for p in _P_VALUES:
            for count in _COUNTS:
                if variant == "ring" and count % p != 0:
                    continue
                base = _BASE_EXEC[(p, count)]
                cid = _case_id(variant, p, count)
                for stack in _STACKS:
                    if variant == "ring" and stack in ("simpler-own",):
                        continue  # simpler-own is mesh-only
                    execute = base[stack]
                    nbytes = count * _BYTES_PER_ELT
                    device_wall = execute * _DEVICE_FRAC[stack]
                    phases: dict[str, float] = {"execute": execute}
                    compile_profile: dict[str, float] | None = None
                    if stack in _COMPILE_S:
                        phases["compile"] = _COMPILE_S[stack]
                        phases["init"] = _INIT_S[stack]
                        compile_profile = _COMPILE_PROFILE[stack]
                    runs.append({
                        "case_id": cid,
                        "stack": stack,
                        "equivalence_hash": "demo",
                        "variant": variant,
                        "p": p,
                        "count": count,
                        "dtype": "fp32",
                        "platform": "a2a3",
                        "correctness": "pass",
                        "execute_s_mean": execute,
                        "execute_s_stdev": execute * 0.03,
                        "execute_s_median": execute * 0.98,
                        "device_wall_s_mean": device_wall,
                        "device_wall_s_median": device_wall * 0.98,
                        "setup_s": _SETUP_S[stack] if stack in _SETUP_S else None,
                        "bw_execute_mb_s": round((nbytes / execute) / 1e6, 3),
                        "wall_s_mean": _SETUP_S[stack] + execute
                        if stack not in ("hccl",) else execute,
                        "wall_s_stdev": execute * 0.03,
                        "phase_means": phases,
                        "compile_profile_means": compile_profile,
                        "n_warmup": 1,
                        "n_timed": 5,
                        "artifact_bundle": str(run_dir / "cases" / cid / stack),
                    })

    results = {
        "campaign": "demo_figures",
        "runs": runs,
        "cases": [],
    }
    out = run_dir / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_demo_pmu(run_dir)
    print(f"wrote {out} ({len(runs)} runs)")
    return out


def main(argv: list[str] | None = None) -> int:
    """Write the synthetic demo campaign."""
    parser = argparse.ArgumentParser(description="Generate synthetic demo results for figure previews")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("results/campaigns/demo_figures/run_001"))
    args = parser.parse_args(argv)
    generate(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
