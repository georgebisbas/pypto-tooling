"""Benchmark driver — pair-mesh and single-stack subcommands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from collectives.artifacts import RunArtifactBundle
from collectives.config import pypto_root, simpler_root
from collectives.equivalence import EquivalenceCase
from collectives.golden import fill_rank_inputs, verify_outputs


def _cmd_validate_case(path: str) -> int:
    case = EquivalenceCase.from_json_file(path)
    case.validate()
    print(f"case_id={case.case_id}")
    print(f"equivalence_hash={case.equivalence_hash()}")
    print(f"n_bytes={case.n_bytes} window_nbytes={case.window_nbytes}")
    return 0


def _devices_dash(device_ids: list[int]) -> str:
    """Convert [0, 1] to '0-1' for simpler CLI."""
    return "-".join(str(d) for d in device_ids)


def _devices_comma(device_ids: list[int]) -> str:
    """Convert [0, 1] to '0,1' for pypto CLI."""
    return ",".join(str(d) for d in device_ids)


def _run_simpler_mesh(case: EquivalenceCase, bundle: RunArtifactBundle) -> tuple[bool, str, float]:
    """Invoke simpler allreduce_distributed/main.py. Returns (ok, error, wall_s)."""
    script = simpler_root() / "examples" / "workers" / "l3" / "allreduce_distributed" / "main.py"
    if not script.is_file():
        return False, f"simpler script not found: {script}", 0.0

    cmd = [
        sys.executable, str(script),
        "-p", case.platform,
        "-d", _devices_dash(case.device_ids),
    ]
    print(f"[simpler] {' '.join(cmd)}")

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=str(simpler_root()),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    wall = time.perf_counter() - t0

    bundle.log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    ok = result.returncode == 0 and "all ranks matched golden" in result.stdout
    error = "" if ok else f"exit={result.returncode}; stderr={result.stderr[-500:]}"
    return ok, error, wall


def _run_pypto_mesh(case: EquivalenceCase, bundle: RunArtifactBundle) -> tuple[bool, str, float]:
    """Invoke pypto test_l3_allreduce.py via pytest. Returns (ok, error, wall_s)."""
    test_file = pypto_root() / "tests" / "st" / "distributed" / "test_l3_allreduce.py"
    if not test_file.is_file():
        return False, f"pypto test not found: {test_file}", 0.0

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    pythonpath = str(pypto_root() / "python")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pythonpath

    cmd = [
        sys.executable, "-m", "pytest", str(test_file),
        "-v", "--platform", case.platform, "-d", _devices_comma(case.device_ids),
        "--timeout", "600",
    ]
    print(f"[pypto] {' '.join(cmd)}")

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=str(pypto_root()), env=env,
    )
    wall = time.perf_counter() - t0

    bundle.log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    ok = result.returncode == 0
    error = "" if ok else f"exit={result.returncode}; stderr={result.stderr[-500:]}"
    return ok, error, wall


def _run_golden_only(case: EquivalenceCase) -> tuple[bool, str]:
    """Compute and verify golden without invoking hardware."""
    from collectives.golden import expected_output_allreduce_sum_v1
    _ = fill_rank_inputs(case)
    expected = expected_output_allreduce_sum_v1(case)
    per_rank = [expected[:] for _ in range(case.p)]
    return verify_outputs(case, per_rank)


def _cmd_pair_mesh(args: argparse.Namespace) -> int:
    case = EquivalenceCase.from_json_file(args.case_file)
    case.validate()
    print(f"case_id={case.case_id} eq_hash={case.equivalence_hash()}")

    stacks = [s.strip() for s in args.stacks.split(",")]
    unknown = [s for s in stacks if s not in ("simpler", "pypto")]
    if unknown:
        print(f"ERROR: unknown stacks: {unknown}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    run_dir = out_path.parent

    bundles: dict[str, RunArtifactBundle] = {}
    for stack in stacks:
        bundles[stack] = RunArtifactBundle(run_dir, case.case_id, stack)
        bundles[stack].ensure_dirs()

    # Phase 0: golden verification (offline)
    print("\n[phase 0] golden verification")
    golden_ok, golden_msg = _run_golden_only(case)
    print(f"  golden: {'OK' if golden_ok else 'FAILED'} — {golden_msg}")
    if not golden_ok:
        print("WARNING: golden formula failed — check golden.py", file=sys.stderr)

    # Phase 1: run each stack
    rows: list[dict] = []
    for stack in stacks:
        bundle = bundles[stack]
        print(f"\n[phase 1] running {stack}...")

        if stack == "simpler":
            ok, err, wall = _run_simpler_mesh(case, bundle)
        else:
            ok, err, wall = _run_pypto_mesh(case, bundle)

        sample = {"round": 0, "wall_s": round(wall, 6), "ok": ok}
        bundle.write_timing([sample])
        bundle.write_manifest(correctness="pass" if ok else "fail", error=err or None)

        rows.append({
            "case_id": case.case_id,
            "stack": stack,
            "equivalence_hash": case.equivalence_hash(),
            "variant": case.variant,
            "p": case.p,
            "count": case.count,
            "dtype": case.dtype,
            "platform": case.platform,
            "correctness": "pass" if ok else "fail",
            "wall_s": round(wall, 6),
            "artifact_bundle": str(bundle.bundle_dir),
        })
        status = "PASS" if ok else "FAIL"
        print(f"  {stack}: wall={wall:.3f}s {status}" + (f" — {err[:120]}" if err else ""))

    # Write results.json
    results = {
        "campaign": args.campaign,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case": case.canonical_dict(),
        "runs": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")

    all_ok = all(r["correctness"] == "pass" for r in rows)
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collective benchmark sweep (pypto-tooling)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-case", help="Validate EquivalenceCase JSON")
    p_val.add_argument("--case-file", required=True)

    p_pair = sub.add_parser("pair-mesh", help="Run same case on simpler and pypto")
    p_pair.add_argument("--case-file", required=True)
    p_pair.add_argument("--stacks", default="simpler,pypto")
    p_pair.add_argument("--campaign", default="default")
    p_pair.add_argument("--profile", default="", help="Comma: l2,pmu,dep")
    p_pair.add_argument("--out", required=True, help="results.json path under results/campaigns/")

    args = parser.parse_args(argv)
    if args.command == "validate-case":
        return _cmd_validate_case(args.case_file)
    if args.command == "pair-mesh":
        return _cmd_pair_mesh(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
