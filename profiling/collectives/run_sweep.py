"""Benchmark driver — pair-mesh and single-stack subcommands."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collectives.artifacts import RunArtifactBundle
from collectives.config import pypto_root, simpler_root
from collectives.equivalence import EquivalenceCase
from collectives.golden import fill_rank_inputs, verify_outputs

_HERE = Path(__file__).resolve().parent
_HCCL_BENCH = _HERE / "hccl_bench.py"


def _cmd_validate_case(path: str) -> int:
    case = EquivalenceCase.from_json_file(path)
    case.validate()
    print(f"case_id={case.case_id}")
    print(f"equivalence_hash={case.equivalence_hash()}")
    print(f"n_bytes={case.n_bytes} window_nbytes={case.window_nbytes}")
    return 0


def _devices_dash(device_ids: list[int]) -> str:
    """Convert [0,1,2,3] to '0-3' for simpler CLI (contiguous ranges only)."""
    if len(device_ids) > 2 and device_ids == list(range(device_ids[0], device_ids[-1] + 1)):
        return f"{device_ids[0]}-{device_ids[-1]}"
    return "-".join(str(d) for d in device_ids)


def _devices_comma(device_ids: list[int]) -> str:
    return ",".join(str(d) for d in device_ids)


def _profile_flags(profile_spec: str) -> dict[str, list[str]]:
    """Translate --profile l2,pmu,dep into per-stack CLI flags."""
    flags: dict[str, list[str]] = {"simpler": [], "pypto": []}
    for tok in profile_spec.split(","):
        tok = tok.strip()
        if tok == "l2":
            flags["pypto"].extend(["--enable-l2-swimlane"])
        elif tok == "pmu":
            flags["pypto"].extend(["--enable-pmu", "2"])
        elif tok == "dep":
            flags["pypto"].extend(["--enable-dep-gen"])
    return flags


ResultOk = tuple[bool, str, float]  # (ok, error, wall_s)

# Phase markers for simpler stdout parsing.
# Each marker records the wall time from the *previous* marker to the line
# containing this string.  The first marker records from process start.
_SIMPLER_MARKERS = [
    ("startup",  "compiling kernels"),            # t0 → "compiling kernels" (imports, arg parse)
    ("compile",  "init worker"),                   # → kernel + orch compilation
    ("init",     "running "),                      # → worker.init + domain alloc + TaskArgs
    ("execute",  "all ranks matched golden"),      # → DAG execution + golden verify
    ("fail",     "golden check FAILED"),
]


def _run_with_phases(
    cmd: list[str], cwd: str, env: dict[str, str], timeout: int,
    markers: list[tuple[str, str]],
) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Run subprocess, capture per-line timestamps, extract phase deltas.

    Returns (ok, error, all_lines, total_wall_s, phases_dict).
    """
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd, env=env,
    )
    lines: list[str] = []
    phase_times: dict[str, float] = {}
    last_phase = "startup"
    last_t = t0

    try:
        for line in proc.stdout:
            now = time.perf_counter()
            lines.append(line)
            for phase_name, marker in markers:
                if marker in line and phase_name not in phase_times:
                    phase_times[phase_name] = now - last_t
                    last_phase = phase_name
                    last_t = now
                    break
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        total = time.perf_counter() - t0
        return False, f"timeout after {timeout}s", lines, total, phase_times

    total = time.perf_counter() - t0
    stderr_text = proc.stderr.read()
    if stderr_text:
        lines.append("\n--- STDERR ---\n" + stderr_text)

    ok = proc.returncode == 0
    # Check for golden match marker
    last_marker = "done" if any("all ranks matched golden" in l for l in lines) else (
        "fail" if any("golden check FAILED" in l for l in lines) else None
    )
    if last_marker == "fail":
        ok = False

    error = "" if ok else f"exit={proc.returncode}"
    if not ok and stderr_text:
        error += f"\nstderr: {stderr_text[-500:]}"

    return ok, error, lines, total, phase_times


def _run_simpler_once(case: EquivalenceCase, extra_flags: list[str] | None = None) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Single simpler invocation. Returns (ok, error, lines, total_s, phases)."""
    if case.variant == "ring":
        script = simpler_root() / "examples" / "workers" / "l3" / "allreduce_ring_distributed" / "main.py"
    else:
        script = simpler_root() / "examples" / "workers" / "l3" / "allreduce_distributed" / "main.py"
    if not script.is_file():
        return False, f"simpler script not found: {script}", [], 0.0, {}

    cmd = [
        sys.executable, str(script),
        "-p", case.platform,
        "-d", _devices_dash(case.device_ids),
    ] + (extra_flags or [])
    return _run_with_phases(
        cmd, str(simpler_root()),
        {**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=600, markers=_SIMPLER_MARKERS,
    )


_PYPTO_MARKERS = [
    ("test_result", "PASSED"),
    ("test_result", "FAILED"),
    ("test_result", "ERROR"),
]

_HCCL_MARKERS = [
    ("execute", "HCCL_ALLREDUCE_OK"),
]


def _run_hccl_once(case: EquivalenceCase, extra_flags: list[str] | None = None) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Single HCCL HcclAllReduce invocation. Returns (ok, error, lines, total_s, phases)."""
    if not _HCCL_BENCH.is_file():
        return False, f"HCCL bench script not found: {_HCCL_BENCH}", [], 0.0, {}
    visible_devices = _devices_comma(case.device_ids)
    cmd = [
        sys.executable, str(_HCCL_BENCH),
        "--count", str(case.count),
        "--dtype", case.dtype,
        "--devices", _devices_comma(case.device_ids),
    ] + (extra_flags or [])
    return _run_with_phases(
        cmd, str(_HERE),
        {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "ASCEND_RT_VISIBLE_DEVICES": visible_devices,
        },
        timeout=300, markers=_HCCL_MARKERS,
    )


def _run_pypto_once(case: EquivalenceCase, extra_flags: list[str] | None = None) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Single pypto pytest invocation. Returns (ok, error, lines, total_s, phases)."""
    if case.variant == "ring":
        return False, "pypto ring allreduce ST not yet implemented", [], 0.0, {}
    test_file = pypto_root() / "tests" / "st" / "distributed" / "test_l3_allreduce.py"
    if not test_file.is_file():
        return False, f"pypto test not found: {test_file}", [], 0.0, {}

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYPTO_COMPILE_PROFILING": "1"}
    # DON'T add pypto/python to PYTHONPATH — pypto is already pip-installed
    # in the Docker image. Prepending the source tree shadows the installed
    # pypto_core .so and causes import errors.

    cmd = [
        sys.executable, "-m", "pytest", str(test_file),
        "-v", "--platform", case.platform, "--device", _devices_comma(case.device_ids),
        "-s",
    ] + (extra_flags or [])
    return _run_with_phases(
        cmd, str(pypto_root()), env,
        timeout=600, markers=_PYPTO_MARKERS,
    )


def _run_stack_multi(
    case: EquivalenceCase,
    stack: str,
    bundle: RunArtifactBundle,
    profile_spec: str,
) -> tuple[bool, str, list[dict[str, Any]], float, float]:
    """Run warmup + timed rounds for one stack. Returns (all_ok, error, samples, mean, stdev)."""
    warmup = case.warmup_rounds
    timed_rounds = case.timed_rounds
    total = warmup + timed_rounds
    pf = _profile_flags(profile_spec)
    extra = pf.get(stack, [])

    if stack == "simpler":
        runner = _run_simpler_once
    elif stack == "pypto":
        runner = _run_pypto_once
    else:
        runner = _run_hccl_once

    samples: list[dict[str, Any]] = []
    all_ok = True
    last_error = ""
    timed_walls: list[float] = []

    for r in range(total):
        label = "warmup" if r < warmup else f"timed-{r - warmup + 1}"
        print(f"  [{stack:>7}] round {r + 1:>2}/{total} ({label:>7})...", end=" ", flush=True)
        ok, err, lines, wall, phases = runner(case, extra)
        status = "✅" if ok else "❌"

        # Build detail line: phases + bandwidth
        details: list[str] = []
        if phases:
            for k, v in phases.items():
                if k not in ("fail",):
                    details.append(f"{k}={v:.2f}s")
        bw = case.n_bytes / wall if wall > 0 else 0
        if bw >= 1e6:
            details.append(f"{bw/1e6:.1f} MB/s")
        elif bw >= 1e3:
            details.append(f"{bw/1e3:.1f} KB/s")
        else:
            details.append(f"{bw:.1f} B/s")
        # Per-rank HCCL timing
        if stack == "hccl":
            for line in lines:
                if "per_rank=" in line:
                    try:
                        ranks = json.loads(line.split("per_rank=")[1].split("]")[0] + "]")
                        details.append(f"ranks={ranks}")
                    except Exception:
                        pass
                    break

        detail_str = ", ".join(details)
        print(f"{wall:.3f}s {status}  [{detail_str}]")
        if not ok and lines:
            for line in lines[-8:]:
                stripped = line.rstrip()
                if stripped:
                    print(f"         {stripped[:120]}")
        sample = {"round": r, "phase": label, "wall_s": round(wall, 6), "ok": ok}
        if phases:
            sample["phases"] = {k: round(v, 6) for k, v in phases.items()}
        sample["bw_mb_s"] = round(bw / 1e6, 6)
        samples.append(sample)
        if not ok:
            all_ok = False
            last_error = err
            if label.startswith("timed"):
                print(f"  [{stack:>7}] ABORTING after timed round failure")
                break
        if r >= warmup:
            timed_walls.append(wall)

    # Write individual samples
    bundle.log_path.write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8"
    )
    bundle.write_timing(samples)

    if timed_walls:
        mean = statistics.mean(timed_walls)
        stdev = statistics.stdev(timed_walls) if len(timed_walls) > 1 else 0.0
    else:
        mean, stdev = 0.0, 0.0

    return all_ok, last_error, samples, mean, stdev


def _print_diagnostics(case: EquivalenceCase, args) -> None:
    """Pre-flight checks — device availability, imports, script paths."""
    stacks = [s.strip() for s in args.stacks.split(",")]
    print(f"\n{'═'*60}")
    print(f"  PRE-FLIGHT DIAGNOSTICS")
    print(f"{'═'*60}")

    # Device count via npu-smi
    try:
        r = subprocess.run(["npu-smi", "info", "-l"], capture_output=True, text=True, timeout=10)
        chips = r.stdout.count("Chip")
        print(f"  npu-smi: {chips} chip(s) visible")
    except Exception:
        print(f"  npu-smi: not available")
    print(f"  requested devices: {case.device_ids} (P={case.p})")

    # HCCL library
    if "hccl" in stacks:
        try:
            lib = ctypes.CDLL("libhccl.so", mode=ctypes.RTLD_GLOBAL)
            print(f"  libhccl.so: loaded ✅")
        except Exception as e:
            print(f"  libhccl.so: FAILED — {str(e)[:50]}")

    # PyPTO import
    if "pypto" in stacks:
        try:
            import pypto as _pypto
            ver = getattr(_pypto, "__version__", "?")
            print(f"  pypto: importable ✅  v{ver}")
        except Exception as e:
            print(f"  pypto: FAILED — {str(e)[:50]}")

    # simpler scripts
    if "simpler" in stacks:
        from collectives.config import simpler_root as _sr
        mesh = _sr() / "examples" / "workers" / "l3" / "allreduce_distributed" / "main.py"
        print(f"  simpler mesh script: {'✅' if mesh.is_file() else '❌ not found'}")

    # Payload size
    nbytes = case.n_bytes
    if nbytes >= 1_000_000:
        size_str = f"{nbytes / 1e6:.1f} MB"
    elif nbytes >= 1_000:
        size_str = f"{nbytes / 1024:.1f} KB"
    else:
        size_str = f"{nbytes} B"
    print(f"  payload: {case.count} × {case.dtype} = {size_str}  ({nbytes} bytes)")
    print(f"{'═'*60}\n")


def _run_golden_only(case: EquivalenceCase) -> tuple[bool, str]:
    from collectives.golden import expected_output_allreduce_sum_v1
    _ = fill_rank_inputs(case)
    expected = expected_output_allreduce_sum_v1(case)
    per_rank = [expected[:] for _ in range(case.p)]
    return verify_outputs(case, per_rank)


def _cmd_pair_mesh(args: argparse.Namespace) -> int:
    case = EquivalenceCase.from_json_file(args.case_file)
    case.validate()

    # ── pre-flight diagnostics ──
    _print_diagnostics(case, args)

    print(f"case_id={case.case_id}  eq_hash={case.equivalence_hash()}")
    print(f"warmup={case.warmup_rounds}  timed={case.timed_rounds}  "
          f"profile={args.profile or 'none'}")

    stacks = [s.strip() for s in args.stacks.split(",")]
    unknown = [s for s in stacks if s not in ("simpler", "pypto", "hccl")]
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

    # Phase 1: multi-round per stack
    rows: list[dict] = []
    nbytes = case.n_bytes
    payload_desc = f"{case.count}×{case.dtype} ({nbytes} B/rank, P={case.p})"
    for stack in stacks:
        bundle = bundles[stack]
        print(f"\n[phase 1] {stack} ({case.warmup_rounds} warmup + {case.timed_rounds} timed)  "
              f"payload: {payload_desc}")

        all_ok, last_error, samples, mean, stdev = _run_stack_multi(
            case, stack, bundle, args.profile,
        )

        bundle.write_manifest(
            correctness="pass" if all_ok else "fail",
            error=last_error or None,
        )

        rows.append({
            "case_id": case.case_id,
            "stack": stack,
            "equivalence_hash": case.equivalence_hash(),
            "variant": case.variant,
            "p": case.p,
            "count": case.count,
            "dtype": case.dtype,
            "platform": case.platform,
            "correctness": "pass" if all_ok else "fail",
            "wall_s_mean": round(mean, 6),
            "wall_s_stdev": round(stdev, 6),
            "n_warmup": case.warmup_rounds,
            "n_timed": case.timed_rounds,
            "artifact_bundle": str(bundle.bundle_dir),
        })
        status = "PASS" if all_ok else "FAIL"
        print(f"  {stack}: mean={mean:.4f}s stdev={stdev:.4f}s {status}")
        if not all_ok and last_error:
            print(f"  first error: {last_error[:200]}")
            print(f"  stopping after {stack} failure; skipping remaining stacks")
            break

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

    # ── summary table ──
    nbytes = case.n_bytes
    print(f"\n{'═'*70}")
    print(f"  RESULTS  ({case.count}×{case.dtype}, {nbytes} B/rank, P={case.p})")
    print(f"{'═'*70}")
    bw_map: dict[str, float] = {}
    for r in rows:
        mean = r["wall_s_mean"]
        stdev = r["wall_s_stdev"]
        bw = nbytes / mean if mean > 0 else 0
        bw_map[r["stack"]] = bw
        ok = r["correctness"]
        bw_str = f"{bw/1e6:.1f} MB/s" if bw >= 1e6 else f"{bw/1e3:.2f} KB/s"
        print(f"  {r['stack']:>7}: {mean:>8.4f}s ±{stdev:.4f}s  {bw_str:>14s}  {'✅' if ok == 'pass' else '❌'}")

    if "hccl" in bw_map and bw_map["hccl"] > 0:
        hccl_bw = bw_map["hccl"]
        for stack in ("simpler", "pypto"):
            if stack in bw_map:
                eff = bw_map[stack] / hccl_bw * 100
                print(f"  {stack:>7} vs HCCL: {eff:>5.1f}% of HCCL bandwidth")
    print(f"{'═'*70}")

    all_ok = all(r["correctness"] == "pass" for r in rows)
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collective benchmark sweep (pypto-tooling)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-case", help="Validate EquivalenceCase JSON")
    p_val.add_argument("--case-file", required=True)

    p_pair = sub.add_parser("pair-mesh", help="Run same case on simpler and pypto")
    p_pair.add_argument("--case-file", required=True)
    p_pair.add_argument("--stacks", default="hccl,simpler,pypto")
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
