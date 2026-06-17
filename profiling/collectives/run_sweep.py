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
from collectives.config import profiling_root, pypto_root, simpler_root, pto_isa_root
from collectives.equivalence import EquivalenceCase
from collectives.golden import fill_rank_inputs, verify_outputs
from collectives.treduce_bench import run_treduce_once

_HERE = Path(__file__).resolve().parent
_HCCL_BENCH = _HERE / "hccl_bench.py"
_HCCL_BENCH_CPP = _HERE / "hccl_bench.cc"
_HCCL_BENCH_BIN = _HERE / ".hccl_bench_bin"


def _resolve_ascend_home() -> Path | None:
    candidates = [
        os.environ.get("ASCEND_HOME_PATH"),
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/latest",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir():
            return path
    return None


def _find_header_roots(ascend_home: Path) -> list[Path]:
    header_roots: list[Path] = []
    for acl_header in ascend_home.rglob("acl/acl.h"):
        include_root = acl_header.parent.parent
        if (include_root / "hccl" / "hccl.h").is_file() and include_root not in header_roots:
            header_roots.append(include_root)
    return header_roots


def _find_library_dirs(ascend_home: Path) -> list[Path]:
    library_dirs: list[Path] = []
    for hccl_lib in ascend_home.rglob("libhccl.so"):
        lib_dir = hccl_lib.parent
        if (lib_dir / "libascendcl.so").is_file() and lib_dir not in library_dirs:
            library_dirs.append(lib_dir)
    return library_dirs


def _ensure_hccl_bench_binary() -> Path:
    if not _HCCL_BENCH_CPP.is_file():
        raise FileNotFoundError(f"HCCL bench source not found: {_HCCL_BENCH_CPP}")

    ascend_home = _resolve_ascend_home()
    if ascend_home is None:
        raise FileNotFoundError(
            "Could not find Ascend toolkit root. Set ASCEND_HOME_PATH to the toolkit installation directory."
        )

    include_dirs = _find_header_roots(ascend_home)
    if not include_dirs:
        raise FileNotFoundError(
            f"Could not find acl/acl.h and hccl/hccl.h under {ascend_home}. "
            "Set ASCEND_HOME_PATH to a toolkit root that contains the public Ascend headers."
        )

    library_dirs = _find_library_dirs(ascend_home)
    if not library_dirs:
        raise FileNotFoundError(
            f"Could not find libhccl.so and libascendcl.so under {ascend_home}. "
            "Set ASCEND_HOME_PATH to a toolkit root that contains the Ascend shared libraries."
        )

    needs_rebuild = (
        not _HCCL_BENCH_BIN.is_file()
        or _HCCL_BENCH_BIN.stat().st_mtime < _HCCL_BENCH_CPP.stat().st_mtime
    )
    if not needs_rebuild:
        return _HCCL_BENCH_BIN

    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-pthread",
        str(_HCCL_BENCH_CPP),
    ]
    for include_dir in include_dirs:
        cmd.extend(["-I", str(include_dir)])
    for library_dir in library_dirs:
        cmd.extend(["-L", str(library_dir)])
        cmd.append(f"-Wl,-rpath,{library_dir}")
    cmd.extend([
        "-lhccl",
        "-lascendcl",
        "-o",
        str(_HCCL_BENCH_BIN),
    ])
    proc = subprocess.run(
        cmd,
        cwd=_HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to build HCCL bench helper:\n{proc.stdout[-2000:]}")
    return _HCCL_BENCH_BIN


def _cmd_validate_case(path: str) -> int:
    case = EquivalenceCase.from_json_file(path)
    case.validate()
    print(f"case_id={case.case_id}")
    print(f"equivalence_hash={case.equivalence_hash()}")
    print(f"n_bytes={case.n_bytes} window_nbytes={case.window_nbytes}")
    return 0


def _apply_case_overrides(case: EquivalenceCase, args: argparse.Namespace) -> EquivalenceCase:
    """Apply CLI overrides to a loaded case file."""
    if getattr(args, "count", None) is not None:
        case.count = args.count
    if getattr(args, "warmup_rounds", None) is not None:
        case.warmup_rounds = args.warmup_rounds
    if getattr(args, "timed_rounds", None) is not None:
        case.timed_rounds = args.timed_rounds

    if case.count <= 0:
        raise ValueError(f"count must be positive, got {case.count}")
    if case.warmup_rounds < 0:
        raise ValueError(f"warmup_rounds must be >= 0, got {case.warmup_rounds}")
    if case.timed_rounds <= 0:
        raise ValueError(f"timed_rounds must be >= 1, got {case.timed_rounds}")

    case.validate()
    return case


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

_CAMPAIGN_STACKS = frozenset({"hccl", "simpler-own"})


def _parse_hccl_per_rank(line: str) -> list[float] | None:
    if "per_rank=" not in line:
        return None
    try:
        payload = line.split("per_rank=", 1)[1].strip()
        end = payload.index("]")
        return [float(x) for x in json.loads(payload[: end + 1])]
    except (ValueError, json.JSONDecodeError):
        return None


def _parse_hccl_setup_s(lines: list[str]) -> float | None:
    for line in lines:
        if "HCCL_COMM_SETUP_OK" in line and "setup_s=" in line:
            try:
                return float(line.split("setup_s=", 1)[1].split()[0])
            except ValueError:
                return None
        if "HCCL_ALLREDUCE_OK" in line and "setup_s=" in line:
            try:
                return float(line.split("setup_s=", 1)[1].split()[0])
            except ValueError:
                return None
    return None


def _extract_execute_s(
    stack: str,
    wall: float,
    phases: dict[str, float],
    lines: list[str],
    *,
    per_rank: list[float] | None = None,
) -> tuple[float, list[float] | None]:
    """Return (execute_s, per_rank_execute_s) for a benchmark round."""
    if stack == "hccl" and per_rank:
        return max(per_rank), per_rank
    if stack == "simpler-own":
        return wall, None
    if phases and "execute" in phases:
        return float(phases["execute"]), None
    return wall, None


def _compute_setup_s(
    stack: str,
    phases: dict[str, float],
    *,
    hccl_setup_s: float | None = None,
) -> float | None:
    if stack == "hccl":
        return hccl_setup_s
    if stack == "simpler-own":
        setup = 0.0
        if "compile" in phases:
            setup += float(phases["compile"])
        if "init" in phases:
            setup += float(phases["init"])
        return setup if setup > 0 else None
    if phases and "init" in phases:
        return float(phases["init"])
    return None


def _format_bw(n_bytes: int, seconds: float) -> str:
    bw = n_bytes / seconds if seconds > 0 else 0
    if bw >= 1e6:
        return f"{bw / 1e6:.1f} MB/s"
    if bw >= 1e3:
        return f"{bw / 1e3:.1f} KB/s"
    return f"{bw:.1f} B/s"


def _build_sample(
    *,
    round_idx: int,
    label: str,
    ok: bool,
    wall_s: float,
    execute_s: float,
    setup_s: float | None,
    n_bytes: int,
    phases: dict[str, float] | None = None,
    per_rank_execute_s: list[float] | None = None,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "round": round_idx,
        "phase": label,
        "wall_s": round(wall_s, 6),
        "execute_s": round(execute_s, 6),
        "ok": ok,
        "bw_execute_mb_s": round((n_bytes / execute_s) / 1e6, 6) if execute_s > 0 else 0.0,
    }
    if setup_s is not None:
        sample["setup_s"] = round(setup_s, 6)
    if phases:
        sample["phases"] = {k: round(v, 6) for k, v in phases.items()}
    if per_rank_execute_s is not None:
        sample["per_rank_execute_s"] = [round(v, 6) for v in per_rank_execute_s]
    return sample

# Phase markers for simpler stdout parsing.
# Each marker records the wall time from the *previous* marker to the line
# containing this string.  The first marker records from process start.
_SIMPLER_MARKERS = [
    ("startup",      "compiling kernels"),        # t0 → "compiling kernels" (imports, arg parse)
    ("compile",      "init worker"),               # → kernel + orch compilation
    ("init",         "running "),                  # → worker.init + domain alloc + TaskArgs
    # execute_done: fires when simpler main.py emits this marker.
    # Today simpler doesn't emit it, so this phase stays empty
    # and execute below captures init → golden_match (DAG + host-side verify).
    # When simpler adds the print, execute captures DAG-only.
    ("execute_done", "SIMPLER_EXECUTE_DONE"),      # → DAG execution complete
    ("execute",      "all ranks matched golden"),  # → DAG + host-side golden verify
    ("fail",         "golden check FAILED"),
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
    if case.count != 256:
        return False, (
            f"simpler supports only count=256 (hardcoded ALLREDUCE_COUNT); "
            f"got {case.count}. Use 'hccl' stack for size sweeps."
        ), [], 0.0, {}
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


_SIMPLER_OWN_MARKERS = [
    ("startup",      "compiling kernels"),
    ("compile",      "init worker"),
    ("execute_done", "SIMPLER_EXECUTE_DONE"),
    ("execute",      "SIMPLER_EXECUTE_DONE"),  # our runner prints this
]


def _run_simpler_own_once(case: EquivalenceCase, extra_flags: list[str] | None = None) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Run one mesh allreduce round in-process (compile/init reused across rounds)."""
    if case.variant != "mesh":
        return False, f"simpler-own only supports mesh variant, got {case.variant}", [], 0.0, {}
    del extra_flags  # reserved for future profile flags

    from collectives.runners.simpler_own import get_mesh_allreduce_session

    try:
        session = get_mesh_allreduce_session(
            case.count,
            case.device_ids,
            case.platform,
            None,
        )
        ok, wall, err = session.execute()
        phases = session.execute_phases(wall)
        return ok, err, [], wall, phases
    except Exception as exc:
        return False, str(exc), [], 0.0, {}


_PYPTO_MARKERS = [
    ("startup", "PYPTO_COMPILE_BEGIN"),
    ("compile", "PYPTO_RUNTIME_INIT_BEGIN"),
    ("init", "PYPTO_RUNTIME_EXECUTE_BEGIN"),
    ("execute", "PYPTO_ALLREDUCE_OK"),
    ("test_result", "PASSED"),
    ("test_result", "FAILED"),
    ("test_result", "ERROR"),
]

_HCCL_MARKERS = [
    ("init", "HCCL_COMM_SETUP_OK"),
    ("execute", "HCCL_ALLREDUCE_OK"),
]

_PTOREDUCE_MARKERS = [
    ("init", "[----------]"),     # gtest test case begin
    ("execute", "[       OK ]"),   # gtest pass
    ("execute", "[  PASSED  ]"),   # gtest pass (alt)
    ("fail", "[  FAILED  ]"),
]


def _run_hccl_once(case: EquivalenceCase, extra_flags: list[str] | None = None) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Single HCCL HcclAllReduce invocation. Returns (ok, error, lines, total_s, phases)."""
    samples, err = _run_hccl_campaign(case, warmup_rounds=0, timed_rounds=1, extra_flags=extra_flags)
    if not samples:
        return False, err or "HCCL campaign produced no samples", [], 0.0, {}
    sample = samples[-1]
    phases = sample.get("phases", {})
    return sample["ok"], err, [], float(sample["wall_s"]), phases


def _run_hccl_campaign(
    case: EquivalenceCase,
    warmup_rounds: int,
    timed_rounds: int,
    extra_flags: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run warmup+timed HCCL rounds in one subprocess. Returns (samples, error)."""
    try:
        bench_bin = _ensure_hccl_bench_binary()
    except (FileNotFoundError, RuntimeError) as exc:
        return [], str(exc)

    visible_devices = _devices_comma(case.device_ids)
    cmd = [
        str(bench_bin),
        "--count", str(case.count),
        "--dtype", case.dtype,
        "--devices", _devices_comma(case.device_ids),
        "--warmup-rounds", str(warmup_rounds),
        "--timed-rounds", str(timed_rounds),
    ] + (extra_flags or [])

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(_HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "ASCEND_RT_VISIBLE_DEVICES": visible_devices,
            "HCCL_OP_EXPANSION_MODE": "AI_CPU",
        },
        timeout=600,
        check=False,
    )
    total_wall = time.perf_counter() - t0
    lines = proc.stdout.splitlines() if proc.stdout else []
    if proc.returncode != 0:
        err = f"exit={proc.returncode}"
        if lines:
            err += f"\n{lines[-1][:200]}"
        return [], err

    setup_s = _parse_hccl_setup_s(lines)
    warmup_lines: list[tuple[int, list[float]]] = []
    timed_lines: list[tuple[int, list[float]]] = []
    for line in lines:
        if line.startswith("HCCL_WARMUP"):
            per_rank = _parse_hccl_per_rank(line)
            if per_rank is None:
                continue
            try:
                round_no = int(line.split("round=", 1)[1].split()[0])
            except ValueError:
                round_no = len(warmup_lines) + 1
            warmup_lines.append((round_no, per_rank))
        elif line.startswith("HCCL_TIMED"):
            per_rank = _parse_hccl_per_rank(line)
            if per_rank is None:
                continue
            try:
                round_no = int(line.split("round=", 1)[1].split()[0])
            except ValueError:
                round_no = len(timed_lines) + 1
            timed_lines.append((round_no, per_rank))

    samples: list[dict[str, Any]] = []
    round_idx = 0
    for round_no, per_rank in warmup_lines:
        execute_s = max(per_rank)
        phases = {"init": setup_s or 0.0, "execute": execute_s} if round_idx == 0 and setup_s else {"execute": execute_s}
        sample_setup = setup_s if round_idx == 0 else None
        samples.append(_build_sample(
            round_idx=round_idx,
            label=f"warmup-{round_no}" if len(warmup_lines) > 1 else "warmup",
            ok=True,
            wall_s=total_wall if round_idx == 0 else execute_s,
            execute_s=execute_s,
            setup_s=sample_setup,
            n_bytes=case.n_bytes,
            phases=phases,
            per_rank_execute_s=per_rank,
        ))
        round_idx += 1

    for round_no, per_rank in timed_lines:
        execute_s = max(per_rank)
        samples.append(_build_sample(
            round_idx=round_idx,
            label=f"timed-{round_no}",
            ok=True,
            wall_s=execute_s,
            execute_s=execute_s,
            setup_s=None,
            n_bytes=case.n_bytes,
            phases={"execute": execute_s},
            per_rank_execute_s=per_rank,
        ))
        round_idx += 1

    if not samples:
        return [], "HCCL campaign produced no HCCL_WARMUP/HCCL_TIMED lines"
    return samples, ""


def _run_simpler_own_campaign(
    case: EquivalenceCase,
    warmup_rounds: int,
    timed_rounds: int,
    extra_flags: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run warmup+timed simpler-own rounds in-process with one session."""
    del extra_flags
    if case.variant != "mesh":
        return [], f"simpler-own only supports mesh variant, got {case.variant}"

    from collectives.runners.simpler_own import close_mesh_allreduce_session, get_mesh_allreduce_session

    samples: list[dict[str, Any]] = []
    try:
        session = get_mesh_allreduce_session(
            case.count,
            case.device_ids,
            case.platform,
            None,
        )
        round_idx = 0
        for r in range(warmup_rounds):
            ok, execute_s, err = session.execute()
            if not ok:
                return samples, err
            phases = session.execute_phases(execute_s)
            setup_s = _compute_setup_s("simpler-own", phases)
            samples.append(_build_sample(
                round_idx=round_idx,
                label="warmup" if warmup_rounds == 1 else f"warmup-{r + 1}",
                ok=True,
                wall_s=execute_s,
                execute_s=execute_s,
                setup_s=setup_s if round_idx == 0 else None,
                n_bytes=case.n_bytes,
                phases=phases,
            ))
            round_idx += 1

        for r in range(timed_rounds):
            ok, execute_s, err = session.execute()
            if not ok:
                return samples, err
            phases = session.execute_phases(execute_s)
            samples.append(_build_sample(
                round_idx=round_idx,
                label=f"timed-{r + 1}",
                ok=True,
                wall_s=execute_s,
                execute_s=execute_s,
                setup_s=None,
                n_bytes=case.n_bytes,
                phases=phases,
            ))
            round_idx += 1
    except Exception as exc:
        return samples, str(exc)
    finally:
        close_mesh_allreduce_session()

    return samples, ""


def _run_pto_isa_once(case: EquivalenceCase, extra_flags: list[str] | None = None) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Single pto-isa TREDUCE invocation. Returns (ok, error, lines, total_s, phases)."""
    return run_treduce_once(case, extra_flags)


def _run_pypto_once(case: EquivalenceCase, extra_flags: list[str] | None = None) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Single pypto pytest invocation. Returns (ok, error, lines, total_s, phases)."""
    if case.count != 256:
        return False, (
            f"pypto supports only count=256 (hardcoded SIZE in test file); "
            f"got {case.count}. Use 'hccl' stack for size sweeps."
        ), [], 0.0, {}
    if case.variant == "ring":
        return False, "pypto ring allreduce ST not yet implemented", [], 0.0, {}
    # Use the composite intrinsic test (pld.tensor.allreduce) to measure
    # the lowering tax. The hand-rolled test_l3_allreduce.py is the reference
    # implementation — comparing it against simpler measures "hand-coded DSL
    # vs hand-coded C++", not the abstraction cost.
    test_file = pypto_root() / "tests" / "st" / "distributed" / "test_l3_tensor_allreduce_intrinsic.py"
    if not test_file.is_file():
        return False, f"pypto test not found: {test_file}", [], 0.0, {}

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYPTO_COMPILE_PROFILING": "1"}
    # DON'T add pypto/python to PYTHONPATH — pypto is already pip-installed
    # in the Docker image. Prepending the source tree shadows the installed
    # pypto_core .so and causes import errors.

    cmd = [
        sys.executable, "-m", "pytest", str(test_file),
        "-v", "--platform", case.platform, "--device", _devices_comma(case.device_ids),
        "-k", f"n_ranks-{case.p}",  # only run the parametrized case matching P
        "-s",
    ] + (extra_flags or [])
    return _run_with_phases(
        cmd, str(pypto_root()), env,
        timeout=600, markers=_PYPTO_MARKERS,
    )


def _format_execute_s(execute_s: float) -> str:
    if execute_s < 0.001:
        return f"{execute_s:.6f}s"
    if execute_s < 0.01:
        return f"{execute_s:.4f}s"
    return f"{execute_s:.3f}s"


def _print_sample_line(stack: str, sample: dict[str, Any], n_bytes: int, lines: list[str] | None = None) -> None:
    execute_s = float(sample["execute_s"])
    details: list[str] = []
    if sample.get("setup_s") is not None:
        details.append(f"setup={float(sample['setup_s']):.2f}s")
    phases = sample.get("phases", {})
    for key, value in phases.items():
        if key not in ("fail",):
            details.append(f"{key}={float(value):.2f}s")
    details.append(_format_bw(n_bytes, execute_s))
    if sample.get("per_rank_execute_s"):
        details.append(f"ranks={sample['per_rank_execute_s']}")
    elif stack == "hccl" and lines:
        for line in lines:
            per_rank = _parse_hccl_per_rank(line)
            if per_rank is not None:
                details.append(f"ranks={per_rank}")
                break
    status = "✅" if sample.get("ok") else "❌"
    print(f"{_format_execute_s(execute_s)} {status}  [{', '.join(details)}]")


def _aggregate_execute_stats(samples: list[dict[str, Any]]) -> tuple[float, float, float | None]:
    timed = [s for s in samples if str(s.get("phase", "")).startswith("timed")]
    execute_values = [float(s["execute_s"]) for s in timed if "execute_s" in s]
    if execute_values:
        mean = statistics.mean(execute_values)
        stdev = statistics.stdev(execute_values) if len(execute_values) > 1 else 0.0
    else:
        mean, stdev = 0.0, 0.0
    setup_s = next((float(s["setup_s"]) for s in samples if s.get("setup_s") is not None), None)
    return mean, stdev, setup_s


def _run_stack_multi(
    case: EquivalenceCase,
    stack: str,
    bundle: RunArtifactBundle,
    profile_spec: str,
) -> tuple[bool, str, list[dict[str, Any]], float, float, float, float | None]:
    """Run warmup + timed rounds for one stack.

    Returns (all_ok, error, samples, execute_s_mean, execute_s_stdev, wall_s_mean, setup_s).
    """
    warmup = case.warmup_rounds
    timed_rounds = case.timed_rounds
    total = warmup + timed_rounds
    pf = _profile_flags(profile_spec)
    extra = pf.get(stack, [])

    samples: list[dict[str, Any]] = []
    all_ok = True
    last_error = ""

    if stack in _CAMPAIGN_STACKS:
        print(f"  [{stack:>7}] campaign ({warmup} warmup + {timed_rounds} timed)...", flush=True)
        if stack == "hccl":
            campaign_samples, err = _run_hccl_campaign(case, warmup, timed_rounds, extra)
        else:
            campaign_samples, err = _run_simpler_own_campaign(case, warmup, timed_rounds, extra)

        if err and not campaign_samples:
            return False, err, [], 0.0, 0.0, 0.0, None

        for sample in campaign_samples:
            label = str(sample.get("phase", ""))
            print(f"  [{stack:>7}] ({label:>7})...", end=" ", flush=True)
            _print_sample_line(stack, sample, case.n_bytes)
            if not sample.get("ok", False):
                all_ok = False
                last_error = err or "campaign round failed"
                if label.startswith("timed"):
                    print(f"  [{stack:>7}] ABORTING after timed round failure")
                    break
        samples = campaign_samples
        if err and all_ok:
            last_error = err
    else:
        if stack == "simpler":
            runner = _run_simpler_once
        elif stack == "pypto":
            runner = _run_pypto_once
        elif stack == "pto-isa":
            runner = _run_pto_isa_once
        else:
            return False, f"unknown stack: {stack}", [], 0.0, 0.0, 0.0, None

        for r in range(total):
            label = "warmup" if r < warmup else f"timed-{r - warmup + 1}"
            print(f"  [{stack:>7}] round {r + 1:>2}/{total} ({label:>7})...", end=" ", flush=True)
            ok, err, lines, wall, phases = runner(case, extra)
            per_rank = None
            for line in lines:
                parsed = _parse_hccl_per_rank(line)
                if parsed is not None:
                    per_rank = parsed
                    break
            execute_s, per_rank_list = _extract_execute_s(stack, wall, phases, lines, per_rank=per_rank)
            setup_s = _compute_setup_s(stack, phases) if r == 0 else None
            sample = _build_sample(
                round_idx=r,
                label=label,
                ok=ok,
                wall_s=wall,
                execute_s=execute_s,
                setup_s=setup_s,
                n_bytes=case.n_bytes,
                phases=phases,
                per_rank_execute_s=per_rank_list,
            )
            if stack == "pypto":
                for line in lines:
                    if line.startswith("PYPTO_COMPILE_PROFILE "):
                        parts = line.strip().split()
                        compile_fields: dict[str, Any] = {}
                        for part in parts[1:]:
                            if "=" not in part:
                                continue
                            key, value = part.split("=", 1)
                            if key == "path":
                                compile_fields[key] = value
                            else:
                                try:
                                    compile_fields[key] = round(float(value), 6)
                                except ValueError:
                                    compile_fields[key] = value
                        sample["compile_profile"] = compile_fields
                        break
            samples.append(sample)
            _print_sample_line(stack, sample, case.n_bytes, lines)
            if not ok and lines:
                for line in lines[-8:]:
                    stripped = line.rstrip()
                    if stripped:
                        print(f"         {stripped[:120]}")
            if not ok:
                all_ok = False
                last_error = err
                if label.startswith("timed"):
                    print(f"  [{stack:>7}] ABORTING after timed round failure")
                    break

    bundle.log_path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    bundle.write_timing(samples)

    execute_mean, execute_stdev, setup_s = _aggregate_execute_stats(samples)
    timed_wall = [
        float(s["wall_s"]) for s in samples if str(s.get("phase", "")).startswith("timed")
    ]
    wall_mean = statistics.mean(timed_wall) if timed_wall else execute_mean
    return all_ok, last_error, samples, execute_mean, execute_stdev, wall_mean, setup_s


def _aggregate_timed_phase_means(samples: list[dict[str, Any]]) -> dict[str, float]:
    timed = [sample for sample in samples if str(sample.get("phase", "")).startswith("timed")]
    if not timed:
        return {}

    phase_totals: dict[str, float] = {}
    phase_counts: dict[str, int] = {}
    for sample in timed:
        phases = sample.get("phases", {})
        for name, value in phases.items():
            phase_totals[name] = phase_totals.get(name, 0.0) + float(value)
            phase_counts[name] = phase_counts.get(name, 0) + 1

    return {
        name: round(phase_totals[name] / phase_counts[name], 6)
        for name in sorted(phase_totals)
        if phase_counts[name] > 0
    }


def _aggregate_timed_compile_profile_means(samples: list[dict[str, Any]]) -> dict[str, float]:
    timed = [sample for sample in samples if str(sample.get("phase", "")).startswith("timed")]
    if not timed:
        return {}

    numeric_totals: dict[str, float] = {}
    numeric_counts: dict[str, int] = {}
    for sample in timed:
        compile_profile = sample.get("compile_profile", {})
        for name, value in compile_profile.items():
            if isinstance(value, (int, float)):
                numeric_totals[name] = numeric_totals.get(name, 0.0) + float(value)
                numeric_counts[name] = numeric_counts.get(name, 0) + 1

    return {
        name: round(numeric_totals[name] / numeric_counts[name], 6)
        for name in sorted(numeric_totals)
        if numeric_counts[name] > 0
    }


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

    # pto-isa treduce binary
    if "pto-isa" in stacks:
        from collectives.treduce_bench import _find_treduce_test_binary
        binary = _find_treduce_test_binary()
        print(f"  pto-isa treduce binary: {'✅' if binary else '❌ not found (build with build_st.py)'}")

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
    return _cmd_pair_impl(args)


def _cmd_cross_variant(args: argparse.Namespace) -> int:
    """Run same (P, count, dtype, devices) with two different algorithms."""
    case_a = EquivalenceCase.from_json_file(args.case_file_a)
    case_b = EquivalenceCase.from_json_file(args.case_file_b)
    case_a = _apply_case_overrides(case_a, args)
    case_b = _apply_case_overrides(case_b, args)

    # Assert same contract
    mismatches: list[str] = []
    if case_a.p != case_b.p:
        mismatches.append(f"P: {case_a.p} vs {case_b.p}")
    if case_a.count != case_b.count:
        mismatches.append(f"count: {case_a.count} vs {case_b.count}")
    if case_a.dtype != case_b.dtype:
        mismatches.append(f"dtype: {case_a.dtype} vs {case_b.dtype}")
    if case_a.device_ids != case_b.device_ids:
        mismatches.append(f"device_ids: {case_a.device_ids} vs {case_b.device_ids}")
    if mismatches:
        print(f"ERROR: cross-variant cases must match on P/count/dtype/devices. Mismatches: {mismatches}",
              file=sys.stderr)
        return 1

    print(f"\n{'═'*70}")
    print(f"  CROSS-VARIANT: {case_a.variant} vs {case_b.variant}")
    print(f"  P={case_a.p}  count={case_a.count}  dtype={case_a.dtype}  devices={case_a.device_ids}")
    print(f"{'═'*70}")

    results_a = _cmd_pair_impl(args, override_case=case_a, label_prefix=case_a.variant)
    if results_a != 0:
        return results_a

    results_b = _cmd_pair_impl(args, override_case=case_b, label_prefix=case_b.variant)
    return results_b


def _cmd_pair_impl(
    args: argparse.Namespace,
    override_case: EquivalenceCase | None = None,
    label_prefix: str | None = None,
) -> int:
    if override_case is not None:
        case = override_case
    else:
        case = EquivalenceCase.from_json_file(args.case_file)
    case = _apply_case_overrides(case, args)

    # ── pre-flight diagnostics ──
    _print_diagnostics(case, args)

    prefix = f"[{label_prefix}] " if label_prefix else ""
    print(f"{prefix}case_id={case.case_id}  eq_hash={case.equivalence_hash()}")
    print(f"{prefix}warmup={case.warmup_rounds}  timed={case.timed_rounds}  "
          f"profile={args.profile or 'none'}")

    stacks = [s.strip() for s in args.stacks.split(",")]
    unknown = [s for s in stacks if s not in ("simpler", "simpler-own", "pypto", "hccl", "pto-isa")]
    if unknown:
        print(f"ERROR: unknown stacks: {unknown}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    if label_prefix:
        # cross-variant: write per-variant results files, e.g. results_mesh.json
        out_path = out_path.parent / f"{out_path.stem}_{label_prefix}{out_path.suffix}"
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

        all_ok, last_error, samples, exec_mean, exec_stdev, wall_mean, setup_s = _run_stack_multi(
            case, stack, bundle, args.profile,
        )

        bundle.write_manifest(
            correctness="pass" if all_ok else "fail",
            error=last_error or None,
        )

        timed_samples = [s for s in samples if str(s.get("phase", "")).startswith("timed")]
        bw_execute = (
            round(statistics.mean(float(s["bw_execute_mb_s"]) for s in timed_samples), 6)
            if timed_samples else 0.0
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
            "execute_s_mean": round(exec_mean, 6),
            "execute_s_stdev": round(exec_stdev, 6),
            "setup_s": round(setup_s, 6) if setup_s is not None else None,
            "bw_execute_mb_s": bw_execute,
            "wall_s_mean": round(wall_mean, 6),
            "wall_s_stdev": round(exec_stdev, 6),
            "phase_means": _aggregate_timed_phase_means(samples),
            "compile_profile_means": _aggregate_timed_compile_profile_means(samples),
            "n_warmup": case.warmup_rounds,
            "n_timed": case.timed_rounds,
            "artifact_bundle": str(bundle.bundle_dir),
        })
        status = "PASS" if all_ok else "FAIL"
        setup_str = f" setup={setup_s:.2f}s" if setup_s is not None else ""
        print(f"  {stack}: execute={exec_mean:.4f}s±{exec_stdev:.4f}s{setup_str} {status}")
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

    # ── summary table (execute_s is primary; setup_s reported once per stack) ──
    nbytes = case.n_bytes
    print(f"\n{'═'*70}")
    print(f"  RESULTS  ({case.count}×{case.dtype}, {nbytes} B/rank, P={case.p})")
    print(f"  Primary metric: execute_s (collective execution only)")
    print(f"{'═'*70}")
    print(f"  {'stack':>11}  {'execute_s':>16}  {'setup_s':>10}  {'bw_execute':>14}  ok")
    print(f"  {'-'*11}  {'-'*16}  {'-'*10}  {'-'*14}  --")
    bw_map: dict[str, float] = {}
    for r in rows:
        exec_mean = r["execute_s_mean"]
        exec_stdev = r["execute_s_stdev"]
        setup_s = r.get("setup_s")
        bw = r.get("bw_execute_mb_s") or (nbytes / exec_mean / 1e6 if exec_mean > 0 else 0)
        bw_map[r["stack"]] = bw * 1e6  # bytes/s for ratio math
        ok = r["correctness"]
        setup_str = f"{setup_s:.2f}s" if setup_s is not None else "—"
        bw_str = _format_bw(nbytes, exec_mean)
        print(f"  {r['stack']:>11}  {exec_mean:>8.4f}s±{exec_stdev:.4f}s  {setup_str:>10}  {bw_str:>14}  "
              f"{'✅' if ok == 'pass' else '❌'}")

    if "hccl" in bw_map and bw_map["hccl"] > 0:
        hccl_bw = bw_map["hccl"]
        for r in rows:
            stack = r["stack"]
            if stack == "hccl":
                continue
            eff = bw_map[stack] / hccl_bw * 100
            print(f"  {stack:>11} vs HCCL execute bandwidth: {eff:>5.1f}%")
    print(f"{'═'*70}")

    all_ok = all(r["correctness"] == "pass" for r in rows)
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collective benchmark sweep (pypto-tooling)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-case", help="Validate EquivalenceCase JSON")
    p_val.add_argument("--case-file", required=True)

    p_pair = sub.add_parser("pair-mesh", help="Run same case on simpler, pypto, and/or hccl")
    p_pair.add_argument("--case-file", required=True)
    p_pair.add_argument("--stacks", default="hccl,simpler,pypto",
                        help="Comma-separated: hccl,simpler,simpler-own,pypto,pto-isa")
    p_pair.add_argument("--campaign", default="default")
    p_pair.add_argument("--profile", default="", help="Comma: l2,pmu,dep")
    p_pair.add_argument("--count", type=int, default=None, help="Override case payload element count (HCCL only; simpler/pypto pinned to 256)")
    p_pair.add_argument("--warmup-rounds", type=int, default=None, help="Override case warmup rounds")
    p_pair.add_argument("--timed-rounds", type=int, default=None, help="Override case timed rounds")
    p_pair.add_argument("--out", required=True, help="results.json path under results/campaigns/")

    p_cross = sub.add_parser("cross-variant", help="Compare two algorithm variants at same (P,count,dtype,devices)")
    p_cross.add_argument("--case-file-a", required=True, help="First variant EquivalenceCase JSON")
    p_cross.add_argument("--case-file-b", required=True, help="Second variant EquivalenceCase JSON")
    p_cross.add_argument("--stacks", default="hccl,simpler",
                         help="Comma-separated: hccl,simpler,simpler-own,pypto,pto-isa")
    p_cross.add_argument("--campaign", default="cross_variant")
    p_cross.add_argument("--profile", default="", help="Comma: l2,pmu,dep")
    p_cross.add_argument("--count", type=int, default=None, help="Override case payload element count")
    p_cross.add_argument("--warmup-rounds", type=int, default=None, help="Override case warmup rounds")
    p_cross.add_argument("--timed-rounds", type=int, default=None, help="Override case timed rounds")
    p_cross.add_argument("--out", required=True, help="results.json path under results/campaigns/")

    args = parser.parse_args(argv)
    if args.command == "validate-case":
        return _cmd_validate_case(args.case_file)
    if args.command == "pair-mesh":
        return _cmd_pair_mesh(args)
    if args.command == "cross-variant":
        return _cmd_cross_variant(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
