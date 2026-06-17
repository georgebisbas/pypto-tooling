"""PTO-ISA TREDUCE benchmark — calls the existing pto-isa treduce gtest binary via mpirun.

The pto-isa treduce test binary must be built beforehand:
    cd {pto_isa_root} && python3 tests/script/build_st.py -r npu -v a3 -t treduce

This module wraps the existing gtest binary with external subprocess timing.
Markers are extracted from gtest output lines.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from collectives.config import pto_isa_root
from collectives.equivalence import EquivalenceCase

_TREDUCE_MARKERS = [
    ("init", "[----------]"),    # gtest test case start
    ("execute", "[       OK ]"),  # gtest pass
    ("execute", "[  PASSED  ]"),  # gtest pass (alternative format)
    ("fail", "[  FAILED  ]"),
]


def _find_treduce_test_binary() -> Path | None:
    """Find the pre-built pto-isa treduce gtest binary."""
    pto_isa = pto_isa_root()
    candidates = [
        pto_isa / "build" / "npu" / "a2a3" / "st" / "comm" / "treduce" / "treduce_test",
        pto_isa / "build" / "npu" / "a5" / "st" / "comm" / "treduce" / "treduce_test",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _build_treduce_test() -> Path | None:
    """Build the pto-isa treduce test via build_st.py. Returns path to binary on success."""
    pto_isa = pto_isa_root()
    build_script = pto_isa / "tests" / "script" / "build_st.py"
    if not build_script.is_file():
        return None

    print(f"[treduce_bench] Building pto-isa treduce test (this may take a while)...", flush=True)
    result = subprocess.run(
        [sys.executable, str(build_script), "-r", "npu", "-v", "a3", "-t", "treduce"],
        cwd=str(pto_isa),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"[treduce_bench] Build FAILED:\n{result.stderr[-800:]}", file=sys.stderr, flush=True)
        return None

    return _find_treduce_test_binary()


def _ensure_treduce_binary() -> Path:
    """Find or build the treduce test binary. Raises FileNotFoundError if unavailable."""
    binary = _find_treduce_test_binary()
    if binary is not None:
        return binary
    binary = _build_treduce_test()
    if binary is not None:
        return binary
    raise FileNotFoundError(
        f"pto-isa treduce test binary not found and build failed. "
        f"Build manually: cd {pto_isa_root()} && "
        f"python3 tests/script/build_st.py -r npu -v a3 -t treduce"
    )


def _case_to_gtest_filter(case: EquivalenceCase) -> str | None:
    """Map an EquivalenceCase to the closest available gtest filter.

    Returns None if no matching test exists for the given (count, dtype, P).
    """
    p = case.p
    count = case.count
    dtype = case.dtype

    # The existing treduce test covers these (dtype, count, P) combinations:
    # See pto-isa/tests/npu/a2a3/comm/st/testcase/treduce/main.cpp
    GTEST_MAP: dict[tuple[str, int, int], str] = {
        # Basic (small tile) — float
        ("fp32", 256, 4):  "TReduce.FloatSmall_Sum_4Ranks",
        ("fp32", 256, 8):  "TReduce.FloatSmall_Sum_8Ranks",
        # Basic — int32 (use for fp32 too since data is bitwise identical at these sizes)
        ("fp32", 512, 2):  "TReduce.Int32Small_Sum",
        ("fp32", 512, 8):  "TReduce.Int32Small_Sum_8Ranks",
        ("fp32", 4096, 2): "TReduce.Int32Large_Sum",
        ("fp32", 4096, 8): "TReduce.Int32Large_Sum_8Ranks",
        # int32 variants
        ("int32", 256, 2):  "TReduce.Int32Small_Max",   # closest: 256-element int32
        ("int32", 256, 8):  "TReduce.Int32Small_Max_8Ranks",
        ("int32", 512, 2):  "TReduce.Int32Small_Sum",
        ("int32", 512, 8):  "TReduce.Int32Small_Sum_8Ranks",
        ("int32", 4096, 2): "TReduce.Int32Large_Sum",
        ("int32", 4096, 8): "TReduce.Int32Large_Sum_8Ranks",
        # Large shape (chunked) — 128×32 int32 = 4096 elems
        ("int32", 4096, 2): "TReduce.LargeShape_Int32_128x32_tile16_Sum",
        ("int32", 4096, 4): "TReduce.LargeShape_Int32_128x32_tile16_Sum_4Ranks",
        # Large shape — 256×64 float = 16384 elems
        ("fp32", 16384, 2): "TReduce.LargeShape_Float_256x64_tile32_Sum",
        # Large shape — 512×32 int32 = 16384 elems
        ("int32", 16384, 2): "TReduce.LargeShape_Int32_512x32_tile64_Sum",
        ("int32", 16384, 8): "TReduce.LargeShape_Int32_512x32_tile64_Sum_8Ranks",
        # Ping-pong double buffering
        ("int32", 4096, 2): "TReduce.PingPong_Int32_128x32_tile16_Sum",
        ("int32", 4096, 4): "TReduce.PingPong_Int32_128x32_tile16_Sum_4Ranks",
        ("fp32", 16384, 2): "TReduce.PingPong_Float_256x64_tile32_Sum",
    }

    # Try exact match first
    key = (dtype, count, p)
    if key in GTEST_MAP:
        return GTEST_MAP[key]

    # Try matching count and P (dtype may differ — int32 test works for fp32 data too)
    for (dt, c, np_), filter_str in GTEST_MAP.items():
        if c == count and np_ == p:
            return filter_str

    return None


def _run_mpirun(case: EquivalenceCase, binary: Path, gtest_filter: str,
                extra_flags: list[str] | None = None) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Run treduce test via mpirun. Returns (ok, error, lines, wall_s, phases)."""
    p = case.p
    device_list = ",".join(str(d) for d in case.device_ids)

    cmd = [
        "mpirun",
        "--allow-run-as-root",
        "-n", str(p),
        str(binary),
        f"--gtest_filter={gtest_filter}",
    ] + (extra_flags or [])

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "ASCEND_RT_VISIBLE_DEVICES": device_list,
        "HCCL_OP_EXPANSION_MODE": "AI_CPU",
    }

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(binary.parent), env=env,
    )

    lines: list[str] = []
    phase_times: dict[str, float] = {}
    last_t = t0

    try:
        for line in proc.stdout:
            now = time.perf_counter()
            lines.append(line)
            # Track gtest phases
            if "[----------]" in line and "init" not in phase_times:
                phase_times["init"] = now - last_t
                last_t = now
            elif ("[       OK ]" in line or "[  PASSED  ]" in line) and "execute" not in phase_times:
                phase_times["execute"] = now - last_t
                last_t = now
            elif "[  FAILED  ]" in line and "fail" not in phase_times:
                phase_times["fail"] = now - last_t
                last_t = now

        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        total = time.perf_counter() - t0
        return False, f"mpirun timeout after 600s", lines, total, phase_times

    total = time.perf_counter() - t0

    stderr_text = proc.stderr.read()
    if stderr_text:
        lines.append("\n--- STDERR ---\n" + stderr_text)

    ok = proc.returncode == 0 and "fail" not in phase_times

    error = "" if ok else f"mpirun exit={proc.returncode}"
    if not ok and stderr_text:
        error += f"\nstderr: {stderr_text[-500:]}"

    return ok, error, lines, total, phase_times


def run_treduce_once(
    case: EquivalenceCase,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str, list[str], float, dict[str, float]]:
    """Run a single TREDUCE benchmark round.

    Args:
        case: EquivalenceCase with (count, dtype, p, device_ids).
        extra_flags: Additional CLI flags (unused for gtest binary).

    Returns:
        (ok, error_message, stdout_lines, total_wall_s, phases_dict)
    """
    # Only fp32 and int32 are supported by the existing treduce test
    if case.dtype not in ("fp32", "int32"):
        return False, (
            f"pto-isa TREDUCE benchmark only supports fp32 and int32 dtypes; "
            f"got {case.dtype}"
        ), [], 0.0, {}

    gtest_filter = _case_to_gtest_filter(case)
    if gtest_filter is None:
        supported = [
            "P=2: count=256(fp32), 512(int32/fp32), 4096(int32/fp32), 16384(int32/fp32)",
            "P=4: count=256(fp32), 4096(int32), 16384(int32)",
            "P=8: count=256(fp32), 512(int32), 4096(int32), 16384(int32)",
        ]
        return False, (
            f"pto-isa TREDUCE: no gtest case for count={case.count}, dtype={case.dtype}, P={case.p}. "
            f"Supported combinations: {'; '.join(supported)}"
        ), [], 0.0, {}

    try:
        binary = _ensure_treduce_binary()
    except FileNotFoundError as exc:
        return False, str(exc), [], 0.0, {}

    return _run_mpirun(case, binary, gtest_filter, extra_flags)
