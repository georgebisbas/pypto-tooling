"""Shared timing-metric parsers for the collective benchmark harness.

Central home for extracting on-device and per-rank timings from runtime
output, so the in-process runners (``pypto_own``, ``simpler_own``) and the
subprocess runners (``simpler``, ``pto-isa``) report the same metrics.
"""

from __future__ import annotations

import json
import re

# The simpler/pypto runtime emits one [STRACE] span per rank with
# ``name=simpler_run.runner_run.device_wall ... dur=<ns> clk=dev``.
_DEVICE_WALL_RE = re.compile(r"device_wall[^\n]*?dur=(\d+)")


def parse_device_wall_s(text: str) -> float | None:
    """Extract the slowest-rank ``device_wall`` span duration (seconds).

    The slowest rank's span is the collective completion. Returns ``None``
    when the runtime did not emit the span in ``text``.
    """
    durs = [int(m) for m in _DEVICE_WALL_RE.findall(text)]
    if not durs:
        return None
    return max(durs) / 1e9


# HCCL microbenchmark prints ``per_rank=[<t0>,<t1>,...]`` on each round line.
_HCCL_PER_RANK_RE = re.compile(r"per_rank=\s*(\[[^\]]*\])")


def parse_hccl_per_rank(line: str) -> list[float] | None:
    """Parse per-rank seconds from an ``HCCL_WARMUP``/``HCCL_TIMED`` line.

    The caller distinguishes warmup from timed rounds by the line prefix; this
    parser is format-only.
    """
    match = _HCCL_PER_RANK_RE.search(line)
    if not match:
        return None
    try:
        values = [float(v) for v in json.loads(match.group(1))]
    except (ValueError, json.JSONDecodeError):
        return None
    return values if values else None
