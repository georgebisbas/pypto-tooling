"""O + N/B bandwidth model — pipelining scorecard for collective timings.

Fits each (stack, P, variant) across a message-size sweep to the two-parameter
collective time model

    T(N) = O + N / B

where ``O`` is the fixed per-collective latency (dispatch, barrier rounds,
neighbour handshakes) and ``B`` is the asymptotic marginal bandwidth (the
steady-state rate a fully-pipelined kernel reaches once the link never idles).
Fitting is ordinary least squares of ``T`` against payload bytes ``N``
(``slope = 1/B``, ``intercept = O``).

The **pipelining score** is the fraction of the largest-payload time that is
actual transfer rather than fixed overhead:

    pipeline_score = (N_max / B) / T(N_max)      in [0, 1]

A stack that already reaches bandwidth-bound steady state at the largest
measured size scores close to 1.0; a stack still dominated by per-collective
latency/barrier cost scores well below — the quantity pipelining (multi-
buffering, async issue, neighbour-local barriers) directly moves.

The **bandwidth efficiency vs HCCL** is ``B / B_hccl`` at the same
(P, variant) — the headline "how close to HCCL's data plane" number.

Timing source preference: ``device_wall_s_mean`` (pure on-device collective)
over ``execute_s_mean`` (dispatch + collective) over wall times, per group.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int32": 4}

# Source precedence — highest wins. device_wall is the pure on-device span.
_TIMING_SOURCES = (
    "device_wall_s_mean",
    "device_wall_s_median",
    "execute_s_mean",
    "execute_s_median",
    "wall_s_mean",
    "wall_s",
)


def n_bytes(count: int, dtype: str) -> int:
    """Payload bytes per rank for a run (count elements of dtype)."""
    return count * _DTYPE_BYTES.get(dtype, 4)


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares ``y = intercept + slope * x``.

    Returns ``(slope, intercept, r2)``. Raises ValueError when not enough
    points or x has zero variance.
    """
    n = len(xs)
    if n < 2:
        raise ValueError("need >= 2 points for a fit")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("zero variance in x — cannot fit")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return slope, intercept, r2


def _run_time(run: dict[str, Any]) -> tuple[str, float | None]:
    """Return ``(source_key, seconds)`` for a run, best available source."""
    for key in _TIMING_SOURCES:
        val = run.get(key)
        if val is None:
            continue
        try:
            value = float(val)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return key, value
    return "", None


@dataclass
class BandwidthModel:
    """Fitted T(N) = O + N/B for one (stack, P, variant) group."""

    stack: str
    p: int
    variant: str
    source: str
    n_points: int
    latency_s: float
    bandwidth_b_s: float
    r2: float
    pipeline_score: float
    largest_n_bytes: int
    largest_t_s: float
    bw_eff_vs_hccl: float | None = None
    latency_ratio_vs_hccl: float | None = None
    # (n_bytes, seconds) points used for the fit — for plotting.
    points: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def group_points(runs: Iterable[dict[str, Any]]) -> dict[tuple[str, int, str], list[tuple[float, float, str]]]:
    """Group runs into (stack, P, variant) buckets of (n_bytes, seconds, source).

    Points keep the best source available per run; the group-level fit later
    requires a consistent source across all points.
    """
    groups: dict[tuple[str, int, str], list[tuple[float, float, str]]] = {}
    for run in runs:
        stack = run.get("stack")
        p = run.get("p")
        count = run.get("count")
        if not stack or p is None or not count:
            continue
        source, seconds = _run_time(run)
        if not source or seconds is None:
            continue
        key = (stack, int(p), str(run.get("variant") or "?"))
        groups.setdefault(key, []).append(
            (float(n_bytes(int(count), run.get("dtype") or "fp32")), seconds, source)
        )
    return groups


def _group_source(points: list[tuple[float, float, str]]) -> str:
    """Highest-precedence source present for every point, else 'mixed'."""
    present = {src for _, _, src in points}
    for source in _TIMING_SOURCES:
        if source in present and all(src == source for _, _, src in points):
            return source
    return "mixed"


def fit_group(
    stack: str,
    p: int,
    variant: str,
    points: list[tuple[float, float, str]],
    hccl_model: BandwidthModel | None = None,
) -> BandwidthModel | None:
    """Fit one (stack, P, variant) group. Returns None when not enough points."""
    pts = sorted(points, key=lambda pt: pt[0])
    if len({pt[0] for pt in pts}) < 3:
        return None
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    try:
        slope, intercept, r2 = _linfit(xs, ys)
    except ValueError:
        return None
    if slope <= 0.0:
        return None
    bandwidth = 1.0 / slope
    latency = max(intercept, 0.0)
    largest_n, largest_t = int(pts[-1][0]), pts[-1][1]
    transfer = largest_n / bandwidth
    pipeline_score = transfer / largest_t if largest_t > 0.0 else 0.0
    model = BandwidthModel(
        stack=stack,
        p=p,
        variant=variant,
        source=_group_source(points),
        n_points=len(pts),
        latency_s=round(latency, 12),
        bandwidth_b_s=round(bandwidth, 6),
        r2=round(r2, 6),
        pipeline_score=round(min(max(pipeline_score, 0.0), 1.0), 6),
        largest_n_bytes=largest_n,
        largest_t_s=round(largest_t, 12),
        points=[[n, t] for n, t, _src in pts],
    )
    if hccl_model is not None:
        if hccl_model.bandwidth_b_s > 0.0:
            model.bw_eff_vs_hccl = round(bandwidth / hccl_model.bandwidth_b_s, 6)
        if hccl_model.latency_s > 0.0:
            model.latency_ratio_vs_hccl = round(latency / hccl_model.latency_s, 6)
    return model


def score_runs(runs: Iterable[dict[str, Any]]) -> list[BandwidthModel]:
    """Fit the O + N/B model for every (stack, P, variant) group in ``runs``."""
    groups = group_points(runs)
    models: list[BandwidthModel] = []
    hccl: dict[tuple[int, str], BandwidthModel] = {}
    # First pass: fit everything; remember HCCL references for (P, variant).
    for (stack, p, variant), points in groups.items():
        model = fit_group(stack, p, variant, points)
        if model is None:
            continue
        models.append(model)
        if stack == "hccl":
            hccl[(p, variant)] = model
    # Second pass: attach HCCL-relative scores.
    for model in models:
        if model.stack == "hccl":
            continue
        ref = hccl.get((model.p, model.variant))
        if ref is None:
            continue
        if ref.bandwidth_b_s > 0.0:
            model.bw_eff_vs_hccl = round(model.bandwidth_b_s / ref.bandwidth_b_s, 6)
        if ref.latency_s > 0.0:
            model.latency_ratio_vs_hccl = round(model.latency_s / ref.latency_s, 6)
    return models


def format_bandwidth(bandwidth_b_s: float) -> str:
    """Human-readable marginal bandwidth."""
    if bandwidth_b_s >= 1e9:
        return f"{bandwidth_b_s / 1e9:.1f} GB/s"
    if bandwidth_b_s >= 1e6:
        return f"{bandwidth_b_s / 1e6:.1f} MB/s"
    return f"{bandwidth_b_s / 1e3:.1f} KB/s"


def format_latency(latency_s: float) -> str:
    """Human-readable per-collective latency."""
    if latency_s >= 1e-3:
        return f"{latency_s * 1e3:.1f} ms"
    if latency_s >= 1e-6:
        return f"{latency_s * 1e6:.1f} µs"
    return f"{latency_s * 1e9:.0f} ns"
