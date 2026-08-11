"""Unit tests for the O + N/B bandwidth-model scorecard (collectives.model)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROFILING = Path(__file__).resolve().parents[1]
if str(_PROFILING) not in sys.path:
    sys.path.insert(0, str(_PROFILING))

from collectives import model as bw_model  # noqa: E402


def _run(stack, p, count, execute, device=None, variant="mesh", dtype="fp32"):
    run = {
        "stack": stack,
        "p": p,
        "count": count,
        "variant": variant,
        "dtype": dtype,
        "execute_s_mean": execute,
    }
    if device is not None:
        run["device_wall_s_mean"] = device
    return run


def test_linfit_exact_line():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [3.0 + 2.0 * x for x in xs]
    slope, intercept, r2 = bw_model._linfit(xs, ys)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(3.0)
    assert r2 == pytest.approx(1.0)


def test_fit_group_recovers_o_and_b():
    # T(N) = 20us + N / 4e9 B/s, at 4 payload sizes.
    latency, bandwidth = 20e-6, 4e9
    points = [
        (4096 * 4, latency + (4096 * 4) / bandwidth, "execute_s_mean"),
        (65536 * 4, latency + (65536 * 4) / bandwidth, "execute_s_mean"),
        (262144 * 4, latency + (262144 * 4) / bandwidth, "execute_s_mean"),
        (1048576 * 4, latency + (1048576 * 4) / bandwidth, "execute_s_mean"),
    ]
    model = bw_model.fit_group("pypto-composite", 8, "mesh", points)
    assert model is not None
    assert model.latency_s == pytest.approx(latency, rel=0.05)
    assert model.bandwidth_b_s == pytest.approx(bandwidth, rel=0.05)
    assert model.r2 > 0.99
    assert model.source == "execute_s_mean"
    # At the largest size the transfer fraction is high (bandwidth-bound).
    assert model.pipeline_score > 0.9


def test_fit_group_needs_three_sizes():
    points = [
        (4096 * 4, 30e-6, "execute_s_mean"),
        (65536 * 4, 90e-6, "execute_s_mean"),
    ]
    assert bw_model.fit_group("hccl", 2, "mesh", points) is None


def test_score_runs_prefers_device_wall():
    # Same execute time but device_wall shrinks with payload (true bandwidth).
    runs = []
    for count in (4096, 65536, 262144, 1048576):
        execute = 200e-6 + (count * 4) / 1e9
        device = 20e-6 + (count * 4) / 4e9
        runs.append(_run("pypto-composite", 8, count, execute, device=device))
        runs.append(_run("hccl", 8, count, execute, device=device))
    models = {m.stack: m for m in bw_model.score_runs(runs)}
    assert "pypto-composite" in models
    assert models["pypto-composite"].source == "device_wall_s_mean"
    assert models["pypto-composite"].bw_eff_vs_hccl is not None
    assert 0.0 < models["pypto-composite"].bw_eff_vs_hccl <= 1.0
    # device_wall 4e9 vs hccl device 4e9 → bandwidth efficiency ~1.0
    assert models["pypto-composite"].bw_eff_vs_hccl == pytest.approx(1.0, rel=0.1)


def test_score_runs_groups_by_p_and_variant():
    runs = []
    for variant in ("mesh", "ring"):
        for p in (2, 4):
            for count in (4096, 65536, 262144, 1048576):
                t = 20e-6 + (count * 4) / 4e9
                runs.append(_run("hccl", p, count, t, variant=variant))
                runs.append(_run("pypto-composite", p, count, t, variant=variant))
    models = bw_model.score_runs(runs)
    keys = {(m.stack, m.p, m.variant) for m in models}
    assert keys == {
        ("hccl", 2, "mesh"), ("hccl", 4, "mesh"),
        ("hccl", 2, "ring"), ("hccl", 4, "ring"),
        ("pypto-composite", 2, "mesh"), ("pypto-composite", 4, "mesh"),
        ("pypto-composite", 2, "ring"), ("pypto-composite", 4, "ring"),
    }


def test_n_bytes():
    assert bw_model.n_bytes(65536, "fp32") == 262144
    assert bw_model.n_bytes(65536, "fp16") == 131072
    assert bw_model.n_bytes(65536, "unknown") == 262144  # default 4 B/elt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
