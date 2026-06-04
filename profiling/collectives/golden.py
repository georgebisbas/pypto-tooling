"""Shared golden reference for allreduce mesh (rank_linear_v1 / allreduce_sum_v1)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collectives.equivalence import EquivalenceCase


def fill_rank_inputs(case: EquivalenceCase) -> list[list[float]]:
    """Per-rank input vectors — matches simpler allreduce_distributed/main.py."""
    if case.input_formula != "rank_linear_v1":
        raise NotImplementedError(f"input_formula {case.input_formula!r}")
    out: list[list[float]] = []
    for rank in range(case.p):
        out.append([float(i + rank * 100) for i in range(case.count)])
    return out


def expected_output_allreduce_sum_v1(case: EquivalenceCase) -> list[float]:
    """Golden vector: sum of all rank inputs element-wise."""
    if case.golden != "allreduce_sum_v1":
        raise NotImplementedError(f"golden {case.golden!r}")
    inputs = fill_rank_inputs(case)
    acc = [0.0] * case.count
    for row in inputs:
        for i, v in enumerate(row):
            acc[i] += v
    return acc


def verify_outputs(
    case: EquivalenceCase,
    per_rank_outputs: list[list[float]],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> tuple[bool, str]:
    """Independent verifier — use for both stacks before timing."""
    expected = expected_output_allreduce_sum_v1(case)
    if len(per_rank_outputs) != case.p:
        return False, f"expected {case.p} rank outputs, got {len(per_rank_outputs)}"
    for r, out in enumerate(per_rank_outputs):
        if len(out) != case.count:
            return False, f"rank {r}: len {len(out)} != count {case.count}"
        for i, (a, b) in enumerate(zip(out, expected, strict=True)):
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                return False, f"rank {r} idx {i}: {a} != {b} (expected)"
    return True, "ok"
