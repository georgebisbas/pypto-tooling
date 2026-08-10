"""Benchmark stack registry — single source of truth for stack capabilities.

Every stack the harness can run is described here. ``kind`` selects the run
path in ``run_sweep.py``:

* ``campaign``   — one in-process session; warmup + timed rounds reuse the
  compile and setup, so ``execute_s`` is a true per-round collective time.
* ``subprocess`` — one subprocess invocation per round (full init each round);
  ``execute_s`` is the best available phase marker.

``variants`` is the set of allreduce algorithm families the stack can run:
``mesh``, ``ring`` (RS+AG), ``twophase`` (mesh RS+AG). ``count_constraint``
lists the allowed payload element counts (``None`` = unbounded).
"""

from __future__ import annotations

from typing import Any

STACK_REGISTRY: dict[str, dict[str, Any]] = {
    "hccl": {
        "kind": "campaign",
        "variants": ("mesh", "ring", "twophase"),
        "count_constraint": None,  # unbounded — any count is valid
        "orch_profile": "mesh_l3_host_domain_v1",
        "description": "CANN HcclAllReduce baseline (algorithm internal to HCCL)",
    },
    "simpler": {
        "kind": "subprocess",
        "variants": ("mesh", "ring", "twophase"),
        "count_constraint": [256],  # ALLREDUCE_COUNT hardcoded in the C++ kernel
        "orch_profile": "mesh_l3_host_domain_v1",
        "description": "hand-written L3 C++ allreduce (--mode mesh|ring|twophase)",
    },
    "simpler-own": {
        "kind": "campaign",
        "variants": ("mesh",),
        "count_constraint": None,
        "orch_profile": "mesh_l3_host_domain_v1",
        "description": "our dynamic-count AIV kernel via simpler KernelCompiler (in-process)",
    },
    "pypto-composite": {
        "kind": "campaign",
        "variants": ("mesh", "ring"),
        "count_constraint": None,
        "orch_profile": "mesh_l3_host_domain_v1",
        "description": "InCore pld.tensor.allreduce composite authored with @pl.jit.host (in-process)",
    },
    "pypto-host": {
        "kind": "campaign",
        "variants": ("mesh", "ring"),
        "count_constraint": None,
        "orch_profile": "mesh_l3_host_builtin_v1",
        "description": "HOST builtin pld.tensor.allreduce authored with @pl.jit.host (in-process)",
        "caveats": ("ring mode: ReduceOp.Sum + FP32 only",),
    },
    "pto-isa": {
        "kind": "subprocess",
        "variants": ("mesh",),
        "count_constraint": None,
        "orch_profile": "mesh_l3_host_domain_v1",
        "description": "pto-isa TREDUCE tile-instruction microbenchmark",
    },
}

CAMPAIGN_STACKS = frozenset(
    name for name, info in STACK_REGISTRY.items() if info["kind"] == "campaign"
)
SUBPROCESS_STACKS = frozenset(
    name for name, info in STACK_REGISTRY.items() if info["kind"] == "subprocess"
)

# Default stack set for the 4-stack apples-to-apples comparison.
DEFAULT_STACKS = ("hccl", "simpler", "pypto-composite", "pypto-host")


def supported_variants(stack: str) -> tuple[str, ...]:
    """Algorithm variants the stack can run (mesh / ring / twophase)."""
    return STACK_REGISTRY[stack]["variants"]


def stack_supports(stack: str, variant: str) -> bool:
    """Whether the stack can run the given algorithm variant."""
    return variant in STACK_REGISTRY[stack]["variants"]


def count_constraint(stack: str) -> list[int] | None:
    """Allowed payload element counts for the stack (None = unbounded)."""
    return STACK_REGISTRY[stack]["count_constraint"]


def count_allowed(stack: str, count: int) -> bool:
    allowed = count_constraint(stack)
    return allowed is None or count in allowed


def orch_profile(stack: str) -> str:
    return STACK_REGISTRY[stack]["orch_profile"]


def stack_caveats(stack: str) -> tuple[str, ...]:
    return tuple(STACK_REGISTRY[stack].get("caveats", ()))


def validate_stack_name(stack: str) -> None:
    """Raise ValueError for an unregistered stack name."""
    if stack not in STACK_REGISTRY:
        raise ValueError(
            f"unknown stack {stack!r}; registered: {', '.join(sorted(STACK_REGISTRY))}"
        )
