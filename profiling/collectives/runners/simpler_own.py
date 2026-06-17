"""Own benchmark runner — compiles & runs our mesh allreduce kernels via simpler.

Unlike simpler's examples/workers/l3/allreduce_distributed/, these kernels accept
arbitrary count at runtime instead of hardcoding ALLREDUCE_COUNT=256.

Usage (direct):
    PYTHONPATH=. python collectives/runners/simpler_own.py \
        --count 65536 --devices 0-3 --platform a2a3

Usage (from harness):
    from collectives.runners.simpler_own import run_mesh_allreduce
    ok, wall_s = run_mesh_allreduce(count=65536, devices=[0,1,2,3], platform="a2a3")
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from collectives.config import simpler_root

_HERE = Path(__file__).resolve().parent
_KERNEL_DIR = _HERE.parent / "kernels"

K_MAX_SUPPORTED_RANKS = 16


def _parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def expected_output(nranks: int, count: int) -> list[float]:
    """output[i] = sum_r (i + r*100) = nranks*i + 100 * nranks*(nranks-1)/2."""
    return [float(nranks * i + 100 * nranks * (nranks - 1) // 2) for i in range(count)]


def build_chip_callable(
    platform: str,
    pto_isa_commit: str | None = None,
) -> Any:
    """Compile the AIV kernel + orchestration shim via simpler's KernelCompiler."""
    from simpler_setup.kernel_compiler import KernelCompiler, CoreCallable, ChipCallable, ArgDirection
    from simpler_setup.runtime_compiler import ensure_pto_isa_root

    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    # Kernel needs access to platform_comm/comm_context.h under src/common/
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    kernel_source = str(_KERNEL_DIR / "aiv" / "allreduce_mesh.cpp")
    if not os.path.isfile(kernel_source):
        raise FileNotFoundError(f"kernel source not found: {kernel_source}")

    kernel_bytes = kc.compile_incore(
        source_path=kernel_source,
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )

    if not platform.endswith("sim"):
        from simpler_setup.runtime_compiler import extract_text_section
        kernel_bytes = extract_text_section(kernel_bytes)

    orch_source = str(_KERNEL_DIR / "orchestration" / "allreduce_mesh_orch.cpp")
    if not os.path.isfile(orch_source):
        raise FileNotFoundError(f"orch source not found: {orch_source}")

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=orch_source,
    )

    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        func_name="allreduce_mesh_orch",
        config_name="allreduce_mesh_orch_config",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def run_mesh_allreduce(
    count: int,
    devices: list[int],
    platform: str = "a2a3",
    pto_isa_commit: str | None = None,
    timed: bool = True,
    warmup: bool = False,
) -> tuple[bool, float, str]:
    """Run mesh allreduce with arbitrary count. Returns (ok, wall_s, error)."""
    from simpler_setup.orchestrator import Worker, CallConfig
    from simpler_setup.orchestrator import CommBufferSpec

    nranks = len(devices)
    if nranks < 2 or nranks > K_MAX_SUPPORTED_RANKS:
        return False, 0.0, f"nranks must be in [2, {K_MAX_SUPPORTED_RANKS}], got {nranks}"
    if count <= 0:
        return False, 0.0, f"count must be positive, got {count}"

    dtype_nbytes = 4  # float32
    signal_tail_nbytes = nranks * 4  # one int32 slot per rank
    scratch_nbytes = count * dtype_nbytes + signal_tail_nbytes
    window_size = max(scratch_nbytes, 4096)

    label = "warmup" if warmup else "bench"
    if not warmup:
        pass  # suppress output during timed runs

    host_inputs = [
        torch.tensor([i + rank * 100 for i in range(count)], dtype=torch.float32).share_memory_()
        for rank in range(nranks)
    ]
    host_outputs = [torch.zeros(count, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    chip_callable = build_chip_callable(platform, pto_isa_commit)

    worker = Worker(
        level=3,
        block_dim=None,
        aicpu_thread_num=None,
        platform=platform,
        pto_isa_commit=pto_isa_commit,
    )

    try:
        worker.init(devices)

        def orch_fn(orch: Any) -> None:
            with orch.allocate_domain(
                name="default",
                workers=list(range(nranks)),
                window_size=window_size,
                buffers=[
                    CommBufferSpec(
                        name="scratch",
                        dtype="float32",
                        count=count,
                        nbytes=scratch_nbytes,
                    ),
                ],
            ) as domain:
                scratch = domain.buffers["scratch"]
                for i in range(nranks):
                    config = CallConfig()
                    chip_handle = orch.get_chip_handle(i)
                    chip_args = chip_handle.create_task_args()
                    chip_args.add_input(host_inputs[i])
                    chip_args.add_output_existing(host_outputs[i])
                    chip_args.add_inout_buffer(scratch, i)
                    chip_args.add_scalar(count)
                    chip_args.add_scalar(nranks)
                    chip_args.add_scalar(chip_handle.comm_context_ptr(i))
                    orch.submit_next_level(chip_handle, chip_args, config, worker=i)

        t0 = time.perf_counter()
        worker.run(orch_fn, args=None, config=CallConfig())
        wall_s = time.perf_counter() - t0

        if not timed:
            expected = torch.tensor(expected_output(nranks, count), dtype=torch.float32)
            for i in range(nranks):
                max_diff = float(torch.max(torch.abs(host_outputs[i] - expected)))
                if max_diff > 1e-3:
                    return False, wall_s, f"chip {i}: max diff = {max_diff:.3e}"

        return True, wall_s, ""
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Mesh allreduce benchmark kernel (dynamic count)")
    parser.add_argument("--count", type=int, required=True, help="Number of float32 elements")
    parser.add_argument("--devices", default="0-1", help="Device range, e.g. '0-1' or '0-3'")
    parser.add_argument("--platform", default="a2a3", help="a2a3, a2a3sim, a5, a5sim")
    parser.add_argument("--pto-isa-commit", default=None, help="pto-isa commit (auto if unset)")
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--timed-rounds", type=int, default=5)
    args = parser.parse_args()

    devices = _parse_device_range(args.devices)

    # Warmup
    for r in range(args.warmup_rounds):
        ok, wall, err = run_mesh_allreduce(args.count, devices, args.platform, args.pto_isa_commit, warmup=True)
        if not ok:
            print(f"WARMUP FAILED round {r}: {err}", file=sys.stderr)
            return 1
        print(f"[warmup {r+1}/{args.warmup_rounds}] {wall:.4f}s")

    # Timed
    walls: list[float] = []
    for r in range(args.timed_rounds):
        ok, wall, err = run_mesh_allreduce(args.count, devices, args.platform, args.pto_isa_commit, timed=True)
        if not ok:
            print(f"TIMED FAILED round {r}: {err}", file=sys.stderr)
            return 1
        walls.append(wall)
        bw = (args.count * 4) / wall if wall > 0 else 0
        bw_str = f"{bw/1e6:.1f} MB/s" if bw >= 1e6 else f"{bw/1e3:.1f} KB/s"
        print(f"[timed {r+1}/{args.timed_rounds}] {wall:.4f}s  {bw_str}")

    if walls:
        mean = sum(walls) / len(walls)
        print(f"mean={mean:.4f}s  n={len(walls)}  count={args.count}  P={len(devices)}")
        print("SIMPLER_EXECUTE_DONE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
