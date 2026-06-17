"""Own benchmark runner — compiles & runs our mesh allreduce kernels via simpler.

Unlike simpler's examples/workers/l3/allreduce_distributed/, these kernels accept
arbitrary count at runtime instead of hardcoding ALLREDUCE_COUNT=256.

Usage (direct):
    PYTHONPATH=. python collectives/runners/simpler_own.py \
        --count 65536 --devices 0-3 --platform a2a3

Usage (from harness):
    from collectives.runners.simpler_own import MeshAllreduceSession
    session = MeshAllreduceSession(count=65536, devices=[0,1,2,3], platform="a2a3")
    try:
        ok, execute_s, err = session.execute()
    finally:
        session.close()

``execute()`` returns ``execute_s``: time for ``worker.run()`` only (collective
execution). Compile and worker init are reported once via ``execute_phases()``.
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

# simpler's Python packages (simpler_setup, simpler) live under the simpler root.
_simpler_root_str = str(simpler_root())
if _simpler_root_str not in sys.path:
    sys.path.insert(0, _simpler_root_str)

_HERE = Path(__file__).resolve().parent
_KERNEL_DIR = _HERE.parent / "kernels"

K_MAX_SUPPORTED_RANKS = 16

_CHIP_CALLABLE_CACHE: dict[tuple[Any, ...], Any] = {}
_ACTIVE_SESSION: Any = None
_ACTIVE_SESSION_KEY: tuple[Any, ...] | None = None


def _parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def expected_output(nranks: int, count: int) -> list[float]:
    """output[i] = sum_r (i + r*100) = nranks*i + 100 * nranks*(nranks-1)/2."""
    return [float(nranks * i + 100 * nranks * (nranks - 1) // 2) for i in range(count)]


def _kernel_cache_key(platform: str, pto_isa_commit: str | None) -> tuple[Any, ...]:
    kernel_source = _KERNEL_DIR / "aiv" / "allreduce_mesh.cpp"
    orch_source = _KERNEL_DIR / "orchestration" / "allreduce_mesh_orch.cpp"
    return (
        platform,
        pto_isa_commit or "",
        kernel_source.stat().st_mtime,
        orch_source.stat().st_mtime,
    )


def _build_chip_callable_uncached(
    platform: str,
    pto_isa_commit: str | None = None,
) -> Any:
    """Compile the AIV kernel + orchestration shim via simpler's KernelCompiler."""
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable
    from simpler_setup.kernel_compiler import KernelCompiler
    from simpler_setup.pto_isa import ensure_pto_isa_root

    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)
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
        from simpler_setup.elf_parser import extract_text_section
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


def build_chip_callable(
    platform: str,
    pto_isa_commit: str | None = None,
) -> Any:
    """Return a cached ChipCallable for (platform, kernel sources, pto_isa_commit)."""
    cache_key = _kernel_cache_key(platform, pto_isa_commit)
    if cache_key not in _CHIP_CALLABLE_CACHE:
        _CHIP_CALLABLE_CACHE[cache_key] = _build_chip_callable_uncached(platform, pto_isa_commit)
    return _CHIP_CALLABLE_CACHE[cache_key]


class MeshAllreduceSession:
    """One compile + one worker init; call execute() many times before close()."""

    def __init__(
        self,
        count: int,
        devices: list[int],
        platform: str = "a2a3",
        pto_isa_commit: str | None = None,
    ) -> None:
        from simpler.task_interface import CallConfig
        from simpler.worker import Worker

        self.count = count
        self.devices = devices
        self.platform = platform
        self.pto_isa_commit = pto_isa_commit
        self.nranks = len(devices)

        if self.nranks < 2 or self.nranks > K_MAX_SUPPORTED_RANKS:
            raise ValueError(f"nranks must be in [2, {K_MAX_SUPPORTED_RANKS}], got {self.nranks}")
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")

        dtype_nbytes = 4
        signal_tail_nbytes = self.nranks * 4
        self.scratch_nbytes = count * dtype_nbytes + signal_tail_nbytes
        self.window_size = max(self.scratch_nbytes, 4096)

        self.host_inputs = [
            torch.tensor([i + rank * 100 for i in range(count)], dtype=torch.float32).share_memory_()
            for rank in range(self.nranks)
        ]
        self.host_outputs = [torch.zeros(count, dtype=torch.float32).share_memory_() for _ in range(self.nranks)]
        self._expected = torch.tensor(expected_output(self.nranks, count), dtype=torch.float32)

        t0 = time.perf_counter()
        chip_callable = build_chip_callable(platform, pto_isa_commit)
        self.compile_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.worker = Worker(
            level=3,
            platform=platform,
            runtime="tensormap_and_ringbuffer",
            device_ids=devices,
            num_sub_workers=0,
            pto_isa_commit=pto_isa_commit,
        )
        self.chip_handle = self.worker.register(chip_callable)
        self.worker.init()
        self.init_s = time.perf_counter() - t1

        self._execute_count = 0
        self._CallConfig = CallConfig

    def execute(self, verify: bool = True) -> tuple[bool, float, str]:
        """Run one allreduce. Returns (ok, execute_s, error).

        ``execute_s`` is ``worker.run()`` wall time only (collective execution).
        """
        t0 = time.perf_counter()
        self.worker.run(self._orch_fn, args=None, config=self._CallConfig())
        execute_s = time.perf_counter() - t0
        self._execute_count += 1

        if not verify:
            return True, execute_s, ""

        for i in range(self.nranks):
            max_diff = float(torch.max(torch.abs(self.host_outputs[i] - self._expected)))
            if max_diff > 1e-3:
                return False, execute_s, f"chip {i}: max diff = {max_diff:.3e}"
        return True, execute_s, ""

    def execute_phases(self, execute_s: float) -> dict[str, float]:
        """Phase breakdown for harness reporting (setup vs execute)."""
        if self._execute_count == 1:
            return {
                "compile": self.compile_s,
                "init": self.init_s,
                "execute": max(execute_s, 0.0),
            }
        return {"execute": execute_s}

    def _orch_fn(self, orch: Any, _args: Any, cfg: Any) -> None:
        from simpler.task_interface import CommBufferSpec, ContinuousTensor, DataType, TaskArgs, TensorArgType
        from simpler_setup.torch_interop import make_tensor_arg

        with orch.allocate_domain(
            name="default",
            workers=list(range(self.nranks)),
            window_size=self.window_size,
            buffers=[
                CommBufferSpec(
                    name="scratch",
                    dtype="float32",
                    count=self.count,
                    nbytes=self.scratch_nbytes,
                ),
            ],
        ) as handle:
            for i in range(self.nranks):
                domain = handle[i]
                chip_args = TaskArgs()
                chip_args.add_tensor(make_tensor_arg(self.host_inputs[i]), TensorArgType.INPUT)
                chip_args.add_tensor(make_tensor_arg(self.host_outputs[i]), TensorArgType.OUTPUT_EXISTING)
                chip_args.add_tensor(
                    ContinuousTensor.make(
                        data=domain.buffer_ptrs["scratch"],
                        shapes=(self.count,),
                        dtype=DataType.FLOAT32,
                        child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )
                chip_args.add_scalar(self.count)
                chip_args.add_scalar(domain.domain_size)
                chip_args.add_scalar(domain.device_ctx)
                orch.submit_next_level(self.chip_handle, chip_args, cfg, worker=i)

    def close(self) -> None:
        self.worker.close()


def _session_key(count: int, devices: list[int], platform: str, pto_isa_commit: str | None) -> tuple[Any, ...]:
    return (count, tuple(devices), platform, pto_isa_commit or "")


def get_mesh_allreduce_session(
    count: int,
    devices: list[int],
    platform: str = "a2a3",
    pto_isa_commit: str | None = None,
) -> MeshAllreduceSession:
    """Return a reused session for repeated harness rounds (same count/devices/platform)."""
    global _ACTIVE_SESSION, _ACTIVE_SESSION_KEY
    key = _session_key(count, devices, platform, pto_isa_commit)
    if _ACTIVE_SESSION is None or _ACTIVE_SESSION_KEY != key:
        close_mesh_allreduce_session()
        _ACTIVE_SESSION = MeshAllreduceSession(count, devices, platform, pto_isa_commit)
        _ACTIVE_SESSION_KEY = key
    return _ACTIVE_SESSION


def close_mesh_allreduce_session() -> None:
    global _ACTIVE_SESSION, _ACTIVE_SESSION_KEY
    if _ACTIVE_SESSION is not None:
        _ACTIVE_SESSION.close()
        _ACTIVE_SESSION = None
        _ACTIVE_SESSION_KEY = None


def run_mesh_allreduce(
    count: int,
    devices: list[int],
    platform: str = "a2a3",
    pto_isa_commit: str | None = None,
    verify: bool = True,
) -> tuple[bool, float, str]:
    """Run mesh allreduce once (compile + init + execute). Prefer MeshAllreduceSession for loops."""
    session = MeshAllreduceSession(count, devices, platform, pto_isa_commit)
    try:
        return session.execute(verify=verify)
    finally:
        session.close()


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

    print("compiling kernels...", flush=True)
    session = MeshAllreduceSession(args.count, devices, args.platform, args.pto_isa_commit)
    print(f"init worker (compile={session.compile_s:.2f}s init={session.init_s:.2f}s)...", flush=True)

    try:
        for r in range(args.warmup_rounds):
            ok, execute_s, err = session.execute()
            if not ok:
                print(f"WARMUP FAILED round {r}: {err}", file=sys.stderr)
                return 1
            print(f"[warmup {r+1}/{args.warmup_rounds}] execute_s={execute_s:.4f}s")

        execute_times: list[float] = []
        for r in range(args.timed_rounds):
            ok, execute_s, err = session.execute()
            if not ok:
                print(f"TIMED FAILED round {r}: {err}", file=sys.stderr)
                return 1
            execute_times.append(execute_s)
            bw = (args.count * 4) / execute_s if execute_s > 0 else 0
            bw_str = f"{bw/1e6:.1f} MB/s" if bw >= 1e6 else f"{bw/1e3:.1f} KB/s"
            print(f"[timed {r+1}/{args.timed_rounds}] execute_s={execute_s:.4f}s  {bw_str}")

        if execute_times:
            mean = sum(execute_times) / len(execute_times)
            print(f"execute_s_mean={mean:.4f}s  n={len(execute_times)}  count={args.count}  P={len(devices)}")
            print("SIMPLER_EXECUTE_DONE")
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
