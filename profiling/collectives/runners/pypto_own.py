"""PyPTO in-process benchmark runners — composite + host builtin allreduce.

Authors the collective with the modern ``@pl.jit`` surface (``@pl.jit.host``
host orchestrator + ``@pl.jit`` chip orchestrator + ``@pl.jit.incore`` step)
calling ``pld.tensor.allreduce`` — either the InCore composite (mesh / ring)
or the HOST builtin (mesh / ring) — then compiles once via
``JITFunction.compile(RunConfig(...))`` and dispatches many rounds through
``DistributedWorker.run``. This is the same compile-once / dispatch-many
shape as ``simpler_own.py``'s ``MeshAllreduceSession``, so ``execute_s`` is a
true per-round collective time with compile + setup amortized.

Programs are generated at call time from ``count`` / ``nranks`` /
``variant`` / ``dtype`` and written to a temp module (``@pl.jit`` requires
on-disk source and its meta inference only recognises a literal
``dtype=pl.XXX`` in ``pld.window`` calls), so the harness passes an
``EquivalenceCase`` and derives everything from it.

Usage (direct):
    PYTHONPATH=. python collectives/runners/pypto_own.py \
        --count 65536 --devices 0-3 --platform a2a3sim --variant mesh --mode composite

Usage (from harness):
    from collectives.runners.pypto_own import get_pypto_session
    session = get_pypto_session(case, mode="composite")
    try:
        ok, execute_s, err = session.execute()
    finally:
        session.close()

``execute()`` returns ``execute_s``: time for one ``rt.run()`` dispatch
(collective execution only). Compile and worker init are reported once via
``execute_phases()``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from collectives.golden import fill_rank_inputs, verify_outputs

K_MAX_SUPPORTED_RANKS = 16
STAGE_CHUNK = 8192

_DTYPE_TO_TORCH = {"fp32": torch.float32, "fp16": torch.float16}
_DTYPE_BYTES = {"fp32": 4, "fp16": 2}
_DTYPE_ATTR = {"fp32": "FP32", "fp16": "FP16"}
# fp16 reductions accumulate more error; the intrinsic STs use a looser bound.
_RTOL = {"fp32": 1e-3, "fp16": 2e-2}
_ATOL = {"fp32": 1e-3, "fp16": 2e-2}

_PROG_CACHE_DIR = Path(tempfile.gettempdir()) / "pypto_own_programs"

_ACTIVE_SESSION: Any = None
_ACTIVE_SESSION_KEY: tuple[Any, ...] | None = None


def _parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def _stage_params(size: int, dtype_bytes: int) -> tuple[int, int]:
    """Return ``(stage_rows, stage_cols)`` for chunked stage-in/out tiles.

    Mirrors the intrinsic STs: the physical stage tile's row byte size must
    be 32-byte aligned, while ``valid_shape`` carries the real (possibly
    non-aligned) extent.
    """
    if size == 1:
        return (32 // dtype_bytes, 1)
    if dtype_bytes == 2:
        alignment = 32 // dtype_bytes
        return (1, min(STAGE_CHUNK, ((size + alignment - 1) // alignment) * alignment))
    return (1, STAGE_CHUNK)


# ---------------------------------------------------------------------------
# Program source generators — one per (mode, variant). The algorithm selector
# and every shape/dtype are baked into the generated source as literals,
# because @pl.jit requires on-disk source and its local-meta inference only
# recognises ``dtype=pl.XXX`` attribute form in ``pld.window`` calls.
# ---------------------------------------------------------------------------


def _render_stage_in(sz: int, stage_rows: int, stage_cols: int, dattr: str) -> str:
    return (
        f"    for col, (data_iter,) in pl.range(0, {sz}, {stage_cols}, init_values=(data,)):\n"
        f"        valid = pl.min({stage_cols}, {sz} - col)\n"
        f"        local = pl.load(inp, [0, col], [{stage_rows}, {stage_cols}], valid_shape=[1, valid])\n"
        f"        data_iter = pl.store(local, [0, col], data_iter)\n"
        f"        staged_data = pl.yield_(data_iter)\n"
    )


def _render_stage_out(sz: int, stage_rows: int, stage_cols: int, dattr: str) -> str:
    return (
        f"    for col, (out_iter,) in pl.range(0, {sz}, {stage_cols}, init_values=(out,)):\n"
        f"        valid = pl.min({stage_cols}, {sz} - col)\n"
        f"        acc = pl.load(data, [0, col], [{stage_rows}, {stage_cols}], valid_shape=[1, valid])\n"
        f"        out_iter = pl.store(acc, [0, col], out_iter)\n"
        f"        staged_out = pl.yield_(out_iter)\n"
    )


def _render_composite_mesh(count: int, nranks: int, dattr: str, dtype_bytes: int) -> str:
    sz, nr = count, nranks
    stage_rows, stage_cols = _stage_params(sz, dtype_bytes)
    return f"""\
import pypto.language as pl
import pypto.language.distributed as pld

@pl.jit.incore
def reduce_step(
    inp: pl.Tensor[[1, {sz}], pl.{dattr}],
    out: pl.Out[pl.Tensor[[1, {sz}], pl.{dattr}]],
    data: pl.InOut[pld.DistributedTensor[[1, {sz}], pl.{dattr}]],
    signal: pl.InOut[pld.DistributedTensor[[{nr}, 1], pl.INT32]],
) -> pl.Tensor[[1, {sz}], pl.{dattr}]:
{_render_stage_in(sz, stage_rows, stage_cols, dattr)}    data = pld.tensor.allreduce(staged_data, signal, op=pld.ReduceOp.Sum)
{_render_stage_out(sz, stage_rows, stage_cols, dattr)}    return staged_out

@pl.jit
def chip_orch(
    inp: pl.Tensor[[1, {sz}], pl.{dattr}],
    out: pl.Out[pl.Tensor[[1, {sz}], pl.{dattr}]],
    data: pl.InOut[pld.DistributedTensor[[1, {sz}], pl.{dattr}]],
    signal: pl.InOut[pld.DistributedTensor[[{nr}, 1], pl.INT32]],
) -> pl.Tensor[[1, {sz}], pl.{dattr}]:
    return reduce_step(inp, out, data, signal)

@pl.jit.host
def host_orch(
    inputs: pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}],
    outputs: pl.Out[pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}]],
) -> pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}]:
    data_buf = pld.alloc_window_buffer({sz} * pl.{dattr}.get_byte())
    signal_buf = pld.alloc_window_buffer({nr} * pl.INT32.get_byte())
    for r in pl.range({nr}):
        data = pld.window(data_buf, [1, {sz}], dtype=pl.{dattr})
        signal = pld.window(signal_buf, [{nr}, 1], dtype=pl.INT32)
        chip_orch(inputs[r], outputs[r], data, signal, device=r)
    return outputs
"""


def _render_composite_ring(count: int, nranks: int, dattr: str, dtype_bytes: int) -> str:
    sz, nr = count, nranks
    stage_rows, stage_cols = _stage_params(sz, dtype_bytes)
    signal_rows = 2 * (nr - 1)  # one row per ring round, one cell per rank
    return f"""\
import pypto.language as pl
import pypto.language.distributed as pld

@pl.jit.incore
def reduce_step(
    inp: pl.Tensor[[1, {sz}], pl.{dattr}],
    out: pl.Out[pl.Tensor[[1, {sz}], pl.{dattr}]],
    data: pl.InOut[pld.DistributedTensor[[1, {sz}], pl.{dattr}]],
    signal: pl.InOut[pld.DistributedTensor[[{signal_rows}, {nr}], pl.INT32]],
) -> pl.Tensor[[1, {sz}], pl.{dattr}]:
{_render_stage_in(sz, stage_rows, stage_cols, dattr)}    data = pld.tensor.allreduce(staged_data, signal, op=pld.ReduceOp.Sum, mode="ring")
{_render_stage_out(sz, stage_rows, stage_cols, dattr)}    return staged_out

@pl.jit
def chip_orch(
    inp: pl.Tensor[[1, {sz}], pl.{dattr}],
    out: pl.Out[pl.Tensor[[1, {sz}], pl.{dattr}]],
    data: pl.InOut[pld.DistributedTensor[[1, {sz}], pl.{dattr}]],
    signal: pl.InOut[pld.DistributedTensor[[{signal_rows}, {nr}], pl.INT32]],
) -> pl.Tensor[[1, {sz}], pl.{dattr}]:
    return reduce_step(inp, out, data, signal)

@pl.jit.host
def host_orch(
    inputs: pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}],
    outputs: pl.Out[pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}]],
) -> pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}]:
    data_buf = pld.alloc_window_buffer({sz} * pl.{dattr}.get_byte())
    signal_buf = pld.alloc_window_buffer({signal_rows} * {nr} * pl.INT32.get_byte())
    for r in pl.range({nr}):
        data = pld.window(data_buf, [1, {sz}], dtype=pl.{dattr})
        signal = pld.window(signal_buf, [{signal_rows}, {nr}], dtype=pl.INT32)
        chip_orch(inputs[r], outputs[r], data, signal, device=r)
    return outputs
"""


def _render_host_mesh(count: int, nranks: int, dattr: str, dtype_bytes: int) -> str:
    sz, nr = count, nranks
    stage_rows, stage_cols = _stage_params(sz, dtype_bytes)
    return f"""\
import pypto.language as pl
import pypto.language.distributed as pld

@pl.jit.incore
def publish_step(
    inp: pl.Tensor[[1, {sz}], pl.{dattr}],
    data: pl.InOut[pld.DistributedTensor[[1, {sz}], pl.{dattr}]],
) -> pld.DistributedTensor[[1, {sz}], pl.{dattr}]:
{_render_stage_in(sz, stage_rows, stage_cols, dattr)}    return staged_data

@pl.jit
def publish_orch(
    inp: pl.Tensor[[1, {sz}], pl.{dattr}],
    data: pl.InOut[pld.DistributedTensor[[1, {sz}], pl.{dattr}]],
) -> pld.DistributedTensor[[1, {sz}], pl.{dattr}]:
    return publish_step(inp, data)

@pl.jit.incore
def consume_step(
    data: pld.DistributedTensor[[1, {sz}], pl.{dattr}],
    out: pl.Out[pl.Tensor[[1, {sz}], pl.{dattr}]],
) -> pl.Tensor[[1, {sz}], pl.{dattr}]:
{_render_stage_out(sz, stage_rows, stage_cols, dattr)}    return staged_out

@pl.jit
def consume_orch(
    data: pld.DistributedTensor[[1, {sz}], pl.{dattr}],
    out: pl.Out[pl.Tensor[[1, {sz}], pl.{dattr}]],
) -> pl.Tensor[[1, {sz}], pl.{dattr}]:
    return consume_step(data, out)

@pl.jit.host
def host_orch(
    inputs: pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}],
    outputs: pl.Out[pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}]],
) -> pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}]:
    data_buf = pld.alloc_window_buffer({sz} * pl.{dattr}.get_byte())
    signal_buf = pld.alloc_window_buffer({nr} * pl.INT32.get_byte())
    for r in pl.range({nr}):
        data = pld.window(data_buf, [1, {sz}], dtype=pl.{dattr})
        publish_orch(inputs[r], data, device=r)
    data = pld.window(data_buf, [1, {sz}], dtype=pl.{dattr})
    signal = pld.window(signal_buf, [{nr}], dtype=pl.INT32)
    data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
    # The @pl.jit specializer does not model pld.tensor.allreduce as
    # shape-preserving, so re-derive the window view to restore the local
    # metadata the consume dispatch needs.
    data = pld.window(data_buf, [1, {sz}], dtype=pl.{dattr})
    for r in pl.range({nr}):
        consume_orch(data, outputs[r], device=r)
    return outputs
"""


def _render_host_ring(count: int, nranks: int, dattr: str, dtype_bytes: int) -> str:
    sz, nr = count, nranks
    stage_rows, stage_cols = _stage_params(sz, dtype_bytes)
    signal_rows = 2 * (nr - 1) + 1  # one extra row for the return barrier
    return f"""\
import pypto.language as pl
import pypto.language.distributed as pld

@pl.jit.incore
def publish_step(
    inp: pl.Tensor[[1, {sz}], pl.{dattr}],
    data: pl.InOut[pld.DistributedTensor[[1, {sz}], pl.{dattr}]],
) -> pld.DistributedTensor[[1, {sz}], pl.{dattr}]:
{_render_stage_in(sz, stage_rows, stage_cols, dattr)}    return staged_data

@pl.jit
def publish_orch(
    inp: pl.Tensor[[1, {sz}], pl.{dattr}],
    data: pl.InOut[pld.DistributedTensor[[1, {sz}], pl.{dattr}]],
) -> pld.DistributedTensor[[1, {sz}], pl.{dattr}]:
    return publish_step(inp, data)

@pl.jit.incore
def consume_step(
    data: pld.DistributedTensor[[1, {sz}], pl.{dattr}],
    out: pl.Out[pl.Tensor[[1, {sz}], pl.{dattr}]],
) -> pl.Tensor[[1, {sz}], pl.{dattr}]:
{_render_stage_out(sz, stage_rows, stage_cols, dattr)}    return staged_out

@pl.jit
def consume_orch(
    data: pld.DistributedTensor[[1, {sz}], pl.{dattr}],
    out: pl.Out[pl.Tensor[[1, {sz}], pl.{dattr}]],
) -> pl.Tensor[[1, {sz}], pl.{dattr}]:
    return consume_step(data, out)

@pl.jit.host
def host_orch(
    inputs: pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}],
    outputs: pl.Out[pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}]],
) -> pl.Tensor[[{nr}, 1, {sz}], pl.{dattr}]:
    data_buf = pld.alloc_window_buffer({sz} * pl.{dattr}.get_byte())
    signal_buf = pld.alloc_window_buffer({signal_rows} * {nr} * pl.INT32.get_byte())
    for r in pl.range({nr}):
        data = pld.window(data_buf, [1, {sz}], dtype=pl.{dattr})
        publish_orch(inputs[r], data, device=r)
    data = pld.window(data_buf, [1, {sz}], dtype=pl.{dattr})
    signal = pld.window(signal_buf, [{signal_rows}, {nr}], dtype=pl.INT32)
    data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
    data = pld.window(data_buf, [1, {sz}], dtype=pl.{dattr})
    for r in pl.range({nr}):
        consume_orch(data, outputs[r], device=r)
    return outputs
"""


def _render_program(mode: str, variant: str, count: int, nranks: int, dtype: str) -> str:
    dattr = _DTYPE_ATTR[dtype]
    dtype_bytes = _DTYPE_BYTES[dtype]
    if mode == "composite":
        if variant == "ring":
            return _render_composite_ring(count, nranks, dattr, dtype_bytes)
        return _render_composite_mesh(count, nranks, dattr, dtype_bytes)
    if mode == "host":
        if variant == "ring":
            return _render_host_ring(count, nranks, dattr, dtype_bytes)
        return _render_host_mesh(count, nranks, dattr, dtype_bytes)
    raise ValueError(f"unknown pypto runner mode: {mode}")


def _load_program_module(mode: str, variant: str, count: int, nranks: int, dtype: str) -> Any:
    """Materialise the generated program module on disk and import it.

    ``@pl.jit`` requires the decorated functions to live in an importable
    module on disk (``inspect.getsource``); the temp file is written once per
    (mode, variant, count, nranks, dtype) and reused.
    """
    src = _render_program(mode, variant, count, nranks, dtype)
    key = f"{mode}_{variant}_{count}_{nranks}_{dtype}"
    mod_name = f"pypto_own_prog_{key}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    _PROG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _PROG_CACHE_DIR / f"{key}.py"
    if not path.is_file() or path.read_text(encoding="utf-8") != src:
        path.write_text(src, encoding="utf-8")

    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _flatten_compile_profile(profile: dict) -> dict[str, float] | None:
    """Flatten a pypto ``CompileProfiler.to_dict()`` into ``{total, passes, codegen, other}``.

    Top-level stages are ``parse``, ``passes`` and ``codegen``; anything not
    attributed to passes/codegen (parse, residual overhead) is lumped into
    ``other``. Returns ``None`` when profiling produced no data.
    """
    if not profile:
        return None
    total = float(profile.get("total_seconds", 0.0))
    top = {
        s.get("name", "?"): float(s.get("seconds", 0.0))
        for s in profile.get("stages", [])
    }
    passes = top.get("passes", 0.0)
    codegen = top.get("codegen", 0.0)
    other = max(0.0, total - passes - codegen)
    return {
        "total": round(total, 6),
        "passes": round(passes, 6),
        "codegen": round(codegen, 6),
        "other": round(other, 6),
    }


_DEVICE_WALL_RE = re.compile(r"device_wall[^\n]*?dur=(\d+)")


def _parse_device_wall_s(text: str) -> float | None:
    """Extract the slowest-rank ``device_wall`` span duration (seconds).

    The runtime emits one ``[STRACE] ... name=simpler_run.runner_run.device_wall
    ... dur=<ns> clk=dev`` span per rank; the slowest rank is the collective
    completion. Returns ``None`` when the runtime did not emit the span.
    """
    durs = [int(m) for m in _DEVICE_WALL_RE.findall(text)]
    if not durs:
        return None
    return max(durs) / 1e9


class PyptoCollectiveSession:
    """One compile + one worker init; call execute() many times before close()."""

    def __init__(
        self,
        case: Any,
        mode: str = "composite",
        pto_isa_commit: str | None = None,
        dfx_dir: str | None = None,
    ) -> None:
        from pypto.ir.distributed_compiled_program import DistributedConfig
        from pypto.runtime import RunConfig

        self.case = case
        self.mode = mode
        self.count = case.count
        self.devices = case.device_ids
        self.nranks = case.p
        self.platform = case.platform
        self.variant = case.variant
        self.dtype = case.dtype
        self.pto_isa_commit = pto_isa_commit

        if self.nranks < 2 or self.nranks > K_MAX_SUPPORTED_RANKS:
            raise ValueError(f"nranks must be in [2, {K_MAX_SUPPORTED_RANKS}], got {self.nranks}")
        if self.count <= 0:
            raise ValueError(f"count must be positive, got {self.count}")
        if self.variant not in ("mesh", "ring"):
            raise ValueError(f"pypto runner supports mesh/ring, got variant={self.variant!r}")
        if self.dtype not in _DTYPE_TO_TORCH:
            raise ValueError(f"unsupported dtype {self.dtype!r}; expected fp32 or fp16")
        if self.variant == "ring" and self.mode == "host" and self.dtype != "fp32":
            raise ValueError("pypto-host ring supports fp32 only")

        torch_dtype = _DTYPE_TO_TORCH[self.dtype]

        # Shared-memory IO buffers must exist BEFORE prepare() forks the chip
        # worker, so the child sees them through the inherited mapping.
        self.inputs = (
            torch.tensor(fill_rank_inputs(case), dtype=torch_dtype)
            .reshape(self.nranks, 1, self.count)
            .share_memory_()
        )
        self.outputs = torch.zeros(
            (self.nranks, 1, self.count), dtype=torch_dtype
        ).share_memory_()

        # Compile once — specialize on sample shapes/dtypes, no dispatch.
        # A thread-local CompileProfiler records per-stage wall clocks
        # (parse / passes / codegen) with negligible overhead; the flattened
        # dict feeds the compile_breakdown figure.
        t0 = time.perf_counter()
        module = _load_program_module(
            self.mode, self.variant, self.count, self.nranks, self.dtype
        )
        dc = DistributedConfig(device_ids=self.devices, num_sub_workers=0)
        if dfx_dir is not None:
            cfg = RunConfig(
                platform=self.platform,
                distributed_config=dc,
                save_kernels_dir=dfx_dir,
            )
        else:
            cfg = RunConfig(platform=self.platform, distributed_config=dc)
        sample_in = torch.zeros((self.nranks, 1, self.count), dtype=torch_dtype)
        sample_out = torch.zeros((self.nranks, 1, self.count), dtype=torch_dtype)

        from pypto.compile_profiling import CompileProfiler  # noqa: PLC0415

        self.compile_profile: dict[str, float] | None = None
        with CompileProfiler() as prof:
            self._compiled = module.host_orch.compile(sample_in, sample_out, config=cfg)
        self.compile_s = time.perf_counter() - t0
        self.compile_profile = _flatten_compile_profile(prof.to_dict())
        self.last_device_wall_s: float | None = None

        # Redirect stderr at the fd level for the worker's lifetime BEFORE
        # prepare(): the chip workers are forked there and inherit fd 2, so a
        # redirect set up after the fork would miss their [STRACE] markers
        # (pypto/runtime/bench.py documents the same requirement for L3). Only
        # runtime stderr is diverted — harness prints go to stdout. Restored in
        # close(); the buffered text is drained per-round by execute().
        self._strace_fh = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._strace_offset = 0
        self._saved_err_fd = os.dup(2)
        os.dup2(self._strace_fh.fileno(), 2)
        try:
            # Prepare the reusable dispatch handle (worker init + registration).
            t1 = time.perf_counter()
            self._rt = self._compiled.prepare()
            self.init_s = time.perf_counter() - t1
        except BaseException:
            self._restore_stderr()
            raise

    def _restore_stderr(self) -> None:
        """Restore fd 2 to its pre-session target; idempotent."""
        if getattr(self, "_saved_err_fd", None) is not None:
            os.dup2(self._saved_err_fd, 2)
            os.close(self._saved_err_fd)
            self._saved_err_fd = None

    def _drain_device_wall(self) -> float | None:
        """Read new ``[STRACE]`` spans since the last drain; return slowest-rank
        ``device_wall`` (seconds) or None when the runtime did not emit one."""
        self._strace_fh.flush()
        self._strace_fh.seek(self._strace_offset)
        text = self._strace_fh.read()
        self._strace_offset = self._strace_fh.tell()
        return _parse_device_wall_s(text)

    def execute(self, config: Any | None = None) -> tuple[bool, float, str]:
        """Run one collective round; returns (ok, execute_s, error).

        ``execute_s`` is the wall time of ``rt.run()`` (host dispatch + device
        collective). The pure on-device collective time — the slowest-rank
        ``device_wall`` STRACE span, when the runtime emits one — is recorded
        on ``self.last_device_wall_s`` (``None`` when unavailable).
        """
        t0 = time.perf_counter()
        self.outputs.zero_()
        self._rt.run(self._compiled, self.inputs, self.outputs, config=config)
        wall = time.perf_counter() - t0
        self.last_device_wall_s = self._drain_device_wall()

        per_rank = [row[0] for row in self.outputs.tolist()]
        ok, msg = verify_outputs(
            self.case, per_rank, rtol=_RTOL[self.dtype], atol=_ATOL[self.dtype]
        )
        return ok, wall, "" if ok else msg

    def execute_batch(self, config: Any | None = None, n: int = 10) -> tuple[bool, float, float | None, str]:
        """Run ``n`` back-to-back rounds; returns (ok, mean_execute_s, mean_device_wall_s, error).

        Amortises per-dispatch host overhead across rounds — a second view of
        the collective cost alongside single-round ``execute()``.
        """
        total = 0.0
        dev_total = 0.0
        dev_count = 0
        for _ in range(n):
            ok, execute_s, err = self.execute(config=config)
            if not ok:
                return False, execute_s, self.last_device_wall_s, err
            total += execute_s
            if self.last_device_wall_s is not None:
                dev_total += self.last_device_wall_s
                dev_count += 1
        mean_dev = dev_total / dev_count if dev_count else None
        return True, total / n, mean_dev, ""

    def execute_phases(self, wall: float) -> dict[str, float]:
        """Phase breakdown: compile + init once, execute per round."""
        return {
            "compile": self.compile_s,
            "init": self.init_s,
            "execute": float(wall),
        }

    def close(self) -> None:
        self._restore_stderr()
        if getattr(self, "_strace_fh", None) is not None:
            self._strace_fh.close()
            self._strace_fh = None
        if getattr(self, "_rt", None) is not None:
            self._rt.close()
            self._rt = None


def _session_key(case: Any, mode: str, dfx_dir: str | None) -> tuple[Any, ...]:
    return (
        mode,
        case.variant,
        case.count,
        tuple(case.device_ids),
        case.platform,
        case.dtype,
        dfx_dir,
    )


def get_pypto_session(
    case: Any,
    mode: str = "composite",
    dfx_dir: str | None = None,
) -> PyptoCollectiveSession:
    """Return a cached session for (case, mode, dfx_dir); rebuilds on key change."""
    global _ACTIVE_SESSION, _ACTIVE_SESSION_KEY
    key = _session_key(case, mode, dfx_dir)
    if _ACTIVE_SESSION is not None and _ACTIVE_SESSION_KEY == key:
        return _ACTIVE_SESSION
    close_pypto_session()
    _ACTIVE_SESSION = PyptoCollectiveSession(case, mode=mode, dfx_dir=dfx_dir)
    _ACTIVE_SESSION_KEY = key
    return _ACTIVE_SESSION


def close_pypto_session() -> None:
    global _ACTIVE_SESSION, _ACTIVE_SESSION_KEY
    if _ACTIVE_SESSION is not None:
        try:
            _ACTIVE_SESSION.close()
        finally:
            _ACTIVE_SESSION = None
            _ACTIVE_SESSION_KEY = None


def main(argv: list[str] | None = None) -> int:
    from collectives.equivalence import EquivalenceCase

    parser = argparse.ArgumentParser(description="pypto in-process collective benchmark")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--devices", default="0-1", help="Device range, e.g. 0-3")
    parser.add_argument("--platform", default="a2a3sim")
    parser.add_argument("--variant", choices=("mesh", "ring"), default="mesh")
    parser.add_argument("--mode", choices=("composite", "host"), default="composite")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1, help="Back-to-back rt.run per round (amortises dispatch)")
    args = parser.parse_args(argv)

    case = EquivalenceCase(
        variant=args.variant,
        p=len(_parse_device_range(args.devices)),
        count=args.count,
        dtype=args.dtype,
        device_ids=_parse_device_range(args.devices),
        platform=args.platform,
        warmup_rounds=1,
        timed_rounds=args.rounds,
    )
    session = get_pypto_session(case, mode=args.mode)
    try:
        for r in range(args.rounds):
            if args.batch > 1:
                ok, execute_s, dev_s, err = session.execute_batch(config=None, n=args.batch)
            else:
                ok, execute_s, err = session.execute()
                dev_s = session.last_device_wall_s
            phases = session.execute_phases(execute_s)
            print(
                f"round {r + 1}: ok={ok} execute={execute_s:.6f}s "
                f"device_wall={dev_s} phases={phases} {err}"
            )
            if not ok:
                return 1
    finally:
        close_pypto_session()
    print("PYPTO_ALLREDUCE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
