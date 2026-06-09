---
name: pypto-vec-tile-row-alignment
description: >-
  Diagnose PyPTO / PTOAS compile failures from Vec tile 32-byte row alignment
  (cols * sizeof(dtype) % 32 == 0). Use when ST or codegen fails on small INT32
  tiles, pl.tile.full([1,1]), alloc_tile, pto.tstore/texpands, distributed comm
  window anchors, or messages about 32-byte row alignment on NPU.
---

# PyPTO NPU Debugging Skills

## Quick diagnostic script

Run `diagnose_npu.py` (repo root) inside any container to get a 10-point
hardware health check in <30 seconds:

```bash
docker cp ~/pypto-tooling/diagnose_npu.py <container>:/opt/pypto/
# Inside container:
python3 /opt/pypto/diagnose_npu.py
```

---

# NPU Device Diagnostics

## NPU error code reference

| Code | Name | Meaning | Typical cause |
|------|------|---------|---------------|
| 507018 | `ACL_ERROR_RT_DEVICE_NOT_EXIST` | Device not found | Wrong `--device`, missing `/dev/davinci*` |
| **507033** | `ACL_ERROR_RT_DEV_SETUP_ERROR` | Device visible but can't open context | Dead device (dmesg: `not working`), CANN/driver mismatch, permissions |
| 507899 | `ACL_ERROR_RT_INTERNAL_ERROR` | Driver internal error (IPC) | CANN 9.0.0 IPC key format rejected by driver <26.0.rc1 |

## Dead vs. busy device (507033)

507033 can mean **dead hardware** or **contention**. Tell them apart:

| Signal | Dead device | Busy device |
|--------|-------------|-------------|
| `dmesg` | `device(N) is not working`, `state=5`, `ret(-6)` | Usually silent |
| `fuser /dev/davinci*` | Empty | Shows PID(s) |
| Other devices | Work fine | Work fine |
| `npu-smi info` | Hangs or shows `NA` for that chip | Shows process list |
| Kernel ret | `-6` (ENXIO) | `-16` (EBUSY) |

**Dead device dmesg signatures:**

```text
[devdrv] [ERROR] device(0) is not working.
[devdrv] [ERROR] devdrv_manager_get_core failed, ret(-6), dev_id(0).
[ascend_udis] [ERROR] udis_check_ucb 80] udis device state is not ready. (udevid=0; state=5)
[ascend_udis] [ERROR] udis_get_device_info 302] Get udis info failed. (udevid=0; ...)
```

`state=5` = hardware management state "unavailable." Not software-fixable — needs
physical reset or replacement.

## Interpreting aclrtSetDevice failures systematically

```python
import ctypes
lib = ctypes.cdll.LoadLibrary("libascendcl.so")
lib.aclInit(None)
cnt = ctypes.c_uint(0)
lib.aclrtGetDeviceCount(ctypes.byref(cnt))  # → count
for dev in range(cnt.value):
    rc = lib.aclrtSetDevice(dev)
    print(f"device {dev}: {rc}")
```

- All devices fail → driver/CANN mismatch or kernel module not loaded
- Only some devices fail → dead hardware
- Fails with LD_PRELOAD=libhccl.so but works without → HCCL preload interference
- Fails with LD_PRELOAD but `fuser` shows PIDs → contention

## Host env leakage into containers

If `LD_LIBRARY_PATH` inside the container contains host paths
(`/usr/local/Ascend/nnal/...`, `/usr/local/python3.12.13/...`), the host
Docker daemon or a systemd environment is injecting them. The Dockerfile
sets a clean `LD_LIBRARY_PATH`; the leakage happens at `docker run` time.

**Impact:** usually harmless for this project (CANN libs take priority),
but can cause silent symbol conflicts if host and container CANN versions
differ.

**Fix:** add `-e LD_LIBRARY_PATH=` to `docker run` to reset, or fix the
host's Docker daemon environment.

## Docker runtime: HCCL, mounts, and `--pid=host`

### Why `--pid=host` is required for distributed/HCCL

simpler's Path-D symmetric-pool setup calls `aclrtIpcMemSetImportPid(myName,
peerPids)` with peerPids harvested via `getpid()` in each rank. Inside a child
PID namespace, `getpid()` returns the *container* PID, but the Ascend devmm
kernel module verifies the *host* PID against the recorded whitelist. Mismatch
→ `aclrtIpcMemImportByKey` returns **507899** and dmesg shows:

```text
[ascend] [devmm] [ERROR] _devmm_ipc_node_open: Wlist verify fail
```

`--pid=host` shares the host PID namespace so `getpid()` == kernel-visible PID.
simpler's own CI runs directly on bare metal, never in Docker, so this is
container-specific.

### `LD_PRELOAD=libhccl.so` — required for HCCL, must NOT be set image-wide

simpler's `host_runtime.so` has **WEAK** undefined references to `HcclGetRootInfo`,
`HcclCommInitRootInfo`, `HcclBarrier`, and `HcclCommDestroy` — but does NOT list
`libhccl.so` in `DT_NEEDED`. Without a global preload, those relocations resolve
to NULL → **SIGSEGV** on `comm_init`. `LD_LIBRARY_PATH` alone is not enough; the
resolution decision happens at `host_runtime.so` load time, before any HCCL call.

**Why it is NOT set image-wide (Docker ENV or bashrc):** setting it globally
injects `libhccl.so` into every process in the container, including VS Code
server's node process → **hang on attach** (see "VS Code attach hang" below).

Set it manually in the shell before running HCCL tests:

```bash
export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so
```

### VS Code "Attach to Running Container" hangs

**Symptom:** attach hangs indefinitely; Dev Containers log shows
`userEnvProbe: loginInteractiveShell` stalled or `taking longer than 10 seconds`.

**Root causes:**

| Cause | Effect |
|-------|--------|
| Ascend base image injects `set_env.sh` into `/etc/profile`, `/etc/bash.bashrc`, `/etc/profile.d/*.sh`, `/root/.bashrc`, etc. | `set_env.sh` takes 10–30s; `userEnvProbe` shell probe times out |
| `LD_PRELOAD=libhccl.so` set as Docker `ENV` or in `bashrc` | VS Code probe captures it and injects into server node process → loads libhccl.so → hang |

**Fix in Dockerfile (both required):**

1. Strip `set_env.sh` from ALL base-image startup files:

```dockerfile
RUN for f in /etc/profile /etc/bash.bashrc /root/.profile /root/.bashrc /root/.bash_profile; do \
      [ -f "$f" ] && sed -i '/set_env\.sh/d' "$f" || true; \
    done && \
    for f in /etc/profile.d/*.sh; do \
      [ -f "$f" ] && sed -i '/set_env\.sh/d' "$f" || true; \
    done
```

2. Do NOT set `LD_PRELOAD` as `ENV` or in `bashrc`. All CANN env vars VS Code
   needs are set via `ENV` instructions; `LD_PRELOAD` is set manually before HCCL
   tests only.

**Verification:** after rebuild, attach completes in <5 seconds with no
`userEnvProbe` warning in the Dev Containers log.

### Never mount `/usr/local/Ascend` into the container

The image has CANN baked in at `/usr/local/Ascend/cann-9.0.0/`. Mounting the
host's `/usr/local/Ascend` shadows the baked-in CANN with the host's (potentially
older or incompatible) version. Only mount the driver:

```bash
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro   # ✅
-v /usr/local/Ascend:/usr/local/Ascend:ro                  # ❌
```

## Driver/CANN compatibility quick check

```bash
# On host:
cat /usr/local/Ascend/driver/version.info 2>/dev/null || \
  modinfo davinci 2>/dev/null | grep ^version

# In container:
cat /usr/local/Ascend/cann-9.0.0/version.cfg 2>/dev/null
```

Per `simpler/docs/install.md`: CANN 9.0.0 requires driver ≥ 26.0.rc1.
Driver 25.5.1 with CANN 9.0.0 → 507033 (device setup) and 507899 (HCCL IPC).

---

# PyPTO Vec tile 32-byte row alignment

## Rule (hardware / PTO-ISA)

For **Vec**, **row-major**, **`none_box`** tiles (typical `pl.tile.full` / `pl.load` /
`pl.store` on UB):

```text
major_dim_bytes = cols * sizeof(T)   # row-major ND
major_dim_bytes % 32 == 0
```

Minimum **static cols** by dtype:

| dtype | sizeof | min cols |
|-------|--------|----------|
| INT32 / FP32 | 4 | **8** |
| FP16 / BF16 | 2 | **16** |
| INT8 | 1 | **32** |

`[1, 1]` INT32 → 4 bytes → **illegal** for Vec DMA/tile ops that assume 32-byte bursts.

**Do not confuse** with `AllocateMemoryAddr` pass: that aligns **UB addresses** to 32 B;
this skill is about **tile row stride** (logical shape), which is independent.

---

## When enforcement appears

| Layer | What fails | Notes |
|-------|------------|-------|
| **pto-isa** | C++ `static_assert` in generated kernel | e.g. A5 `TStore.hpp`: `Cols * sizeof(T) % 32 == 0` |
| **PTOAS** | MLIR `verify()` on ops | Large verifier pass **2026-03-16** (`4051849` in PTOAS). Only explicit `% 32` string in v0.40 binary is often **transpose (A5)**; store/expand failures may surface as other op errors or kernel compile |
| **PyPTO CI** | ST / codegen | **ptoas v0.40** pinned from **2026-05-20** (`pypto` #1417). Older ptoas may not catch the same cases |
| **PyPTO tests** | Documented shapes | `test_scatter_update` (INT32 `cols >= 8`), `test_gather`, `test_l3_notify_wait` |

The rule is **not** a recent PyPTO DSL change; new code paths (e.g. `pl.tile.full`) **expose** it.

---

## Symptom → likely cause

| Symptom | Likely cause |
|---------|----------------|
| GEMM / distributed ST fails after `pl.tile.full([1, 1], INT32)` + `pl.store` | 4-byte row; pad tile or use scalar read/write |
| Error mentions `alloc_tile` / `1×1` / INT32 | Usually the **tile type** on a downstream `pto.texpands` / `pto.tstore`, not `alloc_tile` verify itself |
| AllGather/reduce work; only new anchor InCore fails | New Vec tile shape, not HCCL |
| `test_l3_notify_wait` pattern | Project already avoids `pl.load([1,1])` INT32 — uses `pl.read` / `pl.write` |

---

## Investigation workflow

1. **Capture the first real error** (not a paraphrase):
   - ptoas stderr: `ptoas … -o …` or pytest compile log
   - or Ascend/C++ `static_assert` / `PTO_ASSERT` in generated `kernels/*.cpp`
2. **Find the offending tile shape** in IR or DSL:
   - `pl.tile.full([rows, cols], dtype=…)`
   - `pl.load(…, shapes=[…])` on Vec
   - Generated `.pto`: `!pto.tile_buf<loc=vec, …, rows=…, cols=…>`
3. **Check**: `cols * element_bytes % 32 == 0` (row-major Vec).
4. **Check toolchain versions**:
   - `pypto/.github/workflows/ci.yml` → `PTOAS_VERSION`, `PTO_ISA` commit
   - On atlas: pip-installed `pypto` vs stale tree; **do not** put `…/pypto/python` first on `PYTHONPATH` if testing installed wheel
5. **If unsure which layer failed**:
   ```bash
   strings "$(which ptoas 2>/dev/null || echo ptoas-bin/bin/ptoas)" | rg -i "32-byte|major dimension"
   ```
   Compare with PTOAS `lib/PTO/IR/PTO.cpp` / pto-isa headers for the op in the backtrace.

---

## Fixes (pick one)

### A. Pad the Vec tile (store/load path)

Use the smallest legal width; only use valid region:

```python
COMM_ANCHOR_COLS = 8  # INT32: 8 * 4 = 32 bytes

tile = pl.tile.full([1, COMM_ANCHOR_COLS], dtype=pl.INT32, value=0)
pl.store(tile, [0, 0], scratch)  # logical [1,1] window still OK
```

Reference: `pypto/tests/st/distributed/test_l3_gemm.py`.

### B. Scalar GM access (1×1 signal cells)

Avoid Vec tiles for single INT32 comm slots:

```python
val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
pl.write(out, [0, 0], val)
```

Reference: `pypto/tests/st/distributed/test_l3_notify_wait.py` (comment explains why).

### C. Widen index / scratch tiles in tests

- INT32 index tiles: `cols >= 8` (`test_scatter_update.py`)
- FP16: `cols >= 16`

---

## Distributed ST checklist

- [ ] Dummy comm window + `CollectCommGroups`: anchor must **mutate** window (`InOut`); padded tile or scalar path
- [ ] Window tensor can stay `[1, 1]` in **type** while tile is `[1, 8]` for the Vec op
- [ ] Docker: `--pid=host` for HCCL; `source …/set_env.sh`; `PYTHONPATH` = runtime + examples only when using pip `pypto`
- [ ] After branch pull: `pip install --no-build-isolation -v ".[dev]"` in `/opt/pypto` if compile errors look like missing passes/APIs

---

## Related timeline (for “when did this change?”)

| Date | Event |
|------|--------|
| 2025-12+ | pto-isa A5 `TStore` static_asserts (hardware rule in headers) |
| 2026-03-16 | PTOAS `4051849` — op verifiers aligned with pto-isa |
| 2026-04-27+ | PyPTO gather/scatter tests document `cols * sizeof % 32` |
| 2026-05-06 | `pl.tile.full` (#1274) — easy tiny tiles |
| 2026-05-20 | PyPTO CI ptoas **v0.36 → v0.40** (#1417) |
| 2026-05-26 | `test_l3_notify_wait` — scalar workaround documented |
| 2026-05-28 | GEMM comm anchor `[1,1]` → fix `[1,8]` |

---

## References in repo

| Topic | Location |
|-------|----------|
| INT32 min cols | `pypto/tests/st/runtime/ops/test_scatter_update.py` |
| Gather alignment note | `pypto/tests/st/runtime/ops/test_gather.py` |
| Notify/wait scalar pattern | `pypto/tests/st/distributed/test_l3_notify_wait.py` |
| GEMM anchor fix | `pypto/tests/st/distributed/test_l3_gemm.py` |
| Scatter lowering comments | `pypto/src/ir/transforms/op_conversion_registry.cpp` |
| i32 index tile policy | `pypto/docs/en/dev/passes/12-convert_tensor_to_tile_ops.md` |
| PTOAS manual (transpose) | `PTOAS/docs/PTO_IR_manual.md` (~32-byte major dim) |
| A5 TStore assert | `pto-isa/include/pto/npu/a5/TStore.hpp` |
| CI ptoas pin | `pypto/.github/workflows/ci.yml` → `PTOAS_VERSION` |

---

## Anti-patterns

- Assuming **`[1, 1]` window type** implies a legal **`[1, 1]` Vec tile**
- Blaming HCCL / distributed runtime when failure is at **compile / ptoas / kernel build**
- Using **`pl.load([1, 1], INT32)`** for comm signals when **`pl.read` / `pl.write`** exist
- Interpreting agent summaries as exact ptoas text — always read **raw** `error:` line

---

# Compile-time pypto distributed bugs (DSL / IR passes)

These bugs do not surface as Docker, driver, or HCCL errors. They surface as
**silent semantic failures** — a multi-rank test passes only on rank 0, or a
factory-built program compiles to the wrong shape with no diagnostic. The
triage flow below is the one that worked in the L3 GEMM bring-up
(`feat/l3-allreduce-gemm`).

## Triage order for silent multi-rank correctness failures

When a multi-rank L3 test produces wrong output but no diagnostic:

1. **Dump the frontend IR first**, before suspecting passes, codegen, runtime,
   or the simpler dispatcher. Use `dump_passes=True` on `ir.compile(...)` and
   read `00_frontend.py` (the AST parser's output). If a statement is already
   missing there, the bug is in the parser or the `@pl.program` decorator —
   not in any pass.
2. **Inspect the generated `orchestration/host_orch.py`** for the number of
   `orch.submit_next_level(...)` calls. If a multi-rank host_orch emitted only
   one call, the `CollectCommGroups` pass refused to form a `CommGroup` (see
   the recipe below).
3. Only after the IR and the orchestration look correct, treat it as a runtime
   issue and use the `debugging_skills/SKILL.md` NPU diagnostics or
   `dockerfile_skills/SKILL.md` runtime tips.

## 1. Multi-rank dispatch needs a comm window (`CollectCommGroups` recipe)

**Symptom.** In a multi-rank L3 program, only rank 0's output is correct;
other ranks' output buffers remain at their pre-call values (typically all
zeros). No error, no warning.

**Root cause.**
`pypto/src/ir/transforms/collect_comm_groups_pass.cpp` forms a `CommGroup`
only when the host orchestrator contains at least one
`pld.alloc_window_buffer` paired with a `pld.window` view that is passed
positionally to a dispatched `chip_orch` under a recognised `device=` form
(`ConstInt` or the induction var of `pl.range(pld.world_size())` /
`pl.range(<int>)`). Without that, adjacent `self.chip_orch(...)` dispatches
collapse into a single `orch.submit_next_level` to worker 0 in the generated
`orchestration/host_orch.py` and only rank 0 runs.

**Fix recipe for kernels with no real collectives.** Carry an unused 4-byte
INT32 scratch window through `chip_orch` solely to engage the pass.

```python
@pl.function(type=pl.FunctionType.Orchestration)
def chip_orch(
    self,
    a_shard: pl.Tensor[[m0, k], pl.FP32],
    b: pl.Tensor[[k, n], pl.FP32],
    c_shard: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
    _scratch: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
) -> pl.Tensor[[m0, n], pl.FP32]:
    return self.gemm(a_shard, b, c_shard)

@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
def host_orch(
    self,
    a: pl.Tensor[a_shape, pl.FP32],
    b: pl.Tensor[[k, n], pl.FP32],
    c: pl.Out[pl.Tensor[c_shape, pl.FP32]],
) -> pl.Tensor[c_shape, pl.FP32]:
    scratch_buf = pld.alloc_window_buffer(4)  # 1x1 INT32
    for r in pl.range(pld.world_size()):
        scratch = pld.window(scratch_buf, [1, 1], dtype=pl.INT32)
        self.chip_orch(a[r], b, c[r], scratch, device=r)
    return c
```

**Reference.** First documented in `tests/st/distributed/l3_gemm_programs.py`
("Multi-rank dispatch recipe" section of the module docstring).

## 2. Avoid `pl.range(1)` in the P=1 branch (Simplify single-trip unroll)

**Symptom.** A P=1 program built from a factory that uses
`for r in pl.range(pld.world_size())` fails to codegen with an unresolved
induction-variable reference on the lowered `device=` argument.

**Root cause.** The Simplify pass single-trip-unrolls `for r in pl.range(1)`
but leaves `device=r` on the lowered call. Subsequent lower passes do not
resolve `r` to a constant after unroll.

**Fix.** In the P=1 branch, emit a direct `self.chip_orch(..., device=0)`
call rather than a loop.

```python
if nranks == 1:
    @pl.program
    class FooP1:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, ...):
            self.chip_orch(a[0], b, c[0], device=0)
            return c
    return FooP1
```

## 3. `@pl.program` class-name collision in factory `if/else` branches

**Symptom.** A factory `build_foo(*, nranks)` that returns different program
class shapes per branch silently compiles the wrong shape.

**Root cause.** `python/pypto/language/parser/decorator.py` recovers source via
`inspect.getsourcelines(cls)` and uses `ast.walk` to find `ast.ClassDef`
matching `cls.__name__`. Two `class Foo:` across `if/else` → `ast.walk` returns
the first one.

**Fix.** Give each branch's class a distinct name (`FooP1` vs `FooPN`).

## 4. Per-iteration window rebind inside the dispatch loop

**Symptom.** Multi-rank dispatch with `scratch = pld.window(buf, ...)` bound
**above** the loop emits only one dispatch.

**Fix.** Re-emit `pld.window(buf, ...)` inside the loop body:

```python
for r in pl.range(pld.world_size()):
    scratch = pld.window(scratch_buf, [1, 1], dtype=pl.INT32)  # rebind per iter
    self.chip_orch(a[r], b, c[r], scratch, device=r)
```

## 5. Diagnose parser-level statement drops with frontend IR

**Symptom.** Fewer dispatches in `orchestration/host_orch.py` than DSL source.

**Diagnostic.** Dump frontend IR with `dump_passes=True`:

```python
compiled = ir.compile(
    program, platform="a2a3",
    distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    dump_passes=True, dump_dir="/tmp/pypto_dump",
)
# read /tmp/pypto_dump/<program>/00_frontend.py
```

If missing in `00_frontend.py` → parser/decorator bug. If present there but
missing later → pass bug; bisect numbered files.

## Tensor-argument direction → simpler TensorMap behaviour

| pypto direction       | simpler `TensorArgType` | TensorMap behaviour          |
|-----------------------|--------------------------|------------------------------|
| `pl.Tensor[...]`      | `INPUT`                  | Lookup only                  |
| `pl.Out[pl.Tensor]`   | `OUTPUT_EXISTING`        | Insert only (no lookup)      |
| `pl.InOut[pl.Tensor]` | `INOUT`                  | Lookup + insert              |

Use `Out` when buffer's pre-call contents are irrelevant. Use `InOut` when a
downstream `chip_orch` in the same `host_orch` needs the buffer as input.
