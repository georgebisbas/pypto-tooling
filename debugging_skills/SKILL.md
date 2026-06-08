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
