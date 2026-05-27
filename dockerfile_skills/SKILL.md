# PyPTO Dockerfile construction and debugging

## Anatomy of a correct standalone Dockerfile (zero host deps)

A standalone image requires nothing from the host except the Ascend kernel
driver. Everything else — CANN, source repos, toolchains — is cloned and
built inside the image. No build context; use stdin:

```bash
docker build -t <tag> - < Dockerfile.name
```

### Mandatory patterns

#### 1. ARG-driven paths — every path derives from two ARGs

```dockerfile
ARG CANN_VERSION=9.0.0
ARG INSTALL_PREFIX=/opt

ENV CANN_HOME=/usr/local/Ascend/cann-${CANN_VERSION}
ENV PYPTO_DIR=${INSTALL_PREFIX}/pypto \
    PTO_ISA_DIR=${INSTALL_PREFIX}/pto-isa
```

Never hardcode a path that could vary across CANN versions or install
prefixes. Use `${ENV_VAR}` in RUN instructions so they expand at build time.

#### 2. ARG scope resets after FROM — re-declare after every FROM

```dockerfile
ARG CANN_VERSION=9.0.0
FROM quay.io/ascend/cann:${CANN_VERSION}-910b-ubuntu22.04-py3.12
ARG CANN_VERSION=9.0.0          # ← re-declare
ARG INSTALL_PREFIX=/opt
```

#### 3. Self-referencing ENV — split across multiple ENV instructions

Docker does NOT expand `${VAR}` references within a single ENV block if
`VAR` is defined in that same block. Split into separate ENV instructions:

```dockerfile
ENV PYTHONPATH=/opt/pypto/runtime/python:/opt/pypto/runtime/examples/scripts
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/Ascend/.../site-packages
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/Ascend/.../opp/...
```

#### 4. CANN environment — source at runtime, never mount /usr/local/Ascend

```dockerfile
RUN echo "[ -f ${CANN_HOME}/set_env.sh ] && { source ${CANN_HOME}/set_env.sh 2>/dev/null || true; }" >> /etc/bash.bashrc
```

The image already has CANN baked in. At runtime, mount only the driver:

```bash
docker run ... -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
```

Never mount `/usr/local/Ascend` — it shadows the image's CANN with the
host's (potentially older) version.

#### 5. Python version — use the base image's Python

The CANN base image ships its own Python. Install packages with `pip` (not
`pip3` or `python3 -m pip` — both work, but be consistent). Do NOT install
a different Python version.

---

## Server/dev Dockerfile (host workspace mount)

For live-editing workflows where source is on the host:

```dockerfile
COPY pypto /workspace/hw-native-sys/pypto
```

Runtime mounts the same path so edits are visible inside the container:

```bash
docker run ... -v $HOME/hw-native-sys:/workspace/hw-native-sys
```

### Key differences from standalone

- Uses `docker build -f Dockerfile.name .` (needs build context)
- `COPY`s repos from host instead of cloning
- `SIMPLER_ROOT` points to workspace path, not `/opt`
- Add `git config --system --add safe.directory` for mounted repos
- Early fail-fast validation: `test -f .../pyproject.toml` before building

---

## Simpler submodule handling

### pypto's runtime submodule = simpler

Pypto uses simpler as a git submodule at `runtime/`. Two approaches:

**Standalone image:** Clone pypto, init submodules, then pull simpler's
`origin/main` to bypass stale submodule pins:

```dockerfile
RUN git clone --depth 1 --branch main https://github.com/hw-native-sys/pypto.git "${PYPTO_DIR}" && \
    for i in 1 2 3; do \
      git -C "${PYPTO_DIR}" submodule update --init --recursive && break || sleep 5; \
    done && \
    git -C "${PYPTO_DIR}/runtime" fetch origin main && \
    git -C "${PYPTO_DIR}/runtime" checkout origin/main
```

**Why pull origin/main?** The submodule pin recorded in pypto's tree may
lag behind simpler's main. APIs like `ChipBufferSpec` may only exist in
newer simpler commits. Pulling `origin/main` after submodule init ensures
APIs pypto's `distributed_runner.py` expects are always available.

**Server/dev image:** Require the submodule on host, validate with a
fail-fast check:

```dockerfile
RUN test -f /workspace/hw-native-sys/pypto/runtime/pyproject.toml || \
  { echo 'BUILD ERROR: pypto/runtime/ missing. Run: git -C pypto submodule update --init --recursive'; exit 1; }
```

### standalone simpler Dockerfile (no pypto)

A simpler-only image clones only `hw-native-sys/simpler`, no pypto, no ptoas:

```dockerfile
RUN git clone --depth 1 --branch main https://github.com/hw-native-sys/simpler.git "${SIMPLER_DIR}"
```

It auto-derives its pto-isa commit from simpler's own CI config (different
from pypto's!):

```bash
PTO_ISA_COMMIT=$(grep -oP 'PTO_ISA_COMMIT:\s*\K[a-f0-9]+' .github/workflows/ci.yml | head -1)
```

---

## PTO-ISA clone — correct URLs and fallback

### Primary + fallback pattern (mirrors ci.yml)

```dockerfile
RUN timeout 60 git clone https://github.com/hw-native-sys/pto-isa.git "${PTO_ISA_DIR}" \
      || { rm -rf "${PTO_ISA_DIR}"; timeout 300 git clone https://gitcode.com/luohuan40/pto-isa.git "${PTO_ISA_DIR}"; } && \
    git -C "${PTO_ISA_DIR}" checkout "${PTO_ISA_COMMIT}"
```

**Critical:** The primary is `hw-native-sys/pto-isa` (NOT `PTO-ISA/pto-isa`).
The gitcode.com mirror is `luohuan40/pto-isa`. The `timeout` prevents
network hangs during build.

### Auto-deriving the pto-isa commit

**For pypto** — extract from pypto's own CI config (all `--pto-isa-commit=` occurrences share the same value):

```bash
PTO_ISA_COMMIT=$(grep -oP '(?<=--pto-isa-commit=)[a-f0-9]+' .github/workflows/ci.yml | head -1)
```

**For simpler** — extract from simpler's CI config (different commit!):

```bash
PTO_ISA_COMMIT=$(grep -oP 'PTO_ISA_COMMIT:\s*\K[a-f0-9]+' .github/workflows/ci.yml | head -1)
```

**Allow override via ARG:**

```dockerfile
ARG PTO_ISA_COMMIT=
RUN if [ -z "${PTO_ISA_COMMIT}" ]; then \
      PTO_ISA_COMMIT=$(grep -oP '...' .github/workflows/ci.yml | head -1); \
    fi && \
    git clone ... && git checkout "${PTO_ISA_COMMIT}"
```

---

## Third-party submodules (libbacktrace, msgpack-c)

Fallback pattern when pypto's submodules may not be checked out:

```dockerfile
RUN if [ ! -f 3rdparty/libbacktrace/configure.ac ]; then \
      rm -rf 3rdparty/libbacktrace && \
      git clone --depth 1 --branch macho-bundle-support \
        https://github.com/Hzfengsy/libbacktrace.git 3rdparty/libbacktrace; \
    fi && \
    if [ ! -f 3rdparty/msgpack-c/include/msgpack.hpp ]; then \
      rm -rf 3rdparty/msgpack-c && \
      git clone --depth 1 --branch cpp_master \
        https://github.com/msgpack/msgpack-c.git 3rdparty/msgpack-c; \
    fi
```

These are pypto's own submodules (separate from the runtime/simpler
submodule). They rarely change and the fallback is safe.

---

## Build order matters

1. Clone source repos
2. Clone pto-isa (needed by simpler's kernel compilation)
3. Install PTOAS binary (needed by pypto tests, NOT by simpler-only)
4. Install Python build toolchain (`scikit-build-core`, `nanobind`, `cmake`)
5. Install test deps (`numpy`, `pytest`, `torch` CPU)
6. Build simpler: `pip install --no-build-isolation ./runtime`
7. Build pypto: `pip install --no-build-isolation .[dev]`

Simpler MUST be built before pypto — pypto's `distributed_runner.py` imports
from `simpler.task_interface`.

---

## CI integration (docker-ci.yml)

### Trigger only on Dockerfile changes

```yaml
on:
  pull_request:
    paths:
      - 'Dockerfile.*'
      - '.github/workflows/docker-ci.yml'
```

### Match the existing CI test order exactly

For pypto, the `ci.yml` `system-tests` job runs 4 suites in this order:

1. `pytest tests/st --ignore=tests/st/distributed` (non-distributed)
2. `pytest tests/st/distributed/test_l2_multi_orch.py` (isolated — `Worker(level=2)` leaks state into `level=3` in the same process!)
3. `pytest tests/st/distributed --ignore=test_l2_multi_orch.py`
4. `pytest tests/st/runtime/test_perf_swimlane.py`

Each suite runs in a separate `docker run` so state isolation is guaranteed.

### Pass device range from runner environment

```yaml
-e DEVICE_RANGE    # docker run passes through host env
```

Inside container: `--device="${DEVICE_RANGE}"`

### Check NPU before building

```yaml
- name: Check NPU
  run: npu-smi info
```

### Use the correct runner label

- pypto: `[self-hosted, linux, arm64, npu]`
- simpler: `[self-hosted, a2a3]`

---

## Docker runtime flags (Ascend NPU)

Mandatory for all Ascend containers:

```bash
--privileged --ipc=host
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
-v /dev:/dev
```

Additional for multi-device / HCCL:

```bash
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined
```

---

## Common failures and fixes

### `WORKDIR: command not found` during build

A trailing `&& \` before WORKDIR concatenates it into a shell command.
Every RUN instruction must be a complete, terminated statement.

### `ImportError: cannot import name 'ChipBufferSpec' from 'simpler.task_interface'`

The pinned simpler submodule is stale. Pull `origin/main` after init:

```bash
git -C runtime fetch origin main && git -C runtime checkout origin/main
```

### `fatal error: 'tensor.h' file not found`

Incorrect `SIMPLER_ROOT` or include paths. The runtime resolver
(`env_manager.py`) resolves `SIMPLER_ROOT` by checking local directories
first. Ensure `SIMPLER_ROOT` points to the actual simpler source tree
(`/opt/pypto/runtime` in standalone, `/workspace/hw-native-sys/pypto/runtime`
in server/dev).

### `HcclGetRootInfo` hangs / bootstrap timeout 120s

The HCCL comm init blocks silently. Usually a network interface issue.
HCCL picks the wrong NIC on some machines. The `HCCL_SOCKET_IFNAME` env
var can override, but our CI and Dockerfiles intentionally don't set it
(the runner's default interface is correct).

### `comm_alloc_windows failed with code -1` + `ImportByKey(...) -> 507899`

This is usually an Ascend driver/runtime ABI mismatch, not a pypto/simpler
logic bug. Most common cause: container CANN runtime and host-mounted driver
stack are not a validated pair.

Fast checks:

1. Host: `npu-smi info` (active driver/firmware version)
2. Container: verify mounted `/usr/local/Ascend/driver` metadata matches host
3. Confirm reboot/module reload actually activated the new driver

If versions don't match expected compatibility, fix host driver state first.
Changing Python/test code will not resolve this signature.

### Segfault in `comm_init` after driver upgrade

If `ImportByKey` mismatch disappears but tests now crash with a hard segfault
during `comm_init`, treat this as runtime/process-model instability first.

Strong signal from our incident: tests emitted fork warnings before crash
(`process is multi-threaded, use of fork() may lead to deadlocks`). With
ACL/HCCL loaded, `fork` in multi-threaded parents can crash child comm init.

Triage order:

1. Re-run only on non-busy NPUs (for example `2,3`), avoid contended device 0
2. Reproduce with `spawn`/`forkserver` start method
3. Capture core dump and ACL/HCCL debug logs for vendor escalation

If `spawn` avoids the crash, classify as runtime + process-model issue, not
simpler orchestration logic.

### `aclInit failed` / `aclrtSetDevice failed`

Device files missing. Verify `--privileged` and `-v /dev:/dev` are present.
Also verify `npu-smi info` works on the host.

### `COPY simpler ...` fails with "path not found"

The standalone Dockerfile uses stdin — no build context. Use `git clone`
instead of `COPY`. The `docker build - < Dockerfile` syntax has NO context;
every file must be cloned at build time.

### PTOAS SHA256 mismatch

CI bumped the PTOAS version. Check `ci.yml` for `PTOAS_VERSION` and
`PTOAS_SHA256` and update both ARGs in the Dockerfile.

### `error: custom op 'pto.declare_local_array' is unknown`

This is a PTOAS/toolchain mismatch: pypto emitted a newer op that the current
ptoas binary does not support. Upgrade ptoas to the version pinned in CI.

In our incident this was resolved by moving to PTOAS v0.40 (matching CI).

### `echo` vs `printf` in /etc/bash.bashrc

Use `echo` (not `printf` with single quotes) so `${CANN_HOME}` expands at
build time:

```dockerfile
RUN echo "[ -f ${CANN_HOME}/set_env.sh ] && { source ${CANN_HOME}/set_env.sh 2>/dev/null || true; }" >> /etc/bash.bashrc
```

---

## Compile-time pypto distributed bugs (DSL / IR passes)

These bugs do not surface as Docker, driver, or HCCL errors. They surface as
**silent semantic failures** — a multi-rank test passes only on rank 0, or a
factory-built program compiles to the wrong shape with no diagnostic. The
triage flow below is the one that worked in the L3 GEMM bring-up
(`feat/l3-allreduce-gemm`).

### Triage order for silent multi-rank correctness failures

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
   issue and use the "Common failures and fixes" section above.

### 1. Multi-rank dispatch needs a comm window (`CollectCommGroups` recipe)

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
("Multi-rank dispatch recipe" section of the module docstring). Every passing
multi-rank ST in `tests/st/distributed/` (`test_l3_notify_wait`,
`test_l3_allreduce`, `test_l3_put`, `test_l3_get`) carries a real comm window
because those tests are explicitly testing collectives — the bug is only
observable when the kernel has **no** real comm of its own.

### 2. Avoid `pl.range(1)` in the P=1 branch (Simplify single-trip unroll)

**Symptom.** A P=1 program built from a factory that uses
`for r in pl.range(pld.world_size())` fails to codegen with an unresolved
induction-variable reference on the lowered `device=` argument.

**Root cause.** The Simplify pass single-trip-unrolls `for r in pl.range(1)`
but leaves `device=r` on the lowered call. Subsequent lower passes do not
resolve `r` to a constant after unroll.

**Fix.** In the P=1 branch, emit a direct `self.chip_orch(..., device=0)`
call rather than a loop. Only use `for r in pl.range(pld.world_size())` when
`nranks >= 2`.

```python
if nranks == 1:
    @pl.program
    class FooP1:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, ...):
            self.chip_orch(a[0], b, c[0], device=0)
            return c
    return FooP1

@pl.program
class FooPN:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(self, ...):
        scratch_buf = pld.alloc_window_buffer(4)
        for r in pl.range(pld.world_size()):
            scratch = pld.window(scratch_buf, [1, 1], dtype=pl.INT32)
            self.chip_orch(a[r], b, c[r], scratch, device=r)
        return c
return FooPN
```

### 3. `@pl.program` class-name collision in factory `if/else` branches

**Symptom.** A factory `build_foo(*, nranks)` that returns different program
class shapes per branch silently compiles the wrong shape. The runtime
decorator argument is the correct class object, but the compiled IR matches
a different branch.

**Root cause.**
`python/pypto/language/parser/decorator.py` recovers source via
`inspect.getsourcelines(cls)` and then uses `ast.walk` over the parsed source
to find an `ast.ClassDef` matching `cls.__name__`. If a factory has two
`class Foo:` definitions across an `if/else`, `ast.walk` returns the first
one in source order regardless of which class object was actually passed in.

**Fix.** Give each branch's class a distinct name (`FooP1` vs `FooPN`,
`FooSmall` vs `FooLarge`, etc.). This is a project-wide invariant for any
pypto factory that returns differently-shaped program classes.

**Reference.** First documented in `tests/st/distributed/l3_gemm_programs.py`
(top of module docstring, "The P=1 and P>=2 branches use DISTINCT class
names").

### 4. Per-iteration window rebind inside the dispatch loop

**Symptom.** Multi-rank dispatch loop using `for r in pl.range(pld.world_size())`
with a single `scratch = pld.window(buf, ...)` bound **above** the loop emits
only one dispatch (or the wrong number).

**Root cause.** `CollectCommGroups` identifies windows by their producer
expression. Binding `pld.window(buf, ...)` once outside the loop and reusing
the same SSA value across iterations is **not** an equivalent program — the
pass sees one window consumed once, not P windows consumed once each.

**Fix.** Re-emit the `pld.window(buf, ...)` call inside the loop body so each
iteration produces a fresh window view:

```python
for r in pl.range(pld.world_size()):
    scratch = pld.window(scratch_buf, [1, 1], dtype=pl.INT32)  # rebind per iter
    self.chip_orch(a[r], b, c[r], scratch, device=r)
```

The underlying `scratch_buf` (the `alloc_window_buffer` return value) is
allocated once and reused; only the per-iteration window view is rebound.

### 5. Diagnose parser-level statement drops with frontend IR

**Symptom.** You inspect the generated `orchestration/host_orch.py` (under
the `dump_passes` output directory) and see fewer dispatches than your DSL
source has — or the dispatches are missing arguments.

**Root cause hypothesis.** A statement was dropped by the AST→IR translation
in the parser, or `@pl.program` matched the wrong `ClassDef` (see Sub 3) and
compiled an unrelated body.

**Diagnostic recipe.** Dump the **frontend IR** (the first numbered
`00_frontend.py` produced by `dump_passes=True`) and compare against the
Python source:

```python
import pypto.language as pl
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

program = build_my_program(nranks=2)
compiled = ir.compile(
    program,
    platform="a2a3",
    distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    dump_passes=True,
    dump_dir="/tmp/pypto_dump",
)
# read /tmp/pypto_dump/<program>/00_frontend.py
```

If a statement is already missing in `00_frontend.py`, the bug is upstream of
all passes — parser or decorator. If the statement is present in
`00_frontend.py` but missing in a later numbered file, it's a pass bug;
bisect by reading the numbered files in order.

**Reference.** This is the workflow that pinned down the `@pl.program`
class-name-collision bug during L3 GEMM bring-up. The throwaway
`_dump_l3_gemm_codegen.py` helper that did this used to live at
`tests/st/distributed/_dump_l3_gemm_codegen.py` and has since been retired
(the diagnostic logic is a one-shot `ir.compile(..., dump_passes=True)`
call; no helper needed).

### Tensor-argument direction maps to simpler TensorMap behaviour

For completeness — this is not a bug, but it surfaced often enough during
bring-up to be worth recording.

| pypto direction       | simpler `TensorArgType` | TensorMap behaviour          |
|-----------------------|--------------------------|------------------------------|
| `pl.Tensor[...]`      | `INPUT`                  | Lookup only                  |
| `pl.Out[pl.Tensor]`   | `OUTPUT_EXISTING`        | Insert only (no lookup)      |
| `pl.InOut[pl.Tensor]` | `INOUT`                  | Lookup + insert              |

Use `Out` when the buffer's pre-call contents are irrelevant. Use `InOut`
when a downstream `chip_orch` dispatch in the same `host_orch` needs the
buffer as input. Mismatching `Out` for a value that is read back leaves the
simpler `TensorMap` unable to wire up the dependency, with no error — the
read silently sees stale or uninitialised memory.

---

## Debugging a Dockerfile interactively

1. Build with a target stage or up to the failing line
2. `docker run --rm -it <image> bash` and manually re-run the failing commands
3. Check environments: `env | sort`, `pip list | grep -i simpler`
4. Check paths: `ls /opt/pypto/runtime/python/simpler/task_interface.py`
5. Verify imports: `python -c "from simpler.task_interface import ChipBootstrapConfig"`
6. Verify pto-isa: `git -C /opt/pto-isa rev-parse --short HEAD`
7. Verify simpler: `git -C /opt/pypto/runtime rev-parse --short HEAD`

---

## Quick reference: correct CANN mount

| Mount | Why |
|-------|-----|
| `/usr/local/Ascend/driver` (ro) | ✅ kernel driver user-space libs |
| `/dev` | ✅ NPU device files |
| `/usr/local/bin/npu-smi` (ro) | ✅ device management tool |
| `/usr/local/Ascend` (entire tree) | ❌ shadows image's CANN with host's |

---

## Quick reference: repo URLs

| Repo | URL | Notes |
|------|-----|-------|
| pypto | `https://github.com/hw-native-sys/pypto.git` | main repo |
| simpler | `https://github.com/hw-native-sys/simpler.git` | pypto submodule at `runtime/` |
| pto-isa | `https://github.com/hw-native-sys/pto-isa.git` | kernel ISA headers |
| pto-isa mirror | `https://gitcode.com/luohuan40/pto-isa.git` | fallback when GitHub is down |
| PTOAS | `https://github.com/hw-native-sys/PTOAS/releases/download/${VERSION}/ptoas-bin-aarch64.tar.gz` | binary tarball |
| libbacktrace | `https://github.com/Hzfengsy/libbacktrace.git` | macho-bundle-support branch |
| msgpack-c | `https://github.com/msgpack/msgpack-c.git` | cpp_master branch |
