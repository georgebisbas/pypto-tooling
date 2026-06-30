---
name: pypto-dockerfile-construction
description: >-
  Build and debug standalone/server Ascend NPU Docker images for PyPTO,
  simpler, and pytorch-hccl-tests. Covers ARG-driven paths, git-clone
  patterns, HCCL runtime flags, build order, and common failures.
triggers:
  - "create a Dockerfile"
  - "add to Dockerfile"
  - "update Dockerfile"
  - "Dockerfile is stale"
  - "Dockerfile build fails"
  - "Dockerfile error"
  - "docker build fails"
  - "container fails"
  - "import error in container"
  - "HCCL fails in container"
  - "simpler_init failed"
  - "ChipBufferSpec"
  - "aclInit failed"
---

# PyPTO Dockerfile construction

## When to use this skill

```text
Task starts
├─ User says "update this Dockerfile"?  → "When to update" per-Dockerfile checklist
├─ Creating a NEW Dockerfile?  → "Dockerfile inventory" + "Build-time patterns"
├─ Existing Dockerfile won't BUILD? → "Common failures" symptom table
├─ Container RUNS but tests FAIL? → debugging_skills/SKILL.md first, then return
├─ Need to understand a PATTERN? → Jump to named h3 below
└─ Debugging interactively?        → "Interactive debugging" checklist
```

## Dockerfile inventory

| Dockerfile | What it builds | Build command | Needs NPU? |
|-----------|---------------|---------------|------------|
| `Dockerfile.hw-native-sys.cann9.0` | pypto + simpler + pto-isa + ptoas | `docker build -t img - < Dockerfile...` | Yes (a2a3) |
| `Dockerfile.simpler.cann9.0` | simpler + pto-isa only | `docker build -t img - < Dockerfile...` | Yes (a2a3) |
| `Dockerfile.hw-native-sys.sim.ubuntu22.04` | pypto + pto-isa (sim mode) | `docker build -t img -f Dockerfile... .` | No |
| `Dockerfile.simpler.sim.ubuntu22.04` | simpler + pto-isa (sim mode) | `docker build -t img -f Dockerfile... .` | No |
| `Dockerfile.pytorch-hccl-tests.cann9.0` | HCCL benchmarks | `docker build -t img - < Dockerfile...` | Yes (a2a3) |
| `Dockerfile.server.cann:9.0` | pypto dev workspace | `docker build -t img -f Dockerfile... .` | Yes (a2a3) |

All use `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12` as base image.

## When to update Dockerfiles

Each Dockerfile pins specific external versions. When asked to update one, run the corresponding checklist below. Each check produces a diff command — if it outputs anything, an update is needed.

### `Dockerfile.hw-native-sys.cann9.0` — pypto + simpler + pto-isa + ptoas

| # | Check | Command | What to update if drifted |
|---|-------|---------|--------------------------|
| 1 | pypto `origin/main` ahead of `PYPTO_COMMIT`? | `git -C /path/to/pypto rev-list --count ${PYPTO_COMMIT}..origin/main` | `ARG PYPTO_COMMIT` + header comment example |
| 2 | pto-isa commit in pypto's `ci.yml` changed? | `grep -oP '(?<=--pto-isa-commit=)[a-f0-9]+' pypto/.github/workflows/ci.yml \| head -1` | Auto-derived at build time; update test comments only |
| 3 | PTOAS version or SHA256 changed in pypto CI? | `grep -E 'PTOAS_VERSION\|PTOAS_SHA256' pypto/.github/workflows/ci.yml` | `ARG PTOAS_VERSION` + `ARG PTOAS_SHA256` |
| 4 | pip deps changed in pypto CI Dockerfile? | `diff <(grep 'pip install' pypto/.github/docker/github_ci.Dockerfile) <(grep 'pip install' Dockerfile.hw-native-sys.cann9.0)` | Update pip install RUN lines |
| 5 | Test commands in header comment match pypto CI? | Compare `ci.yml` jobs with Dockerfile comment header | Update test command cheatsheet in header comment |

### `Dockerfile.simpler.cann9.0` — simpler + pto-isa only

| # | Check | Command | What to update if drifted |
|---|-------|---------|--------------------------|
| 1 | simpler `origin/main` ahead of `SIMPLER_COMMIT`? | `git -C /path/to/simpler rev-list --count ${SIMPLER_COMMIT}..origin/main` | `ARG SIMPLER_COMMIT` + header comment |
| 2 | pto-isa commit in simpler's CI changed? | `grep -oP 'PTO_ISA_COMMIT:\s*\K[a-f0-9]+' simpler/.github/workflows/ci.yml \| head -1` | `ARG PTO_ISA_COMMIT` (simpler CI format, not pypto!) |
| 3 | `ENV LD_PRELOAD` absent? | `grep LD_PRELOAD Dockerfile.simpler.cann9.0` — should be comments only | Remove image-wide `ENV LD_PRELOAD` |
| 4 | `set_env.sh` stripped, not re-added to bashrc? | `grep set_env Dockerfile.simpler.cann9.0` — strip `RUN` only, no bashrc append | Align with hw-native-sys pattern |

Note: No PTOAS or pypto dependencies — simpler-only image.

### `Dockerfile.hw-native-sys.sim.ubuntu22.04` — pypto + pto-isa (x86_64 sim)

| # | Check | Command | What to update if drifted |
|---|-------|---------|--------------------------|
| 1 | pypto `origin/main` ahead of `PYPTO_COMMIT`? | Same as hw-native-sys check #1 | `ARG PYPTO_COMMIT` |
| 2 | pto-isa commit in pypto's `ci.yml` changed? | Same as hw-native-sys check #2 | `ARG PTO_ISA_COMMIT` |
| 3 | PTOAS x86_64 version/SHA256 changed? | `grep -E 'PTOAS_VERSION\|PTOAS_SHA256' pypto/.github/workflows/ci.yml` — use the **x86_64** SHA256 from `system-tests-a5sim` job | `ARG PTOAS_VERSION` + `ARG PTOAS_SHA256` (x86_64 binary) |

Note: Uses **x86_64** `ptoas-bin-x86_64.tar.gz` with a **different SHA256** than the aarch64 binary used in hw-native-sys.

### `Dockerfile.simpler.sim.ubuntu22.04` — simpler + pto-isa (x86_64 sim)

| # | Check | Command | What to update if drifted |
|---|-------|---------|--------------------------|
| 1 | simpler `origin/main` ahead of `SIMPLER_COMMIT`? | `git -C /path/to/simpler rev-list --count ${SIMPLER_COMMIT}..origin/main` | `ARG SIMPLER_COMMIT` + header comment |
| 2 | pto-isa commit in simpler's CI changed? | `grep -oP 'PTO_ISA_COMMIT:\s*\K[a-f0-9]+' simpler/.github/workflows/ci.yml \| head -1` | `ARG PTO_ISA_COMMIT` (simpler CI format) |
| 3 | L3 sim pytest flags match `st-sim-a2a3`? | Compare `ci.yml` `pytest examples tests/st --platform a2a3sim` with `scripts/run-simpler-l3-sim.sh` | Update script + Dockerfile header |

Note: No pypto, CANN, or ptoas — simpler-only sim image. Build context must include `scripts/run-simpler-l3-sim.sh` (build from `pypto-tooling/`).

### `Dockerfile.pytorch-hccl-tests.cann9.0` — HCCL benchmarks only

| # | Check | Command | What to update if drifted |
|---|-------|---------|--------------------------|
| 1 | Fork `origin/master` ahead of `PT_HCCL_COMMIT`? | `git -C /path/to/pytorch-hccl-tests rev-list --count ${PT_HCCL_COMMIT}..origin/master` | `ARG PT_HCCL_COMMIT` |
| 2 | `ENV LD_PRELOAD` absent? | `grep LD_PRELOAD Dockerfile.pytorch-hccl-tests.cann9.0` — should be comments only | Remove image-wide `ENV LD_PRELOAD` |
| 3 | `set_env.sh` stripped, not re-added to bashrc? | `grep set_env Dockerfile.pytorch-hccl-tests.cann9.0` — strip `RUN` only, no bashrc append | Align strip `RUN` with hw-native-sys |
| 4 | Editable install (`pip uninstall` + `pip install -e .` after the make target)? | `grep -A1 'pip uninstall' Dockerfile.pytorch-hccl-tests.cann9.0` | Restore uninstall+editable so branch switches work — see below |

**Install contract:** `make ${PT_HCCL_INSTALL_TARGET}` does a non-editable `pip install .` (a flat copy into site-packages). That copy **shadows** a later `pip install -e .` on `sys.path`, so the image must `pip uninstall -y pytorch_hccl_tests` first, then `pip install -e .`. Without it, `git checkout <branch>` inside the container yields `ImportError` against the frozen copy. The build ends with `python -c "import pytorch_hccl_tests as p; print(p.__file__)"` — must print a path under `${PT_HCCL_DIR}`, not site-packages.

**Run recipes** (multi-device needs `--pid=host`; no `LD_PRELOAD`):

```bash
WORLD_SIZE=2 make bidirectional-bw DEVICE=npu   # bibw: point-to-point, exactly 2 ranks
WORLD_SIZE=8 make mbw-mr DEVICE=npu             # mbw_mr: world_size//2 concurrent pairs
```

Pass `WORLD_SIZE` as a **make argument** (or `make -e`) unless the Makefile uses `?=` — `export WORLD_SIZE = 2` shadows an env prefix. Full debugging notes: `issue_pytorch_hccl_tests.md`.

### `Dockerfile.server.cann:9.0` — pypto dev workspace (host mount)

| # | Check | Command | What to update if drifted |
|---|-------|---------|--------------------------|
| 1 | pto-isa commit in pypto's `ci.yml` changed? | Same as hw-native-sys check #2 | The clone command inside the Dockerfile |
| 2 | PTOAS version/SHA256 changed? | Same as hw-native-sys check #3 | `ARG PTOAS_VERSION` + `ARG PTOAS_SHA256` |
| 3 | pip deps changed in pypto CI Dockerfile? | Same as hw-native-sys check #4 | Update pip install RUN lines |

Note: pypto + simpler are mounted from host at runtime, not pinned in the Dockerfile. Only pto-isa and PTOAS are cloned/downloaded at build time.

### Quick bulk check (all Dockerfiles)

```bash
# Clone pypto and simpler if not already present
cd /tmp
git clone --depth 1 https://github.com/hw-native-sys/pypto.git 2>/dev/null || true
git clone --depth 1 https://github.com/hw-native-sys/simpler.git 2>/dev/null || true

# Run all checks at once
cd ~/pypto-tooling
echo "=== pypto origin/main ===" && git -C /tmp/pypto rev-parse --short HEAD
echo "=== simpler origin/main ===" && git -C /tmp/simpler rev-parse --short HEAD
echo "=== pto-isa (pypto CI) ===" && grep -oP '(?<=--pto-isa-commit=)[a-f0-9]+' /tmp/pypto/.github/workflows/ci.yml | head -1
echo "=== pto-isa (simpler CI) ===" && grep -oP 'PTO_ISA_COMMIT:\s*\K[a-f0-9]+' /tmp/simpler/.github/workflows/ci.yml | head -1
echo "=== PTOAS (pypto CI) ===" && grep -E 'PTOAS_VERSION|PTOAS_SHA256' /tmp/pypto/.github/workflows/ci.yml | head -2
echo "=== pip deps (pypto CI Dockerfile) ===" && grep 'pip install' /tmp/pypto/.github/docker/github_ci.Dockerfile
```

---

## Build-time patterns

### 1. ARG-driven paths

Every path derives from two ARGs — never hardcoded:

```dockerfile
ARG CANN_VERSION=9.0.0
ARG INSTALL_PREFIX=/opt

ENV CANN_HOME=/usr/local/Ascend/cann-${CANN_VERSION}
ENV PYPTO_DIR=${INSTALL_PREFIX}/pypto \
    PTO_ISA_DIR=${INSTALL_PREFIX}/pto-isa \
    PTOAS_DIR=${INSTALL_PREFIX}/ptoas-bin
```

### 2. ARG scope resets after FROM

```dockerfile
ARG CANN_VERSION=9.0.0
FROM quay.io/ascend/cann:${CANN_VERSION}-910b-ubuntu22.04-py3.12
ARG CANN_VERSION=9.0.0          # ← re-declare after FROM
ARG INSTALL_PREFIX=/opt
```

### 3. Self-referencing ENV — split across multiple instructions

Docker does NOT expand `${VAR}` within the same ENV block. Split:

```dockerfile
ENV PYTHONPATH=/opt/pypto/runtime/python:/opt/pypto/runtime/examples/scripts
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/Ascend/.../site-packages
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/Ascend/.../opp/...
```

### 4. Build order (executable checklist)

```
[ ] 1. Clone source repos (pypto/simpler)
[ ] 2. Clone pto-isa (needed by simpler kernel compilation)
[ ] 3. Install PTOAS binary (needed by pypto tests; NOT needed by simpler-only)
[ ] 4. pip install scikit-build-core nanobind cmake ninja
[ ] 5. pip install numpy pytest torch (CPU)
[ ] 6. pip install --no-build-isolation ./runtime   ← simpler MUST be built before pypto
[ ] 7. pip install --no-build-isolation .[dev]       ← depends on simpler.task_interface
```

### 5. Simpler submodule (pypto's `runtime/`)

**Standalone image** — clone pypto, init submodules:

```dockerfile
RUN git clone --filter=blob:none https://github.com/hw-native-sys/pypto.git "${PYPTO_DIR}" && \
    git -C "${PYPTO_DIR}" checkout "${PYPTO_COMMIT}" && \
    for i in 1 2 3; do \
      git -C "${PYPTO_DIR}" submodule update --init --recursive && break || sleep 5; \
    done
```

**Server/dev image** — validate submodule exists on host:

```dockerfile
RUN test -f /workspace/hw-native-sys/pypto/runtime/pyproject.toml || \
  { echo 'BUILD ERROR: pypto/runtime/ missing. Run: git -C pypto submodule update --init --recursive'; exit 1; }
```

### 6. PTO-ISA clone (primary + fallback)

```dockerfile
RUN timeout 60 git clone https://github.com/hw-native-sys/pto-isa.git "${PTO_ISA_DIR}" \
      || { rm -rf "${PTO_ISA_DIR}"; timeout 300 git clone https://gitcode.com/luohuan40/pto-isa.git "${PTO_ISA_DIR}"; } && \
    git -C "${PTO_ISA_DIR}" checkout "${PTO_ISA_COMMIT}"
```

Auto-derive commit from CI when not pinned via ARG:

```bash
# For pypto Dockerfiles:
PTO_ISA_COMMIT=$(grep -oP '(?<=--pto-isa-commit=)[a-f0-9]+' .github/workflows/ci.yml | head -1)
# For simpler Dockerfiles (different CI format!):
PTO_ISA_COMMIT=$(grep -oP 'PTO_ISA_COMMIT:\s*\K[a-f0-9]+' .github/workflows/ci.yml | head -1)
```

### 7. Third-party fallback (libbacktrace, msgpack-c)

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

### 8. Python version

Use the CANN base image's Python. Do NOT install a different version.

### 9. Dockerfile types: standalone vs server/dev

| | Standalone (stdin) | Server/dev (context) |
|---|---|---|
| Build | `docker build -t img - < Dockerfile` | `docker build -t img -f Dockerfile .` |
| Source | `git clone` everything | `COPY` from host |
| Use case | CI, reproducible builds | Live editing, workspace mounts |

---

## Runtime patterns

### Docker run flags (all Ascend containers)

```bash
docker run --rm -it --privileged --ipc=host \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /dev:/dev \
    <image>
```

Additional for HCCL / multi-device:

```bash
--pid=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
```

### CANN mount rules

| Mount | OK? | Why |
|-------|-----|-----|
| `/usr/local/Ascend/driver` (ro) | ✅ | Kernel driver user-space libs |
| `/dev` | ✅ | NPU device files |
| `/usr/local/bin/npu-smi` (ro) | ✅ | Device management tool |
| `/usr/local/Ascend` (entire tree) | ❌ | Shadows image's CANN with host's |

### HCCL requirements

`--pid=host` is required because Ascend IPC validates host PIDs, not container PIDs.

`LD_PRELOAD=libhccl.so` is **NOT** set image-wide. Setting it as `ENV` or in `bashrc`
injects libhccl.so into every process including VS Code's server node → hang on attach.

- **pypto / simpler images:** set it manually in the shell before running HCCL tests:
- **pytorch-hccl-tests image:** `LD_PRELOAD` is not needed (torch HCCL path; no `host_runtime.so`).

```bash
export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so
```

→ See `debugging_skills/SKILL.md` → "`LD_PRELOAD=libhccl.so`" and "VS Code attach hang" for details.

### CANN environment

**Do NOT source `set_env.sh` from bashrc or any startup file.** The Ascend base
image already injects it into multiple startup files; if it runs during VS Code's
`userEnvProbe` it takes 10–30s → attach hangs. Strip it at build time:

```dockerfile
RUN for f in /etc/profile /etc/bash.bashrc /root/.profile /root/.bashrc /root/.bash_profile; do \
      [ -f "$f" ] && sed -i '/set_env\.sh/d' "$f" || true; \
    done && \
    for f in /etc/profile.d/*.sh; do \
      [ -f "$f" ] && sed -i '/set_env\.sh/d' "$f" || true; \
    done
```

All required CANN env vars (`ASCEND_HOME_PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`, etc.)
are set via `ENV` instructions — `set_env.sh` is not needed at runtime.

Unset stale `PTO2_RING_*` vars the host may leak in:

```dockerfile
RUN echo "unset PTO2_RING_HEAP PTO2_RING_TASK_WINDOW PTO2_RING_DEP_POOL 2>/dev/null || true" >> /etc/bash.bashrc
```

---

## Common failures

### Decision tree

```text
Error message contains...
├─ "WORKDIR: command not found"    → Trailing &&\ before WORKDIR
├─ "ChipBufferSpec"                → Stale simpler submodule
├─ "aclInit failed" / "aclrtSetDevice" → Missing --privileged or -v /dev:/dev
├─ "HcclGetRootInfo" / timeout 120s → HCCL network issue
├─ "507899" / "comm_alloc_windows"  → CANN/driver mismatch → debugging_skills
├─ "COPY ... path not found"        → Using COPY in stdin build → use git clone
├─ "fatal error: 'tensor.h'"        → Wrong SIMPLER_ROOT
├─ "PTOAS SHA256 mismatch"          → CI bumped version → update ARGs
├─ "custom op ... unknown"          → PTOAS too old → update PTOAS_VERSION
├─ "cannot import name ..." from site-packages after checkout → non-editable copy shadows -e → issue_pytorch_hccl_tests
├─ "Unsupported data type at::kDouble" → fp64 reduce on HCCL → use float32 → issue_pytorch_hccl_tests
├─ WORLD_SIZE=N make ... ignored    → Makefile export shadows env → make arg / ?= → issue_pytorch_hccl_tests
└─ Segfault in comm_init + fork warn → ACL/HCCL + fork → use spawn/forkserver
```

### Symptom → fix table

| Symptom | Cause | Fix |
|---------|-------|-----|
| `WORKDIR: command not found` during build | Trailing `&& \` before WORKDIR | Terminate every RUN before WORKDIR |
| `ImportError: ChipBufferSpec` | Stale simpler submodule pin | `git -C runtime fetch origin main && checkout origin/main` |
| `aclInit` / `aclrtSetDevice` failed | Missing docker run flags | Add `--privileged -v /dev:/dev` |
| `HcclGetRootInfo` hangs (120s) | Wrong NIC / HCCL network | Try different device, check `HCCL_SOCKET_IFNAME` |
| `comm_alloc_windows` + **507899** | CANN 9.0 + driver < 26.0.rc1 | → debugging_skills → "NPU error code reference" |
| segfault in `comm_init` + fork warning | ACL/HCCL loaded + multi-threaded fork | Use spawn/forkserver, avoid busy device 0 |
| `COPY` fails: "path not found" | Stdin build has no build context | Use `git clone` instead of `COPY` |
| `fatal error: 'tensor.h'` | Wrong `SIMPLER_ROOT` | Set to simpler source tree (`/opt/pypto/runtime`) |
| `ImportError: cannot import name 'X'` from `site-packages/...` after `git checkout` | Non-editable `pip install .` copy shadows `pip install -e .` | `pip uninstall -y <pkg> && pip install -e .`; verify `python -c "import <pkg> as p; print(p.__file__)"` is under the repo, not site-packages → `issue_pytorch_hccl_tests.md` |
| `WORLD_SIZE=N make ...` runs wrong rank count | Makefile `export VAR = n` shadows env prefix | Pass as make arg (`make t VAR=n`) or `make -e`; fix Makefile to `VAR ?= n` → `issue_pytorch_hccl_tests.md` |
| `HCCL reduce: Unsupported data type at::kDouble` (ERR02007) | fp64 `dist.reduce` unsupported on HCCL (works on gloo) | Reduce in `float32`; use only HCCL dtypes (int32/fp16/fp32/bf16) → `issue_pytorch_hccl_tests.md` |

For device health issues (dead NPUs, 507033): run `diagnose_npu.py` inside the container, or see `debugging_skills/SKILL.md`.

---

## Interactive debugging

```bash
# 1. Enter the image at the failing stage
docker run --rm -it <image> bash

# 2. Verify environment
env | sort
pip list | grep -i simpler
python -c "from simpler.task_interface import ChipBootstrapConfig"

# 3. Verify git checkouts
git -C /opt/pto-isa rev-parse --short HEAD
git -C /opt/pypto/runtime rev-parse --short HEAD

# 4. NPU health
python3 /opt/pypto/diagnose_npu.py   # if copied in
npu-smi info
```

## Quick reference: repo URLs

| Repo | URL | Notes |
|------|-----|-------|
| pypto | `github.com/hw-native-sys/pypto.git` | main repo |
| simpler | `github.com/hw-native-sys/simpler.git` | pypto submodule at `runtime/` |
| pto-isa | `github.com/hw-native-sys/pto-isa.git` | kernel ISA headers |
| pto-isa mirror | `gitcode.com/luohuan40/pto-isa.git` | fallback |
| PTOAS | `github.com/hw-native-sys/PTOAS/releases/download/${VER}/ptoas-bin-aarch64.tar.gz` | binary tarball |
| libbacktrace | `github.com/Hzfengsy/libbacktrace.git` | `macho-bundle-support` branch |
| msgpack-c | `github.com/msgpack/msgpack-c.git` | `cpp_master` branch |

## See also

- `debugging_skills/SKILL.md` — NPU error codes, dead device diagnosis, Docker runtime tips, distributed bug recipes
- `diagnose_npu.py` — 10-point NPU health check (run inside container)
- `issue_pytorch_hccl_tests.md` — pytorch-hccl-tests benchmark image: editable-install shadow, WORLD_SIZE Makefile shadow, fp64 HCCL reduce crash
- `issue_0.md` — `comm_alloc_windows` / HCCL IPC driver mismatch
- `issue_vscode_summary.md` — VS Code attach hang (`set_env.sh` + `LD_PRELOAD`)
- `build_skills/` — build system debugging
