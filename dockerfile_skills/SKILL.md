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

Note: No PTOAS or pypto dependencies — simpler-only image.

### `Dockerfile.hw-native-sys.sim.ubuntu22.04` — pypto + pto-isa (x86_64 sim)

| # | Check | Command | What to update if drifted |
|---|-------|---------|--------------------------|
| 1 | pypto `origin/main` ahead of `PYPTO_COMMIT`? | Same as hw-native-sys check #1 | `ARG PYPTO_COMMIT` |
| 2 | pto-isa commit in pypto's `ci.yml` changed? | Same as hw-native-sys check #2 | `ARG PTO_ISA_COMMIT` |
| 3 | PTOAS x86_64 version/SHA256 changed? | `grep -E 'PTOAS_VERSION\|PTOAS_SHA256' pypto/.github/workflows/ci.yml` — use the **x86_64** SHA256 from `system-tests-a5sim` job | `ARG PTOAS_VERSION` + `ARG PTOAS_SHA256` (x86_64 binary) |

Note: Uses **x86_64** `ptoas-bin-x86_64.tar.gz` with a **different SHA256** than the aarch64 binary used in hw-native-sys.

### `Dockerfile.pytorch-hccl-tests.cann9.0` — HCCL benchmarks only

| # | Check | Command | What to update if drifted |
|---|-------|---------|--------------------------|
| 1 | Fork `origin/master` ahead of `PT_HCCL_COMMIT`? | `git -C /path/to/pytorch-hccl-tests rev-list --count ${PT_HCCL_COMMIT}..origin/master` | `ARG PT_HCCL_COMMIT` |

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

`--pid=host` is required because Ascend IPC validates host PIDs, not container PIDs. `LD_PRELOAD=libhccl.so` is baked into the image so `host_runtime.so`'s WEAK HCCL symbols resolve at load time.

→ See `debugging_skills/SKILL.md` → "Docker runtime: HCCL, mounts, and --pid=host" for detailed diagnostics.

### CANN environment

Source in `.bashrc` so it's available interactively:

```dockerfile
RUN echo "[ -f ${CANN_HOME}/set_env.sh ] && { source ${CANN_HOME}/set_env.sh 2>/dev/null || true; }" >> /etc/bash.bashrc && \
    echo "unset PTO2_RING_HEAP PTO2_RING_TASK_WINDOW PTO2_RING_DEP_POOL 2>/dev/null || true" >> /etc/bash.bashrc
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
- `build_skills/` — build system debugging
