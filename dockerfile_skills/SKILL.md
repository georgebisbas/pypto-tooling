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

### `echo` vs `printf` in /etc/bash.bashrc

Use `echo` (not `printf` with single quotes) so `${CANN_HOME}` expands at
build time:

```dockerfile
RUN echo "[ -f ${CANN_HOME}/set_env.sh ] && { source ${CANN_HOME}/set_env.sh 2>/dev/null || true; }" >> /etc/bash.bashrc
```

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
