## PyPTO 3.0 (hw-native-sys) Server Test Tutorial

Use this when you want to run PyPTO tests on `hng-atlas01` with the CANN 9.0 image from `Dockerfile.server.cann:9.0`.

### 1) SSH to the server

```bash
ssh -i ~/.ssh/i....
```

### 2) Go to the build context root

The Dockerfile expects `pypto/` to exist in the build context.

```bash
cd ~/hw-native-sys
```

### 3) Build image (recommended)

Rebuild whenever `Dockerfile.server.cann:9.0` changes.

```bash
docker build --no-cache -f "Dockerfile.server.cann:9.0" -t pypto3-dev-env:cann9 .
```

Quick sanity checks:

```bash
docker image inspect pypto3-dev-env:cann9 --format '{{json .Config.Entrypoint}}'
docker image inspect pypto3-dev-env:cann9 --format '{{json .Config.Cmd}}'
```

Expected:

```text
null
["bash"]
```

### 4) Start the container (NPU-enabled)

Recommended mode (with source mount):

```bash
docker run --rm -it --privileged --ipc=host \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /dev:/dev \
  -v "$HOME/hw-native-sys":/workspace/hw-native-sys \
  -w /workspace/hw-native-sys/pypto \
  pypto3-dev-env:cann9
```

Optional mode (without source mount; immutable image validation only):

```bash
docker run --rm -it --privileged --ipc=host \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /dev:/dev \
  pypto3-dev-env:cann9
```

Do not mount `/usr/local/Ascend` from host. The image already contains CANN 9.0.0.

---

## Inside the container: first-time sanity checks

Run these once per session:

```bash
python -c "import pypto; print('pypto ok')"
echo "$ASCEND_HOME_PATH"
echo "$SIMPLER_ROOT"
echo "$PTO_ISA_ROOT"
which ptoas
```

Expected:
- `pypto ok`
- `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0`
- `SIMPLER_ROOT=/workspace/hw-native-sys/simpler`
- `PTO_ISA_ROOT=/opt/pto-isa`
- `ptoas` found in PATH

If you are actively changing Python code in mounted `pypto`, refresh editable install:

```bash
pip install -e ".[dev]"
```

---

## Recommended test workflow

### A) Fast health check (unit tests)

```bash
pytest tests/ut -v
```

### B) Single-device runtime smoke test

```bash
pytest tests/st/runtime/test_matmul.py -v --device=0
```

### C) Full system tests (non-distributed)

```bash
pytest tests/st -v --device="0,1,2,3" --precompile-workers=128 --ignore=tests/st/distributed
```

### D) Distributed system tests

```bash
pytest tests/st/distributed -v --device="0,1,2,3"
```

### E) Focused PTOAS checks

```bash
pytest tests/st/codegen/test_paged_attention.py -v --device=0
```

---

## How to verify NPU is actually being used

From another host terminal:

```bash
watch -n 0.5 npu-smi info
```

Then run runtime tests in the container (for example B, C, or D above). You should see activity while kernels execute.

---

## Common flags explained

- `--device="0,1,2,3"`: expose/select multiple NPU devices for tests.
- `--precompile-workers=128`: parallel precompile workers for system-test setup.
- PTO-ISA commit is auto-derived from `runtime/pto_isa.pin`; use `--pto-isa-commit=<sha>` only to override.
- `--ignore=tests/st/distributed`: run non-distributed system tests only.

Simulator mode (`a2a3sim`) is not supported in this image.

---

## Common issues and fixes

- **Image uses old behavior after Dockerfile updates**
  - Fix:
    ```bash
    docker build --no-cache -f "Dockerfile.server.cann:9.0" -t pypto3-dev-env:cann9 .
    docker image inspect pypto3-dev-env:cann9 --format '{{json .Config.Cmd}}'
    ```
    Confirm:
    ```text
    ["bash"]
    ```

- **`ptoas` binary not found**
  - Check: `which ptoas` and `echo $PTOAS_ROOT`.
  - Expected root: `/opt/ptoas-bin`.

- **`Simpler runtime is not available`**
  - Check: `echo $SIMPLER_ROOT`.
  - If you mount `~/hw-native-sys`, ensure `~/hw-native-sys/simpler` exists on host.

- **No visible NPU activity**
  - Ensure you are running runtime tests (not only codegen/unit tests).
  - Try: `pytest tests/st/runtime/test_matmul.py -v --device=0`
  - Monitor on host: `watch -n 0.5 npu-smi info`

- **CANN symbol/runtime mismatch errors**
  - Cause is often host CANN shadowing image CANN.
  - Do not mount `/usr/local/Ascend` from host.
  - Mount only `/usr/local/Ascend/driver` as shown above.

- **Editable install fails with `dubious ownership`**
  - The image preconfigures safe directories.
  - Temporary fix if needed:
    ```bash
    git config --global --add safe.directory /workspace/hw-native-sys/pypto
    ```

---