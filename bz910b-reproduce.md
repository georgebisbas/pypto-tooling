## PyPTO 3.0 (hw-native-sys) BZ CI Test Tutorial

Use this when you want to run PyPTO tests on `hng-atlas01` with the prebuilt image.

### 1) SSH to the server

```bash
ssh -i ~/.ssh/id_ *server-ip*
```

### 2) Go to the repository

```bash
cd ~/hw-native-sys/pypto
```

### 3) (Optional) Rebuild image

Skip this if `pypto-dev-env:latest` already exists and you do not need Dockerfile changes.

```bash
docker build -f Dockerfile.server -t pypto-dev-env:latest .
```

### 4) Start the container (NPU-enabled)

```bash
docker run --rm -it --privileged -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro -v /usr/local/Ascend:/usr/local/Ascend:ro -v /dev:/dev -v "$PWD":/workspace/pypto -w /workspace/pypto pypto-dev-env:latest
```

---

## Inside the container: first-time sanity checks

Run these once per session:

```bash
python -c "import pypto; print('pypto ok')"
echo "$ASCEND_HOME_PATH"
echo "$SIMPLER_ROOT"
which ptoas
```

Expected:
- `pypto ok`
- `ASCEND_HOME_PATH` set
- `SIMPLER_ROOT=/opt/simpler`
- `ptoas` found in PATH

If you are actively changing code in mounted repo, refresh editable install:
```bash
pip install -e ".[dev]"
```
(If only running prebuilt code from the image, this is usually not necessary.)

---

## Recommended test workflow

### A) Fast health check (unit tests)

```bash
pytest tests/ut -v
```

### B) Compile/codegen-only system checks (no runtime execution)

```bash
pytest tests/st -v --forked --codegen-only
```

### C) Real runtime on NPU (device 0)

```bash
pytest tests/st/runtime/test_matmul.py -v --forked --platform=a2a3 --device=0
```

### D) Full system tests on NPU

```bash
pytest tests/st -v --forked --platform=a2a3 --device=0
```

### E) Focused PTOAS checks (common failure point)

```bash
pytest tests/st/codegen/test_paged_attention.py -v --forked --platform=a2a3 --device=0
```

---

## How to verify NPU is actually being used

From another host terminal:

```bash
watch -n 0.5 npu-smi info
```

Then run runtime tests in container (example C or D above).  
You should see activity while kernels execute.

---

## Common flags explained

- `--forked`: runs each test in isolated subprocess (important for `tests/st` stability).
- `--platform=a2a3`: real NPU mode.
- `--platform=a2a3sim`: simulator mode.
- `--device=0`: selects NPU device index.
- `--codegen-only`: compile/generate only, skip runtime execution.

---

## Common issues and fixes

- **`ptoas binary not found`**
  - In this image it should already be installed.
  - Check: `which ptoas` and `echo $PTOAS_ROOT`.

- **`Simpler runtime is not available`**
  - In this image it should already be configured.
  - Check: `echo $SIMPLER_ROOT` (should be `/opt/simpler`).

- **No visible NPU activity**
  - Ensure you are running runtime tests (not only `--codegen-only`).
  - Run: `pytest tests/st/runtime/test_matmul.py -v --forked --platform=a2a3 --device=0`
  - Monitor on host: `watch -n 0.5 npu-smi info`

- **Editable install fails with `dubious ownership`**
  - Should already be handled in image.
  - Temporary fix if needed:
    ```bash
    git config --global --add safe.directory /workspace/pypto
    ```

---
