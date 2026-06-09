# PyPTO Dev Container Workflow with VS Code

How to edit code in VS Code inside a container and run everything — unit tests,
single-device system tests, and distributed/HCCL tests — on real NPUs.

## Architecture: one container

```
┌──────────────────────────────────────────────┐
│ pypto-dev                                     │
│ docker run -it --pid=host                     │  ← Attach VS Code HERE
│ Edit, explore, run ALL tests (incl. HCCL)     │     Fast attach, ~3 seconds
└──────────────────────────────────────────────┘
```

**Why a single container?** HCCL requires `--pid=host` (Ascend IPC validates
host PIDs against the kernel's pid namespace). Previously this made VS Code's
`userEnvProbe` hang for 10+ seconds because the Ascend base image sourced
`set_env.sh` from `/etc/profile` — the Dockerfile now strips that, so
`bash -i -l` completes in under 2 seconds. VS Code attaches cleanly with
`--pid=host`.

---

## 1. Build the image

```bash
# On host (hng-atlas01 or hng-atlas03)
cd ~/pypto-tooling
git pull

# Build (pin a specific pypto commit via build-arg, or use default = origin/main)
docker build -t pypto3-hw-native-sys:cann9 - < Dockerfile.hw-native-sys.cann9.0

# Verify
docker images pypto3-hw-native-sys:cann9
```

---

## 2. Start the container

```bash
docker rm -f pypto-dev 2>/dev/null || true
docker run --rm -it --name pypto-dev --privileged --ipc=host --pid=host \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /dev:/dev \
    pypto3-hw-native-sys:cann9
```

---

## 3. Attach VS Code

1. In VS Code: `Ctrl+Shift+P` → **"Dev Containers: Attach to Running Container"**
2. Select `/pypto-dev`
3. New VS Code window opens inside the container at `/opt/pypto`
4. Attach completes in ~3 seconds
5. Open integrated terminal → you're at `root@...:/opt/pypto#`

---

## 4. Daily workflow

Everything runs directly in the VS Code terminal — no `docker exec` needed.

### Health check

```bash
npu-smi info
python -c "import pypto; print('pypto ok')"
```

### Unit tests (CPU only, no NPU)

```bash
pytest tests/ut -n auto --maxprocesses 8 -v
```

### Codegen tests (CPU only)

```bash
pytest tests/st/codegen -v --codegen-only \
    --ignore=tests/st/codegen/torch/test_torch_codegen_paged_attention.py
```

### System tests (single-device)

```bash
# Full suite on one working device
pytest tests/st/ -v --device=1 --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24 \
    --ignore=tests/st/distributed --ignore=tests/st/codegen

# Swimlane
pytest tests/st/runtime/framework_and_models/test_perf_swimlane.py \
    -v --device=1 --platform=a2a3 --enable-l2-swimlane --forked \
    --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24

# Dump-tensor (older pto-isa pin)
pytest tests/st/runtime/framework_and_models/test_dump_tag.py \
    -v --device=1 --platform=a2a3 --dump-tensor --pto-isa-commit=2c607938 --forked
```

### Distributed / HCCL tests

```bash
# Full distributed suite
pytest tests/st/distributed/ -v --device="0,1,2,3" \
    --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24 \
    --ignore=tests/st/distributed/test_l2_multi_orch.py

# L2 multi-orch (isolated — Worker level=2 poisons level=3)
pytest tests/st/distributed/test_l2_multi_orch.py -v --device="0,1,2,3" \
    --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24

# HCCL UT + L3 examples
pytest tests/ut/py/test_worker/test_dynamic_alloc_hw.py \
    tests/ut/py/test_worker/test_platform_comm.py \
    -m requires_hardware --platform a2a3 --device 0,1,2,3 -v
```

### Simpler runtime tests (from /opt/pypto/runtime)

```bash
cd /opt/pypto/runtime
pytest tests/ut -m "not requires_hardware" -v --clone-protocol https
pytest tests -m requires_hardware --platform a2a3 --device 0,1,2,3 -v \
    --ignore=tests/ut/py/test_worker/test_dynamic_alloc_hw.py \
    --ignore=tests/ut/py/test_worker/test_platform_comm.py
pytest tests/st/a2a3/ -v --platform a2a3 --device 0,1,2,3 \
    --pto-session-timeout 600 --clone-protocol https --require-pto-isa \
    --pto-isa-commit 016396b57e2c17093f1194e6acd89bb112b0ab24
```

---

## 5. After editing code (rebuild)

```bash
# Full pypto rebuild
pip install --no-build-isolation -v ".[dev]"

# Simpler-only rebuild
pip install --no-build-isolation ./runtime
```

---

## 6. Cleanup

```bash
exit  # inside pypto-dev (--rm auto-removes it)
```

---

## 7. Troubleshooting

### VS Code hangs on attach (>10 seconds)

You may have an older image that still sources `set_env.sh` from `/etc/profile`.
Rebuild from the latest Dockerfile:

```bash
docker build -t pypto3-hw-native-sys:cann9 --no-cache - < Dockerfile.hw-native-sys.cann9.0
```

Verify the fix inside the container:

```bash
time bash -i -l -c 'exit'   # should be <2s
```

If it's still slow, check what's left:

```bash
bash -i -l -x -c 'exit' 2>&1 | grep -E '^\+.*set_env' | head -20
```

### Device 0 fails with 507033

Device 0 is dead on some hng-atlas hosts. Use `--device="1,2,3"` or
run `npu-smi info` to find working devices.

### `comm_alloc_windows` fails with 507899

CANN 9.0.0 + driver < 26.0.rc1 mismatch. Requires host driver upgrade.
Non-HCCL tests unaffected.

### Container name conflict

```bash
docker rm -f pypto-dev
```

### "no space left on device" during build

```bash
docker system prune -a
```

---

## 8. Quick reference card

```bash
# ─── BUILD ───
cd ~/pypto-tooling && git pull
docker build -t pypto3-hw-native-sys:cann9 - < Dockerfile.hw-native-sys.cann9.0

# ─── START ───
docker rm -f pypto-dev 2>/dev/null || true
docker run --rm -it --name pypto-dev --privileged --ipc=host --pid=host \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /dev:/dev \
    pypto3-hw-native-sys:cann9

# ─── VS CODE ───
# Ctrl+Shift+P → "Dev Containers: Attach to Running Container" → /pypto-dev

# ─── TESTS (inside VS Code terminal) ───
pytest tests/ut -n auto --maxprocesses 8 -v
pytest tests/st/ -v --device=1 --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24 \
    --ignore=tests/st/distributed --ignore=tests/st/codegen
pytest tests/st/distributed/ -v --device="0,1,2,3" \
    --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24 \
    --ignore=tests/st/distributed/test_l2_multi_orch.py

# ─── CLEANUP ───
exit   # --rm auto-removes
```
    pypto3-hw-native-sys:cann9

# ─── DISTRIBUTED TESTS (from VS Code terminal) ───
docker exec pypto-hccl pytest tests/st/distributed/ -v \
    --device="0,1,2,3" --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24 \
    --ignore=tests/st/distributed/test_l2_multi_orch.py

exit   # --rm auto-removes
```
