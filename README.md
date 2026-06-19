# pypto-tooling

Dockerfiles and runbooks for PyPTO and simpler development on Ascend 910B.

## Repository Contents

- [README.md](README.md): Repository index and quick usage.
- [Dockerfile.hw-native-sys.cann9.0](Dockerfile.hw-native-sys.cann9.0): Standalone PyPTO image that clones all sources from GitHub at build time.
- [Dockerfile.server.cann:9.0](Dockerfile.server.cann:9.0): Server/dev image that uses a local pypto build context.
- [Dockerfile.simpler.cann9.0](Dockerfile.simpler.cann9.0): Standalone simpler image with pinned commit support and HCCL-safe runtime defaults.
- [Dockerfile.pytorch-hccl-tests.cann9.0](Dockerfile.pytorch-hccl-tests.cann9.0): Standalone HCCL micro-benchmark image (`torchrun` + pytorch-hccl-tests fork).
- [Dockerfile.hw-native-sys.sim.ubuntu22.04](Dockerfile.hw-native-sys.sim.ubuntu22.04): Standalone local simulation image for PyPTO (`a2a3sim`/`a5sim`) on x86_64 without NPU devices.
- [docker-entrypoint-cann.sh](docker-entrypoint-cann.sh): Runtime helper for workspace/runtime symlink handling.
- [bz910b-reproduce.md](bz910b-reproduce.md): Reproduction and test workflow guide.
- [profiling/](profiling/): Personal collective benchmark harness for PyPTO vs simpler L3 collectives.
- [dockerfile_skills/SKILL.md](dockerfile_skills/SKILL.md): Internal Dockerfile construction/debugging notes.
- [dockerfile_skills/issue_0.md](dockerfile_skills/issue_0.md): Detailed issue log for comm_alloc_windows/HCCL environment mismatch.

## Purpose of Each File

- [README.md](README.md): Top-level index of the repository, with build commands and runtime guidance.
- [Dockerfile.hw-native-sys.cann9.0](Dockerfile.hw-native-sys.cann9.0): Build a standalone PyPTO image by cloning repositories during build; best for reproducible CI-like setup without local source dependencies.
- [Dockerfile.server.cann:9.0](Dockerfile.server.cann:9.0): Build a server/dev image from a local hw-native-sys workspace where pypto is copied from build context; best for local iteration.
- [Dockerfile.simpler.cann9.0](Dockerfile.simpler.cann9.0): Build a standalone simpler-only image for runtime and worker validation when pypto is not required.
- [Dockerfile.pytorch-hccl-tests.cann9.0](Dockerfile.pytorch-hccl-tests.cann9.0): Build a standalone HCCL benchmark image for baseline latency/bandwidth/allreduce comparisons (not for pypto/simpler composite tests).
- [Dockerfile.hw-native-sys.sim.ubuntu22.04](Dockerfile.hw-native-sys.sim.ubuntu22.04): Build a standalone simulation-only PyPTO image for local development/testing on CPU-hosted simulators (`a2a3sim`, `a5sim`).
- [docker-entrypoint-cann.sh](docker-entrypoint-cann.sh): Runtime helper script that normalizes runtime layout (workspace/runtime symlink behavior) before launching the container command.
- [bz910b-reproduce.md](bz910b-reproduce.md): Operator runbook for reproducing tests and validating NPU execution on Ascend 910B hosts.
- [profiling/](profiling/): Personal harness: equivalence-driven benchmark drivers, artifact bundles, and figure generation for collectives performance analysis.
- [dockerfile_skills/SKILL.md](dockerfile_skills/SKILL.md): Internal engineering notes and patterns for constructing/debugging these Dockerfiles.
- [dockerfile_skills/issue_0.md](dockerfile_skills/issue_0.md): Historical incident report and root-cause details for the HCCL IPC/driver compatibility issue.

## Recommended Image: `Dockerfile.hw-native-sys.cann9.0`

`Dockerfile.hw-native-sys.cann9.0` is now standalone:

- No local source checkout is required.
- Build works from any directory using stdin input.
- Repositories (`pypto`, `pto-isa`) are cloned during build.
- Key behavior is controlled via build args (`CANN_VERSION`, `INSTALL_PREFIX`, `PYPTO_COMMIT`, `PTO_ISA_COMMIT`).

## Build Commands

Build standalone hw-native-sys image (defaults):

```bash
docker build -t pypto3-hw-native-sys:cann9 - < Dockerfile.hw-native-sys.cann9.0
```

Build standalone hw-native-sys image (custom commit/pinning):

```bash
docker build \
  --build-arg PYPTO_COMMIT=a1b066df02fc938f76b3d38b85fc9fbd0e036d07 \
  --build-arg PTO_ISA_COMMIT=016396b57e2c17093f1194e6acd89bb112b0ab24 \
  -t pypto3-hw-native-sys:cann9 \
  - < Dockerfile.hw-native-sys.cann9.0
```

Build server/dev image (requires local pypto in build context):

```bash
docker build --no-cache -f "Dockerfile.server.cann:9.0" -t pypto3-dev-env:cann9 .
```

Build standalone simpler image (defaults):

```bash
docker build -t simpler-cann9 - < Dockerfile.simpler.cann9.0
```

Build standalone simpler image (custom commit/pinning):

```bash
docker build \
  --build-arg SIMPLER_COMMIT=afb5c5a95cf05d5bb346eaef83a318c6f3164971 \
  --build-arg PTO_ISA_COMMIT=ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8 \
  -t simpler-cann9 \
  - < Dockerfile.simpler.cann9.0
```

Build standalone pytorch-hccl-tests image (defaults):

```bash
docker build -t pytorch-hccl-tests:cann9 - < Dockerfile.pytorch-hccl-tests.cann9.0
```

Build standalone pytorch-hccl-tests image (x86 host / custom ref):

```bash
docker build \
  --build-arg PT_HCCL_INSTALL_TARGET=install-npu-x86 \
  -t pytorch-hccl-tests:cann9 \
  - < Dockerfile.pytorch-hccl-tests.cann9.0
```

Build standalone local simulation image (no NPU required):

```bash
docker build -t pypto3-hw-native-sys:sim -f Dockerfile.hw-native-sys.sim.ubuntu22.04 .
```

Build standalone local simulation image (pinned commits):

```bash
docker build --build-arg PYPTO_COMMIT=a1b066df02fc938f76b3d38b85fc9fbd0e036d07 --build-arg PTO_ISA_COMMIT=016396b57e2c17093f1194e6acd89bb112b0ab24 -t pypto3-hw-native-sys:sim -f Dockerfile.hw-native-sys.sim.ubuntu22.04 .
```

## Runtime Notes (Local Simulation)

This mode is for local CPU-hosted simulator workflows only (`a2a3sim`, `a5sim`):

```bash
docker run --rm -it pypto3-hw-native-sys:sim
```

Inside the container:

```bash
cd /opt/pypto
python -c "import pypto; print('pypto ok')"
which ptoas && ptoas --version

# Unit tests
pytest tests/ut -n auto --maxprocesses 8 -v

# Full ST matrix on simulators (long run)
pytest tests/st -v --forked --platform=a2a3sim,a5sim --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24
```

Optional CI-aligned simulator subset:

```bash
pytest tests/st/runtime/ops/test_assemble.py tests/st/runtime/ops/test_mscatter.py tests/st/runtime/framework_and_models/test_qwen3_decode_scope3_mixed.py tests/st/runtime/control_flow/test_dyn_orch_shape.py::TestDynOrchShapeOperations::test_dyn_orch_paged_attention -v --platform=a5sim --forked --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24 -k "not TestMscatter"
pytest tests/st/runtime/cross_core/test_cross_core.py -v --forked --platform=a5sim --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24
pytest tests/st/runtime/cross_core/test_cross_core.py -v --forked --platform=a2a3sim --pto-isa-commit=016396b57e2c17093f1194e6acd89bb112b0ab24
```

Simulation image scope notes:

- No `--privileged`, no `/dev` mount, and no `/usr/local/Ascend/driver` mount are required.
- Do not use this image for onboard NPU execution (`a2a3`, `a5`); use the CANN-based Dockerfiles instead.

## Runtime Notes (Ascend 910B)

Single-device / non-HCCL run:

```bash
docker run --rm -it \
  --privileged \
  --ipc=host \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /dev:/dev \
  pypto3-hw-native-sys:cann9
```

Multi-device distributed / HCCL run (must include `--pid=host`):

```bash
docker run --rm -it \
  --privileged \
  --ipc=host \
  --pid=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /dev:/dev \
  pypto3-hw-native-sys:cann9
```

The same runtime rule applies to `simpler-cann9` and `pytorch-hccl-tests:cann9`: include `--pid=host` for distributed/HCCL workflows.

Before HCCL tests in **pypto** or **simpler** images, export `LD_PRELOAD` in your shell (not baked into the image):

```bash
export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so
```

The **pytorch-hccl-tests** image uses torch HCCL directly and does not need `LD_PRELOAD`; it still requires `--pid=host` for multi-rank NPU collectives.

Mount only the host driver path at runtime:

```text
/usr/local/Ascend/driver
```

Do not mount /usr/local/Ascend from host, because it can shadow the image's baked CANN version.