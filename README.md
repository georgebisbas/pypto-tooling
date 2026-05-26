# pypto-tooling

Dockerfiles and runbooks for PyPTO and simpler development on Ascend 910B.

## Repository Contents

- [README.md](README.md): Repository index and quick usage.
- [Dockerfile.hw-native-sys.cann9.0](Dockerfile.hw-native-sys.cann9.0): Standalone PyPTO image that clones all sources from GitHub at build time.
- [Dockerfile.server.cann:9.0](Dockerfile.server.cann:9.0): Server/dev image that uses a local pypto build context.
- [Dockerfile.simpler.cann9.0](Dockerfile.simpler.cann9.0): Standalone simpler image with pinned commit support and HCCL-safe runtime defaults.
- [docker-entrypoint-cann.sh](docker-entrypoint-cann.sh): Runtime helper for workspace/runtime symlink handling.
- [bz910b-reproduce.md](bz910b-reproduce.md): Reproduction and test workflow guide.
- [dockerfile_skills/SKILL.md](dockerfile_skills/SKILL.md): Internal Dockerfile construction/debugging notes.
- [dockerfile_skills/issue_0.md](dockerfile_skills/issue_0.md): Detailed issue log for comm_alloc_windows/HCCL environment mismatch.

## Purpose of Each File

- [README.md](README.md): Top-level index of the repository, with build commands and runtime guidance.
- [Dockerfile.hw-native-sys.cann9.0](Dockerfile.hw-native-sys.cann9.0): Build a standalone PyPTO image by cloning repositories during build; best for reproducible CI-like setup without local source dependencies.
- [Dockerfile.server.cann:9.0](Dockerfile.server.cann:9.0): Build a server/dev image from a local hw-native-sys workspace where pypto is copied from build context; best for local iteration.
- [Dockerfile.simpler.cann9.0](Dockerfile.simpler.cann9.0): Build a standalone simpler-only image for runtime and worker validation when pypto is not required.
- [docker-entrypoint-cann.sh](docker-entrypoint-cann.sh): Runtime helper script that normalizes runtime layout (workspace/runtime symlink behavior) before launching the container command.
- [bz910b-reproduce.md](bz910b-reproduce.md): Operator runbook for reproducing tests and validating NPU execution on Ascend 910B hosts.
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
  --build-arg PYPTO_COMMIT=024f4a30 \
  --build-arg PTO_ISA_COMMIT=2c607938 \
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
  --build-arg SIMPLER_COMMIT=896bd025 \
  --build-arg PTO_ISA_COMMIT=50d9c806 \
  -t simpler-cann9 \
  - < Dockerfile.simpler.cann9.0
```

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

The same runtime rule applies to `simpler-cann9`: include `--pid=host` for distributed/HCCL workflows.

Mount only the host driver path at runtime:

```text
/usr/local/Ascend/driver
```

Do not mount /usr/local/Ascend from host, because it can shadow the image's baked CANN version.