# pypto-tooling

Dockerfiles and runbooks for PyPTO and simpler development on Ascend 910B.

## Repository Contents

- [README.md](README.md): Repository index and quick usage.
- [Dockerfile.server.cann:9.0](Dockerfile.server.cann:9.0): Server/dev image that uses a local pypto build context.
- [Dockerfile.hw-native-sys.cann9.0](Dockerfile.hw-native-sys.cann9.0): Standalone hw-native-sys image that clones pypto from GitHub.
- [Dockerfile.simpler.cann9.0](Dockerfile.simpler.cann9.0): Standalone simpler image that clones simpler from GitHub.
- [docker-entrypoint-cann.sh](docker-entrypoint-cann.sh): Runtime helper for workspace/runtime symlink handling.
- [bz910b-reproduce.md](bz910b-reproduce.md): Reproduction and test workflow guide.
- [dockerfile_skills/SKILL.md](dockerfile_skills/SKILL.md): Internal Dockerfile construction/debugging notes.
- [dockerfile_skills/issue_0.md](dockerfile_skills/issue_0.md): Detailed issue log for comm_alloc_windows/HCCL environment mismatch.

## Purpose of Each File

- [README.md](README.md): Top-level index of the repository, with build commands and runtime guidance.
- [Dockerfile.server.cann:9.0](Dockerfile.server.cann:9.0): Build a server/dev image from a local hw-native-sys workspace where pypto is copied from build context; best for local iteration.
- [Dockerfile.hw-native-sys.cann9.0](Dockerfile.hw-native-sys.cann9.0): Build a standalone PyPTO image by cloning repositories during build; best for reproducible CI-like setup without local source dependencies.
- [Dockerfile.simpler.cann9.0](Dockerfile.simpler.cann9.0): Build a standalone simpler-only image for runtime and worker validation when pypto is not required.
- [docker-entrypoint-cann.sh](docker-entrypoint-cann.sh): Runtime helper script that normalizes runtime layout (workspace/runtime symlink behavior) before launching the container command.
- [bz910b-reproduce.md](bz910b-reproduce.md): Operator runbook for reproducing tests and validating NPU execution on Ascend 910B hosts.
- [dockerfile_skills/SKILL.md](dockerfile_skills/SKILL.md): Internal engineering notes and patterns for constructing/debugging these Dockerfiles.
- [dockerfile_skills/issue_0.md](dockerfile_skills/issue_0.md): Historical incident report and root-cause details for the HCCL IPC/driver compatibility issue.

## Build Commands

Build server/dev image (requires local pypto in build context):

```bash
docker build --no-cache -f "Dockerfile.server.cann:9.0" -t pypto3-dev-env:cann9 .
```

Build standalone hw-native-sys image (stdin Dockerfile):

```bash
docker build -t pypto3-hw-native-sys:cann9 - < Dockerfile.hw-native-sys.cann9.0
```

Build standalone simpler image (stdin Dockerfile):

```bash
docker build -t simpler-cann9 - < Dockerfile.simpler.cann9.0
```

## Runtime Note

Mount only the host driver path at runtime:

```text
/usr/local/Ascend/driver
```

Do not mount /usr/local/Ascend from host, because it can shadow the image's baked CANN version.