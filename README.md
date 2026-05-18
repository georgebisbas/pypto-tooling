# pypto-tooling

Utility repository for building and running the PyPTO server/dev container image on Ascend 910B hardware.

## What this repo contains

- `Dockerfile.server.cann:9.0`: Main image definition based on `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12`.
- `bz910b-reproduce.md`: reproduction/test runbook aligned with the current Dockerfile.

## Current image behavior

- Build context should be the `hw-native-sys` root that contains a `pypto/` directory.
- The image includes CANN 9.0.0 and expects only host driver mount (`/usr/local/Ascend/driver`) at runtime.
- The image command is `bash`.

## Quick start

From your host `hw-native-sys` root:

```bash
docker build --no-cache -f "Dockerfile.server.cann:9.0" -t pypto3-dev-env:cann9 .
```

Run with NPU access (recommended):

```bash
docker run --rm -it --privileged --ipc=host \
	-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
	-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
	-v /dev:/dev \
	-v "$HOME/hw-native-sys":/workspace/hw-native-sys \
	-w /workspace/hw-native-sys/pypto \
	pypto3-dev-env:cann9
```

Sanity check inside container:

```bash
python -c "import pypto; print('pypto ok')"
which ptoas
echo "$ASCEND_HOME_PATH"
echo "$SIMPLER_ROOT"
```

## Notes

- Do not mount host `/usr/local/Ascend` into the container; that can shadow the baked CANN version.
- For the full test workflow and troubleshooting, see `bz910b-reproduce.md`.
- The Dockerfile name includes `:`, so pass it to `docker build -f` in quotes.