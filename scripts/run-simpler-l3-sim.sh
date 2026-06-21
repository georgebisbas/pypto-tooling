#!/usr/bin/env bash
# Run simpler L3 example STs on a2a3sim (mirrors simpler CI st-sim-a2a3 flags).
#
# Usage (inside simpler-hw-native-sys:sim container):
#   run-simpler-l3-sim.sh [distributed|all|allreduce]
#
# Environment overrides:
#   SIMPLER_DIR       default /opt/simpler
#   PTO_ISA_COMMIT    default ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8

set -euo pipefail

SIMPLER_DIR="${SIMPLER_DIR:-/opt/simpler}"
PTO_ISA_COMMIT="${PTO_ISA_COMMIT:-ddafa8da9c760ecd13fe9fe2833d6ee55fb20bd8}"
SCOPE="${1:-distributed}"

cd "${SIMPLER_DIR}"

PYTEST_COMMON=(
  pytest
  --platform a2a3sim
  --device 0-15
  -v
  --pto-session-timeout 600
  "--pto-isa-commit=${PTO_ISA_COMMIT}"
  --clone-protocol https
  --require-pto-isa
)

case "${SCOPE}" in
  allreduce)
    TARGETS=(examples/workers/l3/allreduce_distributed)
    ;;
  distributed)
    TARGETS=(
      examples/workers/l3/allreduce_distributed
      examples/workers/l3/allgather_distributed
      examples/workers/l3/reduce_scatter_distributed
      examples/workers/l3/broadcast_distributed
      examples/workers/l3/all_to_all_distributed
      examples/workers/l3/domain_rank_map
      examples/workers/l3/dual_domain_overlap
      examples/workers/l3/ffn_tp_parallel
      examples/workers/l3/ep_dispatch_combine
    )
    ;;
  all)
    TARGETS=(examples/workers/l3)
    ;;
  *)
    echo "Usage: $0 [distributed|all|allreduce]" >&2
    exit 2
    ;;
esac

echo "==> simpler L3 sim: scope=${SCOPE} dir=${SIMPLER_DIR} pto-isa=${PTO_ISA_COMMIT}"
exec "${PYTEST_COMMON[@]}" "${TARGETS[@]}"
