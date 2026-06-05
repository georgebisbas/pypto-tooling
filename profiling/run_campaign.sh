#!/usr/bin/env bash
# -*- mode: shell-script -*-
# run_campaign.sh — full strong-scaling mesh allreduce campaign (P=2,4,8)
#
# Usage (auto-detects local vs Docker layout):
#   cd pypto-tooling/profiling
#   bash run_campaign.sh
#   bash run_campaign.sh --p-values 2 --timed-rounds 2 --warmup-rounds 0
#
# Override paths:
#   PYPTO_ROOT=/custom/pypto SIMPLER_ROOT=/custom/simpler bash run_campaign.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- auto-detect environment ---
TOOLING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"          # pypto-tooling/
WORKSPACE_ROOT="$(dirname "$TOOLING_ROOT")"            # parent of pypto-tooling

if [ -z "${PYPTO_ROOT:-}" ]; then
    if [ -d "$WORKSPACE_ROOT/pypto" ]; then
        PYPTO_ROOT="$WORKSPACE_ROOT/pypto"
    elif [ -d "/opt/pypto" ]; then
        PYPTO_ROOT="/opt/pypto"
    else
        echo "ERROR: cannot find pypto. Set PYPTO_ROOT=/path/to/pypto"
        exit 1
    fi
fi

if [ -z "${SIMPLER_ROOT:-}" ]; then
    # Docker: simpler is a submodule at pypto/runtime
    if [ -d "$PYPTO_ROOT/runtime" ] && [ -f "$PYPTO_ROOT/runtime/pyproject.toml" ]; then
        SIMPLER_ROOT="$PYPTO_ROOT/runtime"
    elif [ -d "$WORKSPACE_ROOT/simpler" ]; then
        SIMPLER_ROOT="$WORKSPACE_ROOT/simpler"
    else
        echo "ERROR: cannot find simpler. Set SIMPLER_ROOT=/path/to/simpler"
        exit 1
    fi
fi

export PYPTO_ROOT SIMPLER_ROOT
echo "PYPTO_ROOT=$PYPTO_ROOT"
echo "SIMPLER_ROOT=$SIMPLER_ROOT"

usage() {
    cat <<'EOF'
Usage: bash run_campaign.sh [campaign] [options]

Options:
  --campaign NAME          Campaign name (default: strong_scaling_mesh)
  --p-values CSV           Comma-separated P values (default: 2,4,8)
  --count N                Override case count passed to run_sweep.py
  --warmup-rounds N        Override warmup rounds passed to run_sweep.py
  --timed-rounds N         Override timed rounds passed to run_sweep.py
  -h, --help               Show this help
EOF
}

CAMPAIGN="strong_scaling_mesh"
P_VALUES_CSV="2,4,8"
COUNT_OVERRIDE=""
WARMUP_OVERRIDE=""
TIMED_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign)
            CAMPAIGN="$2"
            shift 2
            ;;
        --p-values)
            P_VALUES_CSV="$2"
            shift 2
            ;;
        --count)
            COUNT_OVERRIDE="$2"
            shift 2
            ;;
        --warmup-rounds)
            WARMUP_OVERRIDE="$2"
            shift 2
            ;;
        --timed-rounds)
            TIMED_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ "$CAMPAIGN" == "strong_scaling_mesh" ]]; then
                CAMPAIGN="$1"
                shift
            else
                echo "ERROR: unknown argument: $1"
                usage
                exit 1
            fi
            ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/campaigns/${CAMPAIGN}/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

IFS=',' read -r -a P_VALUES <<< "$P_VALUES_CSV"
RESULTS_FILES=()

RUN_SWEEP_EXTRA_ARGS=()
if [[ -n "$COUNT_OVERRIDE" ]]; then
    RUN_SWEEP_EXTRA_ARGS+=(--count "$COUNT_OVERRIDE")
fi
if [[ -n "$WARMUP_OVERRIDE" ]]; then
    RUN_SWEEP_EXTRA_ARGS+=(--warmup-rounds "$WARMUP_OVERRIDE")
fi
if [[ -n "$TIMED_OVERRIDE" ]]; then
    RUN_SWEEP_EXTRA_ARGS+=(--timed-rounds "$TIMED_OVERRIDE")
fi

echo "============================================"
echo " Campaign: ${CAMPAIGN}"
echo " Run dir:  ${RUN_DIR}"
echo " Ranks:    ${P_VALUES[*]}"
if [[ ${#RUN_SWEEP_EXTRA_ARGS[@]} -gt 0 ]]; then
    echo " Overrides: ${RUN_SWEEP_EXTRA_ARGS[*]}"
fi
echo "============================================"

for P in "${P_VALUES[@]}"; do
    CASE_FILE="collectives/cases/mesh_p${P}_n256_fp32.json"
    OUT_FILE="${RUN_DIR}/results_p${P}.json"

    if [ ! -f "$CASE_FILE" ]; then
        echo "ERROR: missing case file $CASE_FILE"
        exit 1
    fi

    echo ""
    echo "--- P=${P} ---"
    python3 -m collectives.run_sweep pair-mesh \
        --case-file "$CASE_FILE" \
        --stacks hccl,simpler,pypto \
        --campaign "$CAMPAIGN" \
        "${RUN_SWEEP_EXTRA_ARGS[@]}" \
        --out "$OUT_FILE" || {
        echo "FATAL: P=${P} failed (exit $?), stopping."
        exit 1
    }
    RESULTS_FILES+=("$OUT_FILE")
done

# Merge all results into one summary
MERGED="${RUN_DIR}/results.json"
echo ""
echo "--- merging results ---"
python3 -c "
import json, sys
from pathlib import Path

files = sys.argv[1:]
merged = {'campaign': '${CAMPAIGN}', 'runs': [], 'cases': []}
for f in files:
    if f == '--':
        continue
    data = json.loads(Path(f).read_text())
    merged['runs'].extend(data.get('runs', []))
    if 'case' in data:
        merged['cases'].append(data['case'])
merged['runs'].sort(key=lambda r: (r.get('p', 0), r.get('stack', '')))
Path('${MERGED}').write_text(json.dumps(merged, indent=2) + '\n')
print(f'  merged {len(merged[\"runs\"])} runs from {len(files)} cases → ${MERGED}')
" -- "${RESULTS_FILES[@]}"

# Summarize
echo ""
echo "--- summary ---"
python3 -m collectives.summarize --run-dir "$RUN_DIR" --emit-report

# Plot
echo ""
echo "--- figures ---"
python3 -m collectives.plot_figures --run-dir "$RUN_DIR" --figures strong_scaling_t_total,paired_stack_ratio,phase_breakdown,compile_breakdown

echo ""
echo "============================================"
echo " Done: ${RUN_DIR}"
echo "   results.json  — merged runs"
echo "   reports/summary.md"
echo "   figures/paired_stack_ratio.png"
echo "   figures/phase_breakdown.png"
echo "   figures/compile_breakdown.png"
echo "============================================"
