#!/usr/bin/env bash
# -*- mode: shell-script -*-
# run_campaign.sh — full strong-scaling mesh allreduce campaign (P=2,4,8)
#
# Usage (auto-detects local vs Docker layout):
#   cd pypto-tooling/profiling
#   bash run_campaign.sh
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

CAMPAIGN="${1:-strong_scaling_mesh}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/campaigns/${CAMPAIGN}/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

P_VALUES=(2 4 8)
RESULTS_FILES=()

echo "============================================"
echo " Campaign: ${CAMPAIGN}"
echo " Run dir:  ${RUN_DIR}"
echo " Ranks:    ${P_VALUES[*]}"
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
