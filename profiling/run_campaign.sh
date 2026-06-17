#!/usr/bin/env bash
# -*- mode: shell-script -*-
# run_campaign.sh — full collective benchmark campaign
#
# Usage:
#   cd pypto-tooling/profiling
#
#   # Strong scaling: one variant, sweep P
#   bash run_campaign.sh --variant mesh --p-values 2,4,8
#   bash run_campaign.sh --variant ring --p-values 2,4,8 --count 65536
#
#   # Single P, cross-variant: mesh vs ring vs HCCL
#   bash run_campaign.sh --mode cross-variant --p-values 4 --count 65536
#
#   # Quick smoke test (small payload, minimal rounds)
#   bash run_campaign.sh --p-values 2 --timed-rounds 2 --warmup-rounds 0
#
#   # Full sweep (all variants, all sizes)
#   bash run_campaign.sh --mode full-sweep
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
Usage: bash run_campaign.sh [options]

Options:
  --mode MODE              Benchmark mode (default: strong-scaling)
                             strong-scaling  — one variant, sweep P
                             cross-variant   — two variants at same P/count
                             full-sweep      — all variants × all sizes
  --variant VARIANT        Algorithm variant: mesh, ring, hccl (default: mesh)
  --variants VA,VB         Two variants for cross-variant mode (default: mesh,ring)
  --p-values CSV           Comma-separated P values (default: 2,4,8)
  --count N                Override case payload element count
  --dtype TYPE             fp32 or fp16 (default: fp32)
  --stacks CSV             Stacks to run: simpler,pypto,hccl (default: hccl,simpler,pypto)
  --warmup-rounds N        Override warmup rounds
  --timed-rounds N         Override timed rounds
  --campaign NAME          Campaign name (default: auto)
  -h, --help               Show this help
EOF
}

MODE="strong-scaling"
VARIANT="mesh"
VARIANTS_CSV="mesh,ring"
P_VALUES_CSV="2,4,8"
COUNT_OVERRIDE=""
DTYPE="fp32"
STACKS="hccl,simpler,pypto"
WARMUP_OVERRIDE=""
TIMED_OVERRIDE=""
CAMPAIGN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --variants)
            VARIANTS_CSV="$2"
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
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --stacks)
            STACKS="$2"
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
        --campaign)
            CAMPAIGN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# Auto campaign name
if [[ -z "$CAMPAIGN" ]]; then
    if [[ "$MODE" == "cross-variant" ]]; then
        CAMPAIGN="cross_variant"
    else
        CAMPAIGN="${MODE}_${VARIANT}"
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/campaigns/${CAMPAIGN}/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

IFS=',' read -r -a P_VALUES <<< "$P_VALUES_CSV"
IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"
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
echo " Campaign:  ${CAMPAIGN}"
echo " Mode:      ${MODE}"
echo " Run dir:   ${RUN_DIR}"
echo " Variant(s): ${VARIANTS[*]}"
echo " P values:  ${P_VALUES[*]}"
echo " Stacks:    ${STACKS}"
echo " Dtype:     ${DTYPE}"
if [[ -n "$COUNT_OVERRIDE" ]]; then
    echo " Count:     ${COUNT_OVERRIDE}"
fi
if [[ ${#RUN_SWEEP_EXTRA_ARGS[@]} -gt 0 ]]; then
    echo " Overrides:  ${RUN_SWEEP_EXTRA_ARGS[*]}"
fi
echo "============================================"

# ── Helper: resolve case file path ──────────────────────────────────────
_resolve_case() {
    local variant="$1" p="$2"
    # Search for case files matching variant, P, dtype.
    # Old naming: mesh_p2_n256_fp32.json
    # New naming: mesh_p2_count256_fp32_a2a3_d0-1.json
    local pattern_new="collectives/cases/${variant}_p${p}_count*_${DTYPE}_*.json"
    local pattern_old="collectives/cases/${variant}_p${p}_n*_${DTYPE}.json"
    local matches=()
    for f in $pattern_new $pattern_old; do
        if [[ -f "$f" ]]; then
            matches+=("$f")
        fi
    done
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo ""
        return 1
    fi
    # Prefer smaller count files (they load faster; --count overrides later)
    echo "${matches[0]}"
}

# ── Mode: strong-scaling ────────────────────────────────────────────────
if [[ "$MODE" == "strong-scaling" ]]; then
    for P in "${P_VALUES[@]}"; do
        CASE_FILE="$(_resolve_case "$VARIANT" "$P")"
        if [[ -z "$CASE_FILE" ]]; then
            echo "ERROR: no case file for variant=${VARIANT} P=${P} dtype=${DTYPE}"
            echo "  Run: python collectives/cases/generate.py --variant ${VARIANT} --p-values ${P}"
            exit 1
        fi
        OUT_FILE="${RUN_DIR}/results_p${P}.json"

        echo ""
        echo "--- P=${P} (${VARIANT}) ---"
        python3 -m collectives.run_sweep pair-mesh \
            --case-file "$CASE_FILE" \
            --stacks "$STACKS" \
            --campaign "$CAMPAIGN" \
            "${RUN_SWEEP_EXTRA_ARGS[@]}" \
            --out "$OUT_FILE" || {
            echo "FATAL: P=${P} failed (exit $?), stopping."
            exit 1
        }
        RESULTS_FILES+=("$OUT_FILE")
    done

# ── Mode: cross-variant ──────────────────────────────────────────────────
elif [[ "$MODE" == "cross-variant" ]]; then
    if [[ ${#VARIANTS[@]} -ne 2 ]]; then
        echo "ERROR: cross-variant mode needs exactly 2 variants (got ${#VARIANTS[@]}: ${VARIANTS[*]})"
        echo "  Use: --variants mesh,ring"
        exit 1
    fi
    VA="${VARIANTS[0]}"
    VB="${VARIANTS[1]}"

    for P in "${P_VALUES[@]}"; do
        CASE_A="$(_resolve_case "$VA" "$P")"
        CASE_B="$(_resolve_case "$VB" "$P")"
        if [[ -z "$CASE_A" ]] || [[ -z "$CASE_B" ]]; then
            echo "ERROR: missing case files for ${VA} vs ${VB} P=${P}"
            echo "  Run: python collectives/cases/generate.py"
            exit 1
        fi
        OUT_FILE="${RUN_DIR}/results_${VA}_vs_${VB}_p${P}.json"

        echo ""
        echo "--- ${VA} vs ${VB}  P=${P} ---"
        python3 -m collectives.run_sweep cross-variant \
            --case-file-a "$CASE_A" \
            --case-file-b "$CASE_B" \
            --stacks "$STACKS" \
            --campaign "$CAMPAIGN" \
            "${RUN_SWEEP_EXTRA_ARGS[@]}" \
            --out "$OUT_FILE" || {
            echo "FATAL: cross-variant P=${P} failed (exit $?), stopping."
            exit 1
        }
        RESULTS_FILES+=("$OUT_FILE")
    done

# ── Mode: full-sweep ─────────────────────────────────────────────────────
elif [[ "$MODE" == "full-sweep" ]]; then
    # Generate all cases if needed
    if ! ls collectives/cases/mesh_p2_n4096_*.json &>/dev/null; then
        echo "--- generating case files ---"
        python3 collectives/cases/generate.py || {
            echo "FATAL: case generation failed"
            exit 1
        }
    fi

    ALL_VARIANTS=("mesh" "ring" "hccl")
    for P in "${P_VALUES[@]}"; do
        for VARIANT in "${ALL_VARIANTS[@]}"; do
            if [[ "$VARIANT" == "hccl" ]]; then
                # HCCL baseline: use mesh case file (same P/count contract)
                CASE_FILE="$(_resolve_case "mesh" "$P")"
            else
                CASE_FILE="$(_resolve_case "$VARIANT" "$P")"
            fi
            if [[ -z "$CASE_FILE" ]]; then
                echo "WARNING: skipping variant=${VARIANT} P=${P} — no case file"
                continue
            fi
            OUT_FILE="${RUN_DIR}/results_${VARIANT}_p${P}.json"

            echo ""
            echo "--- ${VARIANT} P=${P} ---"
            python3 -m collectives.run_sweep pair-mesh \
                --case-file "$CASE_FILE" \
                --stacks "$STACKS" \
                --campaign "$CAMPAIGN" \
                "${RUN_SWEEP_EXTRA_ARGS[@]}" \
                --out "$OUT_FILE" || {
                echo "WARNING: ${VARIANT} P=${P} failed (exit $?) — continuing"
            }
            RESULTS_FILES+=("$OUT_FILE")
        done
    done
fi

# ── Merge all results ────────────────────────────────────────────────────
MERGED="${RUN_DIR}/results.json"
echo ""
echo "--- merging ${#RESULTS_FILES[@]} result files ---"
if [[ ${#RESULTS_FILES[@]} -eq 1 ]]; then
    cp "${RESULTS_FILES[0]}" "$MERGED"
    echo "  single file → ${MERGED}"
else
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
merged['runs'].sort(key=lambda r: (r.get('p', 0), r.get('variant', ''), r.get('stack', '')))
Path('${MERGED}').write_text(json.dumps(merged, indent=2) + '\n')
print(f'  merged {len(merged[\"runs\"])} runs from {len(files)} files → ${MERGED}')
" -- "${RESULTS_FILES[@]}"
fi

# ── Summarize ─────────────────────────────────────────────────────────────
echo ""
echo "--- summary ---"
python3 -m collectives.summarize --run-dir "$RUN_DIR" --emit-report 2>/dev/null || \
    echo "  (summarize skipped — no summarizer or no data)"

# ── Plot ──────────────────────────────────────────────────────────────────
echo ""
echo "--- figures ---"
python3 -m collectives.plot_figures --run-dir "$RUN_DIR" \
    --figures strong_scaling_t_total,paired_stack_ratio,phase_breakdown,compile_breakdown 2>/dev/null || \
    echo "  (plot skipped — no plotter or no data)"

echo ""
echo "============================================"
echo " Done: ${RUN_DIR}"
echo "   results.json  — merged runs"
echo "============================================"
