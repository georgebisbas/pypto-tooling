#!/usr/bin/env python3
"""Generate EquivalenceCase JSON files for a sweep of (variant, P, count, dtype).

Usage:
    python collectives/cases/generate.py                    # generate all cases
    python collectives/cases/generate.py --dry-run           # print without writing
    python collectives/cases/generate.py --variant mesh      # mesh only
    python collectives/cases/generate.py --variant ring      # ring only
    python collectives/cases/generate.py --p-values 2,4      # specific P values
    python collectives/cases/generate.py --min-count 16384   # skip tiny payloads
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from collectives.equivalence import EquivalenceCase

_CASES_DIR = Path(__file__).resolve().parent

# ── Sweep axes ──────────────────────────────────────────────────────────
VARIANTS = ["mesh", "ring"]
P_VALUES = [2, 4, 8]
COUNTS = [
    256,          # 1 KiB   — correctness smoke, overhead-dominant
    4096,         # 16 KiB  — L1-resident, begins to amortize TLOAD overhead
    16384,        # 64 KiB  — UB-sized (~33% of 192 KiB AIV UB), meaningful single-tile BW
    65536,        # 256 KiB — exceeds UB, stresses GM bandwidth
    262144,       # 1 MiB   — production-representative small tensor
    1048576,      # 4 MiB   — production-representative medium tensor
]
DTYPES = ["fp32", "fp16"]

# ── Per-size warmup / timed round heuristics ────────────────────────────
# Small payloads need more rounds for statistical stability; large payloads
# take long enough that fewer rounds suffice.
def _warmup_rounds(count: int) -> int:
    if count <= 4096:
        return 3
    if count <= 65536:
        return 2
    return 1


def _timed_rounds(count: int) -> int:
    if count <= 256:
        return 50    # microsecond scale — need many samples
    if count <= 4096:
        return 20
    if count <= 65536:
        return 10
    if count <= 262144:
        return 5
    return 3         # 4 MiB — 3 samples enough


def generate_cases(
    variants: list[str] | None = None,
    p_values: list[int] | None = None,
    counts: list[int] | None = None,
    dtypes: list[str] | None = None,
    dry_run: bool = False,
    p_max_devices: dict[int, list[int]] | None = None,
) -> list[Path]:
    """Generate EquivalenceCase JSON files. Returns list of written paths."""
    variants = variants or VARIANTS
    p_values = p_values or P_VALUES
    counts = counts or COUNTS
    dtypes = dtypes or DTYPES

    if p_max_devices is None:
        # Default: first P NPUs [0, 1, ..., P-1]
        p_max_devices = {p: list(range(p)) for p in p_values}

    written: list[Path] = []
    skipped: list[str] = []

    for variant in variants:
        for p in p_values:
            for count in counts:
                # Ring constraint: count must be evenly divisible by P
                if variant == "ring" and count % p != 0:
                    skipped.append(f"{variant}_p{p}_count{count}: count % P != 0")
                    continue

                for dtype in dtypes:
                    device_ids = p_max_devices.get(p, list(range(p)))
                    if len(device_ids) != p:
                        skipped.append(
                            f"{variant}_p{p}_count{count}_{dtype}: "
                            f"device_ids length {len(device_ids)} != P={p}"
                        )
                        continue

                    case = EquivalenceCase(
                        variant=variant,
                        p=p,
                        count=count,
                        dtype=dtype,
                        device_ids=list(device_ids),
                        warmup_rounds=_warmup_rounds(count),
                        timed_rounds=_timed_rounds(count),
                    )

                    filename = f"{case.case_id}.json"
                    path = _CASES_DIR / filename

                    if dry_run:
                        print(f"[dry-run] {filename}  size={case.size_tier}  "
                              f"warmup={case.warmup_rounds} timed={case.timed_rounds}")
                    else:
                        path.write_text(
                            json.dumps(case.canonical_dict(), indent=2) + "\n",
                            encoding="utf-8",
                        )

                    written.append(path)

    if skipped:
        print(f"\nSkipped {len(skipped)} combinations (constraint violations):")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate EquivalenceCase JSON files for collective benchmark sweeps"
    )
    parser.add_argument(
        "--variant", choices=VARIANTS, action="append", dest="variants",
        help="Restrict to variant(s). Repeatable. Default: all."
    )
    parser.add_argument(
        "--p-values", type=str, default=None,
        help="Comma-separated P values, e.g. '2,4,8'. Default: 2,4,8."
    )
    parser.add_argument(
        "--min-count", type=int, default=None,
        help="Only generate cases with count >= MIN_COUNT. Useful for skipping 1 KiB smoke."
    )
    parser.add_argument(
        "--max-count", type=int, default=None,
        help="Only generate cases with count <= MAX_COUNT."
    )
    parser.add_argument(
        "--dtype", choices=DTYPES, action="append", dest="dtypes",
        help="Restrict to dtype(s). Repeatable. Default: fp32,fp16."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without writing files."
    )
    args = parser.parse_args(argv)

    # Filter counts
    counts = list(COUNTS)
    if args.min_count is not None:
        counts = [c for c in counts if c >= args.min_count]
    if args.max_count is not None:
        counts = [c for c in counts if c <= args.max_count]

    # Parse P values
    p_values = P_VALUES
    if args.p_values is not None:
        p_values = [int(x.strip()) for x in args.p_values.split(",")]

    # Parse dtypes
    dtypes = args.dtypes if args.dtypes else DTYPES

    written = generate_cases(
        variants=args.variants,
        p_values=p_values,
        counts=counts,
        dtypes=dtypes,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"\nWould generate {len(written)} case files.")
    else:
        print(f"Generated {len(written)} case files in {_CASES_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
