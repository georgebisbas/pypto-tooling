"""Aggregate results.json — paired stack comparison and report emission (E2–E4 stub)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize collective benchmark results")
    parser.add_argument("--run-dir", type=Path, required=True, help=".../run_<timestamp>/")
    parser.add_argument("--emit-report", action="store_true", help="Write reports/summary.md")
    args = parser.parse_args(argv)

    results_path = args.run_dir / "results.json"
    if not results_path.is_file():
        print(f"missing {results_path}")
        return 1

    data = json.loads(results_path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    print(f"runs={len(runs)}")
    # E2: group by equivalence_hash, compute vs_paired_stack for mesh pairs
    if args.emit_report:
        report_dir = args.run_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / "summary.md"
        out.write_text("# Benchmark summary (stub)\n\nRun `plot_figures.py` then refresh this report.\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
