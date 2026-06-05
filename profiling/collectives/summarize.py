"""Aggregate results.json — paired stack comparison and report emission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_runs(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load results.json. Returns (header, runs_list)."""
    path = run_dir / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, data.get("runs", [])


def _group_by_case(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Group runs: {case_id: {stack: run_dict}}."""
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for r in runs:
        cid = r["case_id"]
        groups.setdefault(cid, {})[r["stack"]] = r
    return groups


def _pair_ratio(simpler_ws: float | None, pypto_ws: float | None) -> str:
    """pypto / simpler wall-time ratio as a formatted string."""
    if simpler_ws is None or pypto_ws is None:
        return "—"
    if simpler_ws == 0.0:
        return "∞"
    return f"{pypto_ws / simpler_ws:.2f}×"


def _wall_str(run: dict[str, Any]) -> str:
    """Format wall time with optional stdev."""
    mean = run.get("wall_s_mean") or run.get("wall_s")
    stdev = run.get("wall_s_stdev")
    if mean is None:
        return "—"
    if stdev and stdev > 0:
        return f"{mean:.4f}±{stdev:.4f}"
    return f"{mean:.4f}"


def _print_table(groups: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Print a comparison table and return summary rows for JSON export."""
    print(f"{'case_id':<50} {'simpler':>10} {'pypto':>10} {'ratio':>8} {'ok'}")
    print("-" * 92)

    rows: list[dict[str, Any]] = []
    for cid in sorted(groups):
        stacks = groups[cid]
        s = stacks.get("simpler", {})
        p = stacks.get("pypto", {})
        sw = s.get("wall_s_mean") or s.get("wall_s")
        pw = p.get("wall_s_mean") or p.get("wall_s")
        s_ok = s.get("correctness", "?")
        p_ok = p.get("correctness", "?")
        both_ok = "✅" if s_ok == "pass" and p_ok == "pass" else "❌"

        print(f"{cid:<50} {_wall_str(s):>14} {_wall_str(p):>14} {_pair_ratio(sw, pw):>8}  {both_ok}")

        rows.append({
            "case_id": cid,
            "simpler_wall_s": sw,
            "pypto_wall_s": pw,
            "ratio": round(pw / sw, 4) if sw and pw and sw > 0 else None,
            "simpler_correctness": s_ok,
            "pypto_correctness": p_ok,
            "variant": s.get("variant", p.get("variant", "?")),
            "p": s.get("p", p.get("p", "?")),
            "count": s.get("count", p.get("count", "?")),
        })

    return rows


def _wall_str_fmt(row: dict[str, Any], stack: str) -> str:
    """Format wall time for report table."""
    key = f"{stack}_wall_s"
    val = row.get(key)
    if val is None:
        return "—"
    return f"{val:.4f}"


def _write_report(run_dir: Path, rows: list[dict[str, Any]], header: dict[str, Any]) -> Path:
    """Write reports/summary.md with tables and metadata."""
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    campaign = header.get("campaign", "?")
    timestamp = header.get("timestamp", "?")
    case = header.get("case", {})

    lines = [
        f"# Benchmark summary — {campaign}",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Platform:** {case.get('platform', '?')}",
        f"**Variant:** {case.get('variant', '?')}  "
        f"**P:** {case.get('p', '?')}  "
        f"**Count:** {case.get('count', '?')}  "
        f"**Dtype:** {case.get('dtype', '?')}",
        "",
        "## Per-case comparison",
        "",
        "| case_id | simpler (s) | pypto (s) | ratio | ok |",
        "|---------|-------------|-----------|-------|----|",
    ]

    for r in rows:
        sw = _wall_str_fmt(r, "simpler")
        pw = _wall_str_fmt(r, "pypto")
        ratio = f"{r['ratio']:.2f}×" if r["ratio"] else "—"
        ok = "✅" if r["simpler_correctness"] == "pass" and r["pypto_correctness"] == "pass" else "❌"
        lines.append(f"| {r['case_id']} | {sw} | {pw} | {ratio} | {ok} |")

    # Summary stats
    valid = [r for r in rows if r["ratio"] is not None]
    if valid:
        ratios = [r["ratio"] for r in valid]
        lines += [
            "",
            "## Summary statistics",
            "",
            f"- **Cases:** {len(rows)} total, {len(valid)} with valid ratios",
            f"- **Ratio (pypto/simpler):** "
            f"min={min(ratios):.2f}×  max={max(ratios):.2f}×  "
            f"mean={sum(ratios)/len(ratios):.2f}×",
        ]

    lines += [
        "",
        "## Artifacts",
        "",
        "See `results.json` for raw data and `figures/` for plots.",
        "",
        "Run `python -m collectives.plot_figures --run-dir <dir>` to generate figures.",
    ]

    out = report_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize collective benchmark results")
    parser.add_argument("--run-dir", type=Path, required=True, help=".../run_<timestamp>/")
    parser.add_argument("--emit-report", action="store_true", help="Write reports/summary.md")
    parser.add_argument("--json", type=Path, default=None, help="Write summary rows as JSON")
    args = parser.parse_args(argv)

    try:
        header, runs = _load_runs(args.run_dir)
    except FileNotFoundError as e:
        print(str(e))
        return 1

    print(f"runs={len(runs)}")
    groups = _group_by_case(runs)
    print(f"cases={len(groups)}")
    print()

    rows = _print_table(groups)

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    if args.emit_report:
        path = _write_report(args.run_dir, rows, header)
        print(f"\nwrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
