"""Aggregate results.json — paired stack comparison and report emission."""

from __future__ import annotations

import argparse
import datetime
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


STACK_ORDER = (
    "hccl",
    "simpler",
    "simpler-own",
    "pypto-composite",
    "pypto-host",
    "pto-isa",
)


def _stacks_present(groups: dict[str, dict[str, Any]]) -> list[str]:
    """Stacks present across all cases, in canonical order."""
    seen: list[str] = []
    for stacks in groups.values():
        for name in stacks:
            if name not in seen:
                seen.append(name)
    ordered = [name for name in STACK_ORDER if name in seen]
    ordered += [name for name in seen if name not in STACK_ORDER]
    return ordered


def _ratio_str(numer: float | None, denom: float | None) -> str:
    """Formatted ``numer / denom`` ratio (×), or a dash when undefined."""
    if numer is None or denom is None:
        return "—"
    if denom == 0.0:
        return "∞"
    return f"{numer / denom:.2f}×"


def _primary_time(run: dict[str, Any]) -> float | None:
    """Return primary benchmark time (execute_s preferred, wall_s fallback)."""
    val = run.get("execute_s_mean")
    if val is None:
        val = run.get("wall_s_mean") or run.get("wall_s")
    return float(val) if val is not None else None


def _primary_stdev(run: dict[str, Any]) -> float | None:
    val = run.get("execute_s_stdev")
    if val is None:
        val = run.get("wall_s_stdev")
    return float(val) if val is not None else None


def _time_str(run: dict[str, Any]) -> str:
    """Format primary time with optional stdev."""
    mean = _primary_time(run)
    stdev = _primary_stdev(run)
    if mean is None:
        return "—"
    if stdev and stdev > 0:
        return f"{mean:.4f}±{stdev:.4f}"
    return f"{mean:.4f}"


def _print_table(
    groups: dict[str, dict[str, dict[str, Any]]],
    baseline: str = "simpler",
) -> list[dict[str, Any]]:
    """Print an N-stack comparison table and return summary rows for JSON export.

    Every stack gets a per-case execute_s column; ``pypto-composite`` is also
    shown as a ratio against ``baseline`` (default ``simpler``).
    """
    stacks = _stacks_present(groups)
    width = max(8, max((len(s) for s in stacks), default=8))
    header = f"{'case_id':<46} " + " ".join(f"{s:>{width}}" for s in stacks)
    if "pypto-composite" in stacks and baseline in stacks:
        header += f"  vs {baseline}"
    print(header)
    print("-" * len(header))

    rows: list[dict[str, Any]] = []
    for cid in sorted(groups):
        st = groups[cid]
        cells = [_time_str(st.get(s, {})) for s in stacks]
        line = f"{cid:<46} " + " ".join(f"{c:>{width}}" for c in cells)
        base_run = st.get(baseline, {})
        comp_run = st.get("pypto-composite", {})
        if comp_run and base_run:
            ratio = _ratio_str(_primary_time(comp_run), _primary_time(base_run))
            line += f"  {ratio:>{len(baseline) + 3}}"
        print(line)

        row: dict[str, Any] = {
            "case_id": cid,
            "variant": next(
                (st[s].get("variant") for s in st if st[s].get("variant")), "?"
            ),
            "p": next((st[s].get("p") for s in st if st[s].get("p") is not None), "?"),
            "count": next(
                (st[s].get("count") for s in st if st[s].get("count") is not None), "?"
            ),
            "platform": next(
                (st[s].get("platform") for s in st if st[s].get("platform")), "?"
            ),
            "dtype": next(
                (st[s].get("dtype") for s in st if st[s].get("dtype")), "?"
            ),
        }
        for s in stacks:
            r = st.get(s, {})
            row[f"{s}_execute_s"] = _primary_time(r)
            row[f"{s}_execute_s_median"] = r.get("execute_s_median")
            row[f"{s}_wall_s"] = r.get("wall_s_mean") or r.get("wall_s")
            row[f"{s}_correctness"] = r.get("correctness", "?")
            row[f"{s}_setup_s"] = r.get("setup_s")
            row[f"{s}_bw_mb_s"] = r.get("bw_execute_mb_s")
            row[f"{s}_device_wall_s"] = r.get("device_wall_s_mean")
            row[f"{s}_device_wall_s_median"] = r.get("device_wall_s_median")
        hccl_t = row.get("hccl_execute_s")
        base_t = row.get(f"{baseline}_execute_s")
        for s in stacks:
            if s == "hccl":
                continue
            t = row.get(f"{s}_execute_s")
            row[f"{s}_hccl_eff"] = (
                round(t / hccl_t, 4) if hccl_t and t and hccl_t > 0 else None
            )
            if s == baseline:
                continue
            row[f"{s}_vs_{baseline}"] = (
                round(t / base_t, 4) if base_t and t and base_t > 0 else None
            )
        rows.append(row)

    return rows


def _time_str_fmt(row: dict[str, Any], stack: str) -> str:
    """Format primary time for report table."""
    key = f"{stack}_execute_s"
    val = row.get(key)
    if val is None:
        key = f"{stack}_wall_s"
        val = row.get(key)
    if val is None:
        return "—"
    return f"{val:.4f}"


def _device_time_str_fmt(row: dict[str, Any], stack: str) -> str:
    """Format the pure on-device collective time (device_wall), or a dash."""
    val = row.get(f"{stack}_device_wall_s")
    if val is None:
        return "—"
    return f"{val:.6f}"


def _median_str_fmt(row: dict[str, Any], stack: str) -> str:
    """Format the median execute time, or a dash."""
    val = row.get(f"{stack}_execute_s_median")
    if val is None:
        return "—"
    return f"{val:.4f}"


def _stacks_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Derive the stack list from summary row keys (``<stack>_execute_s``)."""
    stacks: list[str] = []
    for r in rows:
        for key in r:
            if key.endswith("_execute_s"):
                name = key[: -len("_execute_s")]
                if name not in stacks:
                    stacks.append(name)
    return [s for s in STACK_ORDER if s in stacks] + [s for s in stacks if s not in STACK_ORDER]


def _write_report(
    run_dir: Path,
    rows: list[dict[str, Any]],
    header: dict[str, Any],
    baseline: str = "simpler",
) -> Path:
    """Write reports/summary.md with N-stack tables and metadata."""
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    campaign = header.get("campaign", "?")
    timestamp = header.get("timestamp", "?")
    case = header.get("case", {})
    stacks = _stacks_from_rows(rows)

    # Merged multi-P campaigns carry no per-file header; derive the metadata
    # from the summary rows instead (first non-null value per field).
    def _first(*keys: str) -> Any:
        for r in rows:
            for k in keys:
                v = r.get(k)
                if v not in (None, "?"):
                    return v
        return "?"

    platform = case.get("platform") or _first("platform")
    dtype = case.get("dtype") or _first("dtype")
    p_vals = sorted({r["p"] for r in rows if r.get("p") not in (None, "?")})
    count_vals = sorted({r["count"] for r in rows if r.get("count") not in (None, "?")})
    variant = case.get("variant") or _first("variant")
    if not p_vals:
        p_vals = [case.get("p")] if case.get("p") not in (None, "?") else []
    if not count_vals:
        count_vals = [case.get("count")] if case.get("count") not in (None, "?") else []
    p_str = ",".join(map(str, p_vals)) if p_vals else "?"
    count_str = ",".join(map(str, count_vals)) if count_vals else "?"
    if timestamp == "?":
        ts_path = run_dir / "results.json"
        if ts_path.is_file():
            timestamp = datetime.datetime.fromtimestamp(
                ts_path.stat().st_mtime
            ).isoformat()

    lines = [
        f"# Benchmark summary — {campaign}",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Platform:** {platform}",
        f"**Variant:** {variant}  "
        f"**P:** {p_str}  "
        f"**Count:** {count_str}  "
        f"**Dtype:** {dtype}",
        "",
        "## Per-case comparison",
        "",
        "| case_id | " + " | ".join(f"{s} (s)" for s in stacks) + " |",
        "|---------|-" + "|-".join(["----"] * len(stacks)) + "|",
    ]

    for r in rows:
        cells = " | ".join(_time_str_fmt(r, s) for s in stacks)
        lines.append(f"| {r['case_id']} | {cells} |")

    # Per-stack setup + bandwidth table — compile/init overhead vs throughput.
    lines += [
        "",
        "## Setup and bandwidth",
        "",
        "| case_id | " + " | ".join(f"{s} setup(s)/MB/s" for s in stacks) + " |",
        "|---------|-" + "|-".join(["----"] * len(stacks)) + "|",
    ]
    for r in rows:
        cells = []
        for s in stacks:
            setup = r.get(f"{s}_setup_s")
            bw = r.get(f"{s}_bw_mb_s")
            if setup is None and bw is None:
                cells.append("—")
            else:
                cells.append(
                    f"{(setup if setup is not None else float('nan')):.2f}/"
                    f"{(bw if bw is not None else float('nan')):.1f}"
                )
        lines.append(f"| {r['case_id']} | " + " | ".join(cells) + " |")

    # Summary statistics: composite-vs-baseline ratio and HCCL efficiency.
    comp_key = "pypto-composite_vs_" + baseline
    valid = [r for r in rows if r.get(comp_key) is not None]
    if valid:
        ratios = [r[comp_key] for r in valid]
        lines += [
            "",
            "## Summary statistics",
            "",
            f"- **Cases:** {len(rows)} total, {len(valid)} with valid ratios",
            f"- **Ratio (pypto-composite/{baseline}):** "
            f"min={min(ratios):.2f}×  max={max(ratios):.2f}×  "
            f"mean={sum(ratios)/len(ratios):.2f}×",
        ]
    hccl_valid = [r for r in rows if r.get("pypto-composite_hccl_eff") is not None]
    if hccl_valid:
        effs = [r["pypto-composite_hccl_eff"] for r in hccl_valid]
        lines += [
            f"- **HCCL efficiency (pypto-composite):** "
            f"min={min(effs):.2f}×  max={max(effs):.2f}×  "
            f"mean={sum(effs)/len(effs):.2f}×",
        ]

    # Median execute times — robust to the per-round spike pattern seen on the
    # shared NPU box (consecutive rounds can differ 7x; medians resist that).
    median_stacks = [
        s for s in stacks
        if any(r.get(f"{s}_execute_s_median") is not None for r in rows)
    ]
    if median_stacks:
        lines += [
            "",
            "## Median execute times",
            "",
            "Median of the timed-round ``execute_s`` — robust to outlier rounds "
            "(shared-box contention / dispatch spikes).",
            "",
            "| case_id | " + " | ".join(f"{s} med(s)" for s in median_stacks) + " |",
            "|---------|-" + "|-".join(["----"] * len(median_stacks)) + "|",
        ]
        for r in rows:
            cells = " | ".join(_median_str_fmt(r, s) for s in median_stacks)
            lines.append(f"| {r['case_id']} | {cells} |")

    # Device-only comparison: the pure on-device collective (device_wall) vs
    # the dispatch+collective wall time, and vs HCCL's pure collective.
    dev_stacks = [
        s for s in stacks
        if any(r.get(f"{s}_device_wall_s") is not None for r in rows)
    ]
    if dev_stacks:
        lines += [
            "",
            "## Device-time comparison (pure collective)",
            "",
            "`device_wall_s` = slowest-rank on-device span when the runtime "
            "emits STRACE; `—` = not captured.",
            "",
            "| case_id | " + " | ".join(f"{s} dev(s)" for s in dev_stacks) + " |",
            "|---------|-" + "|-".join(["----"] * len(dev_stacks)) + "|",
        ]
        for r in rows:
            cells = " | ".join(_device_time_str_fmt(r, s) for s in dev_stacks)
            lines.append(f"| {r['case_id']} | {cells} |")

        dev_hccl = [
            (r.get("pypto-composite_device_wall_s"), r.get("hccl_execute_s"))
            for r in rows
            if r.get("pypto-composite_device_wall_s") is not None
            and r.get("hccl_execute_s") is not None
        ]
        ratios = [dw / h for dw, h in dev_hccl if h > 0]
        if ratios:
            lines += [
                "",
                f"- **HCCL efficiency (pypto-composite, device-only):** "
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
    """Summarize a campaign run directory's results.json."""
    parser = argparse.ArgumentParser(description="Summarize collective benchmark results")
    parser.add_argument("--run-dir", type=Path, required=True, help=".../run_<timestamp>/")
    parser.add_argument("--baseline", default="simpler", help="Reference stack for paired ratios")
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

    rows = _print_table(groups, baseline=args.baseline)

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    if args.emit_report:
        path = _write_report(args.run_dir, rows, header, baseline=args.baseline)
        print(f"\nwrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
