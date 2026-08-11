"""Generate figures from results.json for reports.

Figure catalog (see notes spec §6b):
  - paired_stack_ratio    — bar chart: pypto/simpler wall-time ratio per case
  - strong_scaling_t_total — T vs P by variant/stack (needs multi-P campaign)
    - phase_breakdown       — stacked bars: startup/compile/init/execute by stack
    - compile_breakdown     — pypto compile sub-stages: passes/codegen/other
  - strong_scaling_efficiency — E(P)
  - message_size_bw_eff   — Campaign B crossover
  - pmu_utilization       — From pmu.csv on anomaly cells
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from collectives import model as bw_model

FIGURE_IDS = (
    "paired_stack_ratio",
    "strong_scaling_t_total",
    "strong_scaling_efficiency",
    "message_size_bw_eff",
    "phase_breakdown",
    "setup_breakdown",
    "compile_breakdown",
    "pmu_utilization",
    "wall_vs_device",
    "bw_model_fit",
)

# Canonical per-stack colors/markers, shared across figures.
STACK_COLORS = {
    "hccl": "#2ecc71",
    "simpler": "#3498db",
    "simpler-own": "#9b59b6",
    "pypto-composite": "#e74c3c",
    "pypto-host": "#e67e22",
    "pto-isa": "#7f8c8d",
}
STACK_MARKERS = {
    "hccl": "D",
    "simpler": "o",
    "simpler-own": "v",
    "pypto-composite": "s",
    "pypto-host": "^",
    "pto-isa": "P",
}


def _primary_time(run: dict) -> float | None:
    """Return primary benchmark time (execute_s preferred, wall_s fallback)."""
    val = run.get("execute_s_mean")
    if val is None:
        val = run.get("wall_s_mean") or run.get("wall_s")
    return float(val) if val is not None else None


def _load_matplotlib():
    """Return pyplot configured for headless PNG rendering, or ``None`` if unavailable."""
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return None


def _plot_paired_stack_ratio(runs: list[dict], fig_dir: Path) -> Path:
    """Bar chart: pypto/simpler ratio per case_id. Falls back to text if no matplotlib."""
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "paired_stack_ratio")

    # Group by case_id
    groups: dict[str, dict[str, float | None]] = {}
    for r in runs:
        cid = r["case_id"]
        groups.setdefault(cid, {})[r["stack"]] = _primary_time(r)

    case_ids = sorted(groups)
    ratios: list[float] = []
    labels: list[str] = []
    stack_names: list[str] = []
    for cid in case_ids:
        base = groups[cid].get("simpler")
        for stack, t in sorted(groups[cid].items()):
            if stack in ("simpler", "hccl"):
                continue
            if base and t and base > 0:
                ratios.append(t / base)
                labels.append(f"{cid.rsplit('_', 2)[0]}\n{stack}")
                stack_names.append(stack)

    if not ratios:
        print("  paired_stack_ratio: no valid pairs")
        return fig_dir / "paired_stack_ratio.png"

    fig, ax = plt.subplots(figsize=(max(6, len(ratios) * 1.2), 4))
    colors = [STACK_COLORS.get(s, "#95a5a6") for s in stack_names]
    bars = ax.bar(range(len(ratios)), ratios, color=colors, edgecolor="white")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="parity (1.0×)")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("stack / simpler execute-time ratio")
    ax.set_title("Paired stack ratio vs simpler (lower = closer to simpler)")
    ax.legend()

    for rect, val in zip(bars, ratios):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                f"{val:.2f}×", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path = fig_dir / "paired_stack_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  paired_stack_ratio → {path}")
    return path


def _plot_strong_scaling_t_total(runs: list[dict], fig_dir: Path) -> Path:
    """Line chart: wall time vs P, one line per stack. Needs multi-P campaign data."""
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "strong_scaling_t_total")

    # Group by (stack, P): compute mean wall time
    groups: dict[str, dict[int, float]] = {}
    for r in runs:
        stack = r["stack"]
        p = r.get("p", 0)
        mean = _primary_time(r) or 0
        if p > 0 and mean > 0:
            groups.setdefault(stack, {})[p] = mean

    if not groups:
        print("  strong_scaling_t_total: no multi-P data")
        return fig_dir / "strong_scaling_t_total.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    for stack in sorted(groups):
        ps = sorted(groups[stack])
        walls = [groups[stack][p] for p in ps]
        ax.plot(ps, walls, marker=STACK_MARKERS.get(stack, "x"),
                color=STACK_COLORS.get(stack), label=stack, linewidth=1.5, markersize=8)

    ax.set_xlabel("Number of ranks (P)")
    ax.set_ylabel("Execute time (s)")
    ax.set_title("Strong scaling: execute time vs P")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = fig_dir / "strong_scaling_t_total.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  strong_scaling_t_total → {path}")
    return path

def _plot_strong_scaling_efficiency(runs: list[dict], fig_dir: Path) -> Path:
    """Parallel efficiency E(P) = T(P_min)*P_min / (T(P)*P), per stack.

    Normalised to the smallest P in the campaign (1.0 there); a flat line at
    1.0 is perfect scaling. Needs a multi-P campaign.
    """
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "strong_scaling_efficiency")

    groups: dict[str, dict[int, float]] = {}
    for r in runs:
        stack = r["stack"]
        p = r.get("p", 0)
        mean = _primary_time(r)
        if p > 0 and mean and mean > 0:
            groups.setdefault(stack, {})[p] = mean

    # Restrict to stacks with >= 2 distinct P values.
    groups = {s: g for s, g in groups.items() if len(g) >= 2}
    if not groups:
        print("  strong_scaling_efficiency: need >=2 P values per stack")
        return fig_dir / "strong_scaling_efficiency.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    for stack in sorted(groups):
        ps = sorted(groups[stack])
        p_min = ps[0]
        t_min = groups[stack][p_min]
        eff = [t_min * p_min / (groups[stack][p] * p) for p in ps]
        ax.plot(ps, eff, marker=STACK_MARKERS.get(stack, "x"),
                color=STACK_COLORS.get(stack), label=stack,
                linewidth=1.5, markersize=8)

    ax.set_xlabel("Number of ranks (P)")
    ax.set_ylabel("Parallel efficiency (normalised to smallest P)")
    ax.set_title("Strong scaling efficiency vs P")
    ax.set_ylim(0.0, 1.15)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = fig_dir / "strong_scaling_efficiency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  strong_scaling_efficiency → {path}")
    return path


def _plot_message_size_bw_eff(runs: list[dict], fig_dir: Path) -> Path:
    """Bandwidth (MB/s) vs payload count per stack — log-x crossover plot.

    Needs a multi-count campaign (message-size sweep)."""
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "message_size_bw_eff")

    groups: dict[str, dict[int, float]] = {}
    for r in runs:
        stack = r["stack"]
        count = r.get("count", 0)
        bw = r.get("bw_execute_mb_s")
        if count > 0 and bw:
            groups.setdefault(stack, {})[count] = float(bw)

    groups = {s: g for s, g in groups.items() if len(g) >= 2}
    if not groups:
        print("  message_size_bw_eff: need >=2 counts per stack")
        return fig_dir / "message_size_bw_eff.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    for stack in sorted(groups):
        counts = sorted(groups[stack])
        bws = [groups[stack][c] for c in counts]
        ax.plot(counts, bws, marker=STACK_MARKERS.get(stack, "x"),
                color=STACK_COLORS.get(stack), label=stack,
                linewidth=1.5, markersize=8)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({c for g in groups.values() for c in g}))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Payload element count (log2)")
    ax.set_ylabel("Bandwidth (MB/s)")
    ax.set_title("Bandwidth vs message size (crossover view)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    path = fig_dir / "message_size_bw_eff.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  message_size_bw_eff → {path}")
    return path


def _plot_bw_model_fit(runs: list[dict], fig_dir: Path) -> Path:
    """Measured T(N) points + fitted O + N/B lines per stack (projected vs HCCL).

    Picks, per stack, the (P, variant) group with the most fitted points (ties
    resolved to the largest P), then plots the raw points and the model line
    on a log-x payload-byte axis. HCCL's fit is included as the reference.
    """
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "bw_model_fit")

    models = bw_model.score_runs(runs)
    if not models:
        print("  bw_model_fit: need >=3 distinct payload sizes per stack "
              "(message-size sweep)")
        return fig_dir / "bw_model_fit.png"

    # Per stack: the fit with the most points (largest P on ties).
    chosen: dict[str, bw_model.BandwidthModel] = {}
    for m in models:
        cur = chosen.get(m.stack)
        if cur is None or (m.n_points, m.p) > (cur.n_points, cur.p):
            chosen[m.stack] = m
    if "hccl" not in chosen:
        print("  bw_model_fit: no HCCL fit to reference — run hccl in the sweep")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for stack in sorted(chosen):
        m = chosen[stack]
        color = STACK_COLORS.get(stack, "#95a5a6")
        marker = STACK_MARKERS.get(stack, "x")
        xs = [pt[0] for pt in m.points]
        ys = [pt[1] for pt in m.points]
        ax.scatter(xs, ys, color=color, marker=marker, s=45,
                   label=f"{stack} (P={m.p}, {m.source})")
        lo, hi = min(xs), max(xs)
        span = [lo * (hi / lo) ** (i / 99) for i in range(100)]
        fit_ys = [m.latency_s + n / m.bandwidth_b_s for n in span]
        eff = f" {m.bw_eff_vs_hccl:.2f}×HCCL" if m.bw_eff_vs_hccl is not None else ""
        ax.plot(span, fit_ys, color=color, linestyle="--", linewidth=1.4,
                label=f"{stack} fit O+N/B (pipe {m.pipeline_score:.2f}{eff})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Payload bytes per rank (log2)")
    ax.set_ylabel("Collective time (s)")
    ax.set_title("T(N) = O + N/B fit — measured points vs bandwidth model")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    path = fig_dir / "bw_model_fit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  bw_model_fit → {path}")
    return path


def _plot_setup_breakdown(runs: list[dict], fig_dir: Path) -> Path:
    """Grouped bars: compile vs init vs execute per (case, stack).

    Uses ``phase_means`` (campaign stacks); subprocess stacks without phase
    data are skipped."""
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "setup_breakdown")

    filtered = [run for run in runs if run.get("phase_means")]
    if not filtered:
        print("  setup_breakdown: no phase data")
        return fig_dir / "setup_breakdown.png"

    def _key(run: dict) -> tuple[int, str, str]:
        return (run.get("p", 0), run.get("case_id", ""), run.get("stack", ""))

    filtered.sort(key=_key)
    labels = [f"P{run.get('p', '?')}\n{run['stack']}" for run in filtered]
    x = range(len(filtered))
    width = 0.25
    offsets = (-width, 0.0, width)
    colors = {"compile": "#3498db", "init": "#9b59b6", "execute": "#2ecc71"}
    keys = ("compile", "init", "execute")

    fig, ax = plt.subplots(figsize=(max(8, len(filtered) * 1.25), 5))
    for key, off in zip(keys, offsets):
        values = [float(run.get("phase_means", {}).get(key, 0.0)) for run in filtered]
        ax.bar([i + off for i in x], values, width=width, color=colors[key],
               edgecolor="white", label=key)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Mean time (s)")
    ax.set_title("Setup breakdown: compile / init / execute")
    ax.legend(ncols=3, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    path = fig_dir / "setup_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  setup_breakdown → {path}")
    return path


def _find_pmu_csvs(run_dir: Path, runs: list[dict]) -> list[Path]:
    """Locate pmu.csv files: prefer artifact-bundle dirs, then a run-dir scan."""
    found: list[Path] = []
    seen: set[Path] = set()
    for r in runs:
        bundle = r.get("artifact_bundle")
        if not bundle:
            continue
        for p in Path(bundle).rglob("pmu.csv"):
            if p not in seen:
                found.append(p)
                seen.add(p)
    for p in run_dir.rglob("pmu.csv"):
        if p not in seen:
            found.append(p)
            seen.add(p)
    return sorted(found)


def _plot_pmu_utilization(runs: list[dict], fig_dir: Path, run_dir: Path) -> Path:
    """Bar chart of numeric PMU columns from collected pmu.csv files.

    Defensive: only fully-numeric columns are plotted (mean across rows).
    Requires runs collected with ``--profile pmu``."""
    import csv

    pmu_files = _find_pmu_csvs(run_dir, runs)
    if not pmu_files:
        path = fig_dir / "pmu_utilization.txt"
        path.write_text(
            "# pmu_utilization\n\nNo pmu.csv found. Collect data with "
            "--profile pmu (pypto stacks route DFX artifacts into the bundle).\n",
            encoding="utf-8",
        )
        print(f"  pmu_utilization → {path} (no pmu data)")
        return path

    plt = _load_matplotlib()
    if plt is None:
        path = fig_dir / "pmu_utilization.txt"
        lines = ["# pmu_utilization (text fallback)", ""]
        for f in pmu_files:
            lines.append(f"  {f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  pmu_utilization → {path} (text)")
        return path

    # Aggregate numeric columns across all pmu files (mean per column).
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for f in pmu_files:
        try:
            with f.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    for name, value in row.items():
                        try:
                            totals[name] = totals.get(name, 0.0) + float(value)
                            counts[name] = counts.get(name, 0) + 1
                        except (TypeError, ValueError):
                            continue
        except OSError as exc:
            print(f"  pmu_utilization: skip {f}: {exc}")
            continue

    if not totals:
        print("  pmu_utilization: pmu.csv present but no numeric columns")
        return fig_dir / "pmu_utilization.png"

    means = {n: totals[n] / counts[n] for n in totals}

    # Utilisation ratios (busy cycles / total cycles) when the total is known.
    ratio_cols = [c for c in ("vec_busy_cycles", "cube_busy_cycles",
                              "scalar_busy_cycles", "mte1_busy_cycles",
                              "mte2_busy_cycles", "mte3_busy_cycles")
                  if c in means]
    if means.get("pmu_total_cycles", 0.0) > 0:
        names = ratio_cols + (["icache_miss_rate"] if "icache_miss" in means and "icache_req" in means else [])
        values = [means[c] / means["pmu_total_cycles"] for c in ratio_cols]
        if "icache_miss_rate" in names:
            values.append(means["icache_miss"] / means["icache_req"] if means["icache_req"] > 0 else 0.0)
        ylabel = "Utilisation (busy cycles / total)"
        title = f"PMU pipe utilisation across {len(pmu_files)} pmu.csv"
    else:
        names = sorted(means)
        values = [means[n] for n in names]
        ylabel = "Mean PMU value"
        title = f"PMU columns across {len(pmu_files)} pmu.csv"

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    ax.bar(range(len(names)), values, color="#16a085", edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    path = fig_dir / "pmu_utilization.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  pmu_utilization → {path} (from {len(pmu_files)} files)")
    return path


def _plot_wall_vs_device(runs: list[dict], fig_dir: Path) -> Path:
    """Grouped bars: ``execute_s`` (dispatch + collective) vs ``device_wall_s``
    (collective only) per pypto (case, stack). Exposes the per-dispatch
    overhead directly."""
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "wall_vs_device")

    filtered = [r for r in runs if r.get("device_wall_s_mean") is not None]
    if not filtered:
        print("  wall_vs_device: no device_wall data")
        return fig_dir / "wall_vs_device.png"

    filtered.sort(key=lambda r: (r.get("p", 0), r.get("case_id", ""), r.get("stack", "")))
    labels = [f"P{run.get('p', '?')}\n{run['stack']}" for run in filtered]
    x = range(len(filtered))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(filtered) * 1.2), 5))
    wall = [float(r["execute_s_mean"]) for r in filtered]
    dev = [float(r["device_wall_s_mean"]) for r in filtered]
    ax.bar([i - width / 2 for i in x], wall, width=width, color="#e74c3c",
           edgecolor="white", label="execute_s (dispatch + collective)")
    ax.bar([i + width / 2 for i in x], dev, width=width, color="#2ecc71",
           edgecolor="white", label="device_wall_s (collective only)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_title("Dispatch overhead vs pure collective (pypto stacks)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    path = fig_dir / "wall_vs_device.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wall_vs_device → {path}")
    return path

def _plot_phase_breakdown(runs: list[dict], fig_dir: Path) -> Path:
    """Stacked bars: canonical phase means per (case, stack)."""
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "phase_breakdown")

    phase_order = ("startup", "compile", "init", "execute")
    stack_order = ("hccl", "simpler", "simpler-own", "pypto-composite", "pypto-host", "pto-isa")
    stack_colors = {
        "startup": "#95a5a6",
        "compile": "#3498db",
        "init": "#9b59b6",
        "execute": "#2ecc71",
    }

    filtered = [run for run in runs if run.get("phase_means")]
    if not filtered:
        print("  phase_breakdown: no phase data")
        return fig_dir / "phase_breakdown.png"

    def _phase_sort_key(run: dict) -> tuple[int, str, int]:
        stack = run.get("stack", "")
        stack_idx = stack_order.index(stack) if stack in stack_order else 99
        return (run.get("p", 0), run.get("case_id", ""), stack_idx)

    filtered.sort(key=_phase_sort_key)
    labels = [f"P{run.get('p', '?')}\n{run['stack']}" for run in filtered]

    fig, ax = plt.subplots(figsize=(max(8, len(filtered) * 1.25), 5))
    bottoms = [0.0] * len(filtered)
    for phase_name in phase_order:
        values = [float(run.get("phase_means", {}).get(phase_name, 0.0)) for run in filtered]
        if not any(values):
            continue
        ax.bar(
            range(len(filtered)),
            values,
            bottom=bottoms,
            color=stack_colors[phase_name],
            edgecolor="white",
            label=phase_name,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_xticks(range(len(filtered)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean phase time (s)")
    ax.set_title("Phase breakdown by stack")
    ax.legend(ncols=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    path = fig_dir / "phase_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  phase_breakdown → {path}")
    return path


def _plot_compile_breakdown(runs: list[dict], fig_dir: Path) -> Path:
    """Stacked bars: pypto compile sub-stages from compile_profile_means."""
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "compile_breakdown")

    filtered = []
    for run in runs:
        if run.get("stack") not in ("pypto-composite", "pypto-host"):
            continue
        profile = run.get("compile_profile_means", {})
        if not profile:
            continue
        total = float(profile.get("total", 0.0))
        passes = float(profile.get("passes", 0.0))
        codegen = float(profile.get("codegen", 0.0))
        other = max(0.0, total - passes - codegen)
        filtered.append({
            "label": f"P{run.get('p', '?')}\n{run['stack']}",
            "passes": passes,
            "codegen": codegen,
            "other": other,
        })

    if not filtered:
        print("  compile_breakdown: no pypto compile profile data")
        return fig_dir / "compile_breakdown.png"

    fig, ax = plt.subplots(figsize=(max(6, len(filtered) * 1.5), 5))
    labels = [row["label"] for row in filtered]
    bottoms = [0.0] * len(filtered)
    colors = {
        "passes": "#1f77b4",
        "codegen": "#ff7f0e",
        "other": "#7f8c8d",
    }
    for key in ("passes", "codegen", "other"):
        values = [float(row[key]) for row in filtered]
        if not any(values):
            continue
        ax.bar(
            range(len(filtered)),
            values,
            bottom=bottoms,
            color=colors[key],
            edgecolor="white",
            label=key,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_xticks(range(len(filtered)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean compile time (s)")
    ax.set_title("PyPTO compile breakdown")
    ax.legend(ncols=3, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    path = fig_dir / "compile_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  compile_breakdown → {path}")
    return path


def _text_fallback(runs: list[dict], fig_dir: Path, fig_id: str) -> Path:
    """Write a text summary when matplotlib is not installed."""
    path = fig_dir / f"{fig_id}.txt"
    lines = [f"# {fig_id} (text fallback — install matplotlib for charts)", ""]
    if fig_id == "phase_breakdown":
        for run in sorted(runs, key=lambda r: (r.get("p", 0), r.get("stack", ""))):
            phases = run.get("phase_means", {})
            if not phases:
                continue
            phase_text = ", ".join(f"{name}={value:.4f}s" for name, value in sorted(phases.items()))
            lines.append(f"  {run['case_id']} {run['stack']}: {phase_text}")
    elif fig_id == "setup_breakdown":
        for run in sorted(runs, key=lambda r: (r.get("p", 0), r.get("stack", ""))):
            phases = run.get("phase_means", {})
            if not phases:
                continue
            parts = ", ".join(
                f"{name}={phases.get(name, 0.0):.4f}s" for name in ("compile", "init", "execute")
            )
            lines.append(f"  {run['case_id']} {run['stack']}: {parts}")
    elif fig_id == "strong_scaling_efficiency":
        for stack in sorted({r.get("stack") for r in runs}):
            pts = sorted(
                (r.get("p"), _primary_time(r))
                for r in runs
                if r.get("stack") == stack and r.get("p") and _primary_time(r)
            )
            if len(pts) < 2:
                continue
            p0, t0 = pts[0]
            line = "  " + stack + ": " + " ".join(
                f"P{p}={t0 * p0 / (t * p):.2f}" for p, t in pts
            )
            lines.append(line)
    elif fig_id == "message_size_bw_eff":
        for stack in sorted({r.get("stack") for r in runs}):
            pts = sorted(
                (r.get("count"), r.get("bw_execute_mb_s"))
                for r in runs
                if r.get("stack") == stack and r.get("count") and r.get("bw_execute_mb_s")
            )
            if len(pts) < 2:
                continue
            line = "  " + stack + ": " + " ".join(
                f"count{c}={bw:.1f}MB/s" for c, bw in pts
            )
            lines.append(line)
    elif fig_id == "compile_breakdown":
        for run in sorted(runs, key=lambda r: (r.get("p", 0), r.get("stack", ""))):
            if run.get("stack") not in ("pypto-composite", "pypto-host"):
                continue
            profile = run.get("compile_profile_means", {})
            if not profile:
                continue
            total = float(profile.get("total", 0.0))
            passes = float(profile.get("passes", 0.0))
            codegen = float(profile.get("codegen", 0.0))
            other = max(0.0, total - passes - codegen)
            lines.append(
                f"  {run['case_id']} pypto: passes={passes:.4f}s, codegen={codegen:.4f}s, other={other:.4f}s, total={total:.4f}s"
            )
    elif fig_id == "wall_vs_device":
        for run in sorted(runs, key=lambda r: (r.get("p", 0), r.get("stack", ""))):
            dev = run.get("device_wall_s_mean")
            if dev is None:
                continue
            wall = _primary_time(run) or 0.0
            lines.append(
                f"  {run['case_id']} {run['stack']}: execute={wall:.6f}s device_wall={float(dev):.6f}s"
            )
    elif fig_id == "bw_model_fit":
        models = bw_model.score_runs(runs)
        if not models:
            lines.append("  (need >=3 distinct payload sizes per stack — message-size sweep)")
        for m in sorted(models, key=lambda m: (m.stack, m.p, m.variant)):
            eff = f" bw={m.bw_eff_vs_hccl:.2f}xHCCL" if m.bw_eff_vs_hccl is not None else ""
            lines.append(
                f"  {m.stack} P={m.p} {m.variant} ({m.source}): "
                f"O={bw_model.format_latency(m.latency_s)} "
                f"B={bw_model.format_bandwidth(m.bandwidth_b_s)} "
                f"r2={m.r2:.3f} pipe@maxN={m.pipeline_score:.2f}{eff}"
            )
    else:
        groups: dict[str, dict[str, float | None]] = {}
        for r in runs:
            groups.setdefault(r["case_id"], {})[r["stack"]] = _primary_time(r)
        for cid in sorted(groups):
            cells = ", ".join(
                f"{s}={t:.4f}" for s, t in sorted(groups[cid].items()) if t is not None
            )
            base = groups[cid].get("simpler")
            ratios = " ".join(
                f"{s}:{t / base:.2f}x"
                for s, t in sorted(groups[cid].items())
                if s not in ("simpler", "hccl") and base and t and base > 0
            )
            lines.append(f"  {cid}: {cells} | vs simpler: {ratios}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {fig_id} → {path} (text)")
    return path


def main(argv: list[str] | None = None) -> int:
    """Load one merged campaign ``results.json`` and emit the requested figures."""
    parser = argparse.ArgumentParser(description="Plot benchmark figures")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--figures", default="paired_stack_ratio",
                        help=f"Comma-separated figure IDs: {', '.join(FIGURE_IDS)}")
    args = parser.parse_args(argv)

    results_path = args.run_dir / "results.json"
    if not results_path.is_file():
        print(f"missing {results_path}")
        return 1

    data = json.loads(results_path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])

    fig_dir = args.run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    requested = [f.strip() for f in args.figures.split(",")]
    unknown = [f for f in requested if f not in FIGURE_IDS]
    if unknown:
        print(f"WARNING: unknown figure IDs: {unknown}")

    for fig_id in requested:
        if fig_id == "paired_stack_ratio":
            _plot_paired_stack_ratio(runs, fig_dir)
        elif fig_id == "strong_scaling_t_total":
            _plot_strong_scaling_t_total(runs, fig_dir)
        elif fig_id == "strong_scaling_efficiency":
            _plot_strong_scaling_efficiency(runs, fig_dir)
        elif fig_id == "message_size_bw_eff":
            _plot_message_size_bw_eff(runs, fig_dir)
        elif fig_id == "phase_breakdown":
            _plot_phase_breakdown(runs, fig_dir)
        elif fig_id == "setup_breakdown":
            _plot_setup_breakdown(runs, fig_dir)
        elif fig_id == "compile_breakdown":
            _plot_compile_breakdown(runs, fig_dir)
        elif fig_id == "pmu_utilization":
            _plot_pmu_utilization(runs, fig_dir, args.run_dir)
        elif fig_id == "wall_vs_device":
            _plot_wall_vs_device(runs, fig_dir)
        elif fig_id == "bw_model_fit":
            _plot_bw_model_fit(runs, fig_dir)
        else:
            print(f"  {fig_id}: not yet implemented")

    print(f"\nfigures written to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
