"""Generate figures from results.json for reports.

Figure catalog (see notes spec §6b):
  - paired_stack_ratio    — bar chart: pypto/simpler wall-time ratio per case
  - strong_scaling_t_total — T vs P by variant/stack (needs multi-P campaign)
    - phase_breakdown       — stacked bars: startup/compile/init/execute by stack
  - strong_scaling_efficiency — E(P)
  - message_size_bw_eff   — Campaign B crossover
  - pmu_utilization       — From pmu.csv on anomaly cells
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

FIGURE_IDS = (
    "paired_stack_ratio",
    "strong_scaling_t_total",
    "phase_breakdown",
    "strong_scaling_efficiency",
    "message_size_bw_eff",
    "pmu_utilization",
)


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
        groups.setdefault(cid, {})[r["stack"]] = r.get("wall_s_mean") or r.get("wall_s")

    case_ids = sorted(groups)
    ratios = []
    labels = []
    for cid in case_ids:
        s = groups[cid].get("simpler")
        p = groups[cid].get("pypto")
        if s and p and s > 0:
            ratios.append(p / s)
            labels.append(cid.rsplit("_", 2)[0])

    if not ratios:
        print("  paired_stack_ratio: no valid pairs")
        return fig_dir / "paired_stack_ratio.png"

    fig, ax = plt.subplots(figsize=(max(6, len(ratios) * 1.2), 4))
    colors = ["#2ecc71" if r <= 1.5 else "#e67e22" if r <= 3.0 else "#e74c3c" for r in ratios]
    bars = ax.bar(range(len(ratios)), ratios, color=colors, edgecolor="white")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="parity (1.0×)")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("pypto / simpler wall-time ratio")
    ax.set_title("Paired stack ratio (lower = pypto closer to simpler)")
    ax.legend()

    for rect, val in zip(bars, ratios):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                f"{val:.2f}×", ha="center", va="bottom", fontsize=8)

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
        mean = r.get("wall_s_mean") or r.get("wall_s", 0)
        if p > 0 and mean > 0:
            groups.setdefault(stack, {})[p] = mean

    if not groups:
        print("  strong_scaling_t_total: no multi-P data")
        return fig_dir / "strong_scaling_t_total.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"simpler": "#3498db", "pypto": "#e74c3c", "hccl": "#2ecc71"}
    markers = {"simpler": "o", "pypto": "s", "hccl": "D"}

    for stack in sorted(groups):
        ps = sorted(groups[stack])
        walls = [groups[stack][p] for p in ps]
        ax.plot(ps, walls, marker=markers.get(stack, "x"),
                color=colors.get(stack, None), label=stack, linewidth=1.5, markersize=8)

    ax.set_xlabel("Number of ranks (P)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Strong scaling: total wall time vs P")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = fig_dir / "strong_scaling_t_total.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  strong_scaling_t_total → {path}")
    return path


def _plot_phase_breakdown(runs: list[dict], fig_dir: Path) -> Path:
    """Stacked bars: canonical phase means per (case, stack)."""
    plt = _load_matplotlib()
    if plt is None:
        return _text_fallback(runs, fig_dir, "phase_breakdown")

    phase_order = ("startup", "compile", "init", "execute")
    stack_order = ("hccl", "simpler", "pypto")
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
    else:
        groups: dict[str, dict[str, float | None]] = {}
        for r in runs:
            groups.setdefault(r["case_id"], {})[r["stack"]] = r.get("wall_s")
        for cid in sorted(groups):
            s = groups[cid].get("simpler")
            p = groups[cid].get("pypto")
            ratio = f"{p/s:.2f}×" if s and p and s > 0 else "—"
            lines.append(f"  {cid}: simpler={s} pypto={p} ratio={ratio}")
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
        elif fig_id == "phase_breakdown":
            _plot_phase_breakdown(runs, fig_dir)
        else:
            print(f"  {fig_id}: not yet implemented")

    print(f"\nfigures written to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
