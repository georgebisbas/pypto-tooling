"""Generate figures from results.json for reports (E4).

Figure catalog (see notes spec §6b):
  - strong_scaling_t_total
  - strong_scaling_efficiency
  - message_size_bw_eff
  - paired_stack_ratio
  - pmu_utilization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIGURE_IDS = (
    "strong_scaling_t_total",
    "strong_scaling_efficiency",
    "message_size_bw_eff",
    "paired_stack_ratio",
    "pmu_utilization",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot benchmark figures")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    results_path = args.run_dir / "results.json"
    if not results_path.is_file():
        print(f"missing {results_path}")
        return 1

    fig_dir = args.run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(results_path.read_text(encoding="utf-8"))
    n_runs = len(data.get("runs", []))
    print(f"figure_ids={FIGURE_IDS}")
    print(f"runs={n_runs} output_dir={fig_dir}")
    print("plot_figures: not implemented (E4). Install requirements.txt and add matplotlib plots.", file=__import__("sys").stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
