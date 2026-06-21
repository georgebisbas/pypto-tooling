from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_hwnative_sys.paths import resolve_workspace_path, workspace_root


def _load_results_json(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = run_dir / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing results.json in {run_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, data.get("runs", [])


def _primary_time(run: dict[str, Any]) -> float | None:
    val = run.get("execute_s_mean")
    if val is None:
        val = run.get("wall_s_mean") or run.get("wall_s")
    return float(val) if val is not None else None


def summarize_profile_impl(run_dir: str) -> dict[str, Any]:
    """Summarize a profiling campaign directory (collectives benchmark or similar)."""
    candidate = Path(run_dir).expanduser()
    if not candidate.is_absolute():
        try:
            candidate = resolve_workspace_path(run_dir)
        except ValueError:
            candidate = (workspace_root() / run_dir).resolve()

    if not candidate.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    anomalies: list[str] = []
    summary: dict[str, Any] = {"run_dir": str(candidate)}

    results_path = candidate / "results.json"
    if results_path.exists():
        header, runs = _load_results_json(candidate)
        summary["campaign"] = header.get("campaign") or header.get("case_file")
        summary["run_count"] = len(runs)

        by_stack: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            by_stack.setdefault(run.get("stack", "unknown"), []).append(run)

        stack_summaries: list[dict[str, Any]] = []
        for stack, stack_runs in sorted(by_stack.items()):
            times = [_primary_time(r) for r in stack_runs]
            valid_times = [t for t in times if t is not None]
            failures = [r for r in stack_runs if r.get("correctness") not in (None, "pass")]
            stack_summaries.append(
                {
                    "stack": stack,
                    "cases": len(stack_runs),
                    "mean_execute_s": round(sum(valid_times) / len(valid_times), 6) if valid_times else None,
                    "failures": len(failures),
                }
            )
            if failures:
                anomalies.append(f"{stack}: {len(failures)} correctness failure(s)")

        summary["stacks"] = stack_summaries

        # Top slowest individual runs
        ranked = sorted(
            [r for r in runs if _primary_time(r) is not None],
            key=lambda r: _primary_time(r) or 0.0,
            reverse=True,
        )[:5]
        summary["slowest_cases"] = [
            {
                "case_id": r.get("case_id"),
                "stack": r.get("stack"),
                "execute_s": _primary_time(r),
                "correctness": r.get("correctness"),
            }
            for r in ranked
        ]

        # Setup vs execute breakdown when present
        setup_execute: list[dict[str, Any]] = []
        for run in runs[:20]:
            setup = run.get("setup_s_mean") or run.get("setup_s")
            execute = run.get("execute_s_mean") or run.get("execute_s")
            if setup is not None or execute is not None:
                setup_execute.append(
                    {
                        "case_id": run.get("case_id"),
                        "stack": run.get("stack"),
                        "setup_s": setup,
                        "execute_s": execute,
                    }
                )
        if setup_execute:
            summary["setup_vs_execute"] = setup_execute[:10]

        zero_bw = [r for r in runs if r.get("bandwidth_gbps") == 0.0]
        if zero_bw:
            anomalies.append(f"{len(zero_bw)} run(s) report zero bandwidth")

    # Optional trace.json presence check
    trace_files = list(candidate.rglob("trace.json"))[:5]
    if trace_files:
        summary["trace_files_found"] = len(trace_files)
        summary["note"] = "Use pypto-tooling profiling scripts for full swimlane; raw trace.json not inlined."
    else:
        summary["trace_files_found"] = 0

    summary["anomalies"] = anomalies
    return summary
