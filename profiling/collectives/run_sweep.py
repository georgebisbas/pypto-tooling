"""Benchmark driver — pair-mesh and single-stack subcommands (E1 stub)."""

from __future__ import annotations

import argparse
import sys

from collectives.equivalence import EquivalenceCase


def _cmd_validate_case(path: str) -> int:
    case = EquivalenceCase.from_json_file(path)
    case.validate()
    print(f"case_id={case.case_id}")
    print(f"equivalence_hash={case.equivalence_hash()}")
    print(f"n_bytes={case.n_bytes} window_nbytes={case.window_nbytes}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collective benchmark sweep (pypto-tooling)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-case", help="Validate EquivalenceCase JSON")
    p_val.add_argument("--case-file", required=True)

    p_pair = sub.add_parser("pair-mesh", help="Run same case on simpler and pypto (E1)")
    p_pair.add_argument("--case-file", required=True)
    p_pair.add_argument("--stacks", default="simpler,pypto")
    p_pair.add_argument("--campaign", default="default")
    p_pair.add_argument("--profile", default="", help="Comma: l2,pmu,dep")
    p_pair.add_argument("--out", required=True, help="results.json path under results/campaigns/")

    args = parser.parse_args(argv)
    if args.command == "validate-case":
        return _cmd_validate_case(args.case_file)
    if args.command == "pair-mesh":
        case = EquivalenceCase.from_json_file(args.case_file)
        case.validate()
        print(f"pair-mesh: case_id={case.case_id} stacks={args.stacks}", file=sys.stderr)
        print("run_sweep pair-mesh: not implemented (E1). Use manual commands in profiling/README.md", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
