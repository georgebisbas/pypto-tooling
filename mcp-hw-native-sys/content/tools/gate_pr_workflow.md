# PR Gate workflow

*MCP-owned — canonical source for the end-to-end PR gate workflow.*

## Overview

The `gate_pr_script` tool generates a self-contained bash script for the full PR gate:

1. **Fetch** origin
2. **Rebase** onto `origin/main` (or specified base branch)
3. **Show commits** that will be pushed
4. **Pre-commit** checks (`pre-commit run --all-files`)
5. **Sim Docker tests** (build + run targeted tests)
6. **Squash** commits (interactive rebase)
7. **Force-push with lease** to origin

## When this applies

- Feature branch is ready for PR review
- All code changes are committed locally
- Tests pass locally in sim Docker

## How to use

Call `gate_pr_script` from the MCP server with:

- `repo` — repository name (e.g. `"pypto"`)
- `test_paths` — list of pytest test files/dirs (repo-relative)
- `extra_test_args` — additional pytest flags (e.g. `"-k allreduce"`)
- `skip_pre_commit` — skip pre-commit checks (default false)
- `skip_tests` — skip sim Docker tests (default false)

Example:

```python
gate_pr_script(
    repo="pypto",
    test_paths=["tests/st/distributed/collectives/test_l3_tensor_allreduce_bidirectional_ring_intrinsic.py"],
    extra_test_args="-k allreduce -v"
)
```

## Precondition checks

Before generating the script, the tool validates:

- Repository exists and is a git repo
- Working tree is clean (no uncommitted changes)
- Current branch is not the base branch
- `origin` remote exists and has the base branch
- Docker image `pypto3-hw-native-sys:sim` exists (unless tests skipped)
- Test paths exist on disk (unless tests skipped)

Errors are returned in the `precondition.errors` field; warnings in `precondition.warnings`.

## Script structure

The generated script uses `set -euo pipefail` and stops on first error. Each step has a colored header.

### Sim Docker test section

The test step follows the [sim Docker workflow](sim_docker_workflow.md) golden rules:

- Mounts the repo via `-v $(pwd):/opt/{repo}`
- Runs `rm -rf build _skbuild` before build
- Uses `pip install --no-build-isolation -e '.[dev]'` (scikit-build-core)
- Never runs `cmake --build build --parallel`
- Uses `--forked --platform=a2a3sim,a5sim --device=0,1,2,3`

## After the gate: agent review

After the script completes successfully, the MCP tool returns `agent_instructions`:

1. Run the **review-bugbot** skill to catch regressions
2. Run the **review-security** skill for security issues
3. If all green, the PR is ready for the **developer gate** (NPU verify)

## Related resources

- `tools/sim_docker_workflow` — sim Docker iteration rules
- `ascend/hccl_container_checklist` — NPU/container verification checklist
- `hw-native-sys://agent/distributed_work_policy` — agent guardrails for distributed work
