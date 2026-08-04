"""Generate a bash script for the PR gate workflow.

Rebase → pre-commit → sim-Docker tests → squash → force-push-with-lease.

The tool validates preconditions (clean tree, branch safety, docker image)
and produces a self-contained script the user can inspect before running.
It does NOT execute any git or docker commands itself.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from mcp_hwnative_sys.paths import load_repos_config, safe_relpath, workspace_root


def _run_command(
    args: list[str],
    cwd: Path | None = None,
    timeout_seconds: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )


def _git(
    repo_path: Path,
    args: list[str],
    timeout_seconds: int = 20,
) -> subprocess.CompletedProcess[str]:
    return _run_command(["git", "-C", str(repo_path), *args], timeout_seconds=timeout_seconds)


def _docker_image_exists(image: str) -> bool:
    proc = _run_command(["docker", "image", "inspect", image], timeout_seconds=15)
    return proc.returncode == 0


_CPP_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl")


def _changed_cpp_files(repo_path: Path, base_branch: str) -> list[str]:
    """Changed C/C++ files on this branch vs ``origin/<base_branch>``."""
    proc = _git(repo_path, ["diff", "--name-only", f"origin/{base_branch}...HEAD"])
    if proc.returncode != 0:
        return []
    return [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip().endswith(_CPP_EXTENSIONS)
    ]


def _resolve_repo_path(repo_name: str) -> Path:
    config = load_repos_config()
    root = workspace_root(config)
    repo_map = config.get("repositories", {})
    relative = repo_map.get(repo_name)
    if relative is None:
        available = ", ".join(sorted(repo_map))
        raise ValueError(f"Unknown repo '{repo_name}'. Available: {available}")
    return (root / str(relative)).resolve()


def _validate_preconditions(
    repo_path: Path,
    base_branch: str,
    test_paths: list[str],
    skip_tests: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Repo path exists and is a git repo
    if not repo_path.exists():
        errors.append(f"Repository path does not exist: {repo_path}")
        return errors, warnings
    if not (repo_path / ".git").exists():
        errors.append(f"Not a git repository: {repo_path}")
        return errors, warnings

    # 2. Working tree is clean
    status = _git(repo_path, ["status", "--porcelain"])
    if status.returncode != 0:
        errors.append(f"git status failed: {status.stderr.strip()}")
        return errors, warnings
    dirty_entries = [line for line in status.stdout.splitlines() if line.strip()]
    if dirty_entries:
        errors.append(
            f"Working tree is dirty ({len(dirty_entries)} entries). "
            "Commit or stash changes before gating."
        )

    # 3. Branch is not the base branch
    branch_result = _git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    if branch_result.returncode != 0:
        errors.append(f"Could not determine current branch: {branch_result.stderr.strip()}")
    else:
        current_branch = branch_result.stdout.strip()
        if current_branch == base_branch:
            errors.append(
                f"Currently on branch '{base_branch}'. Create a feature branch first."
            )
        if current_branch == "HEAD":
            errors.append("Detached HEAD state. Create a branch first.")

    # 4. Remote origin exists and has the base branch
    origin_url = _git(repo_path, ["remote", "get-url", "origin"])
    if origin_url.returncode != 0:
        warnings.append("No 'origin' remote configured. Fetch/rebase steps will fail.")
    else:
        remote_refs = _git(repo_path, ["ls-remote", "--heads", "origin", base_branch])
        if remote_refs.returncode != 0:
            warnings.append(
                f"Could not reach origin to check '{base_branch}' branch."
            )
        elif not remote_refs.stdout.strip():
            errors.append(
                f"Branch '{base_branch}' not found on origin. Is the remote name correct?"
            )

    # 5. Docker image exists (only if tests are not skipped)
    if not skip_tests:
        if not _docker_image_exists("pypto3-hw-native-sys:sim"):
            errors.append(
                "Docker image 'pypto3-hw-native-sys:sim' not found. "
                "Build it first: run pypto-tooling:docker_build_sim."
            )

    # 6. Test paths exist
    if not skip_tests and test_paths:
        for tp in test_paths:
            resolved = (repo_path / tp).resolve()
            if not resolved.exists():
                errors.append(f"Test path not found: {tp}")

    return errors, warnings


def _generate_script(
    repo_path: Path,
    repo_name: str,
    base_branch: str,
    test_paths: list[str],
    extra_test_args: str,
    skip_pre_commit: bool,
    skip_tests: bool,
    push_remote: str,
) -> str:
    """Build the self-contained bash script as a string."""
    mounts = []
    mounts.append(f"-v $(git rev-parse --show-toplevel):/opt/{repo_name}")

    sections: list[str] = []

    sections.append("#!/usr/bin/env bash")
    sections.append("#")
    sections.append(f"# PR gate script for {repo_name}")
    sections.append(f"# Generated by hw-native-sys MCP gate_pr_script")
    sections.append("#")
    sections.append("set -euo pipefail")
    sections.append("")
    sections.append('BOLD="\\033[1m"')
    sections.append('GREEN="\\033[32m"')
    sections.append('CYAN="\\033[36m"')
    sections.append('RESET="\\033[0m"')
    sections.append("")
    sections.append("step() {")
    sections.append('  echo -e "${BOLD}${CYAN}$1${RESET}"')
    sections.append("}")
    sections.append("")

    # Step 1: FETCH
    step_num = 1
    sections.append(f"# {'—' * 60}")
    sections.append(f"# Step {step_num}: Fetch origin")
    sections.append(f"# {'—' * 60}")
    sections.append(f'step "[{step_num}/{7 - (1 if skip_tests else 0) if skip_tests else 7}] Fetching origin ..."')
    sections.append("git fetch origin")
    sections.append("")

    # Step 2: REBASE
    step_num += 1
    sections.append(f"# Step {step_num}: Rebase onto origin/{base_branch}")
    sections.append(f'step "[{step_num}/7] Rebasing onto origin/{base_branch} ..."')
    sections.append(f"git rebase origin/{base_branch}")
    sections.append("")

    # Step 3: Commit log
    step_num += 1
    sections.append(f"# Step {step_num}: Verify commits")
    sections.append(f'step "[{step_num}/7] Commits to push:"')
    sections.append(f"git log origin/{base_branch}..HEAD --oneline --decorate")
    sections.append("")

    # Step 4: Pre-commit
    step_num += 1
    total_steps = 7 if not skip_tests else 6
    if skip_pre_commit:
        sections.append(f"# Step {step_num}: PRE-COMMIT (SKIPPED via --skip-pre-commit)")
    else:
        sections.append(f"# Step {step_num}: Run pre-commit")
        sections.append(f'step "[{step_num}/{total_steps}] Running pre-commit on all files ..."')
        sections.append("pre-commit run --all-files")
    sections.append("")

    # Step 5: Sim Docker tests
    step_num += 1
    if skip_tests:
        sections.append(f"# Step {step_num}: TESTS (SKIPPED via --skip-tests)")
        sections.append("")
    else:
        sections.append(f"# Step {step_num}: Sim Docker tests")
        sections.append(f'step "[{step_num}/{total_steps}] Building and running tests in sim Docker ..."')
        sections.append("")

        # Build the pytest command inside Docker
        mount_str = " ".join(mounts)
        docker_cmd_parts = [
            "docker run --rm --shm-size=4g",
            mount_str,
            "pypto3-hw-native-sys:sim",
            "bash -c \"",
            f"cd /opt/{repo_name}",
            "rm -rf build _skbuild",
            "pip install --no-build-isolation -e '.[dev]'",
            f"PYTHONPATH=/opt/{repo_name}/python:\\$PYTHONPATH \\",
        ]

        if test_paths:
            paths_str = " ".join(test_paths)
            docker_cmd_parts.append(f"  pytest {paths_str} -v --forked --platform=a2a3sim,a5sim --device=0,1,2,3")
        else:
            docker_cmd_parts.append(
                "  pytest tests/st -v --forked --platform=a2a3sim,a5sim --device=0,1,2,3"
            )

        if extra_test_args.strip():
            docker_cmd_parts[-1] = f"{docker_cmd_parts[-1]} {extra_test_args.strip()}"

        docker_cmd_parts.append("\"")

        docker_cmd = " \\\n    ".join(docker_cmd_parts)
        sections.append(docker_cmd)
        sections.append("")

    # Step 6: Squash
    step_num = 6
    sections.append(f"# Step {step_num}: Squash commits (interactive)")
    sections.append(f'step "[{step_num}/{total_steps}] Squashing commits onto origin/{base_branch} (interactive) ..."')
    sections.append(f"git rebase -i origin/{base_branch}")
    sections.append("")

    # Step 7: Push
    step_num = 7
    sections.append(f"# Step {step_num}: Force-push with lease")
    sections.append(f'step "[{step_num}/{total_steps}] Force-pushing to {push_remote} ..."')
    sections.append(f"git push --force-with-lease {push_remote} HEAD")
    sections.append("")

    sections.append("# Done.")
    sections.append(f'echo -e "${{GREEN}}PR gate workflow complete for {repo_name}.${{RESET}}"')

    return "\n".join(sections)


def gate_pr_script_impl(
    repo: str,
    test_paths: list[str] | None = None,
    extra_test_args: str = "",
    base_branch: str = "main",
    skip_pre_commit: bool = False,
    skip_tests: bool = False,
    push_remote: str = "origin",
) -> dict[str, Any]:
    """Generate a bash script for the PR gate workflow.

    Validates preconditions (clean tree, branch safety, docker image) and
    produces a self-contained script. Does not execute any commands.
    """
    if not repo.strip():
        raise ValueError("repo cannot be empty")
    if base_branch.strip() == "":
        raise ValueError("base_branch cannot be empty")
    if not push_remote.strip():
        raise ValueError("push_remote cannot be empty")

    paths = test_paths or []
    resolved_test_paths = [p.strip() for p in paths if p.strip()]

    repo_path = _resolve_repo_path(repo)

    errors, warnings = _validate_preconditions(
        repo_path,
        base_branch,
        resolved_test_paths,
        skip_tests,
    )

    branch_result = _git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])
    current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

    changed_cpp = _changed_cpp_files(repo_path, base_branch)

    root = workspace_root()

    script = _generate_script(
        repo_path=repo_path,
        repo_name=repo,
        base_branch=base_branch,
        test_paths=resolved_test_paths,
        extra_test_args=extra_test_args,
        skip_pre_commit=skip_pre_commit,
        skip_tests=skip_tests,
        push_remote=push_remote,
    )

    total_steps = 7 if not skip_tests else 6
    step_count = total_steps
    step_labels = [
        {"number": 1, "label": "Fetch origin", "command_snippet": "git fetch origin"},
        {"number": 2, "label": f"Rebase onto origin/{base_branch}", "command_snippet": f"git rebase origin/{base_branch}"},
        {"number": 3, "label": "Verify commits", "command_snippet": f"git log origin/{base_branch}..HEAD --oneline"},
        {
            "number": 4,
            "label": "Pre-commit" if not skip_pre_commit else "Pre-commit (skipped)",
            "command_snippet": "pre-commit run --all-files" if not skip_pre_commit else "(skipped)",
        },
        {
            "number": 5,
            "label": "Sim Docker tests" if not skip_tests else "Sim Docker tests (skipped)",
            "command_snippet": "docker run --rm --shm-size=4g ..." if not skip_tests else "(skipped)",
        },
        {"number": 6, "label": "Squash commits", "command_snippet": f"git rebase -i origin/{base_branch}"},
        {"number": 7, "label": f"Force-push to {push_remote}", "command_snippet": f"git push --force-with-lease {push_remote} HEAD"},
    ]
    if skip_tests:
        step_labels = [s for s in step_labels if s["number"] != 5]
        # renumber steps 6,7 to 5,6
        for s in step_labels:
            if s["number"] >= 6:
                s["number"] -= 1

    abs_path = str(repo_path)
    after_push = [
        "Run the review-bugbot skill to catch regressions",
        "Run the review-security skill for security issues",
        "If all green, the PR is ready for the developer gate (NPU verify)",
    ]
    if changed_cpp:
        after_push.insert(
            0, "Run clang-tidy on the changed C++ files (see tools/clang_tidy_workflow)"
        )
    agent_instructions = {
        "after_push": after_push,
        "review_hints": {
            "bugbot_prompt": f"Full Repository Path: {abs_path}\nDiff: branch changes",
            "security_prompt": f"Full Repository Path: {abs_path}\nDiff: branch changes",
        },
    }

    return {
        "repo": repo,
        "repo_path": safe_relpath(repo_path, root),
        "absolute_path": abs_path,
        "branch": current_branch,
        "base_branch": base_branch,
        "precondition": {
            "healthy": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        },
        "test_paths": resolved_test_paths,
        "test_paths_resolved": [
            str((repo_path / p).resolve()) for p in resolved_test_paths
        ],
        "script": script,
        "step_count": step_count,
        "steps": step_labels,
        "changed_cpp_files": changed_cpp,
        "agent_instructions": agent_instructions,
        "notes": [
            "Review the generated script before running it",
            "The squash step requires interactive rebase — review your commits",
            "Run the script from the repo root",
        ],
    }
