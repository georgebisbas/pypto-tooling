from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_hwnative_sys.knowledge import get_repository_meta, register_knowledge
from mcp_hwnative_sys.paths import (
    load_repos_config,
    safe_relpath,
    workspace_root,
)

mcp = FastMCP("hw-native-sys-workflows")

DEFAULT_RISK = "safe-read-only"
RISK_WRITES = "writes-build-artifacts"
RISK_ENV = "environment-sensitive"
RISK_LONG = "long-running"

_BLOCKED_SUBSTRINGS = (
    "git reset --hard",
    "git clean -fdx",
    "sudo rm -rf /",
    "rm -rf /",
    "rm -rf ~",
)


@dataclass(frozen=True)
class RepoConfig:
    name: str
    path: Path


@dataclass(frozen=True)
class TaskSpec:
    key: str
    command: str
    category: str = "misc"
    risk: str = DEFAULT_RISK
    long_running: bool = False
    environment_sensitive: bool = False
    duration_hint: str | None = None
    warning: str | None = None
    prerequisites: tuple[str, ...] = ()
    arch_families: tuple[str, ...] = ()
    requires: dict[str, Any] | None = None
    developer_only: bool = False
    container_notes: str | None = None


def _load_raw_config() -> dict[str, Any]:
    return load_repos_config()


def _load_repositories() -> tuple[Path, list[RepoConfig]]:
    raw_config = _load_raw_config()
    root = workspace_root(raw_config)
    repo_map = raw_config.get("repositories", {})

    repositories: list[RepoConfig] = []
    for name, relative_path in repo_map.items():
        repo_path = (root / str(relative_path)).resolve()
        repositories.append(RepoConfig(name=name, path=repo_path))

    return root, repositories


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y", "on"}
    return False


def _normalize_prerequisites(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                output.append(text)
        return tuple(output)
    return (str(value).strip(),) if str(value).strip() else ()


def _blocked_pattern(command: str) -> str | None:
    lowered = command.lower()
    for pattern in _BLOCKED_SUBSTRINGS:
        if pattern in lowered:
            return pattern
    return None


def _normalize_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                output.append(text)
        return tuple(output)
    return (str(value).strip(),) if str(value).strip() else ()


def _normalize_task_spec(task_key: str, raw_task: Any) -> TaskSpec:
    if isinstance(raw_task, str):
        command = raw_task.strip()
        if not command:
            raise ValueError(f"Task '{task_key}' has an empty command")
        blocked = _blocked_pattern(command)
        if blocked:
            raise ValueError(f"Task '{task_key}' contains blocked command pattern: {blocked}")
        return TaskSpec(key=task_key, command=command)

    if not isinstance(raw_task, dict):
        raise ValueError(f"Task '{task_key}' must be a string or object")

    command = str(raw_task.get("command", "")).strip()
    if not command:
        raise ValueError(f"Task '{task_key}' is missing a non-empty 'command'")
    blocked = _blocked_pattern(command)
    if blocked:
        raise ValueError(f"Task '{task_key}' contains blocked command pattern: {blocked}")

    risk = str(raw_task.get("risk", DEFAULT_RISK)).strip() or DEFAULT_RISK
    category = str(raw_task.get("category", "misc")).strip() or "misc"

    duration_hint_raw = str(raw_task.get("duration_hint", "")).strip()
    warning_raw = str(raw_task.get("warning", "")).strip()
    container_notes_raw = str(raw_task.get("container_notes", "")).strip()
    requires_raw = raw_task.get("requires")
    requires_dict = dict(requires_raw) if isinstance(requires_raw, dict) else None

    return TaskSpec(
        key=task_key,
        command=command,
        category=category,
        risk=risk,
        long_running=_to_bool(raw_task.get("long_running", False)),
        environment_sensitive=_to_bool(raw_task.get("environment_sensitive", False)),
        duration_hint=duration_hint_raw if duration_hint_raw else None,
        warning=warning_raw if warning_raw else None,
        prerequisites=_normalize_prerequisites(raw_task.get("prerequisites", [])),
        arch_families=_normalize_string_list(raw_task.get("arch_families", [])),
        requires=requires_dict,
        developer_only=_to_bool(raw_task.get("developer_only", False)),
        container_notes=container_notes_raw if container_notes_raw else None,
    )


def _merge_task_map(target: dict[str, TaskSpec], source: Any, source_name: str) -> None:
    if not isinstance(source, dict):
        raise ValueError(f"tasks.{source_name} must be an object")

    for key, value in source.items():
        task_key = str(key)
        target[task_key] = _normalize_task_spec(task_key, value)


def _tasks_for_repo(repo_name: str) -> dict[str, TaskSpec]:
    raw_config = _load_raw_config()
    tasks: dict[str, TaskSpec] = {}
    task_config = raw_config.get("tasks", {})

    default_tasks = task_config.get("default", {})
    repo_tasks = task_config.get(repo_name, {})

    _merge_task_map(tasks, default_tasks, "default")
    _merge_task_map(tasks, repo_tasks, repo_name)

    return tasks


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _task_warning(task: TaskSpec) -> str | None:
    warnings: list[str] = []

    if task.warning:
        warnings.append(task.warning)

    if task.risk == RISK_WRITES:
        warnings.append("This task writes build artifacts and may modify the working tree.")
    elif task.risk == RISK_ENV:
        warnings.append("This task depends on environment-specific setup (hardware, drivers, or toolchain).")
    elif task.risk == RISK_LONG:
        warnings.append("This task can run for a long time depending on suite size and machine load.")

    if task.long_running:
        warnings.append("This task is marked long-running.")
    if task.environment_sensitive:
        warnings.append("This task is marked environment-sensitive.")

    merged = _ordered_unique(warnings)
    if not merged:
        return None
    return " ".join(merged)


def _task_to_dict(task: TaskSpec, include_command: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "category": task.category,
        "risk": task.risk,
        "long_running": task.long_running,
        "environment_sensitive": task.environment_sensitive,
        "prerequisites": list(task.prerequisites),
    }

    if include_command:
        payload["command"] = task.command
    if task.duration_hint:
        payload["duration_hint"] = task.duration_hint
    if task.arch_families:
        payload["arch_families"] = list(task.arch_families)
    if task.requires:
        payload["requires"] = task.requires
    if task.developer_only:
        payload["developer_only"] = True
    if task.container_notes:
        payload["container_notes"] = task.container_notes

    warning = _task_warning(task)
    if warning:
        payload["warning"] = warning

    return payload


def _repo_index() -> tuple[Path, dict[str, RepoConfig]]:
    root, repositories = _load_repositories()
    return root, {repo.name: repo for repo in repositories}


def _require_repo(repo_name: str) -> tuple[Path, RepoConfig]:
    root, repo_by_name = _repo_index()
    repo = repo_by_name.get(repo_name)
    if repo is None:
        available = ", ".join(sorted(repo_by_name))
        raise ValueError(f"Unknown repo '{repo_name}'. Available: {available}")
    return root, repo


def _run_command(
    args: list[str],
    cwd: Path | None = None,
    timeout_seconds: int = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )


def _run_shell(command: str, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return _run_command(["bash", "-lc", command], cwd=cwd, timeout_seconds=timeout_seconds)


def _git(repo_path: Path, args: list[str], timeout_seconds: int = 20) -> subprocess.CompletedProcess[str]:
    return _run_command(["git", "-C", str(repo_path), *args], timeout_seconds=timeout_seconds)


def _status_counts(status_lines: list[str]) -> tuple[int, int, int]:
    staged = 0
    unstaged = 0
    untracked = 0

    for line in status_lines:
        if line.startswith("??"):
            untracked += 1
            continue

        if len(line) >= 2:
            if line[0] != " ":
                staged += 1
            if line[1] != " ":
                unstaged += 1

    return staged, unstaged, untracked


def _truncate_text(text: str, max_lines: int = 200, max_chars: int = 20000) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    truncated = "\n".join(lines)
    if len(truncated) > max_chars:
        return truncated[:max_chars] + "\n... (output truncated)"
    return truncated


@mcp.tool()
def list_repositories() -> list[dict[str, Any]]:
    """List configured repositories, architecture metadata, and disk availability."""
    root, repositories = _load_repositories()
    meta = get_repository_meta()

    output: list[dict[str, Any]] = []
    for repo in repositories:
        row: dict[str, Any] = {
            "name": repo.name,
            "path": safe_relpath(repo.path, root),
            "absolute_path": str(repo.path),
            "exists": repo.path.exists(),
            "is_git_repo": (repo.path / ".git").exists(),
        }
        if repo.name in meta:
            row["meta"] = meta[repo.name]
        output.append(row)

    return output


@mcp.tool()
def repository_health(include_clean: bool = True) -> dict[str, Any]:
    """Get branch, dirty state, and ahead/behind for every configured repo."""
    root, repositories = _load_repositories()
    output: list[dict[str, Any]] = []

    for repo in repositories:
        row: dict[str, Any] = {
            "repo": repo.name,
            "path": safe_relpath(repo.path, root),
            "exists": repo.path.exists(),
        }

        if not repo.path.exists():
            row["error"] = "Path does not exist"
            output.append(row)
            continue

        if not (repo.path / ".git").exists():
            row["error"] = "Not a git repository"
            output.append(row)
            continue

        branch_result = _git(repo.path, ["rev-parse", "--abbrev-ref", "HEAD"])
        row["branch"] = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

        status_result = _git(repo.path, ["status", "--porcelain"])
        status_lines = status_result.stdout.splitlines() if status_result.returncode == 0 else []
        staged, unstaged, untracked = _status_counts(status_lines)
        row["staged"] = staged
        row["unstaged"] = unstaged
        row["untracked"] = untracked
        row["clean"] = staged == 0 and unstaged == 0 and untracked == 0

        upstream_result = _git(
            repo.path,
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        )
        if upstream_result.returncode == 0:
            upstream = upstream_result.stdout.strip()
            row["upstream"] = upstream
            ahead_behind = _git(repo.path, ["rev-list", "--left-right", "--count", f"{upstream}...HEAD"])
            if ahead_behind.returncode == 0:
                counts = ahead_behind.stdout.strip().split()
                if len(counts) == 2:
                    row["behind"] = int(counts[0])
                    row["ahead"] = int(counts[1])
        else:
            row["upstream"] = None
            row["behind"] = 0
            row["ahead"] = 0

        commit_result = _git(repo.path, ["log", "-1", "--pretty=%h %cr %s"])
        row["last_commit"] = commit_result.stdout.strip() if commit_result.returncode == 0 else "unknown"

        if repo.name == "pypto" and row.get("branch") == "feat/host-collectives-builtin":
            row["active_program_hints"] = [
                {
                    "program": "host_collectives_program",
                    "plan": "pypto-3.0-notes/pr_plans/33-pypto-host-collectives-builtin-program.md",
                    "memory": "pypto-3.0-notes/memories/host_collectives.md",
                    "agent_verify": "pypto-tooling:host_collectives_ut_sim",
                    "developer_verify": "pypto:host_collectives_st_npu",
                    "route_task": "host_collectives_program",
                }
            ]

        if include_clean or not row["clean"]:
            output.append(row)

    return {
        "workspace_root": str(root),
        "repositories": output,
    }


@mcp.tool()
def search_code(
    query: str,
    repo: str = "all",
    file_glob: str = "*",
    max_results: int = 200,
    use_regex: bool = False,
) -> dict[str, Any]:
    """Search code across one repo or all repos with ripgrep."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    if max_results < 1 or max_results > 2000:
        raise ValueError("max_results must be between 1 and 2000")

    rg_path = shutil.which("rg")
    if rg_path is None:
        raise RuntimeError("ripgrep (rg) is required but was not found in PATH")

    root, repo_by_name = _repo_index()
    targets: list[RepoConfig] = []

    if repo == "all":
        targets = [repo_item for repo_item in repo_by_name.values() if repo_item.path.exists()]
    else:
        requested = [token.strip() for token in repo.split(",") if token.strip()]
        for item in requested:
            _, repo_item = _require_repo(item)
            if repo_item.path.exists():
                targets.append(repo_item)

    if not targets:
        return {"match_count": 0, "matches": []}

    command = [
        rg_path,
        "--line-number",
        "--no-heading",
        "--color",
        "never",
    ]
    if not use_regex:
        command.append("--fixed-strings")
    if file_glob and file_glob != "*":
        command.extend(["-g", file_glob])
    command.append(query)
    command.extend(str(item.path) for item in targets)

    proc = _run_command(command, timeout_seconds=90)
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr.strip() or "search failed")

    matches: list[dict[str, Any]] = []
    target_by_path = sorted(targets, key=lambda item: len(str(item.path)), reverse=True)

    for raw_line in proc.stdout.splitlines():
        parts = raw_line.split(":", 2)
        if len(parts) != 3:
            continue

        file_name, line_no, text = parts
        try:
            line_number = int(line_no)
        except ValueError:
            continue

        file_path = Path(file_name).resolve()
        repo_name = "unknown"
        for candidate in target_by_path:
            try:
                file_path.relative_to(candidate.path)
                repo_name = candidate.name
                break
            except ValueError:
                continue

        matches.append(
            {
                "repo": repo_name,
                "file": safe_relpath(file_path, root),
                "line": line_number,
                "text": text,
            }
        )

        if len(matches) >= max_results:
            break

    return {
        "match_count": len(matches),
        "truncated": len(proc.stdout.splitlines()) > len(matches),
        "matches": matches,
    }


@mcp.tool()
def list_tasks(repo: str, include_command: bool = True) -> dict[str, Any]:
    """List available named tasks for a repository, including metadata and warnings."""
    _, _ = _require_repo(repo)
    tasks = _tasks_for_repo(repo)

    task_items: dict[str, Any] = {}
    for key in sorted(tasks):
        task_items[key] = _task_to_dict(tasks[key], include_command=include_command)

    return {
        "repo": repo,
        "count": len(task_items),
        "tasks": task_items,
    }


@mcp.tool()
def run_task(
    repo: str,
    task: str,
    extra_args: str = "",
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run a configured task in a repository."""
    if timeout_seconds < 1 or timeout_seconds > 7200:
        raise ValueError("timeout_seconds must be between 1 and 7200")

    root, repo_cfg = _require_repo(repo)
    if not repo_cfg.path.exists():
        raise ValueError(f"Repository path does not exist: {repo_cfg.path}")

    tasks = _tasks_for_repo(repo)
    task_spec = tasks.get(task)
    if not task_spec:
        available = ", ".join(sorted(tasks))
        raise ValueError(f"Unknown task '{task}' for repo '{repo}'. Available: {available}")

    command = task_spec.command
    if extra_args.strip():
        command = f"{command} {extra_args.strip()}"

    proc = _run_shell(command, repo_cfg.path, timeout_seconds)
    warning = _task_warning(task_spec)

    return {
        "repo": repo,
        "path": safe_relpath(repo_cfg.path, root),
        "task": task,
        "command": command,
        "task_metadata": _task_to_dict(task_spec, include_command=False),
        "warning": warning,
        "exit_code": proc.returncode,
        "stdout": _truncate_text(proc.stdout),
        "stderr": _truncate_text(proc.stderr),
    }


@mcp.tool()
def run_command(
    repo: str,
    command: str,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Run an ad-hoc shell command in a repository."""
    if not command.strip():
        raise ValueError("command cannot be empty")
    if timeout_seconds < 1 or timeout_seconds > 7200:
        raise ValueError("timeout_seconds must be between 1 and 7200")

    root, repo_cfg = _require_repo(repo)
    if not repo_cfg.path.exists():
        raise ValueError(f"Repository path does not exist: {repo_cfg.path}")

    proc = _run_shell(command, repo_cfg.path, timeout_seconds)
    return {
        "repo": repo,
        "path": safe_relpath(repo_cfg.path, root),
        "command": command,
        "exit_code": proc.returncode,
        "stdout": _truncate_text(proc.stdout),
        "stderr": _truncate_text(proc.stderr),
    }


@mcp.tool()
def explain_task(task: str, repo: str = "pypto") -> dict[str, Any]:
    """Show the exact command and metadata configured for a named task."""
    _, _ = _require_repo(repo)
    tasks = _tasks_for_repo(repo)
    task_spec = tasks.get(task)
    if not task_spec:
        available = ", ".join(sorted(tasks))
        raise ValueError(f"Unknown task '{task}' for repo '{repo}'. Available: {available}")

    return {
        "repo": repo,
        "task": task,
        "details": _task_to_dict(task_spec, include_command=True),
    }


register_knowledge(mcp)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
