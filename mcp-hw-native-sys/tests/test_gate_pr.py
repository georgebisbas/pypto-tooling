"""Tests for the gate_pr_script tool (mcp_hwnative_sys.gate_pr)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mcp_hwnative_sys.gate_pr import _generate_script, gate_pr_script_impl
from mcp_hwnative_sys.paths import load_repos_config, workspace_root




class TestParameterValidation:
    def test_empty_repo_raises(self):
        with pytest.raises(ValueError, match="repo cannot be empty"):
            gate_pr_script_impl("", test_paths=[], base_branch="main")

    def test_empty_base_branch_raises(self):
        with pytest.raises(ValueError, match="base_branch cannot be empty"):
            gate_pr_script_impl("pypto", test_paths=[], base_branch="")

    def test_empty_push_remote_raises(self):
        with pytest.raises(ValueError, match="push_remote cannot be empty"):
            gate_pr_script_impl("pypto", test_paths=[], base_branch="main", push_remote="")

    def test_unknown_repo_raises(self):
        with pytest.raises(ValueError, match="Unknown repo"):
            gate_pr_script_impl("nonexistent-repo-xyz", test_paths=[], base_branch="main")


class TestScriptGeneration:
    """Tests for _generate_script that don't need git or docker."""

    def _make_fake_repo_path(self) -> Path:
        root = workspace_root()
        config = load_repos_config()
        repo_map = config.get("repositories", {})
        first_name = next(iter(repo_map))
        return (root / str(repo_map[first_name])).resolve()

    def test_script_begins_with_shebang(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=["tests/test_foo.py"],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert script.startswith("#!/usr/bin/env bash")

    def test_script_has_set_euo_pipefail(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "set -euo pipefail" in script

    def test_script_contains_fetch(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "git fetch origin" in script

    def test_script_contains_rebase(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="my-base",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "git rebase origin/my-base" in script

    def test_script_contains_push(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="fork-gbisbas",
        )
        assert "git push --force-with-lease fork-gbisbas HEAD" in script

    def test_skip_pre_commit_marks_step_skipped(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=True,
            skip_tests=False,
            push_remote="origin",
        )
        assert "SKIPPED" in script
        assert "pre-commit run --all-files" not in script

    def test_skip_tests_marks_step_skipped(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=True,
            push_remote="origin",
        )
        assert "SKIPPED" in script
        assert "docker run" not in script

    def test_test_paths_appear_in_script(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[
                "tests/st/distributed/test_foo.py",
                "tests/ut/test_bar.py",
            ],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "tests/st/distributed/test_foo.py" in script
        assert "tests/ut/test_bar.py" in script

    def test_extra_test_args_appear_in_script(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=["tests/test_foo.py"],
            extra_test_args="-k allreduce -v",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "-k allreduce -v" in script

    def test_script_contains_commit_log_section(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "git log origin/main..HEAD --oneline" in script

    def test_script_contains_squash_section(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "git rebase -i origin/main" in script

    def test_script_contains_shm_size(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=["tests/test_foo.py"],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "--shm-size=4g" in script

    def test_no_test_paths_defaults_to_st_suite(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "pytest tests/st" in script

    def test_script_ends_with_done_message(self):
        script = _generate_script(
            repo_path=Path("/tmp/fake"),
            repo_name="pypto",
            base_branch="main",
            test_paths=[],
            extra_test_args="",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert "PR gate workflow complete" in script


class TestStructuredResult:
    """Tests for the structured result from gate_pr_script_impl."""

    @staticmethod
    def _setup_clean_git(monkeypatch):
        """Mock _git to simulate a clean feature branch with valid origin."""

        def _mock_git(repo_path, args, timeout_seconds=20):
            cmd_str = " ".join(args)
            if "status --porcelain" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="", stderr=""
                )
            elif "rev-parse --abbrev-ref HEAD" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="feat/test-branch\n", stderr=""
                )
            elif "remote get-url origin" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="https://github.com/org/repo\n", stderr=""
                )
            elif "ls-remote" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="abc123\trefs/heads/main\n", stderr=""
                )
            return subprocess.CompletedProcess(
                args=["git"] + args, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr("mcp_hwnative_sys.gate_pr._git", _mock_git)
        monkeypatch.setattr(
            "mcp_hwnative_sys.gate_pr._docker_image_exists",
            lambda image: True,
        )

    def test_result_has_repo_and_branch(self, monkeypatch):
        self._setup_clean_git(monkeypatch)
        result = gate_pr_script_impl(
            "pypto",
            test_paths=[],
            extra_test_args="",
            base_branch="main",
            skip_pre_commit=False,
            skip_tests=True,
            push_remote="origin",
        )
        assert result["repo"] == "pypto"
        assert result["branch"] == "feat/test-branch"
        assert result["base_branch"] == "main"
        assert "script" in result
        assert "steps" in result
        assert "agent_instructions" in result
        assert "notes" in result

    def test_result_precondition_keys_present(self, monkeypatch):
        self._setup_clean_git(monkeypatch)
        result = gate_pr_script_impl(
            "pypto",
            test_paths=[],
            extra_test_args="",
            base_branch="main",
            skip_pre_commit=False,
            skip_tests=True,
            push_remote="origin",
        )
        assert "precondition" in result
        assert "healthy" in result["precondition"]
        assert "errors" in result["precondition"]
        assert "warnings" in result["precondition"]

    def test_result_agent_instructions_have_review_hints(self, monkeypatch):
        self._setup_clean_git(monkeypatch)
        result = gate_pr_script_impl(
            "pypto",
            test_paths=[],
            extra_test_args="",
            base_branch="main",
            skip_pre_commit=False,
            skip_tests=True,
            push_remote="origin",
        )
        ai = result["agent_instructions"]
        assert "after_push" in ai
        assert "review_hints" in ai
        assert "bugbot_prompt" in ai["review_hints"]
        assert "security_prompt" in ai["review_hints"]
        assert "Full Repository Path" in ai["review_hints"]["bugbot_prompt"]

    def test_result_step_count_with_tests(self, monkeypatch):
        self._setup_clean_git(monkeypatch)
        result = gate_pr_script_impl(
            "pypto",
            test_paths=["tests/test_foo.py"],
            extra_test_args="",
            base_branch="main",
            skip_pre_commit=False,
            skip_tests=False,
            push_remote="origin",
        )
        assert result["step_count"] == 7

    def test_result_step_count_skip_tests(self, monkeypatch):
        self._setup_clean_git(monkeypatch)
        result = gate_pr_script_impl(
            "pypto",
            test_paths=[],
            extra_test_args="",
            base_branch="main",
            skip_pre_commit=False,
            skip_tests=True,
            push_remote="origin",
        )
        assert result["step_count"] == 6

    def test_dirty_tree_reported_as_error(self, monkeypatch):
        def _mock_git(repo_path, args, timeout_seconds=20):
            cmd_str = " ".join(args)
            if "status --porcelain" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout=" M file.txt\n?? new.py\n", stderr=""
                )
            elif "rev-parse --abbrev-ref HEAD" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="feat/test-branch\n", stderr=""
                )
            elif "remote get-url origin" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="https://github.com/org/repo\n", stderr=""
                )
            elif "ls-remote" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="abc123\trefs/heads/main\n", stderr=""
                )
            return subprocess.CompletedProcess(
                args=["git"] + args, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr("mcp_hwnative_sys.gate_pr._git", _mock_git)
        monkeypatch.setattr(
            "mcp_hwnative_sys.gate_pr._docker_image_exists",
            lambda image: True,
        )

        result = gate_pr_script_impl(
            "pypto",
            test_paths=[],
            extra_test_args="",
            base_branch="main",
            skip_pre_commit=False,
            skip_tests=True,
            push_remote="origin",
        )
        assert not result["precondition"]["healthy"]
        assert any("dirty" in e.lower() for e in result["precondition"]["errors"])

    def test_on_base_branch_reported_as_error(self, monkeypatch):
        def _mock_git(repo_path, args, timeout_seconds=20):
            cmd_str = " ".join(args)
            if "status --porcelain" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="", stderr=""
                )
            elif "rev-parse --abbrev-ref HEAD" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="main\n", stderr=""
                )
            elif "remote get-url origin" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="https://github.com/org/repo\n", stderr=""
                )
            elif "ls-remote" in cmd_str:
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="abc123\trefs/heads/main\n", stderr=""
                )
            return subprocess.CompletedProcess(
                args=["git"] + args, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr("mcp_hwnative_sys.gate_pr._git", _mock_git)
        monkeypatch.setattr(
            "mcp_hwnative_sys.gate_pr._docker_image_exists",
            lambda image: True,
        )

        result = gate_pr_script_impl(
            "pypto",
            test_paths=[],
            extra_test_args="",
            base_branch="main",
            skip_pre_commit=False,
            skip_tests=True,
            push_remote="origin",
        )
        assert not result["precondition"]["healthy"]
        assert any(
            "feature branch" in e.lower() for e in result["precondition"]["errors"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
