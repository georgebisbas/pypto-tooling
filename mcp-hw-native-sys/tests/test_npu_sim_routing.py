"""Tests for the NPU-or-sim-Docker build/test routing (mcp_hwnative_sys.server).

Covers the sim-docker redirect builder, heavy-task classification, the
run_command build/test refusal pattern, and the run_task refusal path taken
when no NPU is reachable and the sim image is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_hwnative_sys import server
from mcp_hwnative_sys import ascend_env


def _force_no_npu() -> None:
    ascend_env._NPU_CACHE["available"] = False
    ascend_env._NPU_CACHE["ts"] = 0.0


def _force_npu() -> None:
    ascend_env._NPU_CACHE["available"] = True
    ascend_env._NPU_CACHE["ts"] = 0.0


def test_pypto_redirect_builds_docker_command(monkeypatch):
    _force_no_npu()
    monkeypatch.setattr(server, "npu_available", lambda *a, **k: False)
    redirect = server._sim_docker_redirect("pypto", Path("/tmp/pypto"), "pytest tests/ut -q")
    assert redirect is not None
    docker_command, image = redirect
    assert image == "pypto3-hw-native-sys:sim"
    assert "docker run --rm --shm-size=4g" in docker_command
    assert "-v /tmp/pypto:/opt/pypto" in docker_command
    assert "pytest tests/ut -q" in docker_command
    assert "pip install --no-build-isolation" in docker_command


def test_redirect_per_repo_images():
    _force_no_npu()
    assert server._sim_docker_redirect("simpler", Path("/tmp/s"), "pytest tests/ut")[1] == "simpler-hw-native-sys:sim"
    assert server._sim_docker_redirect("pypto-lib", Path("/tmp/p"), "pytest tests/golden")[1] == "pypto-lib-hw-native-sys:sim"
    assert server._sim_docker_redirect("pto-isa", Path("/tmp/i"), "python3 tests/run_cpu.py")[1] == "pypto3-hw-native-sys:sim"


def test_no_sim_image_repo_returns_none():
    _force_no_npu()
    assert server._sim_docker_redirect("PTOAS", Path("/tmp/PTOAS"), "ninja -C build") is None


def test_heavy_task_classification():
    tasks = server._tasks_for_repo("pypto")
    assert server._needs_npu_or_sim(tasks["unit_tests_fast"]) is True
    assert server._needs_npu_or_sim(tasks["codegen_tests"]) is True
    # lint tasks are not routed to sim docker
    assert server._needs_npu_or_sim(tasks["ruff_fix"]) is False


def test_sim_docker_flag_parsed():
    tasks = server._tasks_for_repo("pypto-tooling")
    assert tasks["host_collectives_ut_sim"].sim_docker is True
    assert tasks["docker_build_sim"].sim_docker is True
    # a plain host test task is not marked containerized
    assert server._tasks_for_repo("pypto")["unit_tests_fast"].sim_docker is False


def test_run_command_refuses_heavy_when_no_npu(monkeypatch):
    _force_no_npu()
    monkeypatch.setattr(server, "npu_available", lambda *a, **k: False)
    with pytest.raises(ValueError, match="sim Docker"):
        server.run_command(repo="pypto", command="pytest tests/ut -q")


def test_run_command_allows_dockerized_heavy_when_no_npu(monkeypatch):
    # A docker run wrapper is the sanctioned sim-Docker path — must not be refused.
    _force_no_npu()
    monkeypatch.setattr(server, "npu_available", lambda *a, **k: False)
    captured: list[str] = []

    def fake_run_shell(command, cwd, timeout_seconds):
        captured.append(command)
        return server._run_command(["printf", "ok"])

    monkeypatch.setattr(server, "_run_shell", fake_run_shell)
    result = server.run_command(
        repo="pypto",
        command="docker run --rm -v /tmp/pypto:/opt/pypto pypto3-hw-native-sys:sim bash -lc 'pytest tests/ut -q'",
    )
    assert result["exit_code"] == 0
    assert len(captured) == 1


def test_run_command_allows_read_only_when_no_npu(monkeypatch):
    _force_no_npu()
    monkeypatch.setattr(server, "npu_available", lambda *a, **k: False)
    captured: list[str] = []

    def fake_run_shell(command, cwd, timeout_seconds):
        captured.append(command)
        return server._run_command(["printf", "ok"])

    monkeypatch.setattr(server, "_run_shell", fake_run_shell)
    result = server.run_command(repo="pypto", command="git status -sb")
    assert result["exit_code"] == 0
    assert captured == ["git status -sb"]


def test_run_command_allows_build_when_npu(monkeypatch):
    _force_npu()
    monkeypatch.setattr(server, "npu_available", lambda *a, **k: True)
    captured: list[str] = []

    def fake_run_shell(command, cwd, timeout_seconds):
        captured.append(command)
        return server._run_command(["printf", "ok"])

    monkeypatch.setattr(server, "_run_shell", fake_run_shell)
    result = server.run_command(repo="pypto", command="pytest tests/ut -q")
    assert result["exit_code"] == 0
    assert captured == ["pytest tests/ut -q"]


def test_run_task_refuses_when_no_npu_and_no_image(monkeypatch):
    _force_no_npu()
    monkeypatch.setattr(server, "npu_available", lambda *a, **k: False)
    monkeypatch.setattr(server, "_docker_image_present", lambda image: False)
    result = server.run_task(repo="pypto", task="unit_tests_fast")
    assert result["exit_code"] == -1
    assert "sim Docker image" in result["note"]
    assert "Refused" in result["stderr"]


def test_run_task_redirects_when_no_npu_and_image_present(monkeypatch):
    _force_no_npu()
    monkeypatch.setattr(server, "npu_available", lambda *a, **k: False)
    monkeypatch.setattr(server, "_docker_image_present", lambda image: True)
    # Poke a tiny shell command through the redirect path by monkeypatching
    # _run_shell to capture what would be executed.
    captured: dict[str, str] = {}

    def fake_run_shell(command, cwd, timeout_seconds):
        captured["command"] = command
        captured["cwd"] = str(cwd)
        return server._run_command(["printf", "ran"])

    monkeypatch.setattr(server, "_run_shell", fake_run_shell)
    result = server.run_task(repo="pypto", task="unit_tests_fast", extra_args="--tb=short")
    assert result["exit_code"] == 0
    assert "docker run" in captured["command"]
    assert "--tb=short" in captured["command"]
    assert "pypto3-hw-native-sys:sim" in captured["command"]
    assert "redirected into sim Docker" in result["note"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
