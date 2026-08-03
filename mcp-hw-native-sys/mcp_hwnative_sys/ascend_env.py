"""Read-only Ascend / CANN environment diagnostics for MCP agents."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _run_capture(command: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return -1, "", str(exc)


def _first_existing_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _guess_hccl_preload(cann_home: str | None) -> str | None:
    # Ascend NPU hosts are aarch64, so the arch-suffixed CANN layout is tried
    # first; the plain lib64 fallback covers x86 CANN installs. Each candidate
    # is existence-checked below, so a wrong guess is harmless.
    if not cann_home:
        return None
    candidates = [
        Path(cann_home) / "aarch64-linux" / "lib64" / "libhccl.so",
        Path(cann_home) / "lib64" / "libhccl.so",
    ]
    ascend = os.environ.get("ASCEND_HOME_PATH", "").strip()
    if ascend:
        candidates.extend(
            [
                Path(ascend) / "aarch64-linux" / "lib64" / "libhccl.so",
                Path(ascend) / "lib64" / "libhccl.so",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _parse_npu_smi_device_count(stdout: str) -> int | None:
    # npu-smi info prints NPU ID lines; count unique IDs when possible
    ids = set(re.findall(r"^\s*(\d+)\s+\d+", stdout, flags=re.MULTILINE))
    if ids:
        return len(ids)
    if "No NPU" in stdout or "not found" in stdout.lower():
        return 0
    return stdout.lower().count("npu") if "npu" in stdout.lower() else None


def _suggest_platform() -> str:
    ascend = os.environ.get("ASCEND_HOME_PATH", "").lower()
    if "950" in ascend or "a5" in ascend:
        return "a5"
    return "a2a3"


# Short-lived cache so run_task/run_command routing does not shell out to
# npu-smi on every call. 10s is long enough to amortize repeated routing
# within one interaction, short enough to notice a device being released.
_NPU_CACHE: dict[str, Any] = {"ts": 0.0, "available": None}


def _detect_npu_available() -> bool:
    """Return True only when at least one NPU device is confirmed reachable."""
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        return False
    code, stdout, stderr = _run_capture([npu_smi, "info"])
    if code != 0:
        return False
    count = _parse_npu_smi_device_count(stdout)
    return bool(count and count > 0)


def npu_available(cache_seconds: float = 10.0) -> bool:
    """Whether at least one NPU device is reachable (cached).

    Anything short of a confirmed device — missing npu-smi, failed npu-smi
    invocation, or an unparseable output — is treated as *not available*.
    Callers use this as the gate for host-side build/test: no confirmed NPU
    means build/test work must run inside the sim Docker images instead.
    """
    now = time.monotonic()
    if _NPU_CACHE["available"] is not None and now - _NPU_CACHE["ts"] < cache_seconds:
        return _NPU_CACHE["available"]
    available = _detect_npu_available()
    _NPU_CACHE["ts"] = now
    _NPU_CACHE["available"] = available
    return available


def npu_env_summary() -> dict[str, Any]:
    """Structured NPU availability facts for task-result envelopes."""
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        return {"available": False, "devices": 0, "reason": "npu-smi not in PATH"}
    code, stdout, stderr = _run_capture([npu_smi, "info"])
    if code != 0:
        return {"available": False, "devices": 0, "reason": f"npu-smi info failed: {stderr[:120]}"}
    count = _parse_npu_smi_device_count(stdout)
    if not count:
        return {"available": False, "devices": 0, "reason": "npu-smi reported no devices"}
    return {"available": True, "devices": count, "reason": "npu-smi ok"}


def ascend_env_check_impl() -> dict[str, Any]:
    """Collect read-only Ascend/CANN environment facts for agent bootstrap."""
    warnings: list[str] = []
    in_docker = Path("/.dockerenv").exists()

    cann_home = _first_existing_env("CANN_HOME", "ASCEND_HOME_PATH")
    if not cann_home:
        warnings.append("CANN_HOME / ASCEND_HOME_PATH not set — NPU tasks will fail.")

    npu_smi = shutil.which("npu-smi")
    device_count: int | None = None
    npu_smi_excerpt = ""
    if npu_smi:
        code, stdout, stderr = _run_capture([npu_smi, "info"])
        if code == 0:
            npu_smi_excerpt = stdout[:2000]
            device_count = _parse_npu_smi_device_count(stdout)
        else:
            warnings.append(f"npu-smi info failed (exit {code}): {stderr[:200]}")
    else:
        warnings.append("npu-smi not in PATH — cannot enumerate devices.")

    hccl_lib = _guess_hccl_preload(cann_home)
    hccl_preload_recommended = hccl_lib is not None

    if in_docker and device_count and device_count >= 2:
        warnings.append("Multi-NPU in Docker: use --pid=host on docker run for HCCL.")
    if hccl_preload_recommended:
        warnings.append(
            "Set LD_PRELOAD to libhccl.so in the test shell only (not image-wide) before HCCL pytest."
        )

    return {
        "in_docker": in_docker,
        "cann_home": cann_home,
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        "npu_smi_available": npu_smi is not None,
        "device_count": device_count,
        "npu_smi_excerpt": npu_smi_excerpt or None,
        "hccl_preload_path": hccl_lib,
        "hccl_preload_recommended": hccl_preload_recommended,
        "pid_host_recommended": in_docker and (device_count or 0) >= 2,
        "suggested_platform": _suggest_platform(),
        "warnings": warnings,
        "mcp_resources": [
            "hw-native-sys://ascend/hccl_container_checklist",
            "hw-native-sys://ascend/platform_decisions",
        ],
    }
