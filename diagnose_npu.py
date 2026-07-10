#!/usr/bin/env python3
"""NPU device diagnostic — run inside a container as root.

Quick health check for Ascend NPU devices. Answers:
  - Which devices are alive?
  - Is LD_PRELOAD=libhccl.so interfering?
  - Are we on a matching CANN+driver pair?
  - What does dmesg say about each chip?
"""

import ctypes
import os
import re
import subprocess
import sys

# ── ANSI helpers ────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"
OK = f"{GREEN}✔{RESET}"
BAD = f"{RED}✘{RESET}"
WARN = f"{YELLOW}⚠{RESET}"

WIDTH = 64  # noqa: SIM112 — kept as constant for readability


def hr(title: str = "", char: str = "─") -> None:
    """Horizontal rule with optional centered title."""
    if title:
        side = (WIDTH - len(title) - 2) // 2
        print(f"\n{DIM}{char * side} {BOLD}{title}{RESET}{DIM} {char * (WIDTH - side - len(title) - 2)}{RESET}")
    else:
        print(f"{DIM}{char * WIDTH}{RESET}")


def info(key: str, val: str) -> None:
    print(f"  {BOLD}{key}:{RESET} {val}")


def ok(msg: str) -> None:
    print(f"  {OK}  {msg}")


def bad(msg: str) -> None:
    print(f"  {BAD}  {msg}")


def warn(msg: str) -> None:
    print(f"  {WARN}  {msg}")


def _dedup_path(path_str: str) -> str:
    """Deduplicate entries in a colon-separated path, preserving order."""
    seen: set[str] = set()
    parts: list[str] = []
    for p in path_str.split(":"):
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            parts.append(p)
    return ":".join(parts)


def _summarize_ld_path(path_str: str) -> str:
    """Show LD_LIBRARY_PATH compactly: count + key directories."""
    deduped = _dedup_path(path_str)
    dirs = deduped.split(":")
    asc_dirs = [d for d in dirs if "/Ascend/" in d]
    other = [d for d in dirs if "/Ascend/" not in d]
    lines = [f"{len(dirs)} entries total ({len(asc_dirs)} Ascend, {len(other)} other)"]
    if asc_dirs:
        lines.append(f"    Ascend roots: {', '.join(sorted(set(d.split('/Ascend/')[1].split('/')[0] for d in asc_dirs if '/Ascend/' in d)))}")
    if other:
        lines.append(f"    Other:       {', '.join(sorted(set(d for d in other if d)))}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
hr("NPU DIAGNOSTIC")
print(f"  Hostname: {os.uname().nodename}")
print(f"  Kernel:   {os.uname().release}")
print(f"  Arch:     {os.uname().machine}")

# ── 1. Environment ──────────────────────────────────────────────────
hr("Environment")

cann_home = os.environ.get("ASCEND_HOME_PATH", "")
ascend_ok = bool(cann_home and os.path.isdir(cann_home))
(ok if ascend_ok else bad)(f"ASCEND_HOME_PATH = {cann_home or 'UNSET'}")

ld_preload = os.environ.get("LD_PRELOAD", "")
if ld_preload:
    info("LD_PRELOAD", os.path.basename(ld_preload))
else:
    ok("LD_PRELOAD not set")

ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if ld_path:
    ld_dedup = _dedup_path(ld_path)
    host_leaked = any(
        p.startswith("/usr/local/") and "Ascend" not in p and "python" not in p
        for p in ld_dedup.split(":")
    )
    info("LD_LIBRARY_PATH", "")
    for line in _summarize_ld_path(ld_path).splitlines():
        print(f"    {line}")
    if host_leaked:
        warn("Host paths detected in LD_LIBRARY_PATH — may cause symbol conflicts")
else:
    warn("LD_LIBRARY_PATH not set")

# ── 2. set_env.sh ───────────────────────────────────────────────────
hr("CANN set_env.sh")

set_env = os.path.join(cann_home, "set_env.sh") if cann_home else ""
if set_env and os.path.isfile(set_env):
    ok(f"Found: {set_env}")
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {set_env} 2>/dev/null && env"],
            capture_output=True, text=True, timeout=3,
            env={**os.environ, "LD_PRELOAD": ""},  # neutralise preload for sourcing
        )
        env_after = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        ld_after = env_after.get("LD_LIBRARY_PATH", "")
        if ld_after:
            info("  adds", f"{len(ld_after.split(':'))} entries to LD_LIBRARY_PATH")
    except subprocess.TimeoutExpired:
        warn("set_env.sh sourcing timed out (harmless — hang in a profile script?)")
    except Exception:
        warn("Could not source set_env.sh")
else:
    bad(f"set_env.sh not found at {set_env}")

# ── 3. Device nodes ─────────────────────────────────────────────────
hr("Device nodes")
for pattern, label in [("/dev/davinci*", "davinci"), ("/dev/svm*", "svm"), ("/dev/hisi_hdc", "hisi_hdc")]:
    try:
        result = subprocess.run(
            f"ls {pattern} 2>/dev/null | wc -l",
            shell=True, capture_output=True, text=True, timeout=3,
        )
        count = result.stdout.strip()
        if count and count != "0":
            ok(f"{label}: {count} node(s)")
        else:
            warn(f"{label}: none found")
    except Exception:
        warn(f"{label}: error checking")

# ── 4. Driver version ───────────────────────────────────────────────
hr("Driver version")
driver_ver = ""
for f in ["/usr/local/Ascend/driver/version.info", "/proc/driver/hisi_hpre/version"]:
    try:
        with open(f) as fh:
            driver_ver = fh.read().strip()[:80]
            break
    except (FileNotFoundError, PermissionError):
        continue
if driver_ver:
    info("driver", driver_ver)
else:
    # Fallback: try modinfo
    try:
        result = subprocess.run(
            ["modinfo", "davinci"], capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.splitlines():
            if line.startswith("version:"):
                driver_ver = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    info("driver", driver_ver or "could not determine")

cann_ver = ""
for f in [f"{cann_home}/version.cfg", f"{cann_home}/version.info"]:
    try:
        with open(f) as fh:
            cann_ver = fh.read().strip()[:80]
            break
    except (FileNotFoundError, PermissionError, TypeError):
        continue
if cann_ver:
    info("CANN", cann_ver)

# ── 5. ACL device sweep (no preload) ────────────────────────────────
hr("Device health (no LD_PRELOAD)")
saved_preload = os.environ.pop("LD_PRELOAD", None)
device_results: dict[int, str] = {}  # dev_id → "ok" | "dead" | "error_<code>"

try:
    lib = ctypes.cdll.LoadLibrary("libascendcl.so")
except OSError as exc:
    bad(f"Cannot load libascendcl.so — Ascend CANN not installed or not in library path")
    info("  detail", str(exc))
    info("  hint", "Build and run from a CANN-based Docker image, or source set_env.sh from a CANN installation")
    if saved_preload:
        os.environ["LD_PRELOAD"] = saved_preload
    sys.exit(0)  # clean exit — diagnostic is still useful up to this point

try:
    lib.aclInit.argtypes = [ctypes.c_char_p]
    lib.aclInit.restype = ctypes.c_int
    lib.aclrtGetDeviceCount.argtypes = [ctypes.c_void_p]
    lib.aclrtGetDeviceCount.restype = ctypes.c_int
    lib.aclrtSetDevice.argtypes = [ctypes.c_int]
    lib.aclrtSetDevice.restype = ctypes.c_int
    lib.aclrtResetDevice.argtypes = [ctypes.c_int]
    lib.aclrtResetDevice.restype = ctypes.c_int
    lib.aclFinalize.restype = ctypes.c_int

    rc = lib.aclInit(None)
    (ok if rc == 0 else bad)(f"aclInit → {rc}")

    cnt = ctypes.c_uint(0)
    rc = lib.aclrtGetDeviceCount(ctypes.byref(cnt))
    total = cnt.value
    (ok if rc == 0 and total > 0 else bad)(f"Device count: {total}")

    if total > 0:
        print()
        # Print device grid
        cols = 4
        print(f"  {'':>7}", end="")
        for dev in range(min(total, 8)):
            print(f"  {BOLD}{dev:>3}{RESET}  ", end="")
            if (dev + 1) % cols == 0 and dev < total - 1:
                print(f"\n  {'':>7}", end="")
        print()

        print(f"  {'state':>7}", end="")
        for dev in range(min(total, 8)):
            rc = lib.aclrtSetDevice(dev)
            if rc == 0:
                label = f"{GREEN} OK {RESET}"
                device_results[dev] = "ok"
                lib.aclrtResetDevice(dev)
            else:
                label = f"{RED}{rc:>5d}{RESET}" if rc != 507033 else f"{RED}DEAD{RESET}"
                device_results[dev] = "dead" if rc == 507033 else f"error_{rc}"
            print(f"  {label}  ", end="")
            if (dev + 1) % cols == 0 and dev < total - 1:
                print(f"\n  {'':>7}", end="")
        print()

    lib.aclFinalize()
finally:
    if saved_preload:
        os.environ["LD_PRELOAD"] = saved_preload

# ── 6. HCCL preload check ───────────────────────────────────────────
hr("HCCL preload check")
if "LD_PRELOAD" in os.environ:
    try:
        lib2 = ctypes.cdll.LoadLibrary("libascendcl.so")
        lib2.aclInit.argtypes = [ctypes.c_char_p]
        lib2.aclInit.restype = ctypes.c_int
        lib2.aclrtSetDevice.argtypes = [ctypes.c_int]
        lib2.aclrtSetDevice.restype = ctypes.c_int
        lib2.aclrtResetDevice.argtypes = [ctypes.c_int]
        lib2.aclrtResetDevice.restype = ctypes.c_int
        lib2.aclFinalize.restype = ctypes.c_int

        lib2.aclInit(None)
        for dev, status in sorted(device_results.items()):
            if status != "ok":
                continue
            rc = lib2.aclrtSetDevice(dev)
            if rc == 0:
                device_results[dev] = "ok_hccl"
                lib2.aclrtResetDevice(dev)
            else:
                device_results[dev] = "broken_by_hccl"
            break  # one healthy device is enough
        lib2.aclFinalize()
    except Exception:
        pass

ok("LD_PRELOAD does not affect device access" if all(
    v in ("ok", "ok_hccl", "dead") for v in device_results.values()
) else bad("LD_PRELOAD breaks device access — unset for non-HCCL tests"))

# ── 7. npu-smi (chip summary only) ──────────────────────────────────
hr("npu-smi (chip summary)")
try:
    result = subprocess.run(
        ["npu-smi", "info", "-m"], capture_output=True, text=True, timeout=5,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            # Only print chip summary lines, skip verbose per-chip detail
            if any(kw in line for kw in ("Chip", "Health", "Temperature", "Power", "HBM", "Frequency", "Process", "npu-smi")):
                continue
            stripped = line.strip()
            if stripped:
                print(f"  {stripped}")
    else:
        warn(f"npu-smi failed: {result.stderr.strip()[:120]}")
except subprocess.TimeoutExpired:
    warn("npu-smi timed out (usually means a dead chip is hanging the query)")
except FileNotFoundError:
    warn("npu-smi not found")
except Exception as e:
    warn(f"npu-smi error: {e}")

# ── 8. Device contention ────────────────────────────────────────────
hr("Device contention")
try:
    result = subprocess.run(
        "fuser /dev/davinci* 2>/dev/null || true",
        shell=True, capture_output=True, text=True, timeout=3,
    )
    pids = result.stdout.strip()
    if pids:
        warn(f"Devices held by PIDs: {pids}")
    else:
        ok("No process holding any /dev/davinci*")
except Exception:
    warn("Could not check device holders")

# ── 9. dmesg — deduplicated errors ──────────────────────────────────
hr("Kernel errors (unique, last 50 ascend lines)")
try:
    result = subprocess.run(
        "dmesg | grep -iE 'ascend.*(error|fail|not working|not ready)' | tail -50",
        shell=True, capture_output=True, text=True, timeout=3,
    )
    raw = result.stdout.strip()
    if raw:
        # Deduplicate by extracting the message pattern (strip timestamps & PIDs)
        seen: set[str] = set()
        for line in raw.splitlines():
            # Normalise: strip leading timestamp + PID noise
            normalized = re.sub(r"^\[\s*\d+\.\d+\]\s*", "", line)  # [123456.789]
            normalized = re.sub(r"<\w+:\d+:\d+:\d+>", "<...>", normalized)  # <python3:pid:tid:...>
            if normalized not in seen:
                seen.add(normalized)
                print(f"  {DIM}{line}{RESET}")
        if len(seen) == 0:
            ok("No kernel errors")
    else:
        ok("No ascend kernel errors in dmesg")
except Exception:
    warn("Could not read dmesg")

# ── 10. Verdict ─────────────────────────────────────────────────────
hr("VERDICT")

ok_count = sum(1 for v in device_results.values() if v in ("ok", "ok_hccl"))
dead_count = sum(1 for v in device_results.values() if v == "dead")
broken_count = sum(1 for v in device_results.values() if v == "broken_by_hccl")
total = len(device_results)

dead_ids = [str(d) for d, v in device_results.items() if v == "dead"]

if total == 0:
    print(f"  {RED}No devices detected at all.{RESET}")
    print(f"  Check: docker run --privileged -v /dev:/dev ?")
elif ok_count == total:
    print(f"  {GREEN}All {total} device(s) healthy.{RESET}")
elif dead_count == 0 and broken_count == 0:
    print(f"  {YELLOW}Some devices report errors but are not confirmed dead.{RESET}")
    print(f"  Run with --device=<working_ids> to skip the bad ones.")
else:
    bad_devs = ", ".join(dead_ids) if dead_ids else "none"
    print(f"  {RED}{dead_count} device(s) dead:{RESET} {bad_devs}")
    print(f"  {GREEN}{ok_count} device(s) healthy{RESET}")
    print()
    if dead_ids:
        working = ",".join(str(d) for d, v in device_results.items() if v in ("ok", "ok_hccl"))
        print(f"  {BOLD}Tests:{RESET} add {CYAN}--device=\"{working}\"{RESET}")
    if broken_count:
        print(f"  {YELLOW}Unset LD_PRELOAD for non-HCCL tests{RESET}")

if dead_ids:
    print()
    print(f"  {DIM}Dead = kernel module reports 'not working' (dmesg).")
    print(f"  Not software-fixable — needs hardware reset / replacement.{RESET}")

print()
