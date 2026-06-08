#!/usr/bin/env python3
"""NPU device diagnostic script — run inside the container as root."""

import ctypes
import os
import subprocess
import sys


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(ok: bool, msg: str) -> None:
    print(f"  {'✅' if ok else '❌'} {msg}")


# ── 1. Environment ──────────────────────────────────────────────────
section("1. Environment")

check("ASCEND_HOME_PATH" in os.environ, "ASCEND_HOME_PATH is set")
print(f"    ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', 'UNSET')}")
print(f"    LD_PRELOAD={os.environ.get('LD_PRELOAD', 'UNSET')}")
print(f"    LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', 'UNSET')}")

# ── 2. CANN set_env.sh ──────────────────────────────────────────────
section("2. CANN set_env.sh")

cann_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
set_env = os.path.join(cann_home, "set_env.sh")
check(os.path.isfile(set_env), f"set_env.sh exists at {set_env}")

# Source it and capture resulting env
try:
    result = subprocess.run(
        f"bash -c 'source {set_env} 2>/dev/null && env'",
        shell=True, capture_output=True, text=True, timeout=10,
    )
    env_after = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    print(f"    LD_LIBRARY_PATH (after source): {env_after.get('LD_LIBRARY_PATH', 'UNSET')[:120]}...")
except Exception as e:
    print(f"    ⚠️  Could not source set_env.sh: {e}")

# ── 3. Device nodes ─────────────────────────────────────────────────
section("3. Device nodes")

for pattern in ["/dev/davinci*", "/dev/svm*", "/dev/hisi*", "/dev/vdavinci*"]:
    try:
        result = subprocess.run(
            f"ls -la {pattern} 2>/dev/null | head -12",
            shell=True, capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            print(f"  {pattern}:")
            for line in result.stdout.strip().splitlines():
                print(f"    {line}")
        else:
            print(f"  {pattern}: (none found)")
    except Exception as e:
        print(f"  {pattern}: error — {e}")

# ── 4. Driver version (from sysfs) ──────────────────────────────────
section("4. Driver version (sysfs)")

for f in ["/proc/driver/hisi_hpre/version", "/sys/module/davinci/version"]:
    try:
        with open(f) as fh:
            print(f"  {f}: {fh.read().strip()}")
    except FileNotFoundError:
        print(f"  {f}: not found")
    except PermissionError:
        print(f"  {f}: permission denied")

# ── 5. Basic ACL: aclInit + getDeviceCount (no preload) ─────────────
section("5. ACL smoke test (no HCCL preload)")

# Temporarily unset LD_PRELOAD for this section
saved_preload = os.environ.pop("LD_PRELOAD", None)
try:
    lib = ctypes.cdll.LoadLibrary("libascendcl.so")
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
    check(rc == 0, f"aclInit(None) → {rc}")

    cnt = ctypes.c_uint(0)
    rc = lib.aclrtGetDeviceCount(ctypes.byref(cnt))
    check(rc == 0 and cnt.value > 0, f"aclrtGetDeviceCount → {rc}, count={cnt.value}")

    if cnt.value > 0:
        print(f"\n  Trying each device (0..{cnt.value - 1}):")
        all_fail = True
        for dev_id in range(min(cnt.value, 8)):
            rc = lib.aclrtSetDevice(dev_id)
            if rc == 0:
                print(f"    ✅ device {dev_id}: OK")
                lib.aclrtResetDevice(dev_id)
                all_fail = False
            else:
                print(f"    ❌ device {dev_id}: {rc}")
        if all_fail:
            print("  ⚠️  ALL devices failed — system-level issue (driver/kernel module)")

    lib.aclFinalize()
finally:
    if saved_preload:
        os.environ["LD_PRELOAD"] = saved_preload

# ── 6. ACL with HCCL preload ────────────────────────────────────────
section("6. ACL smoke test (WITH HCCL preload)")

try:
    lib2 = ctypes.cdll.LoadLibrary("libascendcl.so")
    lib2.aclInit.argtypes = [ctypes.c_char_p]
    lib2.aclInit.restype = ctypes.c_int
    lib2.aclrtSetDevice.argtypes = [ctypes.c_int]
    lib2.aclrtSetDevice.restype = ctypes.c_int
    lib2.aclrtResetDevice.argtypes = [ctypes.c_int]
    lib2.aclrtResetDevice.restype = ctypes.c_int
    lib2.aclFinalize.restype = ctypes.c_int

    rc = lib2.aclInit(None)
    check(rc == 0, f"aclInit(None) → {rc}")

    rc = lib2.aclrtSetDevice(0)
    if rc == 0:
        print(f"    ✅ set device 0: OK (LD_PRELOAD present)")
        lib2.aclrtResetDevice(0)
    else:
        print(f"    ❌ set device 0: {rc} (LD_PRELOAD present)")

    lib2.aclFinalize()
except Exception as e:
    print(f"    ❌ Exception: {e}")

# ── 7. npu-smi ──────────────────────────────────────────────────────
section("7. npu-smi info")

try:
    result = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        print(f"  npu-smi failed: {result.stderr.strip()}")
except FileNotFoundError:
    print("  npu-smi not found")
except Exception as e:
    print(f"  Error: {e}")

# ── 8. Process tree using NPU ───────────────────────────────────────
section("8. Processes using NPU devices")

try:
    result = subprocess.run(
        "fuser /dev/davinci* 2>/dev/null || echo '(none)'",
        shell=True, capture_output=True, text=True, timeout=5,
    )
    print(f"  fuser /dev/davinci*: {result.stdout.strip()}")
except Exception as e:
    print(f"  Error: {e}")

# ── 9. dmesg (last 10 ascend-related lines) ─────────────────────────
section("9. dmesg (ascend/davinci/devmm, last 15 lines)")

try:
    result = subprocess.run(
        "dmesg | grep -iE 'ascend|davinci|devmm|svm|hisi' | tail -15",
        shell=True, capture_output=True, text=True, timeout=5,
    )
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        print("  (no matching lines)")
except Exception as e:
    print(f"  Error: {e}")

# ── 10. Summary ─────────────────────────────────────────────────────
section("10. Summary")

print("""
If ALL devices fail 507033 regardless of LD_PRELOAD:
  → Driver/CANN mismatch or kernel module issue.
    - Check: does host driver version match CANN 9.0.0 requirements?
    - Check: 'dmesg | grep -i ascend' for kernel module load errors.

If devices work WITHOUT LD_PRELOAD but fail WITH it:
  → libhccl.so constructor is breaking ACL init. Try:
    - Unset LD_PRELOAD, run test, set it only for HCCL tests.
    - Or: LD_PRELOAD=libhccl.so:libascendcl.so (order matters).

If some devices work and others don't:
  → Device contention — another process holds those devices.
    Use 'npu-smi info' and 'fuser /dev/davinci*' to find the holder.
""")
