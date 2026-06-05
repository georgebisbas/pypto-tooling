"""HCCL HcclAllReduce microbenchmark — invoked as subprocess by run_sweep.py.

Uses ctypes + threading (one thread per device), matching the HCCL example at
hccl/examples/02_collectives/01_allreduce/main.cc.  Threads share the same
process so no spawn/fork hang issues.

Matches the same (count, dtype, device_ids) contract as the simpler/pypto
runners — same input formula (rank_linear_v1), same golden (allreduce_sum_v1).
"""

import argparse
import ctypes
import json
import os
import sys
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# HCCL C API (libhccl.so)
# ---------------------------------------------------------------------------
HCCL_ROOT_INFO_BYTES = 4104
HCCL_REDUCE_SUM = 0
HCCL_SUCCESS = 0
# HCCL source order in hccl_common.h implies the public enum ordering:
# INT8=0, INT16=1, INT32=2, INT64=3, UINT64=4, FP16=5, FP32=6, ...
# Passing 0 for fp32 makes HcclAllReduce interpret the float buffer as int8,
# which matches the observed byte-wise corruption pattern.
_DTYPE_MAP = {"fp16": 5, "fp32": 6}

_lib = ctypes.CDLL("libhccl.so", mode=ctypes.RTLD_GLOBAL)
_lib.HcclGetRootInfo.argtypes = [ctypes.c_void_p]
_lib.HcclGetRootInfo.restype = ctypes.c_int
_lib.HcclCommInitRootInfo.argtypes = [
    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.HcclCommInitRootInfo.restype = ctypes.c_int
_lib.HcclCommDestroy.argtypes = [ctypes.c_void_p]
_lib.HcclCommDestroy.restype = ctypes.c_int
_lib.HcclAllReduce.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
]
_lib.HcclAllReduce.restype = ctypes.c_int

# ---------------------------------------------------------------------------
# ACL API (libascendcl.so)
# ---------------------------------------------------------------------------
_acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
_acl.aclInit.argtypes = [ctypes.c_char_p]
_acl.aclInit.restype = ctypes.c_int
_acl.aclFinalize.restype = ctypes.c_int
_acl.aclrtGetDeviceCount.restype = ctypes.c_int
_acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
_acl.aclrtSetDevice.restype = ctypes.c_int
_acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
_acl.aclrtMalloc.restype = ctypes.c_int
_acl.aclrtFree.argtypes = [ctypes.c_void_p]
_acl.aclrtFree.restype = ctypes.c_int
_acl.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_acl.aclrtMallocHost.restype = ctypes.c_int
_acl.aclrtFreeHost.argtypes = [ctypes.c_void_p]
_acl.aclrtFreeHost.restype = ctypes.c_int
_acl.aclrtMemcpy.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
]
_acl.aclrtMemcpy.restype = ctypes.c_int
_acl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_acl.aclrtCreateStream.restype = ctypes.c_int
_acl.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
_acl.aclrtDestroyStream.restype = ctypes.c_int
_acl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
_acl.aclrtSynchronizeStream.restype = ctypes.c_int

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _ptr_hex(ptr: ctypes.c_void_p) -> str:
    value = getattr(ptr, "value", ptr)
    if value is None:
        return "0x0"
    return hex(int(value))


def _hexdump(ptr: ctypes.c_void_p, nbytes: int) -> str:
    if getattr(ptr, "value", None) is None or nbytes <= 0:
        return ""
    raw = ctypes.string_at(ptr, nbytes)
    return " ".join(f"{byte:02x}" for byte in raw)


def _rank_thread(rank: int, world_size: int, device_id: int, count: int,
                 dtype_str: str, root_info: ctypes.c_void_p,
                 results: list) -> None:
    """One thread per NPU — init HCCL comm, run HcclAllReduce, verify golden."""
    try:
        ret = _acl.aclrtSetDevice(device_id)
        print(f"[diag] rank {rank} aclrtSetDevice({device_id}) → {ret}", flush=True)
        if ret != 0:
            results.append({"rank": rank, "ok": False,
                            "error": f"aclrtSetDevice({device_id}) → {ret}"})
            return

        elem_size = 4 if dtype_str == "fp32" else 2
        nbytes = count * elem_size
        data_type = _DTYPE_MAP.get(dtype_str, 0)
        print(
            f"[diag] rank {rank} ABI: dtype={dtype_str} data_type={data_type} "
            f"reduce_op={HCCL_REDUCE_SUM} elem_size={elem_size} nbytes={nbytes} "
            f"root_info_ptr={_ptr_hex(root_info)}",
            flush=True,
        )

        # Device buffers
        send_buf = ctypes.c_void_p()
        recv_buf = ctypes.c_void_p()
        ret = _acl.aclrtMalloc(ctypes.byref(send_buf), nbytes, 0)  # ACL_MEM_MALLOC_HUGE_ONLY
        print(f"[diag] rank {rank} aclrtMalloc(send_buf={_ptr_hex(send_buf)}) → {ret}", flush=True)
        if ret != 0:
            results.append({"rank": rank, "ok": False, "error": f"aclrtMalloc(send_buf) → {ret}"})
            return
        ret = _acl.aclrtMalloc(ctypes.byref(recv_buf), nbytes, 0)
        print(f"[diag] rank {rank} aclrtMalloc(recv_buf={_ptr_hex(recv_buf)}) → {ret}", flush=True)
        if ret != 0:
            _acl.aclrtFree(send_buf)
            results.append({"rank": rank, "ok": False, "error": f"aclrtMalloc(recv_buf) → {ret}"})
            return

        # Host input: rank_linear_v1 — allocate pinned host memory
        host_buf = ctypes.c_void_p()
        ret = _acl.aclrtMallocHost(ctypes.byref(host_buf), nbytes)
        print(f"[diag] rank {rank} aclrtMallocHost(host_buf={_ptr_hex(host_buf)}) → {ret}", flush=True)
        if ret != 0:
            _acl.aclrtFree(send_buf)
            _acl.aclrtFree(recv_buf)
            results.append({"rank": rank, "ok": False, "error": f"aclrtMallocHost(host_buf) → {ret}"})
            return
        
        # Fill host buffer with test pattern
        host_view = np.ctypeslib.as_array(ctypes.cast(host_buf, ctypes.POINTER(ctypes.c_float)), (count,))
        for i in range(count):
            host_view[i] = float(i + rank * 100)
        
        # Copy to device (synchronous)
        ret = _acl.aclrtMemcpy(send_buf, nbytes, host_buf, nbytes, ACL_MEMCPY_HOST_TO_DEVICE)
        print(
            f"[diag] rank {rank} aclrtMemcpy H2D kind={ACL_MEMCPY_HOST_TO_DEVICE} "
            f"src={_ptr_hex(host_buf)} dst={_ptr_hex(send_buf)} → {ret}",
            flush=True,
        )
        
        # Verify H2D: read back and check first few elements
        check_buf = ctypes.c_void_p()
        ret = _acl.aclrtMallocHost(ctypes.byref(check_buf), nbytes)
        print(f"[diag] rank {rank} aclrtMallocHost(check_buf={_ptr_hex(check_buf)}) → {ret}", flush=True)
        if ret != 0:
            _acl.aclrtFreeHost(host_buf)
            _acl.aclrtFree(send_buf)
            _acl.aclrtFree(recv_buf)
            results.append({"rank": rank, "ok": False, "error": f"aclrtMallocHost(check_buf) → {ret}"})
            return
        ret = _acl.aclrtMemcpy(check_buf, nbytes, send_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST)
        print(
            f"[diag] rank {rank} aclrtMemcpy D2H verify kind={ACL_MEMCPY_DEVICE_TO_HOST} "
            f"src={_ptr_hex(send_buf)} dst={_ptr_hex(check_buf)} → {ret}",
            flush=True,
        )
        
        # Copy to numpy BEFORE printing (view becomes invalid after free)
        check_view = np.ctypeslib.as_array(ctypes.cast(check_buf, ctypes.POINTER(ctypes.c_float)), (count,))
        check_data = [float(check_view[i]) for i in range(min(4,count))]
        expected_data = [float(i+rank*100) for i in range(min(4,count))]
        print(f"[diag] rank {rank} H2D verify: got={check_data} expected={expected_data}", flush=True)
        
        _acl.aclrtFreeHost(check_buf)
        _acl.aclrtFreeHost(host_buf)

        # Init HCCL comm
        comm = ctypes.c_void_p()
        ret = _lib.HcclCommInitRootInfo(world_size, root_info, device_id, ctypes.byref(comm))
        print(
            f"[diag] rank {rank} HcclCommInitRootInfo(world_size={world_size}, device_id={device_id}, "
            f"root_info={_ptr_hex(root_info)}) comm={_ptr_hex(comm)} → {ret}",
            flush=True,
        )
        if ret != HCCL_SUCCESS:
            results.append({"rank": rank, "ok": False, "error": f"HcclCommInitRootInfo → {ret}"})
            return

        stream = ctypes.c_void_p()
        ret = _acl.aclrtCreateStream(ctypes.byref(stream))
        print(f"[diag] rank {rank} aclrtCreateStream(stream={_ptr_hex(stream)}) → {ret}", flush=True)
        if ret != 0:
            _lib.HcclCommDestroy(comm)
            _acl.aclrtFree(send_buf)
            _acl.aclrtFree(recv_buf)
            results.append({"rank": rank, "ok": False, "error": f"aclrtCreateStream → {ret}"})
            return

        # Timed AllReduce
        t0 = time.perf_counter()
        ret = _lib.HcclAllReduce(send_buf, recv_buf, count, data_type,
                                  HCCL_REDUCE_SUM, comm, stream)
        sync_ret = _acl.aclrtSynchronizeStream(stream)
        wall = time.perf_counter() - t0
        print(
            f"[diag] rank {rank} HcclAllReduce(send={_ptr_hex(send_buf)}, recv={_ptr_hex(recv_buf)}, "
            f"count={count}, data_type={data_type}, reduce_op={HCCL_REDUCE_SUM}, "
            f"comm={_ptr_hex(comm)}, stream={_ptr_hex(stream)}) → {ret}; "
            f"aclrtSynchronizeStream → {sync_ret}; wall={wall:.6f}s",
            flush=True,
        )

        if ret != HCCL_SUCCESS:
            results.append({
                "rank": rank,
                "ok": False,
                "error": (
                    f"HcclAllReduce → {ret} "
                    f"(data_type={data_type}, reduce_op={HCCL_REDUCE_SUM}, "
                    f"visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')}, "
                    f"expansion_mode={os.environ.get('HCCL_OP_EXPANSION_MODE', '<unset>')})"
                ),
            })
            return
        if sync_ret != 0:
            results.append({"rank": rank, "ok": False, "error": f"aclrtSynchronizeStream → {sync_ret}"})
            return

        # Verify golden: allreduce_sum_v1
        # Copy output from device to host
        out_buf = ctypes.c_void_p()
        ret = _acl.aclrtMallocHost(ctypes.byref(out_buf), nbytes)
        print(f"[diag] rank {rank} aclrtMallocHost(out_buf={_ptr_hex(out_buf)}) → {ret}", flush=True)
        ret = _acl.aclrtMemcpy(out_buf, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST)
        print(
            f"[diag] rank {rank} aclrtMemcpy D2H kind={ACL_MEMCPY_DEVICE_TO_HOST} "
            f"src={_ptr_hex(recv_buf)} dst={_ptr_hex(out_buf)} → {ret}",
            flush=True,
        )

        # CRITICAL: Copy data to numpy array BEFORE comparing/freeing
        # (np.ctypeslib.as_array is just a view into out_buf memory)
        out_view = np.ctypeslib.as_array(ctypes.cast(out_buf, ctypes.POINTER(ctypes.c_float)), (count,))
        out_arr = out_view.copy()  # Make a copy so we own the data

        expected_base = 100 * world_size * (world_size - 1) // 2
        max_err = 0.0
        for i in range(count):
            expected = float(world_size * i + expected_base)
            err = abs(float(out_arr[i]) - expected)
            if err > max_err:
                max_err = err

        # Capture diagnostic data BEFORE freeing
        if max_err >= 1e-3:
            raw = [float(out_arr[i]) for i in range(min(16, count))]
            exp = [float(world_size * i + expected_base) for i in range(min(16, count))]
            print(f"[diag] rank {rank} GOLDEN MISMATCH max_err={max_err:.6f}")
            print(f"[diag]   output[0:16]={raw}")
            print(f"[diag]   expected[0:16]={exp}", flush=True)

        # Now safe to free
        _acl.aclrtFreeHost(out_buf)
        _lib.HcclCommDestroy(comm)
        _acl.aclrtDestroyStream(stream)
        _acl.aclrtFree(send_buf)
        _acl.aclrtFree(recv_buf)

        results.append({
            "rank": rank, "ok": max_err < 1e-3, "wall_s": wall,
            "max_err": max_err if max_err >= 1e-3 else None,
        })

    except Exception as e:
        import traceback
        results.append({"rank": rank, "ok": False,
                        "error": f"{e}\n{traceback.format_exc()}"})


def main() -> int:
    parser = argparse.ArgumentParser(description="HCCL AllReduce benchmark")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--devices", required=True)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.devices.split(",")]
    world_size = len(device_ids)
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>")
    expansion_mode = os.environ.get("HCCL_OP_EXPANSION_MODE", "<unset>")

    print(
        f"[diag] ABI constants: HCCL_ROOT_INFO_BYTES={HCCL_ROOT_INFO_BYTES} "
        f"HCCL_REDUCE_SUM={HCCL_REDUCE_SUM} "
        f"DTYPE_MAP={json.dumps(_DTYPE_MAP, sort_keys=True)} "
        f"ACL_MEMCPY_HOST_TO_DEVICE={ACL_MEMCPY_HOST_TO_DEVICE} "
        f"ACL_MEMCPY_DEVICE_TO_HOST={ACL_MEMCPY_DEVICE_TO_HOST} "
        f"ASCEND_RT_VISIBLE_DEVICES={visible_devices} "
        f"HCCL_OP_EXPANSION_MODE={expansion_mode}",
        flush=True,
    )

    # Init ACL, generate root info on device 0 (matches HCCL example pattern)
    ret = _acl.aclInit(None)
    print(f"[diag] aclInit → {ret}")
    if ret != 0:
        print(f"ERROR: aclInit failed: {ret}")
        return 1

    ret = _acl.aclrtSetDevice(0)
    print(f"[diag] aclrtSetDevice(0) → {ret}")
    if ret != 0:
        print(f"ERROR: aclrtSetDevice(0) failed: {ret}")
        return 1

    device_count = ctypes.c_uint32()
    if hasattr(_acl, "aclrtGetDeviceCount"):
        _acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        ret = _acl.aclrtGetDeviceCount(ctypes.byref(device_count))
        print(f"[diag] aclrtGetDeviceCount → {ret}; count={device_count.value}")

    root_info_buf = ctypes.c_void_p()
    ret = _acl.aclrtMallocHost(ctypes.byref(root_info_buf), HCCL_ROOT_INFO_BYTES)
    print(f"[diag] aclrtMallocHost(root_info_buf={_ptr_hex(root_info_buf)}, {HCCL_ROOT_INFO_BYTES}) → {ret}")
    if ret != 0:
        print(f"ERROR: aclrtMallocHost failed: {ret}")
        return 1

    ctypes.memset(root_info_buf, 0, HCCL_ROOT_INFO_BYTES)
    print(f"[diag] root_info pre-HcclGetRootInfo hex[0:32]={_hexdump(root_info_buf, 32)}")

    ret = _lib.HcclGetRootInfo(root_info_buf)
    print(f"[diag] HcclGetRootInfo → {ret}")
    if ret != HCCL_SUCCESS:
        print(f"HcclGetRootInfo failed: {ret}")
        return 1
    print(f"[diag] root_info post-HcclGetRootInfo hex[0:32]={_hexdump(root_info_buf, 32)}")

    # One thread per device
    print(f"[diag] spawning {world_size} thread(s) for devices {device_ids}")
    results: list[dict] = []
    threads = []
    for rank, dev in enumerate(device_ids):
        t = threading.Thread(target=_rank_thread,
                             args=(rank, world_size, dev, args.count, args.dtype,
                                   root_info_buf, results),
                             daemon=True)
        t.start()
        threads.append(t)
        print(f"[diag]   thread rank={rank} dev={dev} started")

    for i, t in enumerate(threads):
        t.join(timeout=60)
        print(f"[diag]   thread rank={i} joined (alive={t.is_alive()})")

    _acl.aclrtFreeHost(root_info_buf)
    _acl.aclFinalize()

    all_ok = all(r.get("ok") for r in results) and len(results) == world_size
    if all_ok:
        walls = [r["wall_s"] for r in results]
        print(f"HCCL: mean={sum(walls)/len(walls):.6f}s "
              f"per_rank={json.dumps([round(w, 6) for w in walls])}")
        print("HCCL_ALLREDUCE_OK")
        return 0
    else:
        for r in results:
            if not r.get("ok"):
                print(f"HCCL rank {r['rank']}: {r.get('error', 'unknown')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

