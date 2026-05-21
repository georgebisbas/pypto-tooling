**Issue: `comm_alloc_windows` / `allocate_domain` fails on hng-atlas01 — CANN 9.0.0 IPC key format rejected by driver 25.5.1**

---

**Environment**

| Component | Version |
|---|---|
| Host | hng-atlas01 |
| Hardware | Ascend 910B (8× NPU, all HCCS-connected) |
| Host driver | 25.5.1 (`V100R001C23SPC006B220`) |
| Host CANN | 8.5.0 (only version installed) |
| Docker image CANN | 9.0.0 (baked into `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12`) |

---

**Symptom**

Any test that calls `comm_alloc_windows` or `allocate_domain` (which allocates shared HBM windows across chip processes via HCCL IPC) fails with:

```
RuntimeError: comm_alloc_windows failed with code -1
[comm rank 1] ipc: ImportByKey(peer=0 pid=...) -> 507899
[comm rank 0] ipc: ImportByKey(peer=1 pid=...) -> 507899
```

Error 507899 = "driver error:internal error" from `rtsIpcMemImportByKey` inside the kernel driver.

Affected tests in simpler:
- `tests/ut/py/test_worker/test_dynamic_alloc_hw.py::test_two_rank_allocate_release_round_trip`
- `tests/ut/py/test_worker/test_platform_comm.py::test_two_rank_comm_lifecycle`

All other hardware tests (non-HCCL) pass normally.

---

**Root cause**

`comm_alloc_windows` shares device HBM buffers between chip subprocesses using the ACL IPC API:

1. Rank A calls `aclrtIpcMemGetExportKey` → gets an opaque key
2. Rank A sends the key to Rank B via a file
3. Rank B calls `aclrtIpcMemImportByKey` → **fails 507899**

CANN 9.0.0's `aclrtIpcMemGetExportKey` generates a key with an internal `nameLen` field set to **64**. Driver 25.5.1's `rtsIpcMemImportByKey` rejects any key where `nameLen >= 64` — it expects `nameLen < 64`. This is confirmed by `aclrtIpcMemSetAttr`:

```
"value 64 for parameter nameLen is invalid. Expected value: [0, 64)"
```

This is a **breaking change introduced in driver 25.5.1** that is incompatible with the IPC key format CANN 9.0.0 produces. The driver and CANN were not co-released, creating an incompatible pair.

All workarounds at the application level were exhausted (different malloc flags, buffer sizes, `EnablePeerAccess`, `P2PMap`, raw VA sharing) — all return 507899. The failure is at the driver ABI level.

---

**Why CANN 8.5.0 is not a workaround**

Simpler requires CANN ≥ 9.0.0 because `libopapi.so` must export `aclnnShmemSdmaStarsQuery` for the SDMA workspace allocation — this symbol is absent in CANN 8.5.0. The two constraints are mutually exclusive on this machine:

| | `aclnnShmemSdmaStarsQuery` present | IPC keys accepted by driver 25.5.1 |
|---|---|---|
| CANN 9.0.0 | ✅ | ❌ |
| CANN 8.5.0 | ❌ | ✅ (likely) |

---

**Requested action**

One of the following is needed:

1. **(Preferred) Downgrade driver to the version co-released with CANN 9.0.0** — the Huawei release notes for CANN 9.0.0 should list the compatible driver version. This restores full test coverage without any code changes.

2. **Install CANN 9.0.0 alongside 8.5.0 on the host** — not sufficient by itself (driver mismatch remains), but needed if/when a compatible driver is available.

3. **Coordinate a compatible CANN + driver upgrade** — upgrade both to a pair that Huawei has validated together.

Until one of the above is done, `comm_alloc_windows` and `allocate_domain` tests will remain permanently broken on this machine. All other pypto/simpler tests are unaffected.

Completed: *Add driver/CANN version findings to SKILL.md* (12/12)