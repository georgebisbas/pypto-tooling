# Task Submit: Resource Isolation and Graph Dispatch for a Shared NPU Cluster

> A reference guide covering both the host-level device-locking CLI and the
> on-device PTO2 runtime orchestration model. Written for compiler engineers
> and HPC practitioners who share an Ascend cluster and need to understand
> resource contention, scheduling, and failure modes.

---

## 1. The Problem: Shared Hardware Without Isolation

Our cluster has **many developers sharing a fixed set of NPU cards**. A single
Ascend die exposes ~24 clusters (each = 1 AIC + 2 AIV), device memory, and a
bounded pool of AICPU threads. Without a locking protocol, two processes
grabbing the same device simultaneously produce:

- **Silent correctness bugs** — device memory is not partitioned by address
  space; one process's `rtMalloc` region can alias another's tensor workspace.
- **Noisy performance measurements** — a colocated kernel steals memory
  bandwidth, L2 cache lines, and AICore cycles.
- **Cascading device errors** — `507018` (`aclrtSynchronizeStreamWithTimeout
  failed`) from an unrelated user's deadlocked job looks identical to your own
  bug.

The `task-submit` CLI is a **host-side queue with per-device exclusive locks**.
It ensures at most one process owns a given device at any time. This is *not* a
job scheduler — it's a lightweight wrapper that acquires a lock, forks a child,
and releases the lock on exit.

---

## 2. The Two Layers of "Submit"

"Task submit" means two different things at two different stack levels. They
are **independent** — you can use the CLI without the PTO2 runtime, and you can
run PTO2 submits without the CLI (on hardware you own).

```
┌───────────────────────────────────────────────────────────────────────┐
│  HOST: task-submit CLI                                                │
│  ─────────────────────                                                │
│  A queue wrapper that acquires exclusive access to NPU device(s)      │
│  before launching a child process.                                    │
│                                                                       │
│  Mechanism:   per-device lock file + in-memory queue                  │
│  Where:       shared dev box, CI runners                              │
│  What it guarantees: at most one process on device N at a time        │
│  Must-use on: any command touching rtMalloc / aclrt*                  │
│  Irrelevant for: sim platforms (a2a3sim, a5sim)                       │
│                                                                       │
│    $ task-submit --device auto --device-num 2 \                      │
│        --run "pytest ... --platform a2a3 --device \$TASK_DEVICE"     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                │  child process runs unlocked on the
                                │  device it was assigned
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  DEVICE: PTO2 Runtime submit_task                                     │
│  ─────────────────────────────                                        │
│  A task-graph builder and scheduler running on the device's AICPU     │
│  (ARM cores on the NPU itself).                                       │
│                                                                       │
│  Mechanism:   ring buffers + TensorMap dependency tracking +          │
│               lock-free ready queues + register-based handshake       │
│  Where:       AICPU thread 3 (orchestrator), threads 0-2 (scheduler)  │
│  What it guarantees: dependency-correct, deadlock-free dispatch       │
│                                                                       │
│    rt_submit_aic_task(rt, FUNC_QK, args);                              │
│    rt_submit_aiv_task(rt, FUNC_SF, args);                              │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: The `task-submit` CLI

### 3.1 When You Must Use It

The rule is simple: **any command that touches a real NPU on the shared
dev box or CI must be wrapped in `task-submit`.**

| You're running | Platform | Use task-submit? |
|---|---|---|
| C++/Python UT via sim | `a2a3sim`, `a5sim` | Never |
| pypto-lib example on device | `a2a3`, `a5` | Always |
| ST test on device | `a2a3`, `a5` | Always |
| Perf benchmark / stress test | `a2a3`, `a5` | Always |
| Manual `npu-smi` / `rtMalloc` script | any | Always |
| Compile-only / codegen-only | N/A | Never |
| Your own dedicated hardware | any | Optional (but still good practice) |

### 3.2 Quick Command Reference

```bash
# ── Discovery ───────────────────────────────────────────────────────
task-submit --list                          # who holds what right now
npu-smi info | head -40                     # per-chip memory/soc/health

# ── Auto device (prefer this) ───────────────────────────────────────
task-submit --device auto --device-num 1 \
    --run "python my_script.py --device \$TASK_DEVICE"

# ── Pinned device ───────────────────────────────────────────────────
task-submit --device 8,9 --run "pytest ... --platform a2a3 --device 8-9"

# ── Timeouts ────────────────────────────────────────────────────────
# --timeout  = max wall-clock wait in the queue before giving up
# --max-time = hard kill after N seconds of execution
task-submit --device auto --device-num 2 \
    --timeout 1800 --max-time 7200 \
    --run "stress_harness.sh"

# ── Waiting / canceling ─────────────────────────────────────────────
task-submit --wait <task-id>                # block until task finishes
task-submit --cancel <task-id>              # remove pending task
```

**Critical:** Inside `--run`, always use `$TASK_DEVICE` for the device ID.
`--device auto` picks different cards each time, and hardcoding a number
defeats the purpose.

### 3.3 Long-Running Work: Lock Once, Not Per-Iteration

```bash
# WRONG — re-acquires the lock 50 times, may bounce between devices
for i in $(seq 1 50); do
    task-submit --device auto --run "pytest ... --device \$TASK_DEVICE"
done

# RIGHT — one lock for the entire loop, same device throughout
task-submit --device 10,11 --timeout 7200 --max-time 7200 \
    --run "/tmp/my_stress.sh 50 10 11"
```

The first pattern not only wastes lock-acquisition overhead but also risks
landing on a card that another user's job *just* finished with, inheriting
partially-reclaimed device state (residual memory, pending AICPU callbacks).

### 3.4 Is It Automatic?

**No.** The Python harness does **not** automatically invoke `task-submit`.

- `pypto.runtime.run()` — never uses task-submit.
- `tests/st/` harness — opt-in via `--execute-via-task-submit` flag (CI
  always sets this; local runs default to direct execution).
- CI pipelines — always wrapped in task-submit.
- Dev box — you wrap commands manually.

### 3.5 Arch Precheck: Avoid Wrong-Platform Cascades

Running `--platform a2a3` on an a5 machine (or vice versa) produces cryptic
`507018` / `507899` errors that look indistinguishable from genuine bugs.
Always gate onboard invocations:

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
task-submit --device auto --device-num 1 \
    --run "pytest ... --platform a2a3 --device \$TASK_DEVICE"
```

The precheck is fast (~600 ms cold, ~5 ms cached) and refuses a wrong-arch
invocation *before* any device lock is acquired.

---

## 4. The `507018` Triage: Classify Before You Conclude

`507018` is a generic host-side error code — several **distinct** on-device
mechanisms all report the same number. Never call it a "deadlock" or "OOM"
from the host error alone. Read the device log first.

### 4.1 Where the Device Log Lives

The default path is:

```
~/ascend/log/debug/device-<id>/device-<pid>_<timestamp>.log
```

This directory is shared by **every user and process on the box**. Finding
your run means guessing by PID and timestamp — and racing other writers.

**Always** redirect the device log before invoking task-submit:

```bash
LOGDIR="$PWD/outputs/<case>/ascend"
mkdir -p "$LOGDIR"
export ASCEND_PROCESS_LOG_PATH="$LOGDIR"
task-submit --device auto --device-num 1 --run "python ..."
# Your isolated log is now at: $LOGDIR/device-*/device-*.log
```

The directory **must exist** before the run (the driver calls `fopen`, not
`mkdir`). Both `export` and `--env ASCEND_PROCESS_LOG_PATH=...` work
identically — task-submit inherits the caller's environment.

### 4.2 Error Signature Taxonomy

| Device-log signature | Mechanism | Meaning | Action |
|---|---|---|---|
| `FATAL: Task Allocator Deadlock` / `Provable head-of-line` | Ring/heap structural deadlock | Task window too small for the scope depth; deadlock is *provable*, not timeout-based | Increase `ring_task_window` |
| `Timeout (N cycles): producer/consumers ...` | SPIN wait on specific producer | Consumer spin-waited 500 ms+ on a producer that never completed | Trace the dependency chain |
| `HandleTaskTimeout` / `kill aicpu-sd` | OS op-execute timeout | The op ran longer than 45s (default) — **not necessarily a deadlock**. Could be a large compute, a stall, or a race | Profile, check expected runtime |
| `log_stall_diagnostics` (cores idle, `state=WAIT fanin 0/N`, `completed` frozen) | Forward-progress stall | No dedicated detector; intermittent races (often contention-triggered) | Same as above |
| No detector fired, only `HandleTaskTimeout` | Long/stalled op | Not a capacity/deadlock bug — the hardware detectors would have caught it | Look for a race or unexpectedly large input |

**Decisive rule:** If the device log shows zero hits for "Task Allocator
Deadlock" and "Timeout (cycles)", and only `HandleTaskTimeout` fired, it
is **not** a deadlock or capacity bug. Capacity exhaustion trips its own
dedicated detector (500 ms backstop, or immediate structural check).

### 4.3 Deadlock Detection Architecture

The runtime has **four independent detection layers**, each with its own
trigger and scope:

| Layer | Trigger | Timeout | Scope |
|---|---|---|---|
| **Structural deadlock** | Head task COMPLETED with scope still open | Immediate (provable) | One ring |
| **Ring/heap wall-clock** | Task/heap allocator spin-waits with no watermark progress | 500 ms | One ring |
| **TensorMap pool** | Entry pool exhausted, zero entries freed per reclaim attempt | 500 ms | Global |
| **Scope capacity** | Single scope submits `>= window_size` tasks | Immediate (structural) | One ring depth |

The structural check (layer 1) is load-bearing: a ring full of tasks that
are completed but whose scope hasn't ended is a *provable* deadlock — the
orchestrator (which calls `scope_end`) is the only thread that can advance
the watermark, and it's blocked. No timeout needed.

### 4.4 Key Timing Constants

| Parameter | Value | Where defined |
|---|---|---|
| Ring/heap deadlock timeout | 500 ms | `pto_ring_buffer.h:61` |
| BLOCKED warn interval | 10,000 spins | `pto_ring_buffer.h:55` |
| Scheduler wall-clock timeout | 10,000 ms | `platform_config.h:75` |
| STARS op-execute timeout | 45,000 ms | `platform_config.h:38` |
| Host stream-sync timeout | 50,000 ms | `platform_config.h:40` |

**Timeout ordering invariant** (enforced at runtime):
`scheduler < op-execute < stream-sync`. The scheduler timeout has a 1500 ms
guard margin below op-execute. On sim platforms this ordering check is
bypassed.

---

## 5. The PTO2 Runtime: How the Device-Side Orchestrator Works

When your program runs on hardware (whether through task-submit the CLI or
directly), the following architecture is what dispatches your kernels.

### 5.1 Execution Layers

```
                         ┌──────────────────────┐
                         │   HOST (x86/ARM)     │
                         │ compile, alloc, init │
                         │ rtMalloc, rtMemcpy   │
                         │ launch AICPU threads │
                         └──────────┬───────────┘
                                    │ device memory (GM)
                         ┌──────────▼───────────┐
                         │ AICPU (NPU ARM cores) │
                         │                       │
                         │ Thread 3: Orchestrator│
                         │   builds task graph   │
                         │   calls rt_submit_*   │
                         │                       │
                         │ Threads 0-2: Scheduler│
                         │   poll COND register  │
                         │   pop ready queues    │
                         │   write DATA_MAIN_BASE│
                         │                       │
                         │ Shared Memory (GM):   │
                         │  ├─ TaskRing[4]       │
                         │  ├─ HeapRing[4]       │
                         │  ├─ DepListPool[4]    │
                         │  ├─ TensorMap         │
                         │  └─ PTO2TaskDescriptor│
                         └──────────┬───────────┘
                                    │ register handshake
                         ┌──────────▼───────────┐
                         │ AICore (compute)      │
                         │   poll DATA_MAIN_BASE │
                         │ │ if task_id changed: │
                         │ │   ACK → execute     │
                         │ │   FIN → idle        │
                         └──────────────────────┘
```

### 5.2 Hardware Model

| Arch | Cores | Clusters | AICPU threads |
|---|---|---|---|
| a2a3 (Ascend 910B) | 24 AIC + 48 AIV | 24 (1 AIC + 2 AIV each) | 4 (3 sched + 1 orch) |
| a5 (Ascend 950) | 36 AIC + 72 AIV | 36 (1 AIC + 2 AIV each) | 7 (6 sched + 1 orch) |

AIC cores handle matrix multiplication (Cube units). AIV cores handle
vector/scalar operations. They share L2 cache and memory bandwidth within
a cluster, so co-dispatching a mixed task (AIC + AIV together on the same
cluster) has locality benefits — both the planned motivation and the reason
the runtime guarantees atomic cluster dispatch.

### 5.3 The Submit API

```cpp
// A cluster is 1 AIC + 2 AIV. Each slot is independently active or idle.
struct MixedKernels {
    int32_t aic_kernel_id{INVALID_KERNEL_ID};   // -1 = unused
    int32_t aiv0_kernel_id{INVALID_KERNEL_ID};
    int32_t aiv1_kernel_id{INVALID_KERNEL_ID};
};

// Create one node in the task graph.
void rt_submit_task(PTO2Runtime* rt, const MixedKernels& mixed, Arg* args, int32_t n);

// Convenience — these expand to rt_submit_task with a single-slot mask.
void rt_submit_aic_task(PTO2Runtime* rt, int32_t kernel_id, Arg* args, int32_t n);
void rt_submit_aiv_task(PTO2Runtime* rt, int32_t kernel_id, Arg* args, int32_t n);
```

Every submit creates exactly one `MixedTask` in the graph. The scheduler
later dispatches all active slots of that task atomically to one cluster.

### 5.4 What Happens Inside `submit_task`

The orchestrator runs a 6-step pipeline per submit call:

| Step | Operation | Back-pressure? |
|---|---|---|
| 0 | `sync_tensormap` — lazy-prune entries for tasks whose slot was reclaimed | No |
| 1 | `alloc` — allocate a slot from the task ring buffer | **Yes** — blocks if ring full |
| 2 | `init` — fill task descriptor (kernel IDs, active mask) and payload (args, scalars) | No |
| 3 | `lookup` — for each INPUT/INOUT arg, `TensorMap::lookup(addr)` finds the producer task | No |
| 4 | `insert` — register OUTPUT/INOUT args as new producers in TensorMap | **Yes** — blocks if pool exhausted |
| 5 | `wire + publish` — record fanin edges, wire fanout to consumers, push ready tasks to scheduler queues | No |

### 5.5 Automatic Dependency Discovery (TensorMap)

Dependencies are **not declared**. The runtime infers them from memory
addresses:

- **INPUT/INOUT**: `TensorMap::lookup(base_addr)` returns the last task
  that wrote to that address range. If found, a dependency edge is created.
- **OUTPUT/INOUT**: `TensorMap::insert(base_addr, task_id)` registers this
  task as the new producer.

Overlap detection handles sub-regions: a write to `[0, 16]` of a buffer
followed by a read of `[8, 24]` correctly identifies the overlapping
producer.

Stale entries are cleaned through a three-layer defense:

1. **Chain truncation** — bucket chains are in descending task_id order, so
   the first stale entry truncates the entire tail (O(1) unlinking).
2. **Periodic batch** — every 64 retired tasks, per-task entry chains are
   freed in bulk (no full-pool scan).
3. **Back-pressure** — if the pool is exhausted, the orchestrator blocks
   until replaced tasks free enough entries.

### 5.6 Resource Shapes and Scheduling

Tasks are classified by which cores they need and queued in **per-shape**
lock-free MPMC (Vyukov) queues:

| Shape | Active mask | What it occupies |
|---|---|---|
| `AIC` | AIC only | 1 AIC core |
| `AIV` | AIV0 or AIV1 | 1 AIV core |
| `MIX` | AIC + any AIV(s) | 1 entire cluster (AIC + AIV0 + AIV1) |

Cluster dispatch is **atomic**: for a MIX task needing AIC + 2 AIV, all
three cores must be available simultaneously before any of them are
launched. No partial dispatch.

Each scheduler thread owns a disjoint subset of clusters, assigned at
init time during a parallel handshake phase. The `CoreTracker` structure
encodes per-cluster idle/running/pending state in a 3-bit bitmask per
cluster, enabling O(1) cluster-state queries.

### 5.7 The Scheduler Main Loop

```
while (!orchestrator_done || any_tasks_remaining) {
    Phase 1 — Completion:
        for each core this thread owns:
            poll COND register (volatile uint32_t*)
            if FIN detected for task T, subtask S:
                task->completed_subtasks++
                if completed_subtasks == total_required:
                    on_task_complete(T):
                        mark task_state = COMPLETED
                        acquire fanout_lock
                        traverse fanout list → increment consumers' fanin_refcount
                        mark task_state = CONSUMED
                        advance last_task_alive watermark (try-lock per ring)

    Phase 2 — Dispatch:
        drain ready queues in occupancy order:
            sync_start (Tier-0) → MIX → AIC/AIV
            idle cores first, then pending slots
        for each ready task:
            build PTO2DispatchPayload from TaskDescriptor
            write payload to Handshake.task
            write dispatch_seq to DATA_MAIN_BASE (register doorbell)

    // Only if BOTH normal lanes are empty:
    Phase 2b — Early dispatch (speculative):
        pop from early_dispatch_queues
        pre-stage payloads without launching (gated on src_payload)
        when producer completes → ring per-core doorbell → consumer launches
}
```

### 5.8 The Dispatch Payload

```cpp
struct alignas(64) PTO2DispatchPayload {
    uint64_t function_bin_addr;           // kernel entry point address
    PTO2LocalContext local_context;       // SPMD: block_idx, block_num
    void *src_payload;                    // early-dispatch gate (nullptr = immediate)
    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
    PTO2GlobalContext global_context;     // sub_block_id for multi-invocation
};
```

- `function_bin_addr` — resolved at dispatch time from `func_id_to_addr_[]`.
- `src_payload` — `nullptr` means "execute immediately"; non-null means
  "wait for doorbell from the producer before starting." This enables
  speculative pre-staging (Section 5.10).
- `alignas(64)` — exactly one cache line, no false sharing between cores.

### 5.9 The Handshake Protocol (DATA_MAIN_BASE / COND)

The scheduler and AICore communicate through two hardware registers:

```
DATA_MAIN_BASE  (offset 0xA0 a2a3, 0xD0 a5)   AICPU → AICore
COND            (offset 0x4C8 a2a3, 0x5108 a5)  AICore → AICPU

COND register encoding:  [bit31 = state | bits30:0 = task_id]
  TASK_ACK_STATE = 0  → "I received task N"
  TASK_FIN_STATE = 1  → "I completed task N"

Idle sentinel:  AICORE_IDLE_TASK_ID = 0x7FFFFFFF
Exit sentinel:  AICORE_EXIT_TASK_ID = 0x7FFFFFFE
```

**Dispatch sequence:**
1. Scheduler writes `dispatch_seq` to `DATA_MAIN_BASE`. AICore detects the
   changed value, reads `Handshake.task` for the payload.
2. AICore writes `ACK(task_id)` to `COND` — "I have the payload."
3. AICore executes the kernel.
4. AICore writes `FIN(task_id)` to `COND` — "I am done."

The scheduler polls `cond_ptr` (a `volatile uint32_t*` precomputed per core
at init) in a tight spin loop. No shared-memory polling — all register I/O.

**Multi-ring fix:** Since task IDs are 64-bit (upper 32 = ring_id, lower 32
= local_id) but `DATA_MAIN_BASE` is 32-bit, a per-core monotonic
`dispatch_seq` replaces task_id in register writes to avoid truncation
collisions.

### 5.10 Early Dispatch (Speculative Pre-Staging)

When a scheduler thread has idle cores but no ready tasks, it can
**speculatively pre-stage** a consumer's payload in a core's slot *before*
the producer finishes. The consumer's bootstrap routine will spin on the
`src_payload` doorbell. When the producer completes and publishes its
fanout, the scheduler rings the per-core doorbell — the consumer launches
immediately with no register-poll dispatch overhead.

**Gate:** A producer must have **published** all its blocks before any
consumer can be speculatively staged. For an SPMD producer with 50 blocks on
24 cores, this means all 50 blocks' payloads and MMIO dispatch tokens must be
visible before the consumer can pre-occupy resources — otherwise the
consumer's rendezvous would wedge, waiting for a producer release that can
never happen because the producer's remaining blocks can't launch.

### 5.11 sync_start: SPMD Atomic Launch

Some tasks require all SPMD blocks to launch simultaneously (e.g., `MPI_Barrier`
equivalent). The `sync_start` drain protocol achieves this through:

1. **Single election** — a CAS on `sync_start_pending` makes drains mutually
   exclusive; at most one cohort drains at a time.
2. **All-or-nothing** — the elected thread verifies *global* available-core
   count across all scheduler threads. If short, it retries.
3. **Parallel stage** — all threads barrier, then each CAS-claims a block
   range and stages its own cores. Gates all blocks on a shared `pending_task`.
4. **Rendezvous launch** — when `running_slot_count` reaches
   `popcount(staged_core_mask)` AND the producer has released, all cores'
   doorbells are rung together.

### 5.12 Scopes and Buffer Lifetime

```cpp
PTO2_SCOPE(rt) {
    auto qk = rt_submit_aic_task(FUNC_QK, args);   // output lives in this scope
    auto sf = rt_submit_aiv_task(FUNC_SF, args);
    // qk and sf outputs are valid here
}
// scope_end → scope reference released → buffers eligible for reclamation
```

Buffers live in per-scope ring-buffer slots. Each scope has its own
TaskRing, HeapRing, and DepListPool (4 depth levels total), enabling
independent reclamation without waiting for sibling/deeper scopes.

**Critical invariant:** `TaskOutputTensors` references returned by submit
must not outlive their `PTO2_SCOPE`. The underlying storage is a ring-buffer
slot that gets reused. There is no runtime check for this — it's a static
contract verified by code review.

### 5.13 Ring Buffer Flow Control

The orchestrator and scheduler communicate through bounded ring buffers.
When the orchestrator produces faster than the scheduler can consume:

```
active_tasks = submitted - completed
if active_tasks >= window_size - 1:
    ORCHESTRATOR BLOCKS
```

This back-pressure is *normal* — it prevents unbounded memory growth. The
orchestrator spin-waits with periodic diagnostic logging (every 10,000 spins)
and a 500 ms deadlock backstop.

**Sizing rule:** `task_window_size` must exceed the maximum live tasks in
any single scope. A safe choice is `2 × max_tasks_per_scope`, or the default
of 16384 for production.

---

## 6. How PyPTO Programs Reach the Device

### 6.1 Compilation Pipeline

```
Python @pl.program
  │
  ▼ IR passes → C++ codegen
  │
  ▼ Generated orchestration .cpp + kernel .cpp
  │
  ▼ compiled → orchestration .so + kernel .o → ChipCallable
  │
  ▼ uploaded to device (device_malloc + H2D copy)
  │
  ▼ AICPU orchestrator: dlopen(orch.so) → dlsym("aicpu_orchestration_entry")
  │   calls rt_submit_aic_task / rt_submit_aiv_task in user order
  │
  ▼ Scheduler dispatches to AICore
```

The OS-level orchestration loading on AICPU:

1. SO binary is copied from device memory to a filesystem path (tries
   `/usr/lib64/aicpu_kernels/`, `/usr/lib64`, `/lib64`, `/var/tmp`, `/tmp`).
2. `dlopen(RTLD_LAZY | RTLD_LOCAL)` — then `unlink` the temp file immediately
   (the image is mmap'd; this avoids stale `.so` files if the worker exits
   via `os._exit()`).
3. `dlsym("aicpu_orchestration_entry")` → function pointer.
4. Optional: `dlsym("aicpu_orchestration_config")` for arg-count validation.
5. Optional: `dlsym("framework_bind_runtime")` for SO-local runtime pointer.
6. Call the entry function within a scope.

A per-callable_id table keeps SO handles warm across runs (one dlopen per
callable, reused indefinitely). Since only Thread 3 (the orchestrator)
touches this table, no locking needed.

### 6.2 Test Harness Integration

The ST test harness has two execution modes:

**Default (direct):** Compile on CPU, execute on device via `DeviceRunner.run()`.
The test process holds the device for both compile and execute.

**`--execute-via-task-submit` (opt-in):** Compile on CPU (no card needed),
then for each case spawn:

```
task-submit --device auto --run \
    'python -m pypto.runtime.execute_artifact --work-dir <wd> \
       --platform a2a3 --device-id $TASK_DEVICE'
```

The child process re-binds the pre-compiled `.o`/`.so` artifacts (no
recompilation), runs on the allocated device, and reports via a sentinel
line:

```
PYPTO_EXEC_RESULT=PASS device=8    # success
PYPTO_EXEC_RESULT=FAIL             # numerical / device failure
PYPTO_EXEC_RESULT=INFRA            # setup / cache failure (retry-safe)
```

**Batching:** To avoid per-artifact cold-start (torch/pypto import + NPU init),
the harness groups compiled artifacts into batches. One `task-submit` task
runs the entire batch in a single hot process, reusing one `ChipWorker` device
session. Artifacts within a batch must share the same `(platform, runtime)`
pair — mixing them would trigger `halMemCtl EACCES` from concurrent
`ChipWorker.init()` calls.

---

## 7. Practical Workflows

### 7.1 I Want to Run My New Kernel on Hardware

```bash
# 1. Check what's free
task-submit --list

# 2. Arch precheck
.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1

# 3. Isolated device log
mkdir -p outputs/my_test/ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/my_test/ascend"

# 4. Run
task-submit --device auto --device-num 1 \
    --timeout 1800 --max-time 1800 \
    --run "python my_kernel.py -p a2a3 -d \$TASK_DEVICE"
```

### 7.2 I'm Debugging a `507018` Failure

1. **Find your device log.** If you set `ASCEND_PROCESS_LOG_PATH`, it's
   there. Otherwise check `~/ascend/log/debug/device-*/`.
2. **Grep for the detector signatures** — not the error code:

```bash
rg -c 'FATAL: Task Allocator' device-*.log
rg -c 'Timeout.*cycles.*producer' device-*.log
rg -c 'HandleTaskTimeout' device-*.log
rg -c 'log_stall_diagnostics' device-*.log
```

3. **Classify:**
   - Only `HandleTaskTimeout` with no other hits → not a deadlock. The op
     was slow or stalled. Increase `PLATFORM_OP_EXECUTE_TIMEOUT_US` to
     measure true on-device duration.
   - `FATAL: Task Allocator Deadlock` → ring/heap capacity deadlock.
     Increase `ring_task_window` or reduce max tasks per scope.
   - Both → the deadlock detector fired first, the op-timeout is the
     cascading kill.

### 7.3 I Need to Stress-Test a Flaky Race Condition

```bash
# Hold the lock for the whole loop — one device, one process
task-submit --device 12,13 --timeout 7200 --max-time 7200 \
    --run "bash -c '
        for i in \$(seq 1 500); do
            python my_test.py -p a2a3 -d \$TASK_DEVICE || echo \"FAIL at \$i\"
        done
    '"
```

**Do not** spawn 500 individual task-submit invocations — each one acquires
and releases the lock, and you may bounce between devices. Reproducibility
requires the same device across all iterations.

### 7.4 I'm in a Docker Container Without `task-submit`

The CANN Docker images do not include `task-submit`. This is normal — the
container owns its devices exclusively. Run your commands directly.

```bash
# Inside the container — no task-submit needed
python -m pytest tests/ut/ -v
python -m pytest tests/st/ --platform a2a3 --device 0
```

---

## 8. Anti-Patterns

| Pattern | Why it's broken | What to do instead |
|---|---|---|
| `pytest --device 8` without task-submit | Races other users for card 8 | Wrap in `task-submit --device auto` |
| `for i in ...; do task-submit ...; done` | Lock acquired/released N times; device may change | One `task-submit` wrapping the whole loop |
| Claiming "X% repro rate" from unlocked runs | Device state inherited from prior user | Check `task-submit --list` at time of run |
| Calling every 507018 a deadlock | Generic error code, many root causes | Read device log; classify by detector signature |
| Skipping arch precheck | Wrong-arch 507018 looks identical to a real bug | Always run precheck before onboard |
| Fishing for device logs in `~/ascend/log/debug/` | Shared directory, guessing PID/timestamp | Set `ASCEND_PROCESS_LOG_PATH` per-run |
| Mixing platforms in a task-submit batch | `halMemCtl EACCES` from concurrent init | Group by `(platform, runtime)` in batch manifest |

---

## 9. Summary: Key Constants for Sizing and Debugging

| Parameter | Default | Tune when |
|---|---|---|
| `ring_task_window` | 16384 | Deadlock → increase; memory pressure → decrease |
| Ring depth (PTO2_MAX_RING_DEPTH) | 4 | Number of nested scopes that need independent ring resources |
| `heap_per_ring` | 256 MB | Output buffers exceed heap → increase |
| `dep_list_pool_entries` | 16384 per ring | Fanout count per task × active tasks > pool |
| `tensormap_pool_size` | 65536 entries | Entry pool exhaustion deadlock |
| `PLATFORM_OP_EXECUTE_TIMEOUT_US` | 45,000,000 (45s) | Long-running kernels killed prematurely |
| `PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES` | 500 ms | Tuning deadlock detection sensitivity |
| `PLATFORM_MAX_BLOCKDIM` | 24 (a2a3) / 36 (a5) | Hardware-dependent |
| `PLATFORM_MAX_AICPU_THREADS` | 4 (a2a3) / 7 (a5) | Hardware-dependent |

---

## 10. Further Reading

- `simpler/src/a2a3/runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md` — full PTO2 runtime design
- `simpler/src/a2a3/runtime/tensormap_and_ringbuffer/docs/SUBMIT_BY_CLUSTER.md` — cluster-dispatch requirements and acceptance criteria
- `simpler/src/a2a3/runtime/tensormap_and_ringbuffer/docs/MULTI_RING.md` — multi-ring scope isolation
- `simpler/.claude/rules/running-onboard.md` — dev-box onboard policy
- `pypto/python/pypto/runtime/execute_artifact.py` — CLI entry point for the device-side harness
- `pypto/tests/st/harness/core/test_runner.py` — batch submitter and pipeline state machine
- `pypto/tests/ut/runtime/test_task_submit_dispatch.py` — CLI construction and result-classification unit tests