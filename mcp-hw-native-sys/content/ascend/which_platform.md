# Ascend platform chooser (hw-native-sys)

*MCP-owned decision tree — canonical platform docs live in sibling repos.*

## Quick map

| Goal | `platform` flag | Hardware |
|------|-----------------|----------|
| CI / dev without NPU | `a2a3sim` or `a5sim` | CPU simulator |
| Training / 910B / 910C inference | `a2a3` | Real NPU (910B, 910_93) |
| Ascend950 decode / prefill | `a5` | Real NPU (950DT, 950PR) |

## PyPTO backend mapping

| CLI / pytest | `BackendType` | Simpler tree | PTO arch |
|--------------|---------------|--------------|----------|
| `a2a3`, `a2a3sim` | `Ascend910B` | `simpler/src/a2a3/` | `--pto-arch a3` |
| `a5`, `a5sim` | `Ascend950` | `simpler/src/a5/` | `--pto-arch a5` |

One backend handler covers both A2 (910B) and A3 (910_93 / 910C) on the a2a3 code path.

## When to use simulator

- IR passes, codegen round-trips, most `tests/ut/`
- `tests/st` with `--platform=a2a3sim,a5sim` (no device bind)
- Agent gate tasks (Docker sim) — not a substitute for HCCL multi-NPU STs

## When real NPU is required

- PyPTO `tests/st/distributed/` multi-rank HCCL (separate from simpler `examples/`)
- Performance tuning with PMU / Insight
- pypto-lib model scripts with `-p a2a3 -d <id>`

**Note:** Simpler L3 `examples/workers/l3/*_distributed/` STs **do run on `a2a3sim`**
in CI (`st-sim-a2a3`). See `hw-native-sys://simpler/l3_distributed_collectives`.

## CANN / SocVersion (AscendC layer)

| Alias | Chips | Notes |
|-------|-------|-------|
| A2 | 910B1–B4 | DAV_2201 |
| A3 | 910_93 / 910C | Same a2a3 PyPTO path as A2 |
| A5 | 950DT, 950PR | arch35 — Regbase, SIMT, FP8 in AscendC; PyPTO primary path is PTO ISA |

See `hw-native-sys://ascend/cann_mapping` and enriched `ascend-architectures.md`.

## Decision flow

```text
Need HCCL multi-rank?
├─ YES → real NPU, a2a3 (today's distributed STs), >=2 devices
│         + container: --pid=host, LD_PRELOAD libhccl.so (see hccl_container_checklist)
└─ NO  → Need A5-specific kernel behavior (fractal V2C, larger L0C)?
          ├─ YES → a5 or a5sim
          └─ NO  → a2a3sim for fast iteration; a2a3 for hardware smoke
```

## Related MCP resources

- `hw-native-sys://ascend/arch_families` — handler diffs, block_dim limits
- `hw-native-sys://ascend/alignment_rules` — GM alignment per arch
- `hw-native-sys://simpler/l3_distributed_collectives` — L3 collective ST hardware requirement
- `route_task` with `npu_tuning` or `ascend_arch`
