# GM alignment rules (A2A3 vs A5)

*MCP-owned summary — see `BackendHandler` headers and `ascend-architectures.md` for full tables.*

## Why alignment matters

PyPTO mixed AIC↔AIV kernels on **A2A3** round-trip through **GM**. Misaligned inner dimensions trigger PH001 perf hints and extra traffic. **A5** uses on-chip fractal pipes but still has minimum GM granularity.

## A2A3 (`a2a3` / `a2a3sim`) — 512-byte GM granularity

`BackendHandler910B::GetGmAccessGranularityBytes()` → **512**

| Dtype | Minimum inner dim (elements) | Preferred for L2 line |
|-------|------------------------------|------------------------|
| FP32 (4 B) | 128 | 128+ |
| BF16 (2 B) | 256 | 256+ |
| FP16 (2 B) | 256 | 256+ |

**Rule of thumb:** trailing GM dimension × element_size should be a **multiple of 512 bytes**.

Mixed kernels (`SplitMode::None` on A2A3) may need **both AIV lanes** (`RequiresNoSplitDualAivDispatch`).

## A5 (`a5` / `a5sim`) — 128-byte minimum, 512-byte preferred

`BackendHandler950::GetGmAccessGranularityBytes()` → **128**  
`GetL2CacheLineBytes()` → **512** (both archs)

| Dtype | Minimum inner dim | Preferred |
|-------|-------------------|-----------|
| FP32 | 32 | 128 |
| BF16 / FP16 | 64 | 256 |

AIV→cube handoff may insert **fractal adapter** `tile.move` (`RequiresVtoCFractalAdapt`).

## Memory hierarchy (tuning lens)

| Level | Role |
|-------|------|
| **GM** | Global tensors; MTE2/MTE3; HCCL windows |
| **L1 / UB** | Operand staging; vector working set |
| **L0A / L0B / L0C** | Cube matmul tiles |
| **Pipes** | MTE1 (L1→L0), MTE2 (GM→L1), MTE3 (UB→GM), cube, vector |

L0C budget: **128 KiB** (A2A3) vs **256 KiB** (A5) — affects matmul tile sizing.

## block_dim sweeps

| Arch | `PLATFORM_MAX_BLOCKDIM` | Notes |
|------|-------------------------|-------|
| a2a3 | 24 | 1 unit = 1 AIC + 2 AIV |
| a5 | 36 | Same 3 cores per block |

Distributed STs often use `block_dim=3` as smoke — not an optimum. Sweep toward arch max subject to memory.

## MCP queries

- `explain_abstraction BackendHandler910B` or `BackendHandler950`
- `route_task npu_tuning`
- `hw-native-sys://ascend/arch_families`
