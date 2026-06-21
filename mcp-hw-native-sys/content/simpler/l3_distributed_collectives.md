# Simpler L3 distributed collectives — CI and platform markers

*MCP-owned — canonical examples live in `simpler/examples/workers/l3/`.*

## Sim CI runs these (confirmed)

The `st-sim-a2a3` job in [`.github/workflows/ci.yml`](../../../../simpler/.github/workflows/ci.yml) runs:

```bash
pytest examples tests/st --platform a2a3sim --device 0-15 -v
```

L3 comm-domain examples under `examples/workers/l3/` with
`@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", ...])` **are selected and
executed** on `a2a3sim`. CI logs (e.g. run `27813083617`, job `82307725017`,
2026-06-19) show **PASS** for:

| Test | Sim CI |
|------|--------|
| `test_allreduce_distributed[onephase]` | PASS |
| `test_allreduce_distributed[twophase]` | PASS |
| `test_allreduce_distributed[ring]` | PASS |
| `test_allreduce_distributed_multi_rank[4-onephase]` | PASS |
| `test_allreduce_distributed_multi_rank[4-twophase]` | PASS |
| `test_allreduce_distributed_multi_rank[4-ring]` | PASS |
| `test_allgather_distributed[2/4]` | PASS |
| `test_reduce_scatter_distributed[2/4]` | PASS |
| `test_broadcast_distributed[2/4]` | PASS |
| `test_all_to_all_distributed[2/4]` | PASS |
| `test_domain_rank_map` | PASS |
| `test_dual_domain_overlap` | PASS |
| `test_ep_dispatch_combine` | PASS |
| `test_ffn_tp_parallel` | PASS |

**Do not remove `a2a3sim` from `@pytest.mark.platforms` on these STs** without
an explicit CI/workflow change — sim is the primary gate for PRs.

## Local Docker reproduction (verified 2026-06-20)

Image: [`Dockerfile.simpler.sim.ubuntu22.04`](../../../../pypto-tooling/Dockerfile.simpler.sim.ubuntu22.04)

```bash
cd pypto-tooling
docker build -t simpler-hw-native-sys:sim -f Dockerfile.simpler.sim.ubuntu22.04 .

# L3 distributed subset — use --shm-size (default Docker /dev/shm is too small for forked L3 + torch)
docker run --rm --shm-size=4g simpler-hw-native-sys:sim \
  /opt/pypto-tooling/scripts/run-simpler-l3-sim.sh distributed
```

**Results (local run):**

| Phase | Source | Outcome |
|-------|--------|---------|
| A | Image-baked simpler `286aa7e` | **14/14 PASS** (`distributed` scope) |
| B | Mount `simpler/` + `pip install -e '.[test]'` ([#1092](https://github.com/hw-native-sys/simpler/pull/1092) consolidation branch) | **6/6 PASS** allreduce (`onephase`/`twophase`/`ring` × 2- and 4-rank); **18/18 PASS** full `distributed` scope |

Without `--shm-size=4g`, parallel L3 subprocesses can fail with
`RuntimeError: No space left on device` from torch shared-memory (`/dev/shm`) —
environment limitation, not a collective algorithm failure.

Local branch mount:

```bash
docker run --rm --shm-size=4g -v /path/to/simpler:/opt/simpler simpler-hw-native-sys:sim
# inside: pip install --no-build-isolation -e '.[test]'
#         /opt/pypto-tooling/scripts/run-simpler-l3-sim.sh distributed
```

Helper script scopes: `distributed` | `allreduce` | `all` — see
[`scripts/run-simpler-l3-sim.sh`](../../../../pypto-tooling/scripts/run-simpler-l3-sim.sh).

## What sim exercises vs hardware

| Aspect | `a2a3sim` | `a2a3` / `a5` |
|--------|-----------|---------------|
| Comm-domain allocation + forked L3 workers | Yes (thread-based sim HCCL window) | Yes (real HCCL IPC) |
| Golden ST correctness | Yes — CI gate | Yes — `st-onboard-a2a3` / `st-onboard-a5` |
| PMU / real timing / CANN quirks | No | Yes |

## Platform markers (do not narrow without cause)

Typical pattern for distributed collectives and comm-domain demos:

```python
@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim"])
```

`allreduce_distributed` 2-rank test also lists `"a5"`.

## Related

- [simpler #1092](https://github.com/hw-native-sys/simpler/pull/1092) — consolidate allreduce variants (`--mode`); CI green
- `simpler/docs/comm-domain.md` — domain allocation API
- `simpler/docs/testing.md` — `--platform` / marker semantics
- `hw-native-sys://ascend/which_platform` — general sim vs NPU decision tree
