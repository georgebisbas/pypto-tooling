"""EquivalenceCase — single source of truth for apples-to-apples pypto vs simpler runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

ORCH_PROFILES: dict[str, dict[str, Any]] = {
    "mesh_l3_host_domain_v1": {
        "description": "L3 HOST: 1 comm domain, P chip submits, 0 sub-workers, 4-phase mesh InCore",
        "linqu_level": "L3_HOST",
        "num_sub_workers": 0,
        "comm_domains_per_execute": 1,
        "chip_submissions": "P",
        "incore_phases": 4,
    },
}


@dataclass
class EquivalenceCase:
    """All knobs that must match between simpler and pypto for a paired comparison."""

    variant: str  # mesh | ring | hccl
    p: int
    count: int
    dtype: str = "fp32"
    device_ids: list[int] = field(default_factory=list)
    platform: str = "a2a3"
    runtime: str = "tensormap_and_ringbuffer"
    input_formula: str = "rank_linear_v1"
    golden: str = "allreduce_sum_v1"
    orch_profile: str = "mesh_l3_host_domain_v1"
    orch_tier: str = "logical"  # logical | strict_window
    block_dim: int | None = None
    aicpu_thread_num: int | None = None
    warmup_rounds: int = 3
    timed_rounds: int = 20
    measure: str = "execute_only"

    def __post_init__(self) -> None:
        if not self.device_ids:
            self.device_ids = list(range(self.p))
        if len(self.device_ids) != self.p:
            raise ValueError(f"len(device_ids)={len(self.device_ids)} != p={self.p}")

    @property
    def n_bytes(self) -> int:
        nbytes_per_elem = 4 if self.dtype == "fp32" else 2 if self.dtype == "fp16" else 4
        return self.count * nbytes_per_elem

    @property
    def window_nbytes(self) -> int:
        return max(self.n_bytes, 4096)

    @property
    def case_id(self) -> str:
        dev = "-".join(str(d) for d in self.device_ids)
        return f"{self.variant}_p{self.p}_count{self.count}_{self.dtype}_{self.platform}_d{dev}"

    def canonical_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_bytes"] = self.n_bytes
        d["window_nbytes"] = self.window_nbytes
        return d

    def equivalence_hash(self) -> str:
        blob = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def validate(self) -> None:
        if self.variant not in ("mesh", "ring", "hccl"):
            raise ValueError(f"unknown variant: {self.variant}")
        if self.p < 2:
            raise ValueError(f"p must be >= 2, got {self.p}")
        if self.count <= 0:
            raise ValueError(f"count must be positive, got {self.count}")
        if self.variant == "ring" and self.count % self.p != 0:
            raise ValueError(f"ring requires count % p == 0, got {self.count} % {self.p}")
        if self.orch_profile not in ORCH_PROFILES:
            raise ValueError(f"unknown orch_profile: {self.orch_profile}")
        if self.variant == "mesh" and self.orch_profile != "mesh_l3_host_domain_v1":
            raise ValueError("mesh variant requires mesh_l3_host_domain_v1 orch profile")

    @classmethod
    def from_json_file(cls, path: str) -> EquivalenceCase:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def validate_orch_parity(manifest: dict[str, Any], case: EquivalenceCase) -> list[str]:
    """Best-effort checks on manifest / deps after a run. Returns warning strings."""
    warnings: list[str] = []
    profile = ORCH_PROFILES.get(case.orch_profile)
    if not profile:
        warnings.append(f"unknown orch_profile {case.orch_profile}")
        return warnings
    deps = (manifest.get("profiling") or {}).get("deps")
    if deps and case.p > 0:
        # Full validation deferred to E1 (parse deps.json task count).
        pass
    return warnings
