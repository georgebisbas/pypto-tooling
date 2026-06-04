"""Run artifact bundle layout and manifest.json writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunArtifactBundle:
    """Directory for one (case_id, stack) execution under a campaign run."""

    run_dir: Path
    case_id: str
    stack: str

    @property
    def bundle_dir(self) -> Path:
        return self.run_dir / "cases" / self.case_id / self.stack

    @property
    def log_path(self) -> Path:
        return self.bundle_dir / "run.log"

    @property
    def timing_path(self) -> Path:
        return self.bundle_dir / "timing.json"

    @property
    def manifest_path(self) -> Path:
        return self.bundle_dir / "manifest.json"

    @property
    def profiling_dir(self) -> Path:
        return self.bundle_dir / "profiling"

    @property
    def upstream_dir(self) -> Path:
        return self.bundle_dir / "upstream"

    def ensure_dirs(self) -> None:
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.profiling_dir.mkdir(parents=True, exist_ok=True)
        self.upstream_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(
        self,
        *,
        correctness: str = "unknown",
        error: str | None = None,
        profiling: dict[str, str | None] | None = None,
        upstream_work_dir: str | None = None,
    ) -> Path:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "stack": self.stack,
            "correctness": correctness,
            "logs": {"run": "run.log"},
            "timing": "timing.json",
            "profiling": profiling or {},
        }
        if error:
            payload["error"] = error
        if upstream_work_dir:
            payload["upstream"] = {"work_dir": upstream_work_dir}
        self.manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return self.manifest_path

    def write_timing(self, samples: list[dict[str, Any]]) -> Path:
        self.timing_path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        return self.timing_path
