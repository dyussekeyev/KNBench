"""Run provenance sidecar for KNBench.

The sidecar keeps reproducibility metadata out of the canonical results schema.
It is written next to the results file as ``<stem>.provenance.json`` and never
contains an API key, document text or model response.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from knbench.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_TASK_PROMPT
from knbench.schemas import RunResults

DEPENDENCIES: Final = (
    "pydantic",
    "pydantic-settings",
    "httpx",
    "python-dotenv",
)
SIDECAR_SCHEMA_VERSION: Final = "1.0"


class ProvenanceFile(BaseModel):
    """Machine-readable provenance for one completed or partial run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SIDECAR_SCHEMA_VERSION
    run_id: str
    generated_at: datetime
    python_version: str
    code_revision: str | None = None
    code_dirty: bool | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    dataset_sha256: str
    models: list[str]
    judge_model: str
    target_prompt_sha256: dict[str, str] = Field(default_factory=dict)
    judge_prompt_sha256: list[str] = Field(default_factory=list)


def provenance_path(results_path: Path) -> Path:
    """Return the sidecar path for a results file."""
    source = Path(results_path)
    return source.with_name(f"{source.stem}.provenance.json")


def build_provenance(
    results: RunResults, *, generated_at: datetime | None = None
) -> ProvenanceFile:
    """Build a provenance document from results and the local environment."""
    revision, dirty = _git_metadata()
    target_hashes: dict[str, str] = {}
    for scenario in results.scenarios:
        task_prompt = scenario.task_prompt or DEFAULT_TASK_PROMPT
        for variant in scenario.variants:
            prompt = (
                f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                f"{variant.document_text}\n\n{task_prompt}"
            )
            target_hashes[f"{scenario.scenario_id}/{variant.variant_id}"] = (
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            )

    judge_hashes: set[str] = set()
    for record in results.records:
        evaluation = record.evaluation
        if evaluation is None or evaluation.judge_meta is None:
            continue
        judge_hashes.add(evaluation.judge_meta.anonymized_prompt_sha256)

    return ProvenanceFile(
        run_id=results.run_id,
        generated_at=generated_at or datetime.now(timezone.utc),
        python_version=platform.python_version(),
        code_revision=revision,
        code_dirty=dirty,
        dependency_versions=_dependency_versions(),
        dataset_sha256=results.run_config.scenarios_sha256,
        models=list(results.run_config.models),
        judge_model=results.run_config.judge_model,
        target_prompt_sha256=target_hashes,
        judge_prompt_sha256=sorted(judge_hashes),
    )


def write_provenance(results_path: Path, results: RunResults) -> Path:
    """Write the provenance sidecar atomically and return its path."""
    target = provenance_path(results_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            build_provenance(results).model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_metadata() -> tuple[str | None, bool | None]:
    revision = _git_output(["rev-parse", "HEAD"])
    if revision is None:
        return None, None
    status = _git_output(["status", "--porcelain"])
    return revision, bool(status)


def _git_output(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
