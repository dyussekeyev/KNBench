"""Tests for the provenance sidecar (owner decision, option C)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from knbench.provenance import (
    ProvenanceFile,
    build_provenance,
    provenance_path,
    write_provenance,
)
from knbench.schemas import (
    BenchmarkScenario,
    DocumentVariant,
    InjectedRisk,
    RunConfig,
    RunResults,
)


def _results() -> RunResults:
    scenario = BenchmarkScenario(
        scenario_id="test-1",
        domain="g_cloud",
        injected_risks=[
            InjectedRisk(
                risk_id="jurisdiction",
                category="jurisdiction",
                description="A foreign regulator may demand data.",
            )
        ],
        variants=[
            DocumentVariant(
                variant_id="v1",
                vendor_name="Vendor",
                origin_country="Country",
                document_text="Vendor stores data locally.",
            )
        ],
        task_prompt="Assess.",
    )
    return RunResults(
        run_id="run-provenance",
        started_at=datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 11, 11, 10, 1, tzinfo=timezone.utc),
        run_config=RunConfig(
            models=["openai/gpt-6.1-sol"],
            judge_model="anthropic/claude-opus-5.5",
            concurrency=1,
            max_retries=1,
            timeout_s=5.0,
            judge_votes=1,
            anonymize=True,
            scenarios_path="dataset.json",
            scenarios_sha256="a" * 64,
            app_version="0.1.0",
        ),
        scenarios=[scenario],
        records=[],
    )


def test_provenance_path_is_a_sidecar() -> None:
    assert provenance_path(Path("output/raw_results.json")) == Path(
        "output/raw_results.provenance.json"
    )


def test_build_provenance_records_hashes_without_prompts() -> None:
    provenance = build_provenance(_results())

    assert provenance.run_id == "run-provenance"
    assert provenance.dataset_sha256 == "a" * 64
    assert provenance.models == ["openai/gpt-6.1-sol"]
    assert provenance.judge_model == "anthropic/claude-opus-5.5"
    assert len(provenance.target_prompt_sha256["test-1/v1"]) == 64
    assert provenance.dependency_versions["pydantic"] != "not-installed"
    assert provenance.judge_prompt_sha256 == []


def test_write_provenance_is_atomic_and_contains_no_content(tmp_path: Path) -> None:
    results_path = tmp_path / "raw_results.json"
    results_path.write_text("{}", encoding="utf-8")

    sidecar = write_provenance(results_path, _results())

    assert sidecar == tmp_path / "raw_results.provenance.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    ProvenanceFile.model_validate(payload)
    text = sidecar.read_text(encoding="utf-8")
    assert "Vendor stores data locally." not in text
    assert "raw_response" not in text
    assert "sk-or-v1-" not in text
    assert list(tmp_path.glob("*.tmp")) == []
