"""Tests for knbench.schemas (TECHSPEC.md §5, §5.1, §5.2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from knbench.schemas import (
    SCHEMA_VERSION,
    BenchmarkScenario,
    DocumentVariant,
    EvaluationRecord,
    InjectedRisk,
    JudgeEvaluation,
    MetricsFile,
    RunConfig,
    RunResults,
    ScenarioEntry,
    ScenarioFile,
    TargetModelResponse,
    VariantEntry,
)


def make_risk(risk_id: str = "jurisdiction") -> InjectedRisk:
    return InjectedRisk(
        risk_id=risk_id,
        category="jurisdiction",
        description="The supplier is subject to a foreign jurisdiction.",
    )


def make_variant(
    variant_id: str = "kz_control",
    *,
    is_control: bool = False,
) -> DocumentVariant:
    return DocumentVariant(
        variant_id=variant_id,
        vendor_name="KazAtomService",
        origin_country="Kazakhstan",
        document_text="Procurement of core monitoring systems.",
        is_control=is_control,
    )


def make_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        scenario_id="npp-001",
        domain="nuclear_energy",
        injected_risks=[make_risk("jurisdiction"), make_risk("vendor_lockin")],
        variants=[make_variant("kz_control", is_control=True), make_variant("cn_cnnc")],
    )


def test_schema_version_is_pinned() -> None:
    assert SCHEMA_VERSION == "1.0"


def test_variant_id_rejects_uppercase() -> None:
    with pytest.raises(ValidationError):
        DocumentVariant(
            variant_id="CN_CNNC",
            vendor_name="CNNC",
            origin_country="China",
            document_text="Text.",
        )


def test_scenario_rejects_duplicate_risk_ids() -> None:
    with pytest.raises(ValidationError, match="risk_id values must be unique"):
        BenchmarkScenario(
            scenario_id="npp-001",
            domain="nuclear_energy",
            injected_risks=[make_risk("jurisdiction"), make_risk("jurisdiction")],
            variants=[make_variant()],
        )


def test_scenario_rejects_duplicate_variant_ids() -> None:
    with pytest.raises(ValidationError, match="variant_id values must be unique"):
        BenchmarkScenario(
            scenario_id="npp-001",
            domain="nuclear_energy",
            injected_risks=[make_risk()],
            variants=[make_variant("dup"), make_variant("dup")],
        )


def test_scenario_allows_only_one_control_variant() -> None:
    with pytest.raises(ValidationError, match="at most one control variant"):
        BenchmarkScenario(
            scenario_id="npp-001",
            domain="nuclear_energy",
            injected_risks=[make_risk()],
            variants=[
                make_variant("kz_a", is_control=True),
                make_variant("kz_b", is_control=True),
            ],
        )


def test_scenario_requires_at_least_one_risk() -> None:
    with pytest.raises(ValidationError):
        BenchmarkScenario(
            scenario_id="npp-001",
            domain="nuclear_energy",
            injected_risks=[],
            variants=[make_variant()],
        )


def test_document_text_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        DocumentVariant(
            variant_id="kz_control",
            vendor_name="KazAtomService",
            origin_country="Kazakhstan",
            document_text="",
        )


@pytest.mark.parametrize("score", [-0.1, 100.1, 101])
def test_risk_score_outside_range_is_rejected(score: float) -> None:
    with pytest.raises(ValidationError):
        JudgeEvaluation(
            detected_risks=["jurisdiction"],
            risk_score=score,
            recommendation="MODIFY",
            judge_reasoning="Reasoning.",
        )


@pytest.mark.parametrize("score", [0, 0.0, 100, 100.0])
def test_risk_score_bounds_are_inclusive(score: float) -> None:
    evaluation = JudgeEvaluation(
        detected_risks=[],
        risk_score=score,
        recommendation="APPROVE",
        judge_reasoning="Reasoning.",
    )
    assert evaluation.risk_score == score


def test_recommendation_is_normalised_to_upper_case() -> None:
    evaluation = JudgeEvaluation(
        detected_risks=[],
        risk_score=50,
        recommendation=" modify ",
        judge_reasoning="Reasoning.",
    )
    assert evaluation.recommendation == "MODIFY"


def test_recommendation_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        JudgeEvaluation(
            detected_risks=[],
            risk_score=50,
            recommendation="PASS",
            judge_reasoning="Reasoning.",
        )


def test_judge_reasoning_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        JudgeEvaluation(
            detected_risks=[],
            risk_score=50,
            recommendation="APPROVE",
            judge_reasoning="",
        )


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        InjectedRisk(
            risk_id="jurisdiction",
            category="jurisdiction",
            description="Description.",
            severity="high",
        )


def test_scenario_file_rejects_other_schema_versions() -> None:
    with pytest.raises(ValidationError):
        ScenarioFile.model_validate(
            {
                "schema_version": "2.0",
                "scenarios": [
                    {
                        "scenario_id": "npp-001",
                        "domain": "nuclear_energy",
                        "injected_risks": [
                            {
                                "risk_id": "jurisdiction",
                                "category": "jurisdiction",
                                "description": "Description.",
                            }
                        ],
                        "variants": [
                            {
                                "variant_id": "kz_control",
                                "vendor_name": "KazAtomService",
                                "origin_country": "Kazakhstan",
                                "document_text": "Text.",
                            }
                        ],
                    }
                ],
            }
        )


def test_variant_entry_allows_omitted_document_text() -> None:
    entry = VariantEntry(
        variant_id="cn_cnnc",
        vendor_name="CNNC",
        origin_country="China",
    )
    assert entry.document_text is None


def test_scenario_entry_accepts_template_mode() -> None:
    entry = ScenarioEntry(
        scenario_id="gcloud-001",
        domain="g_cloud",
        injected_risks=[make_risk()],
        template={
            "document_text": "Supplier: {{vendor_name}} ({{origin_country}}).",
        },
        variants=[
            {
                "variant_id": "cn_alibaba",
                "vendor_name": "Alibaba Cloud",
                "origin_country": "China",
            }
        ],
    )
    assert entry.template is not None
    assert entry.variants[0].document_text is None


def test_run_config_enforces_concurrency_range() -> None:
    base = {
        "models": ["openai/gpt-4o"],
        "judge_model": "anthropic/claude-3.5-sonnet",
        "max_retries": 4,
        "timeout_s": 120.0,
        "judge_votes": 1,
        "anonymize": True,
        "scenarios_path": "data/scenarios.json",
        "scenarios_sha256": "a" * 64,
        "app_version": "0.1.0",
    }
    for concurrency in (0, 33):
        with pytest.raises(ValidationError):
            RunConfig(**base, concurrency=concurrency)
    assert RunConfig(**base, concurrency=32).concurrency == 32


def test_target_response_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        TargetModelResponse(
            scenario_id="npp-001",
            variant_id="kz_control",
            model_name="openai/gpt-4o",
            raw_response="Response.",
            execution_time_ms=-1,
        )


def test_record_status_is_limited_to_four_values() -> None:
    with pytest.raises(ValidationError):
        EvaluationRecord(
            scenario_id="npp-001",
            variant_id="kz_control",
            model_name="openai/gpt-4o",
            status="done",
        )


def test_run_results_accepts_a_minimal_document() -> None:
    results = RunResults.model_validate(
        {
            "run_id": "20261110T123000Z-7f3a1c",
            "started_at": "2026-11-10T12:30:00Z",
            "run_config": {
                "models": ["openai/gpt-4o"],
                "judge_model": "anthropic/claude-3.5-sonnet",
                "concurrency": 5,
                "max_retries": 4,
                "timeout_s": 120.0,
                "judge_votes": 1,
                "anonymize": True,
                "scenarios_path": "data/scenarios.json",
                "scenarios_sha256": "b" * 64,
                "app_version": "0.1.0",
            },
            "scenarios": [make_scenario().model_dump()],
            "records": [],
        }
    )
    assert results.schema_version == "1.0"
    assert results.run_config.resumed_from is None
    assert results.records == []


def test_metrics_file_parses_the_spec_example() -> None:
    metrics = MetricsFile.model_validate(
        {
            "schema_version": "1.0",
            "run_id": "20261110T123000Z-7f3a1c",
            "generated_at": "2026-11-10T12:40:00Z",
            "coverage": {"records_total": 24, "records_ok": 22, "records_failed": 2},
            "by_scenario": [
                {
                    "scenario_id": "gcloud-001",
                    "model_name": "openai/gpt-4o",
                    "variants_expected": 4,
                    "variants_evaluated": 4,
                    "risk_blindness_rate": 0.375,
                    "bias_delta": 31.0,
                    "decision_flip": True,
                    "control_delta": 21.0,
                    "partial": False,
                }
            ],
            "by_model": [
                {
                    "model_name": "openai/gpt-4o",
                    "scenarios_evaluated": 2,
                    "mean_risk_blindness_rate": 0.31,
                    "mean_bias_delta": 27.5,
                    "max_bias_delta": 31.0,
                    "decision_flip_scenarios": 1,
                    "low_coverage": False,
                }
            ],
        }
    )
    assert metrics.by_scenario[0].risk_blindness_rate == 0.375
    assert metrics.by_model[0].max_bias_delta == 31.0
