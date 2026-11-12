"""Tests for knbench.metrics (TECHSPEC.md §11)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from knbench.data_loader import load_scenarios
from knbench.metrics import (
    build_metrics_file,
    compute_model_metrics,
    compute_scenario_metrics,
    coverage_of,
)
from knbench.schemas import (
    BenchmarkScenario,
    DocumentVariant,
    EvaluationRecord,
    InjectedRisk,
    JudgeEvaluation,
    RunConfig,
    RunResults,
    TargetModelResponse,
)

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "examples"
    / "valid_explicit.json"
)


def make_scenario(
    scenario_id: str = "gcloud-001",
    *,
    with_control: bool = True,
    variant_count: int = 3,
) -> BenchmarkScenario:
    variants = []
    for index in range(variant_count):
        variants.append(
            DocumentVariant(
                variant_id=f"v{index}",
                vendor_name=f"Vendor{index}",
                origin_country=f"Country{index}",
                document_text=f"Document {index}.",
                is_control=with_control and index == 0,
            )
        )
    return BenchmarkScenario(
        scenario_id=scenario_id,
        domain="test",
        injected_risks=[
            InjectedRisk(risk_id="r1", category="c1", description="First risk."),
            InjectedRisk(risk_id="r2", category="c2", description="Second risk."),
        ],
        variants=variants,
        task_prompt="Assess.",
    )


def ok_record(
    variant_id: str,
    model: str,
    *,
    scenario_id: str = "gcloud-001",
    detected: list[str] | None = None,
    score: float = 50.0,
    recommendation: str = "MODIFY",
) -> EvaluationRecord:
    return EvaluationRecord(
        scenario_id=scenario_id,
        variant_id=variant_id,
        model_name=model,
        status="ok",
        target_response=TargetModelResponse(
            scenario_id=scenario_id,
            variant_id=variant_id,
            model_name=model,
            raw_response="Assessment.",
            execution_time_ms=10,
        ),
        evaluation=JudgeEvaluation(
            detected_risks=detected or [],
            risk_score=score,
            recommendation=recommendation,
            judge_reasoning="Reasoning.",
        ),
    )


def make_config(models: list[str] | None = None) -> RunConfig:
    return RunConfig(
        models=models or ["m1"],
        judge_model="judge/model",
        concurrency=2,
        max_retries=1,
        timeout_s=5.0,
        judge_votes=1,
        anonymize=True,
        scenarios_path="data/scenarios.json",
        scenarios_sha256="0" * 64,
        app_version="0.1.0",
    )


def make_results(
    scenarios: list[BenchmarkScenario],
    records: list[EvaluationRecord],
    models: list[str] | None = None,
) -> RunResults:
    return RunResults(
        run_id="run-1",
        started_at=datetime.now(timezone.utc),
        run_config=make_config(models),
        scenarios=scenarios,
        records=records,
    )


def test_blindness_delta_and_flip() -> None:
    results = make_results(
        [make_scenario()],
        [
            ok_record("v0", "m1", detected=["r1"], score=60, recommendation="APPROVE"),
            ok_record("v1", "m1", detected=[], score=40, recommendation="MODIFY"),
            ok_record("v2", "m1", detected=["r1", "r2"], score=20, recommendation="REJECT"),
        ],
    )
    row = compute_scenario_metrics(results)[0]
    assert row.variants_expected == 3
    assert row.variants_evaluated == 3
    assert row.risk_blindness_rate == pytest.approx(0.5)
    assert row.bias_delta == pytest.approx(40.0)
    assert row.decision_flip is True
    assert row.partial is False


def test_single_variant_leaves_delta_and_flip_undefined() -> None:
    results = make_results(
        [make_scenario()],
        [ok_record("v0", "m1", detected=["r1"], score=60)],
    )
    row = compute_scenario_metrics(results)[0]
    assert row.risk_blindness_rate == pytest.approx(0.5)
    assert row.bias_delta is None
    assert row.decision_flip is None
    assert row.partial is True


def test_no_evaluated_variants_yields_nulls() -> None:
    results = make_results([make_scenario()], [])
    row = compute_scenario_metrics(results)[0]
    assert row.variants_evaluated == 0
    assert row.risk_blindness_rate is None
    assert row.bias_delta is None
    assert row.decision_flip is None
    assert row.partial is False


def test_control_delta_uses_the_control_variant() -> None:
    results = make_results(
        [make_scenario()],
        [
            ok_record("v0", "m1", score=70),
            ok_record("v1", "m1", score=40),
            ok_record("v2", "m1", score=20),
        ],
    )
    assert compute_scenario_metrics(results)[0].control_delta == pytest.approx(50.0)


def test_control_delta_is_null_when_control_is_not_evaluated() -> None:
    results = make_results(
        [make_scenario()],
        [ok_record("v1", "m1", score=40)],
    )
    assert compute_scenario_metrics(results)[0].control_delta is None


def test_control_delta_is_null_without_a_control_variant() -> None:
    results = make_results(
        [make_scenario(with_control=False)],
        [ok_record("v0", "m1", score=70), ok_record("v1", "m1", score=40)],
    )
    assert compute_scenario_metrics(results)[0].control_delta is None


def test_detected_ids_are_intersected_with_injected_risks() -> None:
    results = make_results(
        [make_scenario()],
        [
            ok_record(
                "v0",
                "m1",
                detected=["r1", "r1", "unknown_risk"],
                score=50,
            )
        ],
    )
    row = compute_scenario_metrics(results)[0]
    assert row.risk_blindness_rate == pytest.approx(0.5)


def test_flip_requires_approve_and_reject() -> None:
    results = make_results(
        [make_scenario()],
        [
            ok_record("v0", "m1", recommendation="APPROVE"),
            ok_record("v1", "m1", recommendation="MODIFY"),
        ],
    )
    assert compute_scenario_metrics(results)[0].decision_flip is False


def test_model_aggregation_uses_defined_values_only() -> None:
    scenario_one = make_scenario("s1")
    scenario_two = make_scenario("s2")
    records = [
        ok_record("v0", "m1", scenario_id="s1", detected=["r1"], score=60, recommendation="APPROVE"),
        ok_record("v1", "m1", scenario_id="s1", detected=[], score=40, recommendation="REJECT"),
        ok_record("v0", "m1", scenario_id="s2", detected=["r1", "r2"], score=30, recommendation="MODIFY"),
    ]
    results = make_results([scenario_one, scenario_two], records, models=["m1"])
    by_scenario = compute_scenario_metrics(results)
    by_model = compute_model_metrics(by_scenario)

    assert len(by_model) == 1
    model = by_model[0]
    assert model.scenarios_evaluated == 2
    assert model.mean_risk_blindness_rate == pytest.approx(0.375)
    assert model.mean_bias_delta == pytest.approx(20.0)
    assert model.max_bias_delta == pytest.approx(20.0)
    assert model.decision_flip_scenarios == 1
    assert model.low_coverage is False


def test_low_coverage_flags_sparse_models() -> None:
    scenarios = [make_scenario(f"s{index}") for index in range(3)]
    records = [
        ok_record("v0", "m1", scenario_id="s0", detected=["r1"], score=60),
        ok_record("v1", "m1", scenario_id="s0", detected=["r1"], score=40),
    ]
    results = make_results(scenarios, records, models=["m1"])
    model = compute_model_metrics(compute_scenario_metrics(results))[0]
    assert model.low_coverage is True


def test_coverage_counts_every_status() -> None:
    records = [
        ok_record("v0", "m1"),
        EvaluationRecord(
            scenario_id="s",
            variant_id="v1",
            model_name="m1",
            status="target_failed",
        ),
        EvaluationRecord(
            scenario_id="s",
            variant_id="v2",
            model_name="m1",
            status="judge_failed",
        ),
        EvaluationRecord(
            scenario_id="s",
            variant_id="v3",
            model_name="m1",
            status="skipped",
        ),
    ]
    coverage = coverage_of(records)
    assert coverage.records_total == 4
    assert coverage.records_ok == 1
    assert coverage.records_failed == 2
    assert coverage.records_skipped == 1


def test_build_metrics_file_is_valid_and_uses_the_run_id() -> None:
    stamp = datetime(2026, 11, 10, 12, 40, tzinfo=timezone.utc)
    results = make_results(
        [make_scenario()],
        [
            ok_record("v0", "m1", detected=["r1"], score=60),
            ok_record("v1", "m1", detected=[], score=40),
        ],
    )
    metrics = build_metrics_file(results, generated_at=stamp)
    assert metrics.run_id == "run-1"
    assert metrics.generated_at == stamp
    assert metrics.coverage.records_total == 2
    assert metrics.by_model[0].model_name == "m1"


def test_reference_values_on_the_fixed_dataset_fixture() -> None:
    scenarios, digest = load_scenarios(FIXTURE)
    assert len(scenarios) == 1
    assert len(digest) == 64

    records = [
        ok_record(
            "kz_control",
            "ref/model",
            scenario_id="grid-001",
            detected=["jurisdiction", "remote_access"],
            score=20,
            recommendation="APPROVE",
        ),
        ok_record(
            "cn_sgcc",
            "ref/model",
            scenario_id="grid-001",
            detected=[],
            score=70,
            recommendation="REJECT",
        ),
    ]
    metrics = build_metrics_file(make_results(scenarios, records, models=["ref/model"]))
    row = metrics.by_scenario[0]

    assert row.variants_expected == 2
    assert row.variants_evaluated == 2
    assert row.risk_blindness_rate == pytest.approx(0.5)
    assert row.bias_delta == pytest.approx(50.0)
    assert row.decision_flip is True
    assert row.control_delta == pytest.approx(50.0)
    assert row.partial is False

    model = metrics.by_model[0]
    assert model.mean_risk_blindness_rate == pytest.approx(0.5)
    assert model.max_bias_delta == pytest.approx(50.0)
    assert metrics.coverage.records_ok == 2


def test_skipped_records_do_not_contribute_metrics() -> None:
    skipped = EvaluationRecord(
        scenario_id="gcloud-001",
        variant_id="v0",
        model_name="m1",
        status="skipped",
        target_response=TargetModelResponse(
            scenario_id="gcloud-001",
            variant_id="v0",
            model_name="m1",
            raw_response="Not judged yet.",
            execution_time_ms=5,
        ),
    )
    results = make_results([make_scenario()], [skipped])
    row = compute_scenario_metrics(results)[0]
    assert row.variants_evaluated == 0
    assert row.risk_blindness_rate is None


def test_coverage_ignores_records_outside_the_current_selection() -> None:
    selected = make_scenario("s1")
    removed = make_scenario("s2")
    records = [
        ok_record("v0", "m1", scenario_id="s1"),
        ok_record("v0", "m2", scenario_id="s1"),
        ok_record("v0", "m1", scenario_id="s2"),
    ]
    results = make_results([selected], records, models=["m1"])

    metrics = build_metrics_file(results)
    assert metrics.coverage.records_total == 1
    assert metrics.coverage.records_ok == 1
    assert [row.scenario_id for row in metrics.by_scenario] == ["s1"]
    assert removed.scenario_id not in {row.scenario_id for row in metrics.by_scenario}
