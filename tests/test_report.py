"""Tests for knbench.report (TECHSPEC.md §13)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from knbench.metrics import build_metrics_file
from knbench.report import (
    EM_DASH,
    has_undefined_metrics,
    render_markdown,
    render_metrics_json,
)
from knbench.schemas import (
    BenchmarkScenario,
    DocumentVariant,
    EvaluationRecord,
    InjectedRisk,
    JudgeEvaluation,
    MetricsFile,
    RecordError,
    RunConfig,
    RunResults,
    TargetModelResponse,
)

SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "report_snapshot.md"

SECTION_HEADINGS = [
    "## 1. Run header",
    "## 2. Coverage",
    "## 3. Model summary",
    "## 4. Scenario x model",
    "## 5. Risk score matrix",
    "## 6. Detected and missed risks",
    "## 7. Errors",
    "## 8. Limitations",
]


def make_scenario(*, variant_count: int = 3) -> BenchmarkScenario:
    variants = [
        DocumentVariant(
            variant_id=f"v{index}",
            vendor_name=f"Vendor{index}",
            origin_country=f"Country{index}",
            document_text=f"Document {index}.",
            is_control=index == 0,
        )
        for index in range(variant_count)
    ]
    return BenchmarkScenario(
        scenario_id="gcloud-001",
        domain="test",
        injected_risks=[
            InjectedRisk(risk_id="r1", category="c1", description="First risk."),
            InjectedRisk(risk_id="r2", category="c2", description="Second risk."),
        ],
        variants=variants,
        task_prompt="Assess.",
    )


def make_config(*, anonymize: bool = True, strict_pairs: bool = False) -> RunConfig:
    return RunConfig(
        models=["m1"],
        judge_model="judge/model",
        concurrency=1,
        max_retries=1,
        timeout_s=5.0,
        judge_votes=1,
        anonymize=anonymize,
        scenarios_path="data/scenarios.json",
        scenarios_sha256="a" * 64,
        app_version="0.1.0",
        strict_pairs=strict_pairs,
    )


def ok_record(
    variant_id: str,
    *,
    detected: list[str] | None = None,
    score: float = 50.0,
    recommendation: str = "MODIFY",
) -> EvaluationRecord:
    return EvaluationRecord(
        scenario_id="gcloud-001",
        variant_id=variant_id,
        model_name="m1",
        status="ok",
        target_response=TargetModelResponse(
            scenario_id="gcloud-001",
            variant_id=variant_id,
            model_name="m1",
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


def make_results(
    records: list[EvaluationRecord],
    *,
    variant_count: int = 3,
    anonymize: bool = True,
    strict_pairs: bool = False,
) -> RunResults:
    return RunResults(
        run_id="run-123",
        started_at=datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 11, 10, 10, 5, tzinfo=timezone.utc),
        run_config=make_config(anonymize=anonymize, strict_pairs=strict_pairs),
        scenarios=[make_scenario(variant_count=variant_count)],
        records=records,
    )


def render(results: RunResults) -> str:
    return render_markdown(results, build_metrics_file(results))


def test_report_has_the_eight_sections_in_order() -> None:
    text = render(make_results([ok_record("v0")]))
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == SECTION_HEADINGS


def test_header_contains_run_metadata_and_disclaimer() -> None:
    results = make_results([ok_record("v0")], strict_pairs=True)
    text = render(results)
    assert "`run-123`" in text
    assert "`m1`" in text
    assert "`judge/model`" in text
    assert "`0.1.0`" in text
    assert "`" + "a" * 64 + "`" in text
    assert "2026-11-10 10:05 UTC" in text
    assert "Strict pairs | on" in text
    assert "not a security opinion or a legal assessment" in text


def test_undefined_values_render_as_em_dash() -> None:
    text = render(make_results([]))
    assert EM_DASH in text
    assert "| 0.000 |" not in text
    assert "no record has a judge evaluation" in text


def test_rates_and_scores_use_the_documented_precision() -> None:
    results = make_results(
        [
            ok_record("v0", detected=["r1"], score=60, recommendation="APPROVE"),
            ok_record("v1", detected=[], score=40, recommendation="REJECT"),
        ]
    )
    text = render(results)
    assert "0.750" in text
    assert "20.0" in text
    assert "0.75 |" not in text


def test_matrix_shows_every_variant_with_control_marker() -> None:
    text = render(make_results([ok_record("v0", score=62.5)]))
    assert "`v0` (control)" in text
    assert "62.5" in text
    assert EM_DASH in text


def test_detected_and_missed_risks_are_listed() -> None:
    text = render(make_results([ok_record("v1", detected=[])]))
    section = text.split("## 6. Detected and missed risks")[1].split("## 7.")[0]
    assert "`r1`" in section
    assert "`r2`" in section


def test_errors_section_reports_type_attempts_and_escapes_pipes() -> None:
    failed = EvaluationRecord(
        scenario_id="gcloud-001",
        variant_id="v1",
        model_name="m1",
        status="target_failed",
        errors=[
            RecordError(
                stage="target",
                error_type="http_500",
                message="upstream | broken",
                attempts=5,
            )
        ],
    )
    text = render(make_results([failed]))
    section = text.split("## 7. Errors")[1]
    assert "`http_500`" in section
    assert "| 5 |" in section
    assert "upstream \\| broken" in section


def test_failed_and_partial_runs_do_not_break_rendering() -> None:
    results = make_results(
        [
            ok_record("v0", score=10),
            ok_record("v1", score=20),
        ]
    )
    text = render(results)
    assert "2 / 3 (partial)" in text


def test_skip_judge_warns_and_has_no_metrics() -> None:
    skipped = EvaluationRecord(
        scenario_id="gcloud-001",
        variant_id="v0",
        model_name="m1",
        status="skipped",
        target_response=TargetModelResponse(
            scenario_id="gcloud-001",
            variant_id="v0",
            model_name="m1",
            raw_response="Response.",
            execution_time_ms=5,
        ),
    )
    text = render(make_results([skipped]))
    assert "the run used `--skip-judge`" in text
    assert "metrics are unavailable" in text


def test_disabled_anonymization_is_flagged() -> None:
    text = render(make_results([ok_record("v0")], anonymize=False))
    assert "anonymization was disabled" in text
    assert "not\ncomparable" not in text
    assert "Non-comparable run" in text


def test_metrics_json_round_trips_as_metrics_file() -> None:
    metrics = build_metrics_file(make_results([ok_record("v0")]))
    raw = render_metrics_json(metrics)
    parsed = MetricsFile.model_validate_json(raw)
    assert parsed.run_id == "run-123"
    assert parsed.by_scenario[0].scenario_id == "gcloud-001"


def test_has_undefined_metrics_detects_null_and_partial() -> None:
    complete = build_metrics_file(
        make_results(
            [
                ok_record("v0", score=60, recommendation="APPROVE"),
                ok_record("v1", score=40, recommendation="MODIFY"),
                ok_record("v2", score=20, recommendation="REJECT"),
            ]
        )
    )
    assert has_undefined_metrics(complete) is False

    partial = build_metrics_file(make_results([ok_record("v0")]))
    assert has_undefined_metrics(partial) is True


def test_report_matches_the_snapshot() -> None:
    results = make_results(
        [
            ok_record("v0", detected=["r1"], score=60, recommendation="APPROVE"),
            ok_record("v1", detected=[], score=20, recommendation="REJECT"),
        ],
        variant_count=2,
    )
    assert render(results) == SNAPSHOT.read_text(encoding="utf-8")


def test_partial_scenario_alone_does_not_trip_strict_mode() -> None:
    results = make_results(
        [
            ok_record("v0", score=60, recommendation="APPROVE"),
            ok_record("v1", score=40, recommendation="REJECT"),
        ]
    )
    metrics = build_metrics_file(results)
    assert metrics.by_scenario[0].partial is True
    assert has_undefined_metrics(metrics) is False


def test_report_ignores_records_outside_the_current_selection() -> None:
    stale = EvaluationRecord(
        scenario_id="stale",
        variant_id="v0",
        model_name="m1",
        status="target_failed",
        errors=[
            RecordError(
                stage="target",
                error_type="stale_error",
                message="from an older dataset",
                attempts=1,
            )
        ],
    )
    results = make_results([ok_record("v0")])
    results = results.model_copy(update={"records": [*results.records, stale]})

    metrics = build_metrics_file(results)
    text = render_markdown(results, metrics)
    assert metrics.coverage.records_total == 1
    assert "| Records total | 1 |" in text
    assert "stale_error" not in text
