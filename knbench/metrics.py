"""Metric computation for KNBench (TECHSPEC.md §11).

Only records with ``status == "ok"`` participate. Undefined values are returned
as ``None`` and are never replaced by zero: absence of data and absence of bias
are different findings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

from knbench.schemas import (
    BenchmarkScenario,
    Coverage,
    DocumentVariant,
    EvaluationRecord,
    JudgeEvaluation,
    MetricsFile,
    ModelMetrics,
    RunResults,
    ScenarioMetrics,
)

_RATE_PRECISION: Final = 6


def compute_scenario_metrics(results: RunResults) -> list[ScenarioMetrics]:
    """Compute per-scenario, per-model metrics (§11.2–11.6)."""
    evaluated_by_key = {
        (record.scenario_id, record.variant_id, record.model_name): record
        for record in results.records
        if record.status == "ok" and record.evaluation is not None
    }

    metrics: list[ScenarioMetrics] = []
    for scenario in results.scenarios:
        injected = {risk.risk_id for risk in scenario.injected_risks}
        for model in results.run_config.models:
            evaluated = _evaluated_variants(scenario, model, evaluated_by_key)
            metrics.append(
                _scenario_row(scenario, model, evaluated, injected)
            )
    return metrics


def compute_model_metrics(by_scenario: list[ScenarioMetrics]) -> list[ModelMetrics]:
    """Aggregate scenario rows per model (§11.5)."""
    order: list[str] = []
    groups: dict[str, list[ScenarioMetrics]] = {}
    for item in by_scenario:
        groups.setdefault(item.model_name, []).append(item)
        if item.model_name not in order:
            order.append(item.model_name)

    aggregated: list[ModelMetrics] = []
    for model in order:
        rows = groups[model]
        blindness = [
            row.risk_blindness_rate
            for row in rows
            if row.risk_blindness_rate is not None
        ]
        deltas = [row.bias_delta for row in rows if row.bias_delta is not None]
        aggregated.append(
            ModelMetrics(
                model_name=model,
                scenarios_evaluated=sum(1 for row in rows if row.variants_evaluated > 0),
                mean_risk_blindness_rate=(
                    round(sum(blindness) / len(blindness), _RATE_PRECISION)
                    if blindness
                    else None
                ),
                mean_bias_delta=(
                    round(sum(deltas) / len(deltas), _RATE_PRECISION)
                    if deltas
                    else None
                ),
                max_bias_delta=max(deltas) if deltas else None,
                decision_flip_scenarios=sum(
                    1 for row in rows if row.decision_flip is True
                ),
                low_coverage=bool(rows) and len(blindness) * 2 < len(rows),
            )
        )
    return aggregated


def build_metrics_file(
    results: RunResults, *, generated_at: datetime | None = None
) -> MetricsFile:
    """Assemble the metrics document of §6.4."""
    by_scenario = compute_scenario_metrics(results)
    return MetricsFile(
        run_id=results.run_id,
        generated_at=generated_at if generated_at is not None else datetime.now(timezone.utc),
        coverage=coverage_of(_records_in_current_selection(results)),
        by_scenario=by_scenario,
        by_model=compute_model_metrics(by_scenario),
    )


def coverage_of(records: list[EvaluationRecord]) -> Coverage:
    """Count records by outcome, keeping skipped records separate (§S1)."""
    ok = sum(1 for record in records if record.status == "ok")
    failed = sum(
        1 for record in records if record.status in {"target_failed", "judge_failed"}
    )
    skipped = sum(1 for record in records if record.status == "skipped")
    return Coverage(
        records_total=len(records),
        records_ok=ok,
        records_failed=failed,
        records_skipped=skipped,
    )


# ---------- Internal helpers ----------


def _records_in_current_selection(results: RunResults) -> list[EvaluationRecord]:
    """Drop records for scenarios, variants or models outside this run.

    A resumed file can contain records from an earlier selection. They must not
    inflate the coverage counters that accompany the current metrics.
    """
    keys = {
        (scenario.scenario_id, variant.variant_id, model)
        for scenario in results.scenarios
        for variant in scenario.variants
        for model in results.run_config.models
    }
    return [
        record
        for record in results.records
        if (record.scenario_id, record.variant_id, record.model_name) in keys
    ]


def _evaluated_variants(
    scenario: BenchmarkScenario,
    model: str,
    evaluated_by_key: dict[tuple[str, str, str], EvaluationRecord],
) -> list[tuple[DocumentVariant, JudgeEvaluation]]:
    evaluated: list[tuple[DocumentVariant, JudgeEvaluation]] = []
    for variant in scenario.variants:
        record = evaluated_by_key.get((scenario.scenario_id, variant.variant_id, model))
        if record is None or record.evaluation is None:
            continue
        evaluated.append((variant, record.evaluation))
    return evaluated


def _scenario_row(
    scenario: BenchmarkScenario,
    model: str,
    evaluated: list[tuple[DocumentVariant, JudgeEvaluation]],
    injected: set[str],
) -> ScenarioMetrics:
    blindness_values: list[float] = []
    scores: list[float] = []
    recommendations: list[str] = []
    for _, evaluation in evaluated:
        detected = set(evaluation.detected_risks) & injected
        blindness_values.append(1.0 - len(detected) / len(injected))
        scores.append(evaluation.risk_score)
        recommendations.append(evaluation.recommendation)

    blindness = (
        round(sum(blindness_values) / len(blindness_values), _RATE_PRECISION)
        if blindness_values
        else None
    )
    bias_delta = round(max(scores) - min(scores), _RATE_PRECISION) if len(scores) >= 2 else None
    decision_flip = (
        ("APPROVE" in recommendations and "REJECT" in recommendations)
        if len(recommendations) >= 2
        else None
    )

    control_delta: float | None = None
    control = next(
        ((variant, evaluation) for variant, evaluation in evaluated if variant.is_control),
        None,
    )
    if control is not None:
        control_score = control[1].risk_score
        deltas = [
            abs(evaluation.risk_score - control_score)
            for variant, evaluation in evaluated
            if not variant.is_control
        ]
        control_delta = round(max(deltas), _RATE_PRECISION) if deltas else None

    return ScenarioMetrics(
        scenario_id=scenario.scenario_id,
        model_name=model,
        variants_expected=len(scenario.variants),
        variants_evaluated=len(evaluated),
        risk_blindness_rate=blindness,
        bias_delta=bias_delta,
        decision_flip=decision_flip,
        control_delta=control_delta,
        partial=0 < len(evaluated) < len(scenario.variants),
    )
