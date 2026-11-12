"""Report rendering for KNBench (TECHSPEC.md §13).

The Markdown report has eight sections in a fixed order. Undefined numbers are
rendered as an em dash, never as ``0``: a missing measurement and a measured
zero are different findings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from knbench.schemas import (
    EvaluationRecord,
    MetricsFile,
    RunResults,
    ScenarioMetrics,
)

EM_DASH: Final = "\u2014"

RecordKey = tuple[str, str, str]


def render_metrics_json(metrics: MetricsFile) -> str:
    """Serialise the metrics document of §6.4."""
    return metrics.model_dump_json(indent=2)


def has_undefined_metrics(metrics: MetricsFile) -> bool:
    """Return True when a mandatory metric is undefined (§12.3, EXECPLAN S4).

    ``partial`` scenarios keep their values and do not trigger strict mode;
    ``control_delta`` is optional and therefore ignored.
    """
    for row in metrics.by_scenario:
        if (
            row.risk_blindness_rate is None
            or row.bias_delta is None
            or row.decision_flip is None
        ):
            return True
    for model in metrics.by_model:
        if model.mean_risk_blindness_rate is None or model.mean_bias_delta is None:
            return True
    return False


def render_markdown(results: RunResults, metrics: MetricsFile) -> str:
    """Render the eight report sections of §13."""
    lines: list[str] = ["# KNBench report", ""]
    lines.append(
        "> **Research instrument.** KNBench measures how a model reacts to the same "
        "supplier facts under different national labels. This report is not a security "
        "opinion or a legal assessment of any supplier."
    )
    lines.append("")
    lines.extend(_header_section(results))
    lines.extend(_coverage_section(results, metrics))
    lines.extend(_model_summary_section(metrics))
    lines.extend(_scenario_section(metrics))
    lines.extend(_score_matrix_section(results))
    lines.extend(_risk_section(results))
    lines.extend(_errors_section(results))
    lines.extend(_limitations_section(results, metrics))
    return "\n".join(lines).rstrip() + "\n"


# ---------- Sections ----------


def _header_section(results: RunResults) -> list[str]:
    config = results.run_config
    stamp = results.finished_at or results.started_at
    lines = [
        "## 1. Run header",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Run ID | `{results.run_id}` |",
        f"| Date | {_format_date(stamp)} |",
        f"| Models | {_code_list(config.models)} |",
        f"| Judge model | `{config.judge_model}` |",
        f"| Version | `{config.app_version}` |",
        f"| Dataset SHA-256 | `{config.scenarios_sha256}` |",
        f"| Anonymization | {'enabled' if config.anonymize else 'disabled'} |",
        f"| Strict pairs | {'on' if config.strict_pairs else 'off'} |",
        "",
    ]
    if not config.anonymize:
        lines.extend(
            [
                "> **Warning:** anonymization was disabled with `--no-anonymize`. "
                "The judge saw vendor and country names, so these results are not "
                "comparable with a normal run.",
                "",
            ]
        )
    return lines


def _coverage_section(results: RunResults, metrics: MetricsFile) -> list[str]:
    coverage = metrics.coverage
    variants = sum(len(scenario.variants) for scenario in results.scenarios)
    lines = [
        "## 2. Coverage",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Records total | {coverage.records_total} |",
        f"| Records judged successfully | {coverage.records_ok} |",
        f"| Failed records | {coverage.records_failed} |",
        f"| Skipped records | {coverage.records_skipped} |",
        f"| Scenarios | {len(results.scenarios)} |",
        f"| Variants in dataset | {variants} |",
        f"| Models | {len(results.run_config.models)} |",
        f"| Sample size (judged records) | {coverage.records_ok} |",
        "",
    ]
    if coverage.records_skipped > 0:
        lines.extend(
            [
                "> **Warning:** the run used `--skip-judge`. Target responses were "
                "collected, but metrics are unavailable until the judge runs.",
                "",
            ]
        )
    if coverage.records_ok == 0:
        lines.extend(
            [
                "> **Warning:** no record has a judge evaluation, so every metric "
                "below is undefined.",
                "",
            ]
        )
    return lines


def _model_summary_section(metrics: MetricsFile) -> list[str]:
    lines = [
        "## 3. Model summary",
        "",
        "| Model | Mean risk blindness rate | Mean bias delta | Max bias delta | Decision flip scenarios |",
        "|-------|--------------------------|-----------------|----------------|-------------------------|",
    ]
    for model in metrics.by_model:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{model.model_name}`",
                    _format_rate(model.mean_risk_blindness_rate),
                    _format_score(model.mean_bias_delta),
                    _format_score(model.max_bias_delta),
                    str(model.decision_flip_scenarios),
                ]
            )
            + " |"
        )
    if not metrics.by_model:
        lines.append(f"| {EM_DASH} | {EM_DASH} | {EM_DASH} | {EM_DASH} | {EM_DASH} |")
    lines.append("")

    sparse = [model.model_name for model in metrics.by_model if model.low_coverage]
    if sparse:
        lines.extend(
            [
                "Low coverage: " + ", ".join(f"`{name}`" for name in sparse) + ". "
                "Means use only the scenarios with defined values.",
                "",
            ]
        )
    return lines


def _scenario_section(metrics: MetricsFile) -> list[str]:
    lines = [
        "## 4. Scenario x model",
        "",
        "| Scenario | Model | Variants evaluated / expected | Risk blindness rate | Bias delta | Decision flip | Control delta |",
        "|----------|-------|-------------------------------|---------------------|------------|---------------|---------------|",
    ]
    for row in metrics.by_scenario:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.scenario_id}`",
                    f"`{row.model_name}`",
                    _variants_cell(row),
                    _format_rate(row.risk_blindness_rate),
                    _format_score(row.bias_delta),
                    _format_flip(row.decision_flip),
                    _format_score(row.control_delta),
                ]
            )
            + " |"
        )
    if not metrics.by_scenario:
        lines.append("| " + " | ".join([EM_DASH] * 7) + " |")
    lines.append("")
    return lines


def _score_matrix_section(results: RunResults) -> list[str]:
    lines = ["## 5. Risk score matrix", ""]
    lookup = _record_lookup(results)
    models = results.run_config.models

    for scenario in results.scenarios:
        lines.append(f"### `{scenario.scenario_id}`")
        lines.append("")
        lines.append("| Variant | " + " | ".join(f"`{model}`" for model in models) + " |")
        lines.append("|---------|" + "|".join("------" for _ in models) + "|")
        for variant in scenario.variants:
            cells = [_format_score(_score_of(lookup, scenario.scenario_id, variant.variant_id, model)) for model in models]
            label = f"`{variant.variant_id}`"
            if variant.is_control:
                label += " (control)"
            lines.append("| " + label + " | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _risk_section(results: RunResults) -> list[str]:
    lines = ["## 6. Detected and missed risks", ""]
    lookup = _record_lookup(results)

    for scenario in results.scenarios:
        lines.append(f"### `{scenario.scenario_id}`")
        lines.append("")
        lines.append("| Variant | Model | Detected | Missed |")
        lines.append("|---------|-------|----------|--------|")
        rows = 0
        injected = {risk.risk_id for risk in scenario.injected_risks}
        for variant in scenario.variants:
            for model in results.run_config.models:
                record = lookup.get((scenario.scenario_id, variant.variant_id, model))
                if record is None or record.evaluation is None:
                    continue
                detected = [risk for risk in record.evaluation.detected_risks if risk in injected]
                missed = sorted(injected - set(detected))
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{variant.variant_id}`",
                            f"`{model}`",
                            _id_list(detected),
                            _id_list(missed),
                        ]
                    )
                    + " |"
                )
                rows += 1
        if rows == 0:
            lines.append(f"| {EM_DASH} | {EM_DASH} | {EM_DASH} | {EM_DASH} |")
        lines.append("")
    return lines


def _errors_section(results: RunResults) -> list[str]:
    lines = [
        "## 7. Errors",
        "",
        "| Scenario | Variant | Model | Stage | Error type | Attempts | Message |",
        "|----------|---------|-------|-------|------------|----------|---------|",
    ]
    current = _current_keys(results)
    rows = 0
    for record in results.records:
        if (record.scenario_id, record.variant_id, record.model_name) not in current:
            continue
        if not record.errors:
            continue
        for error in record.errors:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{record.scenario_id}`",
                        f"`{record.variant_id}`",
                        f"`{record.model_name}`",
                        error.stage,
                        f"`{error.error_type}`",
                        str(error.attempts),
                        _cell(error.message),
                    ]
                )
                + " |"
            )
            rows += 1
    if rows == 0:
        lines.append("No failed records.")
    lines.append("")
    return lines


def _limitations_section(results: RunResults, metrics: MetricsFile) -> list[str]:
    judged = metrics.coverage.records_ok
    scenarios = len(results.scenarios)
    lines = [
        "## 8. Limitations",
        "",
        f"- **Sample size.** {judged} judged record(s) across {scenarios} scenario(s). "
        "KNBench reports descriptive differences only; it makes no claim of statistical "
        "significance.",
        "- **Anonymization boundaries.** Direct vendor names, country names, aliases and "
        "model names are replaced before judging, but indirect hints (regulation style, "
        "contract clauses, document structure) may still identify a supplier.",
        "- **Prompt sensitivity.** Results depend on the task prompt, the system prompt, "
        "the sampling parameters and the judge model. A single run is one measurement, "
        "not a stable property of a model.",
        "- **Synthetic documents.** The dataset uses synthetic procurement texts. Real "
        "classified, internal or personal material requires explicit approval and a "
        "data-processing assessment.",
        "- **No legal force.** This report is a research instrument, not a security "
        "opinion or a legal assessment of a supplier.",
        "",
    ]
    if not results.run_config.anonymize:
        lines.append(
            "- **Non-comparable run.** Anonymization was disabled, so scores reflect "
            "vendor identity and must not be compared with anonymized runs."
        )
        lines.append("")
    return lines


# ---------- Formatting helpers ----------


def _record_lookup(results: RunResults) -> dict[RecordKey, EvaluationRecord]:
    current = _current_keys(results)
    return {
        (record.scenario_id, record.variant_id, record.model_name): record
        for record in results.records
        if (record.scenario_id, record.variant_id, record.model_name) in current
        and record.status == "ok"
        and record.evaluation is not None
    }


def _current_keys(results: RunResults) -> set[RecordKey]:
    return {
        (scenario.scenario_id, variant.variant_id, model)
        for scenario in results.scenarios
        for variant in scenario.variants
        for model in results.run_config.models
    }


def _score_of(
    lookup: dict[RecordKey, EvaluationRecord],
    scenario_id: str,
    variant_id: str,
    model: str,
) -> float | None:
    record = lookup.get((scenario_id, variant_id, model))
    if record is None or record.evaluation is None:
        return None
    return record.evaluation.risk_score


def _format_rate(value: float | None) -> str:
    return EM_DASH if value is None else f"{value:.3f}"


def _format_score(value: float | None) -> str:
    return EM_DASH if value is None else f"{value:.1f}"


def _format_flip(value: bool | None) -> str:
    if value is None:
        return EM_DASH
    return "yes" if value else "no"


def _variants_cell(row: ScenarioMetrics) -> str:
    cell = f"{row.variants_evaluated} / {row.variants_expected}"
    return f"{cell} (partial)" if row.partial else cell


def _id_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else EM_DASH


def _code_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def _format_date(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()
