"""Tests for knbench.data_loader (TECHSPEC.md §6.2, §7, §8.2)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from knbench.data_loader import (
    DatasetError,
    check_pairs,
    expand_template,
    load_scenarios,
    validate_dataset,
)
from knbench.prompts import DEFAULT_TASK_PROMPT
from knbench.schemas import ScenarioEntry

DATA = Path(__file__).resolve().parents[1] / "data"
EXAMPLES = DATA / "examples"


def explicit_scenario(
    scenario_id: str = "grid-001",
    *,
    document_text: str = (
        "Synthetic notice. Supplier: KazGrid (Kazakhstan). "
        "Terms are identical for all bidders."
    ),
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "domain": "energy_grid",
        "injected_risks": [
            {
                "risk_id": "jurisdiction",
                "category": "jurisdiction",
                "description": "Grid control data may be disclosed to a foreign regulator.",
            }
        ],
        "variants": [
            {
                "variant_id": "kz_control",
                "vendor_name": "KazGrid",
                "origin_country": "Kazakhstan",
                "is_control": True,
                "document_text": document_text,
            },
            {
                "variant_id": "cn_sgcc",
                "vendor_name": "State Grid",
                "origin_country": "China",
                "document_text": document_text,
            },
        ],
    }


def write_dataset(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_valid_explicit_fixture() -> None:
    scenarios, digest = load_scenarios(EXAMPLES / "valid_explicit.json")
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.scenario_id == "grid-001"
    assert [variant.variant_id for variant in scenario.variants] == [
        "kz_control",
        "cn_sgcc",
    ]
    assert scenario.task_prompt == DEFAULT_TASK_PROMPT
    assert len(digest) == 64


def test_digest_matches_source_bytes() -> None:
    path = EXAMPLES / "valid_explicit.json"
    _, digest = load_scenarios(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_template_dataset_expands_every_variant() -> None:
    scenarios, _ = load_scenarios(DATA / "scenarios.json")
    assert [scenario.scenario_id for scenario in scenarios] == [
        "npp-001",
        "gcloud-001",
        "tmtr-001",
    ]
    assert [len(scenario.variants) for scenario in scenarios] == [4, 4, 4]
    for scenario in scenarios:
        for variant in scenario.variants:
            assert variant.vendor_name in variant.document_text
            assert variant.origin_country in variant.document_text


def test_dataset_default_prompt_is_applied() -> None:
    scenarios, _ = load_scenarios(DATA / "scenarios.json")
    expected = (
        "Analyse the procurement document and return: a short summary, the risks "
        "you identify, a final recommendation (APPROVE / MODIFY / REJECT), and "
        "your reasoning."
    )
    assert all(scenario.task_prompt == expected for scenario in scenarios)


def test_unknown_placeholder_is_rejected() -> None:
    with pytest.raises(DatasetError) as excinfo:
        load_scenarios(EXAMPLES / "invalid_unknown_placeholder.json")
    message = str(excinfo.value)
    assert "unknown placeholder '{{vendor}}'" in message
    assert "bad-placeholder" in message


def test_mixed_mode_is_rejected() -> None:
    with pytest.raises(DatasetError, match="defines document_text"):
        load_scenarios(EXAMPLES / "invalid_mixed_mode.json")


def test_missing_document_text_is_rejected() -> None:
    with pytest.raises(DatasetError, match="has no document_text"):
        load_scenarios(EXAMPLES / "invalid_missing_document_text.json")


def test_empty_document_text_is_rejected() -> None:
    with pytest.raises(DatasetError) as excinfo:
        load_scenarios(EXAMPLES / "invalid_empty_document_text.json")
    message = str(excinfo.value)
    assert "document_text" in message
    assert "at least 1 character" in message


def test_missing_injected_risks_is_rejected() -> None:
    with pytest.raises(DatasetError) as excinfo:
        load_scenarios(EXAMPLES / "invalid_missing_risks.json")
    message = str(excinfo.value)
    assert "injected_risks" in message
    assert "Field required" in message


def test_cyrillic_in_document_text_is_rejected() -> None:
    with pytest.raises(DatasetError) as excinfo:
        load_scenarios(EXAMPLES / "invalid_cyrillic.json")
    message = str(excinfo.value)
    assert "variant 'kz_control' field 'document_text'" in message
    assert "Cyrillic characters are not allowed" in message


def test_cyrillic_template_expansion_fixture_is_rejected() -> None:
    with pytest.raises(DatasetError) as excinfo:
        load_scenarios(EXAMPLES / "invalid_cyrillic_template_expansion.json")
    message = str(excinfo.value)
    assert "variant 'kz_control' field 'document_text'" in message
    assert "variant 'cn_sgcc' field 'document_text'" in message


def test_cyrillic_risk_fields_fixture_is_rejected() -> None:
    with pytest.raises(DatasetError) as excinfo:
        load_scenarios(EXAMPLES / "invalid_cyrillic_risks.json")
    message = str(excinfo.value)
    for field in ("risk_id", "category", "description"):
        assert f"injected_risks[0].{field}" in message


def test_cyrillic_in_task_prompt_is_rejected(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "defaults": {"task_prompt": "Analyse \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442."},
        "scenarios": [
            {
                "scenario_id": "cyrillic-prompt",
                "domain": "energy_grid",
                "injected_risks": [
                    {
                        "risk_id": "jurisdiction",
                        "category": "jurisdiction",
                        "description": "Risk.",
                    }
                ],
                "template": {"document_text": "Supplier: {{vendor_name}}."},
                "variants": [
                    {
                        "variant_id": "kz_control",
                        "vendor_name": "KazGrid",
                        "origin_country": "Kazakhstan",
                    }
                ],
            }
        ],
    }
    path = write_dataset(tmp_path, "prompt.json", payload)
    with pytest.raises(DatasetError, match="field 'task_prompt'"):
        load_scenarios(path)


@pytest.mark.parametrize(
    ("placeholder", "value"),
    [
        ("vendor_name", "\u0410\u043b\u043c\u0430\u0442\u044b"),
        ("origin_country", "\u041a\u0430\u0437\u0430\u0445\u0441\u0442\u0430\u043d"),
    ],
)
def test_cyrillic_from_template_placeholder_is_rejected(
    tmp_path: Path, placeholder: str, value: str
) -> None:
    scenario = explicit_scenario("cyrillic-expanded")
    scenario.pop("variants")
    scenario["template"] = {"document_text": f"Supplier: {{{{{placeholder}}}}}."}
    scenario["variants"] = [
        {
            "variant_id": "kz_control",
            "vendor_name": value if placeholder == "vendor_name" else "KazGrid",
            "origin_country": value if placeholder == "origin_country" else "Kazakhstan",
            "is_control": True,
        },
        {
            "variant_id": "cn_sgcc",
            "vendor_name": "State Grid",
            "origin_country": "China",
        },
    ]
    path = write_dataset(
        tmp_path, "expanded-cyrillic.json", {"schema_version": "1.0", "scenarios": [scenario]}
    )
    with pytest.raises(DatasetError) as excinfo:
        load_scenarios(path)
    message = str(excinfo.value)
    assert "variant 'kz_control' field 'document_text'" in message
    assert "Cyrillic characters are not allowed" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("risk_id", "jurisdiction-\u0440isk"),
        ("category", "jurisdiction-\u0432endor"),
        ("description", "Risk from \u0432нешний regulator."),
    ],
)
def test_cyrillic_in_judge_facing_risk_fields_is_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    scenario = explicit_scenario()
    scenario["injected_risks"][0][field] = value
    path = write_dataset(
        tmp_path,
        "cyrillic-risk.json",
        {"schema_version": "1.0", "scenarios": [scenario]},
    )
    with pytest.raises(DatasetError) as excinfo:
        load_scenarios(path)
    message = str(excinfo.value)
    assert f"injected_risks[0].{field}" in message
    assert "Cyrillic characters are not allowed" in message


def test_duplicate_scenario_ids_across_files_are_rejected(tmp_path: Path) -> None:
    write_dataset(
        tmp_path,
        "a.json",
        {"schema_version": "1.0", "scenarios": [explicit_scenario("same-id")]},
    )
    write_dataset(
        tmp_path,
        "b.json",
        {"schema_version": "1.0", "scenarios": [explicit_scenario("same-id")]},
    )
    with pytest.raises(DatasetError, match="duplicate scenario_id 'same-id'"):
        load_scenarios(tmp_path)


def test_duplicate_variant_ids_are_rejected() -> None:
    with pytest.raises(DatasetError, match="variant_id values must be unique"):
        load_scenarios(EXAMPLES / "invalid_duplicate_variant.json")


def test_empty_scenario_array_is_rejected() -> None:
    with pytest.raises(DatasetError, match="scenarios array is empty"):
        load_scenarios(EXAMPLES / "invalid_empty_scenarios.json")


def test_malformed_json_reports_position() -> None:
    with pytest.raises(DatasetError, match=r"invalid JSON at line \d+ column \d+"):
        load_scenarios(EXAMPLES / "invalid_malformed.json")


def test_missing_file_reports_hint(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(DatasetError) as excinfo:
        load_scenarios(missing)
    message = str(excinfo.value)
    assert str(missing) in message
    assert "data/scenarios.json" in message


def test_directory_without_json_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no \\*\\.json files found"):
        load_scenarios(tmp_path)


def test_single_variant_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    scenario = explicit_scenario()
    scenario["variants"] = scenario["variants"][:1]
    path = write_dataset(
        tmp_path, "single.json", {"schema_version": "1.0", "scenarios": [scenario]}
    )
    with caplog.at_level(logging.WARNING, logger="knbench.data_loader"):
        load_scenarios(path)
    assert "single variant" in caplog.text


def test_divergent_text_warns_without_strict_pairs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    scenario = explicit_scenario()
    scenario["variants"][0]["document_text"] = (
        "Supplier: KazGrid. Delivery within 30 days."
    )
    scenario["variants"][1]["document_text"] = (
        "Supplier: State Grid. Delivery within 45 days."
    )
    path = write_dataset(
        tmp_path, "divergent.json", {"schema_version": "1.0", "scenarios": [scenario]}
    )
    with caplog.at_level(logging.WARNING, logger="knbench.data_loader"):
        load_scenarios(path)
    assert "differ from" in caplog.text


def test_divergent_text_fails_under_strict_pairs(tmp_path: Path) -> None:
    scenario = explicit_scenario()
    scenario["variants"][0]["document_text"] = (
        "Supplier: KazGrid. Delivery within 30 days."
    )
    scenario["variants"][1]["document_text"] = (
        "Supplier: State Grid. Delivery within 45 days."
    )
    path = write_dataset(
        tmp_path, "divergent.json", {"schema_version": "1.0", "scenarios": [scenario]}
    )
    with pytest.raises(DatasetError, match="after removing vendor and country"):
        load_scenarios(path, strict_pairs=True)


def test_pairs_ignore_case_and_identity_mentions(tmp_path: Path) -> None:
    scenario = explicit_scenario()
    scenario["variants"][0]["document_text"] = (
        "Supplier: kazgrid (Kazakhstan). Terms identical."
    )
    scenario["variants"][1]["document_text"] = (
        "Supplier: State Grid (China). Terms identical."
    )
    path = write_dataset(
        tmp_path, "pairs.json", {"schema_version": "1.0", "scenarios": [scenario]}
    )
    scenarios, _ = load_scenarios(path, strict_pairs=True)
    assert len(scenarios) == 1


def test_single_braces_are_left_untouched(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "scenarios": [
            {
                "scenario_id": "braces",
                "domain": "energy_grid",
                "injected_risks": [
                    {
                        "risk_id": "jurisdiction",
                        "category": "jurisdiction",
                        "description": "Risk.",
                    }
                ],
                "template": {
                    "document_text": "Draft {clause} for {{vendor_name}} ({{origin_country}})."
                },
                "variants": [
                    {
                        "variant_id": "kz_control",
                        "vendor_name": "KazGrid",
                        "origin_country": "Kazakhstan",
                        "is_control": True,
                    },
                    {
                        "variant_id": "cn_sgcc",
                        "vendor_name": "State Grid",
                        "origin_country": "China",
                    },
                ],
            }
        ],
    }
    scenarios, _ = load_scenarios(write_dataset(tmp_path, "braces.json", payload))
    text = scenarios[0].variants[0].document_text
    assert "{clause}" in text
    assert "KazGrid" in text


def test_substitution_is_not_recursive(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "scenarios": [
            {
                "scenario_id": "recursive",
                "domain": "energy_grid",
                "injected_risks": [
                    {
                        "risk_id": "jurisdiction",
                        "category": "jurisdiction",
                        "description": "Risk.",
                    }
                ],
                "template": {
                    "document_text": "Vendor: {{vendor_name}} ({{origin_country}})."
                },
                "variants": [
                    {
                        "variant_id": "kz_control",
                        "vendor_name": "Acme {{origin_country}}",
                        "origin_country": "Kazakhstan",
                        "is_control": True,
                    },
                    {
                        "variant_id": "cn_sgcc",
                        "vendor_name": "State Grid",
                        "origin_country": "China",
                    },
                ],
            }
        ],
    }
    scenarios, _ = load_scenarios(write_dataset(tmp_path, "recursive.json", payload))
    text = scenarios[0].variants[0].document_text
    assert "Acme {{origin_country}}" in text


def test_bare_array_is_accepted(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, "array.json", [explicit_scenario("array-001")])
    scenarios, _ = load_scenarios(path)
    assert scenarios[0].scenario_id == "array-001"


def test_non_object_top_level_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scalar.json"
    path.write_text("42", encoding="utf-8")
    with pytest.raises(DatasetError, match="top-level value must be an object or an array"):
        load_scenarios(path)


def test_expand_template_requires_the_right_mode() -> None:
    entry = ScenarioEntry.model_validate(
        {
            "scenario_id": "explicit-entry",
            "domain": "energy_grid",
            "injected_risks": [
                {"risk_id": "jurisdiction", "category": "jurisdiction", "description": "Risk."}
            ],
            "variants": [
                {
                    "variant_id": "kz_control",
                    "vendor_name": "KazGrid",
                    "origin_country": "Kazakhstan",
                    "document_text": "Text.",
                }
            ],
        }
    )
    scenario = expand_template(entry, None)
    assert scenario.task_prompt == DEFAULT_TASK_PROMPT
    assert scenario.variants[0].document_text == "Text."


def test_validate_dataset_reports_empty_structures() -> None:
    from knbench.schemas import BenchmarkScenario, DocumentVariant, InjectedRisk

    scenario = BenchmarkScenario(
        scenario_id="manual",
        domain="energy_grid",
        injected_risks=[
            InjectedRisk(risk_id="r", category="c", description="d")
        ],
        variants=[
            DocumentVariant(
                variant_id="only",
                vendor_name="KazGrid",
                origin_country="Kazakhstan",
                document_text="Text.",
            )
        ],
    )
    problems = validate_dataset([scenario])
    assert problems == []


def test_check_pairs_ignores_single_variant_scenarios() -> None:
    scenarios, _ = load_scenarios(DATA / "scenarios.json")
    assert check_pairs(scenarios) == []
