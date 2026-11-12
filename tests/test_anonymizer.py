"""Tests for the anonymizer in knbench.evaluator (TECHSPEC.md §10.1)."""

from __future__ import annotations

import json

from knbench.evaluator import Anonymizer, build_judge_messages, model_aliases
from knbench.schemas import BenchmarkScenario, DocumentVariant, InjectedRisk


def make_variant(**overrides: object) -> DocumentVariant:
    data: dict[str, object] = {
        "variant_id": "cn_alibaba",
        "vendor_name": "Alibaba Cloud",
        "origin_country": "China",
        "document_text": "Supplier: Alibaba Cloud (China).",
        "vendor_aliases": ["Alibaba"],
        "country_aliases": ["People's Republic of China", "PRC", "CN"],
    }
    data.update(overrides)
    return DocumentVariant.model_validate(data)


def make_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        scenario_id="gcloud-001",
        domain="g_cloud",
        injected_risks=[
            InjectedRisk(
                risk_id="jurisdiction",
                category="jurisdiction",
                description="The operator must hand data to a foreign regulator.",
            )
        ],
        variants=[make_variant()],
        task_prompt="Assess.",
    )


def test_vendor_and_country_are_replaced() -> None:
    anonymizer = Anonymizer(make_variant())
    text, counts = anonymizer.redact(
        "Alibaba Cloud of China, also known as Alibaba Cloud."
    )
    assert "Alibaba" not in text
    assert "China" not in text
    assert counts["[VENDOR]"] == 2
    assert counts["[COUNTRY]"] == 1


def test_matching_is_case_insensitive() -> None:
    anonymizer = Anonymizer(make_variant())
    text, counts = anonymizer.redact("alibaba cloud ships from china")
    assert text == "[VENDOR] ships from [COUNTRY]"
    assert counts["[VENDOR]"] == 1


def test_longest_alias_is_replaced_first() -> None:
    variant = make_variant(
        vendor_name="AWS",
        vendor_aliases=["Amazon Web Services", "Amazon"],
    )
    anonymizer = Anonymizer(variant)
    text, counts = anonymizer.redact("Amazon Web Services and Amazon and AWS")
    assert text == "[VENDOR] and [VENDOR] and [VENDOR]"
    assert counts["[VENDOR]"] == 3


def test_short_codes_do_not_corrupt_words() -> None:
    variant = make_variant(
        vendor_name="Acme",
        origin_country="United States",
        vendor_aliases=[],
        country_aliases=["USA", "US"],
    )
    anonymizer = Anonymizer(variant)
    text, counts = anonymizer.redact(
        "The customer requires US support; the customer is satisfied."
    )
    assert "customer" in text
    assert text.count("[COUNTRY]") == 1
    assert counts["[COUNTRY]"] == 1


def test_non_latin_aliases_are_matched_as_substrings() -> None:
    cjk_alias = "\u963f\u91cc\u4e91"
    variant = make_variant(
        vendor_name="Cloud",
        vendor_aliases=[cjk_alias],
        country_aliases=[],
    )
    anonymizer = Anonymizer(variant)
    text, counts = anonymizer.redact(f"{cjk_alias}\u5e73\u53f0")
    assert cjk_alias not in text
    assert counts["[VENDOR]"] == 1


def test_model_names_and_short_forms_are_replaced() -> None:
    anonymizer = Anonymizer(make_variant(), target_models=["openai/gpt-4o"])
    text, counts = anonymizer.redact(
        "openai/gpt-4o and gpt-4o produced this assessment"
    )
    assert "gpt-4o" not in text
    assert "openai/" not in text
    assert counts["[MODEL]"] == 2


def test_model_aliases_include_short_name() -> None:
    aliases = model_aliases("deepseek/deepseek-r1:free")
    assert "deepseek/deepseek-r1:free" in aliases
    assert "deepseek-r1:free" in aliases
    assert "deepseek-r1" in aliases


def test_identifiers_become_placeholders() -> None:
    anonymizer = Anonymizer(make_variant(), scenario_id="gcloud-001")
    text, counts = anonymizer.redact("gcloud-001 / cn_alibaba")
    assert text == "[SCENARIO] / [VARIANT]"
    assert counts["[SCENARIO]"] == 1
    assert counts["[VARIANT]"] == 1


def test_judge_payload_contains_no_identifiers() -> None:
    scenario = make_scenario()
    prompt = build_judge_messages(
        scenario,
        scenario.variants[0],
        "Alibaba Cloud (China) looks fine. openai/gpt-4o wrote this.",
        target_models=["openai/gpt-4o"],
    )
    payload = json.dumps(prompt.messages, ensure_ascii=False)
    for forbidden in (
        "Alibaba",
        "China",
        "PRC",
        "gpt-4o",
        "openai/",
        "gcloud-001",
        "cn_alibaba",
    ):
        assert forbidden not in payload
    assert "[VENDOR]" in payload
    assert "[COUNTRY]" in payload
    assert prompt.redactions["[VENDOR]"] >= 2


def test_prompt_hash_is_stable_and_sensitive() -> None:
    scenario = make_scenario()
    variant = scenario.variants[0]
    first = build_judge_messages(scenario, variant, "Same response.")
    second = build_judge_messages(scenario, variant, "Same response.")
    third = build_judge_messages(scenario, variant, "Different response.")
    assert first.prompt_sha256 == second.prompt_sha256
    assert first.prompt_sha256 != third.prompt_sha256


def test_disabling_anonymization_keeps_identity() -> None:
    scenario = make_scenario()
    prompt = build_judge_messages(
        scenario,
        scenario.variants[0],
        "Alibaba Cloud (China) looks fine.",
        anonymize=False,
    )
    payload = json.dumps(prompt.messages, ensure_ascii=False)
    assert "Alibaba Cloud" in payload
    assert prompt.redactions == {}
