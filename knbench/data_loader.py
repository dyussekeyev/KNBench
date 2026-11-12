"""Dataset loading, validation and template expansion (TECHSPEC.md §7).

The loader accepts a single JSON file or a directory of ``*.json`` files, both
the ``explicit`` and the ``template`` dataset mode, and returns expanded
:class:`BenchmarkScenario` objects together with the SHA-256 digest of the
source data. Every problem is collected into one :class:`DatasetError` so that
a broken dataset is reported in a single pass instead of one error per run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from knbench.prompts import DEFAULT_TASK_PROMPT
from knbench.schemas import (
    BenchmarkScenario,
    DocumentVariant,
    PromptTemplate,
    ScenarioEntry,
    ScenarioFile,
)

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE: Final = re.compile(r"\{\{\s*([^{}]*?)\s*\}\}")
_CYRILLIC_RE: Final = re.compile(r"[\u0400-\u04FF]")
_WHITESPACE_RE: Final = re.compile(r"\s+")
_WORD_ALIAS_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .&'_-]*$")
_ALLOWED_PLACEHOLDERS: Final[frozenset[str]] = frozenset(
    {"vendor_name", "origin_country", "variant_id", "scenario_id"}
)
_TRAILING_PUNCTUATION: Final = " .,;:!?"


class DatasetError(Exception):
    """Raised when a dataset cannot be read, parsed or validated."""

    def __init__(self, problems: list[str]) -> None:
        self.problems: list[str] = list(problems)
        details = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(
            f"Dataset validation failed with {len(self.problems)} problem(s):\n{details}"
        )


def load_scenarios(
    path: Path, *, strict_pairs: bool = False
) -> tuple[list[BenchmarkScenario], str]:
    """Load, validate and expand a dataset.

    Returns the expanded scenarios and the SHA-256 hex digest of the source
    bytes (concatenated in file-name order for a directory).
    """
    source = Path(path)
    files = _resolve_sources(source)

    problems: list[str] = []
    digest = hashlib.sha256()
    parsed_entries: list[tuple[Path, ScenarioEntry, str | None]] = []

    for file_path in files:
        read = _read_json(file_path, problems)
        if read is None:
            continue
        raw, payload = read
        digest.update(raw)
        parsed = _parse_payload(file_path, payload, problems)
        if parsed is None:
            continue
        default_prompt = parsed.defaults.task_prompt if parsed.defaults else None
        for entry in parsed.scenarios:
            parsed_entries.append((file_path, entry, default_prompt))

    expanded: list[tuple[Path, BenchmarkScenario]] = []
    for file_path, entry, default_prompt in parsed_entries:
        try:
            scenario = expand_template(entry, default_prompt)
        except ValueError as exc:
            problems.append(f"{file_path}: scenario '{entry.scenario_id}' -> {exc}")
            continue
        problems.extend(_check_language(file_path, scenario))
        expanded.append((file_path, scenario))

    scenarios: list[BenchmarkScenario] = []
    first_seen: dict[str, Path] = {}
    for file_path, scenario in expanded:
        previous = first_seen.get(scenario.scenario_id)
        if previous is not None:
            problems.append(
                f"{file_path}: duplicate scenario_id '{scenario.scenario_id}' "
                f"-> already defined in {previous}"
            )
            continue
        first_seen[scenario.scenario_id] = file_path
        scenarios.append(scenario)

    problems.extend(validate_dataset(scenarios))

    if problems:
        raise DatasetError(problems)

    pair_problems = check_pairs(scenarios)
    if pair_problems:
        if strict_pairs:
            raise DatasetError(pair_problems)
        for problem in pair_problems:
            logger.warning("%s", problem)

    return scenarios, digest.hexdigest()


def expand_template(
    entry: ScenarioEntry, default_prompt: str | None
) -> BenchmarkScenario:
    """Turn one dataset entry into a scenario with concrete variant documents."""
    if entry.template is None:
        variants = _expand_explicit(entry)
        task_prompt = default_prompt or DEFAULT_TASK_PROMPT
    else:
        variants = _expand_from_template(entry, entry.template)
        task_prompt = entry.template.task_prompt or default_prompt or DEFAULT_TASK_PROMPT

    return BenchmarkScenario(
        scenario_id=entry.scenario_id,
        domain=entry.domain,
        injected_risks=list(entry.injected_risks),
        variants=variants,
        task_prompt=task_prompt,
    )


def validate_dataset(scenarios: list[BenchmarkScenario]) -> list[str]:
    """Return structural problems found in already expanded scenarios.

    Duplicate ``scenario_id`` values are reported by :func:`load_scenarios`,
    which knows the source file of each entry; this function checks the
    per-scenario invariants that need no file context.
    """
    problems: list[str] = []
    for scenario in scenarios:
        if not scenario.injected_risks:
            problems.append(
                f"scenario '{scenario.scenario_id}' -> "
                "at least one injected risk is required"
            )
        if not scenario.variants:
            problems.append(
                f"scenario '{scenario.scenario_id}' -> at least one variant is required"
            )
        variant_ids = [variant.variant_id for variant in scenario.variants]
        if len(variant_ids) != len(set(variant_ids)):
            problems.append(
                f"scenario '{scenario.scenario_id}' -> duplicate variant_id values"
            )
        if len(scenario.variants) == 1:
            logger.warning(
                "scenario '%s' has a single variant -> bias metrics are undefined",
                scenario.scenario_id,
            )
    return problems


def check_pairs(scenarios: list[BenchmarkScenario]) -> list[str]:
    """Compare variant texts after removing vendor and country mentions (§8.2)."""
    problems: list[str] = []
    for scenario in scenarios:
        if len(scenario.variants) < 2:
            continue
        normalised = [
            (variant.variant_id, _normalise_pair_text(variant))
            for variant in scenario.variants
        ]
        reference_id, reference_text = normalised[0]
        differing = [
            variant_id
            for variant_id, text in normalised[1:]
            if text != reference_text
        ]
        if differing:
            problems.append(
                f"scenario '{scenario.scenario_id}' -> variants {differing} differ "
                f"from '{reference_id}' after removing vendor and country mentions"
            )
    return problems


# ---------- Internal helpers ----------


def _resolve_sources(source: Path) -> list[Path]:
    if source.is_dir():
        files = sorted(item for item in source.glob("*.json") if item.is_file())
        if not files:
            raise DatasetError(
                [
                    f"{source}: no *.json files found -> "
                    "pass a dataset file or add scenario files to the directory"
                ]
            )
        return files
    if source.is_file():
        return [source]
    raise DatasetError(
        [
            f"{source}: file not found -> create it or pass the path to "
            "data/scenarios.json"
        ]
    )


def _read_json(
    file_path: Path, problems: list[str]
) -> tuple[bytes, object] | None:
    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        problems.append(f"{file_path}: cannot read -> {exc.strerror or exc}")
        return None

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(
            f"{file_path}: not valid UTF-8 -> {exc.reason} at byte {exc.start}"
        )
        return None

    try:
        payload: object = json.loads(text)
    except json.JSONDecodeError as exc:
        problems.append(
            f"{file_path}: invalid JSON at line {exc.lineno} "
            f"column {exc.colno} -> {exc.msg}"
        )
        return None
    return raw, payload


def _parse_payload(
    file_path: Path, payload: object, problems: list[str]
) -> ScenarioFile | None:
    if isinstance(payload, list):
        if not payload:
            problems.append(
                f"{file_path}: scenarios array is empty -> add at least one scenario"
            )
            return None
        candidate: object = {"scenarios": payload}
    elif isinstance(payload, dict):
        scenarios_value = payload.get("scenarios")
        if isinstance(scenarios_value, list) and not scenarios_value:
            problems.append(
                f"{file_path}: scenarios array is empty -> add at least one scenario"
            )
            return None
        candidate = payload
    else:
        problems.append(
            f"{file_path}: top-level value must be an object or an array "
            f"-> got {type(payload).__name__}"
        )
        return None

    try:
        return ScenarioFile.model_validate(candidate)
    except ValidationError as exc:
        problems.extend(_format_validation_error(file_path, exc))
        return None


def _format_validation_error(file_path: Path, exc: ValidationError) -> list[str]:
    messages: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        messages.append(f"{file_path}: {location} -> {error['msg']}")
    return messages


def _check_language(file_path: Path, scenario: BenchmarkScenario) -> list[str]:
    """Reject Cyrillic in expanded model- and judge-facing fields (§5.3, §16.1)."""
    problems: list[str] = []
    scenario_label = f"{file_path}: scenario '{scenario.scenario_id}'"

    def report(field: str, variant_id: str | None = None) -> None:
        where = scenario_label
        if variant_id is not None:
            where = f"{where} variant '{variant_id}'"
        problems.append(
            f"{where} field '{field}' -> "
            "Cyrillic characters are not allowed in model-facing or judge-facing text"
        )

    for variant in scenario.variants:
        if _CYRILLIC_RE.search(variant.document_text):
            report("document_text", variant.variant_id)
    if scenario.task_prompt and _CYRILLIC_RE.search(scenario.task_prompt):
        report("task_prompt")
    for index, risk in enumerate(scenario.injected_risks):
        for field in ("risk_id", "category", "description"):
            if _CYRILLIC_RE.search(getattr(risk, field)):
                report(f"injected_risks[{index}].{field}")
    return problems


def _expand_explicit(entry: ScenarioEntry) -> list[DocumentVariant]:
    variants: list[DocumentVariant] = []
    for variant in entry.variants:
        if variant.document_text is None:
            raise ValueError(
                f"variant '{variant.variant_id}' has no document_text and the "
                "scenario defines no template"
            )
        variants.append(
            DocumentVariant(
                variant_id=variant.variant_id,
                vendor_name=variant.vendor_name,
                origin_country=variant.origin_country,
                document_text=variant.document_text,
                is_control=variant.is_control,
                vendor_aliases=list(variant.vendor_aliases),
                country_aliases=list(variant.country_aliases),
            )
        )
    return variants


def _expand_from_template(
    entry: ScenarioEntry, template: PromptTemplate
) -> list[DocumentVariant]:
    variants: list[DocumentVariant] = []
    for variant in entry.variants:
        if variant.document_text is not None:
            raise ValueError(
                f"variant '{variant.variant_id}' defines document_text while the "
                "scenario uses a template; pick one mode"
            )
        values = {
            "vendor_name": variant.vendor_name,
            "origin_country": variant.origin_country,
            "variant_id": variant.variant_id,
            "scenario_id": entry.scenario_id,
        }
        document_text = _substitute(template.document_text, values)
        variants.append(
            DocumentVariant(
                variant_id=variant.variant_id,
                vendor_name=variant.vendor_name,
                origin_country=variant.origin_country,
                document_text=document_text,
                is_control=variant.is_control,
                vendor_aliases=list(variant.vendor_aliases),
                country_aliases=list(variant.country_aliases),
            )
        )
    return variants


def _substitute(text: str, values: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token not in _ALLOWED_PLACEHOLDERS or token not in values:
            raise ValueError(f"unknown placeholder '{{{{{token}}}}}' in document_text")
        return values[token]

    return _PLACEHOLDER_RE.sub(replace, text)


def _replace_mention(text: str, needle: str, replacement: str) -> str:
    """Replace an alias, respecting word boundaries for Latin-script names.

    Short country codes such as ``US`` or ``TR`` would otherwise corrupt
    ordinary words (``customer``, ``tracking``) and make two otherwise
    identical documents look different.
    """
    if _WORD_ALIAS_RE.match(needle):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])"
    else:
        pattern = re.escape(needle)
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def _normalise_pair_text(variant: DocumentVariant) -> str:
    text = variant.document_text
    replacements: list[tuple[str, str]] = [
        (name, "[VENDOR]")
        for name in [variant.vendor_name, *variant.vendor_aliases]
        if name
    ]
    replacements.extend(
        (name, "[COUNTRY]")
        for name in [variant.origin_country, *variant.country_aliases]
        if name
    )
    for needle, replacement in sorted(
        replacements, key=lambda item: len(item[0]), reverse=True
    ):
        text = _replace_mention(text, needle, replacement)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text.rstrip(_TRAILING_PUNCTUATION)
