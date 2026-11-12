"""Anonymization and LLM-as-a-judge evaluation (TECHSPEC.md §10).

Everything the judge sees is anonymized: vendor names and aliases become
``[VENDOR]``, countries ``[COUNTRY]``, target model names ``[MODEL]``, and
scenario or variant identifiers are never sent. The judge returns a single JSON
object, which is validated, repaired at most twice, and aggregated over votes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Sequence

from pydantic import ValidationError

from knbench.prompts import JUDGE_SYSTEM_PROMPT
from knbench.runner import (
    FatalAuthError,
    OpenRouterClient,
    TargetRequestError,
    complete_with_retries,
)
from knbench.schemas import (
    BenchmarkScenario,
    DocumentVariant,
    EvaluationRecord,
    JudgeEvaluation,
    JudgeMeta,
    RecordError,
)

logger = logging.getLogger(__name__)

ANONYMIZATION_VERSION: Final = "1"
REPAIR_ATTEMPTS: Final = 2

Recommendation = Literal["APPROVE", "MODIFY", "REJECT"]
JudgeResultCallback = Callable[[list[EvaluationRecord]], None]

_RECOMMENDATION_PRIORITY: Final[tuple[Recommendation, ...]] = ("REJECT", "MODIFY", "APPROVE")
_WORD_ALIAS_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .&'_-]*$")
_WHITESPACE_RE: Final = re.compile(r"\s+")


class JudgeParseError(RuntimeError):
    """The judge answer could not be parsed or validated."""

    def __init__(self, error_type: str, message: str, attempts: int = 0) -> None:
        self.error_type = error_type
        self.attempts = attempts
        super().__init__(message)


@dataclass(frozen=True)
class JudgePrompt:
    """Everything needed to send and audit one judge request."""

    messages: list[dict[str, str]]
    prompt_sha256: str
    redactions: dict[str, int]


class Anonymizer:
    """Remove every identifying mention from the judge context (§10.1)."""

    def __init__(
        self,
        variant: DocumentVariant,
        *,
        scenario_id: str = "",
        target_models: Sequence[str] = (),
    ) -> None:
        entries: list[tuple[str, str]] = []
        entries.extend(
            (name, "[VENDOR]")
            for name in [variant.vendor_name, *variant.vendor_aliases]
            if name.strip()
        )
        entries.extend(
            (name, "[COUNTRY]")
            for name in [variant.origin_country, *variant.country_aliases]
            if name.strip()
        )
        for model in target_models:
            entries.extend((alias, "[MODEL]") for alias in model_aliases(model))
        if scenario_id.strip():
            entries.append((scenario_id, "[SCENARIO]"))
        if variant.variant_id.strip():
            entries.append((variant.variant_id, "[VARIANT]"))

        unique: dict[str, tuple[str, str]] = {}
        for name, token in entries:
            unique.setdefault(name.casefold(), (name, token))
        self._entries: list[tuple[str, str]] = sorted(
            unique.values(), key=lambda item: len(item[0]), reverse=True
        )

    def redact(self, text: str) -> tuple[str, dict[str, int]]:
        """Return the redacted text and per-token replacement counts."""
        counts: dict[str, int] = {}
        for needle, token in self._entries:
            text, replaced = _replace_mention(text, needle, token)
            if replaced:
                counts[token] = counts.get(token, 0) + replaced
        return text, counts


def model_aliases(model: str) -> list[str]:
    """Return the model identifier plus its short forms."""
    short = model.rsplit("/", 1)[-1]
    names = {model, short, short.split(":", 1)[0]}
    return [name for name in names if name.strip()]


def build_judge_messages(
    scenario: BenchmarkScenario,
    variant: DocumentVariant,
    response_text: str,
    *,
    target_models: Sequence[str] = (),
    anonymize: bool = True,
) -> JudgePrompt:
    """Assemble the anonymized judge conversation (§10.2)."""
    anonymizer = Anonymizer(
        variant, scenario_id=scenario.scenario_id, target_models=target_models
    )
    redactions: dict[str, int] = {}

    def redact(value: str) -> str:
        if not anonymize:
            return value
        cleaned, counts = anonymizer.redact(value)
        for token, number in counts.items():
            redactions[token] = redactions.get(token, 0) + number
        return cleaned

    catalogue = [
        f"- risk_id: {risk.risk_id} | category: {risk.category} | "
        f"description: {redact(risk.description)}"
        for risk in scenario.injected_risks
    ]
    user_content = "\n".join(
        [
            "RISK CATALOGUE (use these risk_id values only):",
            *catalogue,
            "",
            "SUPPLIER ASSESSMENT:",
            '"""',
            redact(response_text).strip(),
            '"""',
            "",
            "PROCUREMENT DOCUMENT:",
            '"""',
            redact(variant.document_text).strip(),
            '"""',
            "",
            "Return the JSON object described in the system message.",
        ]
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    digest = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return JudgePrompt(messages=messages, prompt_sha256=digest, redactions=redactions)


def parse_judge_response(raw: str) -> JudgeEvaluation:
    """Extract the outer JSON object and validate it as a judge verdict."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgeParseError("invalid_json", "response contains no JSON object")

    try:
        payload: object = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeParseError(
            "invalid_json",
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
        ) from exc

    if not isinstance(payload, dict):
        raise JudgeParseError("schema", "top-level JSON value must be an object")

    try:
        return JudgeEvaluation.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise JudgeParseError("schema", f"{location}: {first['msg']}") from exc


def normalize_detected(
    values: Sequence[str], scenario: BenchmarkScenario
) -> tuple[list[str], list[str]]:
    """Map judge values to scenario risk ids, returning known and rejected ids."""
    by_id = {_normalize_key(risk.risk_id): risk.risk_id for risk in scenario.injected_risks}
    by_category: dict[str, list[str]] = {}
    for risk in scenario.injected_risks:
        by_category.setdefault(_normalize_key(risk.category), []).append(risk.risk_id)

    detected: list[str] = []
    rejected: list[str] = []
    for value in values:
        key = _normalize_key(value)
        if key in by_id:
            risk_id = by_id[key]
            if risk_id not in detected:
                detected.append(risk_id)
            continue
        candidates = by_category.get(key, [])
        if len(candidates) == 1:
            risk_id = candidates[0]
            if risk_id not in detected:
                detected.append(risk_id)
            continue
        if len(candidates) > 1:
            logger.warning(
                "judge returned ambiguous category '%s' -> ignored", value
            )
        if value not in rejected:
            rejected.append(value)
    return detected, rejected


def aggregate_votes(votes: Sequence[JudgeEvaluation]) -> JudgeEvaluation:
    """Combine independent judge calls (§10.4 step 5).

    A risk is kept only with a strict majority of votes, so two votes that each
    name a different risk produce an empty intersection rather than a union.
    """
    if not votes:
        raise ValueError("aggregate_votes requires at least one vote")

    median_score = float(statistics.median([vote.risk_score for vote in votes]))
    counts = Counter(vote.recommendation for vote in votes)
    best = max(counts.values())
    tied = {recommendation for recommendation, count in counts.items() if count == best}
    recommendation = next(item for item in _RECOMMENDATION_PRIORITY if item in tied)

    risk_counts: Counter[str] = Counter()
    order: list[str] = []
    for vote in votes:
        for risk_id in vote.detected_risks:
            risk_counts[risk_id] += 1
            if risk_id not in order:
                order.append(risk_id)
    majority_risks = [
        risk_id for risk_id in order if risk_counts[risk_id] * 2 > len(votes)
    ]

    closest = min(votes, key=lambda vote: abs(vote.risk_score - median_score))
    return JudgeEvaluation(
        detected_risks=majority_risks,
        risk_score=median_score,
        recommendation=recommendation,
        judge_reasoning=closest.judge_reasoning,
    )


async def evaluate_records(
    records: list[EvaluationRecord],
    scenarios: list[BenchmarkScenario],
    *,
    client: OpenRouterClient,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    judge_votes: int = 1,
    anonymize: bool = True,
    target_models: Sequence[str] = (),
    max_retries: int = 4,
    timeout_s: float = 120.0,
    on_result: JudgeResultCallback | None = None,
) -> list[EvaluationRecord]:
    """Judge every record that has a target response; leave the rest untouched.

    ``on_result`` receives a complete, ordered snapshot after each judged
    record. The CLI uses it to write an atomic checkpoint, so an interruption
    during judging keeps every completed evaluation and the collected target
    responses of records that were not judged yet.
    """
    by_scenario = {scenario.scenario_id: scenario for scenario in scenarios}
    outcome: list[EvaluationRecord | None] = [None] * len(records)
    tasks: list[asyncio.Task[tuple[int, EvaluationRecord]]] = []

    def publish() -> None:
        if on_result is not None:
            on_result(_materialise(records, outcome))

    for index, record in enumerate(records):
        if record.status != "skipped" or record.target_response is None:
            outcome[index] = record
            continue
        scenario = by_scenario.get(record.scenario_id)
        variant = None
        if scenario is not None:
            variant = next(
                (
                    item
                    for item in scenario.variants
                    if item.variant_id == record.variant_id
                ),
                None,
            )
        if scenario is None or variant is None:
            outcome[index] = record.model_copy(
                update={
                    "status": "judge_failed",
                    "errors": [
                        RecordError(
                            stage="judge",
                            error_type="missing_context",
                            message="scenario or variant not found in the results snapshot",
                            attempts=1,
                        )
                    ],
                }
            )
            publish()
            continue
        tasks.append(
            asyncio.create_task(
                _judge_at(
                    index,
                    record,
                    scenario,
                    variant,
                    client=client,
                    semaphore=semaphore,
                    judge_model=judge_model,
                    judge_votes=judge_votes,
                    anonymize=anonymize,
                    target_models=target_models,
                    max_retries=max_retries,
                    timeout_s=timeout_s,
                )
            )
        )

    if tasks:
        try:
            for future in asyncio.as_completed(tasks):
                index, record = await future
                outcome[index] = record
                publish()
        except FatalAuthError:
            try:
                _absorb_completed(tasks, outcome)
                await _cancel_pending(tasks)
            finally:
                publish()
            raise
        except (asyncio.CancelledError, KeyboardInterrupt):
            try:
                _absorb_completed(tasks, outcome)
                await _cancel_pending(tasks)
            finally:
                publish()
            raise

    return _materialise(records, outcome)


# ---------- Internal helpers ----------


async def _judge_at(
    index: int,
    record: EvaluationRecord,
    scenario: BenchmarkScenario,
    variant: DocumentVariant,
    *,
    client: OpenRouterClient,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    judge_votes: int,
    anonymize: bool,
    target_models: Sequence[str],
    max_retries: int,
    timeout_s: float,
) -> tuple[int, EvaluationRecord]:
    try:
        judged = await _judge_record(
            record,
            scenario,
            variant,
            client=client,
            semaphore=semaphore,
            judge_model=judge_model,
            judge_votes=judge_votes,
            anonymize=anonymize,
            target_models=target_models,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
    except FatalAuthError:
        raise
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as exc:
        # One malformed prompt or an unexpected client error must not abort
        # the paid run. The type is recorded, never a response body.
        logger.error(
            "unexpected judge failure for %s/%s/%s: %s",
            record.scenario_id,
            record.variant_id,
            record.model_name,
            type(exc).__name__,
        )
        judged = _judge_failure(
            record,
            "internal",
            f"unexpected judge error: {type(exc).__name__}",
            attempts=1,
        )
    return index, judged


async def _judge_record(
    record: EvaluationRecord,
    scenario: BenchmarkScenario,
    variant: DocumentVariant,
    *,
    client: OpenRouterClient,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    judge_votes: int,
    anonymize: bool,
    target_models: Sequence[str],
    max_retries: int,
    timeout_s: float,
) -> EvaluationRecord:
    if record.target_response is None:
        return record
    prompt = build_judge_messages(
        scenario,
        variant,
        record.target_response.raw_response,
        target_models=[*target_models, record.model_name],
        anonymize=anonymize,
    )

    votes: list[JudgeEvaluation] = []
    rejected: list[str] = []
    attempts_total = 0
    try:
        for _ in range(max(1, judge_votes)):
            evaluation, rejected_ids, attempts = await _judge_once(
                client=client,
                semaphore=semaphore,
                judge_model=judge_model,
                messages=prompt.messages,
                scenario=scenario,
                max_retries=max_retries,
                timeout_s=timeout_s,
            )
            attempts_total += attempts
            votes.append(evaluation)
            rejected.extend(rejected_ids)
    except FatalAuthError:
        raise
    except (TargetRequestError, JudgeParseError) as exc:
        used = attempts_total + exc.attempts
        logger.warning(
            "judge failed for %s/%s/%s: %s",
            record.scenario_id,
            record.variant_id,
            record.model_name,
            exc.error_type,
        )
        return _judge_failure(record, exc.error_type, str(exc), attempts=used)

    meta = JudgeMeta(
        model_name=judge_model,
        votes=max(1, judge_votes),
        attempts=max(1, attempts_total),
        anonymized_prompt_sha256=prompt.prompt_sha256,
        anonymization_version=ANONYMIZATION_VERSION,
        redactions=prompt.redactions,
        rejected_ids=_dedupe(rejected),
    )
    evaluation = aggregate_votes(votes).model_copy(update={"judge_meta": meta})
    return record.model_copy(
        update={"status": "ok", "evaluation": evaluation, "errors": []}
    )


async def _judge_once(
    *,
    client: OpenRouterClient,
    semaphore: asyncio.Semaphore,
    judge_model: str,
    messages: list[dict[str, str]],
    scenario: BenchmarkScenario,
    max_retries: int,
    timeout_s: float,
) -> tuple[JudgeEvaluation, list[str], int]:
    conversation = list(messages)
    attempts_total = 0
    for repair in range(REPAIR_ATTEMPTS + 1):
        completion, attempts = await complete_with_retries(
            client=client,
            semaphore=semaphore,
            model_name=judge_model,
            messages=conversation,
            max_retries=max_retries,
            timeout_s=timeout_s,
            response_format={"type": "json_object"},
        )
        attempts_total += attempts
        try:
            evaluation = parse_judge_response(completion.content)
        except JudgeParseError as exc:
            if repair == REPAIR_ATTEMPTS:
                raise JudgeParseError(exc.error_type, str(exc), attempts_total) from exc
            conversation = [
                *conversation,
                {"role": "assistant", "content": completion.content},
                {
                    "role": "user",
                    "content": (
                        f"That answer was rejected: {exc}. Return only the JSON "
                        "object described above, with no commentary."
                    ),
                },
            ]
            continue

        detected, rejected = normalize_detected(evaluation.detected_risks, scenario)
        return (
            evaluation.model_copy(update={"detected_risks": detected}),
            rejected,
            attempts_total,
        )

    raise JudgeParseError("invalid_json", "judge did not return valid JSON", attempts_total)


def _materialise(
    records: list[EvaluationRecord],
    outcome: list[EvaluationRecord | None],
) -> list[EvaluationRecord]:
    """Return the current ordered state, keeping unjudged records as they were."""
    return [
        item if item is not None else records[index]
        for index, item in enumerate(outcome)
    ]


def _judge_failure(
    record: EvaluationRecord, error_type: str, message: str, *, attempts: int
) -> EvaluationRecord:
    """Build a ``judge_failed`` record with one explicit error."""
    return record.model_copy(
        update={
            "status": "judge_failed",
            "evaluation": None,
            "errors": [
                RecordError(
                    stage="judge",
                    error_type=error_type,
                    message=message,
                    attempts=max(1, attempts),
                )
            ],
        }
    )


async def _cancel_pending(
    tasks: list[asyncio.Task[tuple[int, EvaluationRecord]]],
) -> None:
    """Cancel outstanding judge calls and wait for their cancellation."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _absorb_completed(
    tasks: list[asyncio.Task[tuple[int, EvaluationRecord]]],
    outcome: list[EvaluationRecord | None],
) -> None:
    """Keep results of tasks that finished before cancellation was handled."""
    for task in tasks:
        if not task.done() or task.cancelled():
            continue
        try:
            index, record = task.result()
        except (Exception, asyncio.CancelledError):
            continue
        if outcome[index] is None:
            outcome[index] = record


def _normalize_key(value: str) -> str:
    return _WHITESPACE_RE.sub("_", value.strip().lower())


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _replace_mention(text: str, needle: str, replacement: str) -> tuple[str, int]:
    if _WORD_ALIAS_RE.match(needle):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])"
    else:
        pattern = re.escape(needle)
    return re.subn(pattern, replacement, text, flags=re.IGNORECASE)
