"""Execution engine: OpenRouter client, retries, checkpoints (TECHSPEC.md §9).

The runner never retries authentication failures, records per-record failures
instead of aborting a run, and writes results atomically so that an interrupted
run leaves a valid file behind.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import httpx
from pydantic import ValidationError

from knbench.config import (
    DEFAULT_APP_NAME,
    DEFAULT_BASE_URL,
    Settings,
    get_settings,
    require_api_key,
)
from knbench.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_TASK_PROMPT
from knbench.schemas import (
    SCHEMA_VERSION,
    BenchmarkScenario,
    DocumentVariant,
    EvaluationRecord,
    RecordError,
    RequestMeta,
    RunConfig,
    RunResults,
    TargetModelResponse,
)

logger = logging.getLogger(__name__)

TEMPERATURE: Final = 0.1
TOP_P: Final = 0.9
BACKOFF_BASE: Final = 1.0
BACKOFF_JITTER: Final = 0.5
BACKOFF_CAP: Final = 30.0
MAX_TOTAL_WAIT_S: Final = 180.0
CHECKPOINT_EVERY: Final = 10

_RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 502, 503, 504})
_AUTH_STATUSES: Final[frozenset[int]] = frozenset({401, 403})
_CLIENT_ERROR_STATUSES: Final[frozenset[int]] = frozenset({400, 404})


class FatalAuthError(RuntimeError):
    """Raised on HTTP 401/403; the run must stop with exit code 3."""


class TargetRequestError(RuntimeError):
    """A single target request failed after the retry budget was spent."""

    def __init__(self, error_type: str, message: str, attempts: int) -> None:
        self.error_type = error_type
        self.attempts = attempts
        super().__init__(message)


class ResumeError(RuntimeError):
    """Raised when a results file cannot be used for ``--resume``."""


class CatalogError(RuntimeError):
    """Raised when the OpenRouter model catalog cannot be read or lacks an id."""


@dataclass(frozen=True)
class ChatCompletion:
    """Raw outcome of one HTTP request, retries excluded."""

    status_code: int
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retry_after: float | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ModelCatalog:
    """The model ids advertised by OpenRouter at preflight time."""

    model_ids: frozenset[str]
    catalog_date: str


class OpenRouterClient:
    """Thin asynchronous wrapper around the OpenRouter chat-completions API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        app_name: str = DEFAULT_APP_NAME,
        app_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": app_name,
        }
        if app_url:
            headers["HTTP-Referer"] = app_url
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout_s,
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> OpenRouterClient:
        """Build a client from configuration, failing fast without a key."""
        resolved = settings if settings is not None else get_settings()
        return cls(
            require_api_key(resolved),
            base_url=resolved.openrouter_base_url,
            app_name=resolved.openrouter_app_name,
            app_url=resolved.openrouter_app_url,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        timeout_s: float,
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletion:
        """Send one chat-completions request and normalise the response."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = await self._http.post(
            "chat/completions", json=payload, timeout=timeout_s
        )
        content, prompt_tokens, completion_tokens = _extract_completion(response)
        return ChatCompletion(
            status_code=response.status_code,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retry_after=_parse_retry_after(response.headers.get("retry-after")),
            error_message=None if response.status_code < 400 else content,
        )

    async def fetch_model_catalog(self, *, timeout_s: float = 30.0) -> ModelCatalog:
        """Read ``GET /models`` and return its ids and retrieval date."""
        try:
            response = await self._http.get("models", timeout=timeout_s)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise CatalogError(
                "cannot read the OpenRouter model catalog: "
                f"{type(exc).__name__}. Check OPENROUTER_BASE_URL and network access."
            ) from exc

        if response.status_code != 200:
            raise CatalogError(
                f"OpenRouter model catalog returned HTTP {response.status_code}. "
                "Check OPENROUTER_BASE_URL and try again."
            )

        try:
            payload: object = response.json()
        except ValueError as exc:
            raise CatalogError(
                "OpenRouter model catalog is not valid JSON. "
                "Check OPENROUTER_BASE_URL."
            ) from exc
        if not isinstance(payload, dict):
            raise CatalogError(
                "OpenRouter model catalog has an unexpected shape. "
                "Check OPENROUTER_BASE_URL."
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise CatalogError(
                "OpenRouter model catalog has no data array. "
                "Check OPENROUTER_BASE_URL."
            )

        ids: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if isinstance(model_id, str) and model_id.strip():
                ids.append(model_id.strip())
        if not ids:
            raise CatalogError(
                "OpenRouter model catalog contains no model ids. "
                "Check OPENROUTER_BASE_URL."
            )

        return ModelCatalog(
            model_ids=frozenset(ids),
            catalog_date=datetime.now(timezone.utc).date().isoformat(),
        )

    async def require_models(
        self, model_ids: Sequence[str], *, timeout_s: float = 30.0
    ) -> ModelCatalog:
        """Fail fast unless every requested id exists in the current catalog."""
        catalog = await self.fetch_model_catalog(timeout_s=timeout_s)
        missing = [
            model_id
            for model_id in dict.fromkeys(model_ids)
            if model_id not in catalog.model_ids
        ]
        if missing:
            raise CatalogError(
                f"OpenRouter model catalog ({catalog.catalog_date}) does not list: "
                f"{', '.join(missing)}. Check --models, --judge-model or "
                "OPENROUTER_BASE_URL."
            )
        logger.info(
            "model catalog: %d id(s), catalog date %s",
            len(catalog.model_ids),
            catalog.catalog_date,
        )
        return catalog


async def run_model(
    prompt: str,
    model_name: str,
    *,
    client: OpenRouterClient,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
    timeout_s: float = 120.0,
    scenario_id: str = "",
    variant_id: str = "",
) -> TargetModelResponse:
    """Query one target model and return its recorded response.

    Raises :class:`FatalAuthError` on 401/403 and :class:`TargetRequestError`
    when the retry budget is exhausted (TECHSPEC.md §9.3).
    """
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    started = time.perf_counter()
    completion, attempts = await complete_with_retries(
        client=client,
        semaphore=semaphore,
        model_name=model_name,
        messages=messages,
        max_retries=max_retries,
        timeout_s=timeout_s,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return TargetModelResponse(
        scenario_id=scenario_id,
        variant_id=variant_id,
        model_name=model_name,
        raw_response=completion.content,
        execution_time_ms=elapsed_ms,
        request_meta=RequestMeta(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            attempts=attempts,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        ),
    )


async def run_benchmark(
    scenarios: list[BenchmarkScenario],
    models: list[str],
    config: RunConfig,
    *,
    resume_from: RunResults | None = None,
    client: OpenRouterClient | None = None,
    output_path: Path | None = None,
    on_checkpoint: Callable[[RunResults], None] | None = None,
    run_id: str | None = None,
    preflight: bool = False,
) -> RunResults:
    """Query every (scenario, variant, model) triple and record the outcomes.

    Successful target calls are stored with ``status="skipped"``: judging is a
    separate stage (TECHSPEC.md §10), and ``--skip-judge`` stops here.
    """
    resume_records: list[EvaluationRecord] = []
    if resume_from is not None:
        resume_records = prepare_resume(resume_from, scenarios, models)
        config = config.model_copy(update={"resumed_from": resume_from.run_id})

    owned_client = client is None
    active_client = client if client is not None else OpenRouterClient.from_settings()
    if preflight:
        try:
            await active_client.require_models(
                [*models, config.judge_model], timeout_s=config.timeout_s
            )
        except BaseException:
            if owned_client:
                await active_client.aclose()
            raise
    # The CLI generates the id before the run so the log file can be named after
    # it (TECHSPEC.md §14.3); direct callers keep the generated default.
    active_run_id = run_id if run_id is not None else new_run_id()
    started_at = datetime.now(timezone.utc)
    semaphore = asyncio.Semaphore(config.concurrency)

    records: dict[tuple[str, str, str], EvaluationRecord] = {}
    for record in resume_records:
        records[_record_key(record)] = record

    pending = [
        (scenario, variant, model)
        for scenario in scenarios
        for variant in scenario.variants
        for model in models
        if records.get((scenario.scenario_id, variant.variant_id, model)) is None
        or records[(scenario.scenario_id, variant.variant_id, model)].status != "ok"
    ]

    def snapshot(finished_at: datetime | None) -> RunResults:
        return RunResults(
            run_id=active_run_id,
            started_at=started_at,
            finished_at=finished_at,
            run_config=config,
            scenarios=scenarios,
            records=sorted(records.values(), key=_record_key),
        )

    def checkpoint() -> None:
        current = snapshot(None)
        if output_path is not None:
            write_results_atomic(output_path, current)
        if on_checkpoint is not None:
            on_checkpoint(current)

    try:
        if not pending:
            logger.info("nothing to do: every record is already complete")
            finished = snapshot(datetime.now(timezone.utc))
            if output_path is not None:
                write_results_atomic(output_path, finished)
            if on_checkpoint is not None:
                on_checkpoint(finished)
            return finished

        tasks = [
            asyncio.create_task(
                _execute_triple(scenario, variant, model, active_client, semaphore, config)
            )
            for scenario, variant, model in pending
        ]
        completed = 0
        try:
            for future in asyncio.as_completed(tasks):
                record = await future
                records[_record_key(record)] = record
                completed += 1
                if record.status != "skipped":
                    logger.warning(
                        "record failed: %s/%s/%s -> %s",
                        record.scenario_id,
                        record.variant_id,
                        record.model_name,
                        record.errors[0].error_type if record.errors else "unknown",
                    )
                if completed % CHECKPOINT_EVERY == 0:
                    logger.info("progress: %d/%d records", completed, len(tasks))
                    checkpoint()
        except FatalAuthError:
            try:
                await _cancel_pending(tasks)
            finally:
                # Keep the records that finished before the fatal response.
                checkpoint()
            raise
        except (asyncio.CancelledError, KeyboardInterrupt):
            try:
                await _cancel_pending(tasks)
            finally:
                # An interrupted run must still leave a parseable results file
                # with every record completed so far (§18).
                checkpoint()
            raise

        finished = snapshot(datetime.now(timezone.utc))
        if output_path is not None:
            write_results_atomic(output_path, finished)
        if on_checkpoint is not None:
            on_checkpoint(finished)
        logger.info("run finished: %d records", len(finished.records))
        return finished
    finally:
        if owned_client:
            await active_client.aclose()


def new_run_id() -> str:
    """Return a sortable, unique run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def write_results_atomic(path: Path, results: RunResults) -> None:
    """Write results through a temporary file and rename atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, target)


def load_results(path: Path) -> RunResults:
    """Read a results file, raising :class:`ResumeError` when unusable."""
    source = Path(path)
    if not source.is_file():
        raise ResumeError(f"{source}: results file not found")
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ResumeError(f"{source}: cannot read -> {exc.strerror or exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResumeError(
            f"{source}: invalid JSON at line {exc.lineno} column {exc.colno} -> {exc.msg}"
        ) from exc

    if isinstance(payload, dict):
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ResumeError(
                f"{source}: unsupported schema_version {version!r}, "
                f"expected {SCHEMA_VERSION!r}"
            )

    try:
        return RunResults.model_validate(payload)
    except ValidationError as exc:
        raise ResumeError(
            f"{source}: not a valid results file -> {exc.error_count()} problem(s)"
        ) from exc


def prepare_resume(
    resume: RunResults,
    scenarios: list[BenchmarkScenario],
    models: list[str],
) -> list[EvaluationRecord]:
    """Validate a resume snapshot and keep records for the current triples.

    The returned list is safe to merge into the new run: it excludes records for
    scenarios, variants and models that are no longer selected. A changed
    overlapping scenario or variant is a hard error, because reusing its old
    response would mix two different datasets in one report.
    """
    _check_resumable(resume, models)
    _check_resume_snapshot(resume, scenarios)

    snapshot_pairs = {
        (scenario.scenario_id, variant.variant_id)
        for scenario in resume.scenarios
        for variant in scenario.variants
    }
    current_keys = {
        (scenario.scenario_id, variant.variant_id, model)
        for scenario in scenarios
        for variant in scenario.variants
        for model in models
    }

    seen: set[tuple[str, str, str]] = set()
    kept: list[EvaluationRecord] = []
    for record in resume.records:
        key = _record_key(record)
        if key in seen:
            raise ResumeError(
                f"cannot resume from run {resume.run_id}: duplicate record for "
                f"{record.scenario_id}/{record.variant_id}/{record.model_name}"
            )
        seen.add(key)
        if (record.scenario_id, record.variant_id) not in snapshot_pairs:
            raise ResumeError(
                f"cannot resume from run {resume.run_id}: record "
                f"{record.scenario_id}/{record.variant_id}/{record.model_name} "
                "does not match any variant in that run"
            )
        if record.model_name not in resume.run_config.models:
            raise ResumeError(
                f"cannot resume from run {resume.run_id}: record model "
                f"'{record.model_name}' is not listed in that run"
            )
        if key in current_keys:
            kept.append(record)

    if len(kept) != len(resume.records):
        logger.info(
            "resume: keeping %d of %d record(s) for the current selection",
            len(kept),
            len(resume.records),
        )
    return kept


# ---------- Internal helpers ----------


async def complete_with_retries(
    *,
    client: OpenRouterClient,
    semaphore: asyncio.Semaphore,
    model_name: str,
    messages: list[dict[str, str]],
    max_retries: int,
    timeout_s: float,
    response_format: dict[str, str] | None = None,
) -> tuple[ChatCompletion, int]:
    attempts = 0
    waited = 0.0
    empty_retries = 0
    allowed_attempts = max_retries + 1

    while True:
        attempts += 1
        try:
            async with semaphore:
                completion = await client.complete(
                    model=model_name,
                    messages=messages,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    timeout_s=timeout_s,
                    response_format=response_format,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            error_type = (
                "timeout" if isinstance(exc, httpx.TimeoutException) else "transport"
            )
            waited = await _wait_or_fail(
                error_type, str(exc), attempts, allowed_attempts, waited
            )
            continue

        status = completion.status_code
        if status in _AUTH_STATUSES:
            raise FatalAuthError(
                f"OpenRouter rejected the credentials with HTTP {status}. "
                "Check OPENROUTER_API_KEY."
            )
        if status in _CLIENT_ERROR_STATUSES:
            raise TargetRequestError(
                f"http_{status}",
                completion.error_message or f"OpenRouter returned {status}",
                attempts,
            )
        if status in _RETRYABLE_STATUSES:
            error_type = f"http_{status}"
            message = completion.error_message or f"OpenRouter returned {status}"
            if attempts >= allowed_attempts:
                raise TargetRequestError(error_type, message, attempts)
            delay = completion.retry_after
            if delay is None:
                delay = _backoff(attempts)
            waited = await _wait_or_fail(error_type, message, attempts, allowed_attempts, waited, delay)
            continue
        if status >= 400:
            raise TargetRequestError(
                f"http_{status}",
                completion.error_message or f"OpenRouter returned {status}",
                attempts,
            )

        if not completion.content.strip():
            empty_retries += 1
            if empty_retries > 1 or attempts >= allowed_attempts:
                raise TargetRequestError(
                    "empty_content", "OpenRouter returned empty content", attempts
                )
            waited = await _wait_or_fail(
                "empty_content", "empty content", attempts, allowed_attempts, waited
            )
            continue

        return completion, attempts


async def _wait_or_fail(
    error_type: str,
    message: str,
    attempts: int,
    allowed_attempts: int,
    waited: float,
    delay: float | None = None,
) -> float:
    """Sleep before the next attempt, or raise when the budget is exhausted."""
    if attempts >= allowed_attempts:
        raise TargetRequestError(error_type, message, attempts)
    pause = _backoff(attempts) if delay is None else delay
    if waited + pause > MAX_TOTAL_WAIT_S:
        raise TargetRequestError(
            "retry_budget",
            f"{message}; retry budget of {MAX_TOTAL_WAIT_S:.0f}s exhausted",
            attempts,
        )
    await _sleep(pause)
    return waited + pause


def _backoff(attempt: int) -> float:
    # 2.0 ** attempt keeps the expression typed as float; int ** int is Any.
    return min(BACKOFF_BASE * 2.0**attempt + _jitter(), BACKOFF_CAP)


def _jitter() -> float:
    return random.uniform(0.0, BACKOFF_JITTER)


async def _sleep(delay: float) -> None:
    await asyncio.sleep(delay)


async def _execute_triple(
    scenario: BenchmarkScenario,
    variant: DocumentVariant,
    model_name: str,
    client: OpenRouterClient,
    semaphore: asyncio.Semaphore,
    config: RunConfig,
) -> EvaluationRecord:
    prompt = f"{variant.document_text}\n\n{scenario.task_prompt or DEFAULT_TASK_PROMPT}"
    try:
        response = await run_model(
            prompt,
            model_name,
            client=client,
            semaphore=semaphore,
            max_retries=config.max_retries,
            timeout_s=config.timeout_s,
            scenario_id=scenario.scenario_id,
            variant_id=variant.variant_id,
        )
    except TargetRequestError as exc:
        return EvaluationRecord(
            scenario_id=scenario.scenario_id,
            variant_id=variant.variant_id,
            model_name=model_name,
            status="target_failed",
            errors=[
                RecordError(
                    stage="target",
                    error_type=exc.error_type,
                    message=str(exc),
                    attempts=exc.attempts,
                )
            ],
        )
    return EvaluationRecord(
        scenario_id=scenario.scenario_id,
        variant_id=variant.variant_id,
        model_name=model_name,
        status="skipped",
        target_response=response,
    )


async def _cancel_pending(tasks: list[asyncio.Task[EvaluationRecord]]) -> None:
    """Cancel outstanding requests and wait for them to unwind."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _check_resumable(resume: RunResults, models: list[str]) -> None:
    missing = sorted(set(models) - set(resume.run_config.models))
    if missing:
        raise ResumeError(
            f"cannot resume from run {resume.run_id}: models {missing} were not "
            "part of that run"
        )


def _check_resume_snapshot(
    resume: RunResults, scenarios: list[BenchmarkScenario]
) -> None:
    """Reject changed content for scenarios that exist in both snapshots."""
    previous_by_id: dict[str, BenchmarkScenario] = {}
    for previous_scenario in resume.scenarios:
        if previous_scenario.scenario_id in previous_by_id:
            raise ResumeError(
                f"cannot resume from run {resume.run_id}: duplicate scenario "
                f"'{previous_scenario.scenario_id}' in the previous snapshot"
            )
        previous_by_id[previous_scenario.scenario_id] = previous_scenario

    problems: list[str] = []
    for scenario in scenarios:
        previous = previous_by_id.get(scenario.scenario_id)
        if previous is None:
            continue
        changes = _scenario_changes(previous, scenario)
        if changes:
            problems.append(
                f"scenario '{scenario.scenario_id}': {', '.join(changes)}"
            )

    if problems:
        raise ResumeError(
            f"cannot resume from run {resume.run_id}: dataset changed since that "
            f"run -> {'; '.join(problems)}"
        )


def _scenario_changes(
    previous: BenchmarkScenario, current: BenchmarkScenario
) -> list[str]:
    changes: list[str] = []
    if previous.domain != current.domain:
        changes.append("domain")
    if previous.task_prompt != current.task_prompt:
        changes.append("task_prompt")
    if [risk.model_dump() for risk in previous.injected_risks] != [
        risk.model_dump() for risk in current.injected_risks
    ]:
        changes.append("injected_risks")

    previous_variants = {variant.variant_id: variant for variant in previous.variants}
    for variant in current.variants:
        old = previous_variants.get(variant.variant_id)
        if old is None:
            continue
        fields = _variant_changes(old, variant)
        if fields:
            changes.append(
                f"variant '{variant.variant_id}' ({', '.join(fields)})"
            )
    return changes


def _variant_changes(previous: DocumentVariant, current: DocumentVariant) -> list[str]:
    fields: list[str] = []
    if previous.vendor_name != current.vendor_name:
        fields.append("vendor_name")
    if previous.origin_country != current.origin_country:
        fields.append("origin_country")
    if previous.document_text != current.document_text:
        fields.append("document_text")
    if previous.is_control != current.is_control:
        fields.append("is_control")
    if previous.vendor_aliases != current.vendor_aliases:
        fields.append("vendor_aliases")
    if previous.country_aliases != current.country_aliases:
        fields.append("country_aliases")
    return fields


def _record_key(record: EvaluationRecord) -> tuple[str, str, str]:
    return (record.scenario_id, record.variant_id, record.model_name)


def _extract_completion(response: httpx.Response) -> tuple[str, int | None, int | None]:
    try:
        payload: object = response.json()
    except ValueError:
        return response.text, None, None
    if not isinstance(payload, dict):
        return "", None, None

    content = ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                raw_content = message.get("content")
                if isinstance(raw_content, str):
                    content = raw_content

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    usage = payload.get("usage")
    if isinstance(usage, dict):
        raw_prompt = usage.get("prompt_tokens")
        raw_completion = usage.get("completion_tokens")
        if isinstance(raw_prompt, int):
            prompt_tokens = raw_prompt
        if isinstance(raw_completion, int):
            completion_tokens = raw_completion
    return content, prompt_tokens, completion_tokens


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
