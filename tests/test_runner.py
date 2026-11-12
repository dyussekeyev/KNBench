"""Tests for knbench.runner (TECHSPEC.md §9)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
import respx

from knbench import runner
from knbench.runner import (
    CatalogError,
    FatalAuthError,
    OpenRouterClient,
    TargetRequestError,
    run_benchmark,
    run_model,
)
from knbench.schemas import (
    BenchmarkScenario,
    DocumentVariant,
    InjectedRisk,
    RunConfig,
)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"


def completion_payload(content: str = "response text") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }


def make_config(**overrides: Any) -> RunConfig:
    data: dict[str, Any] = {
        "models": ["test/model"],
        "judge_model": "judge/model",
        "concurrency": 2,
        "max_retries": 2,
        "timeout_s": 5.0,
        "judge_votes": 1,
        "anonymize": True,
        "scenarios_path": "data/scenarios.json",
        "scenarios_sha256": "0" * 64,
        "app_version": "0.1.0",
    }
    data.update(overrides)
    return RunConfig(**data)


def make_scenario(variant_count: int = 2) -> BenchmarkScenario:
    variants = [
        DocumentVariant(
            variant_id=f"v{index}",
            vendor_name=f"Vendor{index}",
            origin_country="Country",
            document_text=f"Document {index}.",
            is_control=index == 0,
        )
        for index in range(variant_count)
    ]
    return BenchmarkScenario(
        scenario_id="sc-001",
        domain="test",
        injected_risks=[InjectedRisk(risk_id="r1", category="c1", description="Risk.")],
        variants=variants,
        task_prompt="Assess the document.",
    )


def no_wait(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace sleeping with recording and remove jitter."""
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(runner, "_sleep", fake_sleep)
    monkeypatch.setattr(runner, "_jitter", lambda: 0.0)
    return delays


async def test_successful_call_records_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload("done"))
            )
            result = await run_model(
                "prompt",
                "test/model",
                client=client,
                semaphore=asyncio.Semaphore(1),
                scenario_id="sc-001",
                variant_id="v0",
            )
    finally:
        await client.aclose()

    assert result.raw_response == "done"
    assert result.scenario_id == "sc-001"
    assert result.variant_id == "v0"
    assert result.request_meta is not None
    assert result.request_meta.attempts == 1
    assert result.request_meta.prompt_tokens == 12
    assert result.execution_time_ms >= 0


async def test_catalog_accepts_known_models() -> None:
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.get(MODELS_URL).mock(
                return_value=httpx.Response(
                    200,
                    json={"data": [{"id": "test/model"}, {"id": "judge/model"}]},
                )
            )
            catalog = await client.require_models(["test/model", "judge/model"])
    finally:
        await client.aclose()

    assert catalog.model_ids == frozenset({"test/model", "judge/model"})
    assert len(catalog.catalog_date) == 10
    assert route.call_count == 1


async def test_catalog_reports_every_missing_model() -> None:
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            respx.get(MODELS_URL).mock(
                return_value=httpx.Response(200, json={"data": [{"id": "test/model"}]})
            )
            with pytest.raises(CatalogError) as excinfo:
                await client.require_models(["test/model", "missing/one", "missing/two"])
    finally:
        await client.aclose()

    message = str(excinfo.value)
    assert "missing/one" in message
    assert "missing/two" in message
    assert "catalog" in message.lower()


async def test_catalog_failure_is_actionable() -> None:
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            respx.get(MODELS_URL).mock(return_value=httpx.Response(503))
            with pytest.raises(CatalogError) as excinfo:
                await client.require_models(["test/model"])
    finally:
        await client.aclose()

    message = str(excinfo.value)
    assert "HTTP 503" in message
    assert "OPENROUTER_BASE_URL" in message


async def test_sampling_parameters_are_sent_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            await run_model(
                "prompt",
                "test/model",
                client=client,
                semaphore=asyncio.Semaphore(1),
            )
    finally:
        await client.aclose()

    payload = json.loads(route.calls[0].request.content)
    assert payload["temperature"] == 0.1
    assert payload["top_p"] == 0.9
    assert payload["model"] == "test/model"
    assert payload["messages"][0]["role"] == "system"


async def test_retry_after_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                side_effect=[
                    httpx.Response(429, headers={"Retry-After": "2"}),
                    httpx.Response(200, json=completion_payload()),
                ]
            )
            result = await run_model(
                "prompt",
                "test/model",
                client=client,
                semaphore=asyncio.Semaphore(1),
                max_retries=4,
            )
    finally:
        await client.aclose()

    assert delays[0] == 2.0
    assert route.call_count == 2
    assert result.request_meta is not None
    assert result.request_meta.attempts == 2


async def test_502_retries_until_the_budget_is_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(return_value=httpx.Response(502))
            with pytest.raises(TargetRequestError) as excinfo:
                await run_model(
                    "prompt",
                    "test/model",
                    client=client,
                    semaphore=asyncio.Semaphore(1),
                    max_retries=4,
                )
    finally:
        await client.aclose()

    error = excinfo.value
    assert error.error_type == "http_502"
    assert error.attempts == 5
    assert route.call_count == 5


async def test_401_is_fatal_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(return_value=httpx.Response(401))
            with pytest.raises(FatalAuthError, match="401"):
                await run_model(
                    "prompt",
                    "test/model",
                    client=client,
                    semaphore=asyncio.Semaphore(1),
                    max_retries=4,
                )
    finally:
        await client.aclose()

    assert route.call_count == 1


async def test_400_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(return_value=httpx.Response(400))
            with pytest.raises(TargetRequestError) as excinfo:
                await run_model("prompt", "test/model", client=client, semaphore=asyncio.Semaphore(1))
    finally:
        await client.aclose()

    assert excinfo.value.error_type == "http_400"
    assert excinfo.value.attempts == 1
    assert route.call_count == 1


async def test_empty_content_is_retried_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload(""))
            )
            with pytest.raises(TargetRequestError) as excinfo:
                await run_model("prompt", "test/model", client=client, semaphore=asyncio.Semaphore(1))
    finally:
        await client.aclose()

    assert excinfo.value.error_type == "empty_content"
    assert excinfo.value.attempts == 2
    assert route.call_count == 2


async def test_empty_content_recovers_after_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                side_effect=[
                    httpx.Response(200, json=completion_payload("")),
                    httpx.Response(200, json=completion_payload("second attempt")),
                ]
            )
            result = await run_model("prompt", "test/model", client=client, semaphore=asyncio.Semaphore(1))
    finally:
        await client.aclose()

    assert result.raw_response == "second attempt"
    assert route.call_count == 2


async def test_transport_errors_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                side_effect=[
                    httpx.ConnectError("boom"),
                    httpx.Response(200, json=completion_payload("recovered")),
                ]
            )
            result = await run_model("prompt", "test/model", client=client, semaphore=asyncio.Semaphore(1))
    finally:
        await client.aclose()

    assert result.raw_response == "recovered"
    assert route.call_count == 2


async def test_retry_budget_stops_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    no_wait(monkeypatch)
    monkeypatch.setattr(runner, "MAX_TOTAL_WAIT_S", 0.0)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            respx.post(CHAT_URL).mock(
                return_value=httpx.Response(429, headers={"Retry-After": "5"})
            )
            with pytest.raises(TargetRequestError) as excinfo:
                await run_model("prompt", "test/model", client=client, semaphore=asyncio.Semaphore(1))
    finally:
        await client.aclose()

    assert excinfo.value.error_type == "retry_budget"


async def test_concurrency_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    no_wait(monkeypatch)
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json=completion_payload())

    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            respx.post(CHAT_URL).mock(side_effect=handler)
            await run_benchmark(
                [make_scenario(4)],
                ["m1", "m2"],
                make_config(models=["m1", "m2"], concurrency=2),
                client=client,
            )
    finally:
        await client.aclose()

    assert 1 <= peak <= 2


async def test_benchmark_records_partial_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        if "Document 0." in prompt:
            return httpx.Response(502)
        return httpx.Response(200, json=completion_payload("ok"))

    try:
        with respx.mock:
            respx.post(CHAT_URL).mock(side_effect=handler)
            results = await run_benchmark(
                [make_scenario(2)],
                ["m1"],
                make_config(models=["m1"], max_retries=2),
                client=client,
            )
    finally:
        await client.aclose()

    by_variant = {record.variant_id: record for record in results.records}
    assert by_variant["v0"].status == "target_failed"
    assert by_variant["v0"].errors[0].error_type == "http_502"
    assert by_variant["v0"].errors[0].attempts == 3
    assert by_variant["v1"].status == "skipped"
    assert by_variant["v1"].target_response is not None


async def test_fatal_error_aborts_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    no_wait(monkeypatch)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            respx.post(CHAT_URL).mock(return_value=httpx.Response(403))
            with pytest.raises(FatalAuthError):
                await run_benchmark(
                    [make_scenario(2)],
                    ["m1"],
                    make_config(models=["m1"]),
                    client=client,
                )
    finally:
        await client.aclose()
