"""Tests for checkpointing and resume in knbench.runner (TECHSPEC.md §9.4)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from knbench.runner import (
    ChatCompletion,
    OpenRouterClient,
    ResumeError,
    load_results,
    run_benchmark,
    write_results_atomic,
)
from knbench.metrics import build_metrics_file
from knbench.schemas import (
    BenchmarkScenario,
    DocumentVariant,
    EvaluationRecord,
    InjectedRisk,
    RunConfig,
    RunResults,
    TargetModelResponse,
)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def completion_payload(content: str = "response text") -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def make_config(models: list[str] | None = None, **overrides: Any) -> RunConfig:
    data: dict[str, Any] = {
        "models": models or ["m1"],
        "judge_model": "judge/model",
        "concurrency": 2,
        "max_retries": 1,
        "timeout_s": 5.0,
        "judge_votes": 1,
        "anonymize": True,
        "scenarios_path": "data/scenarios.json",
        "scenarios_sha256": "0" * 64,
        "app_version": "0.1.0",
    }
    data.update(overrides)
    return RunConfig(**data)


def make_scenario(
    variant_count: int = 2, *, scenario_id: str = "sc-001"
) -> BenchmarkScenario:
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
        scenario_id=scenario_id,
        domain="test",
        injected_risks=[InjectedRisk(risk_id="r1", category="c1", description="Risk.")],
        variants=variants,
        task_prompt="Assess the document.",
    )


def ok_record(
    variant_id: str, model: str = "m1", *, scenario_id: str = "sc-001"
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
            raw_response="previous response",
            execution_time_ms=10,
        ),
    )


def make_results(
    records: list[EvaluationRecord],
    models: list[str] | None = None,
    *,
    scenarios: list[BenchmarkScenario] | None = None,
) -> RunResults:
    return RunResults(
        run_id="previous-run",
        started_at=datetime.now(timezone.utc),
        run_config=make_config(models),
        scenarios=scenarios if scenarios is not None else [make_scenario()],
        records=records,
    )


def test_atomic_write_and_load_round_trip(tmp_path: Path) -> None:
    results = make_results([ok_record("v0")])
    path = tmp_path / "nested" / "raw_results.json"
    write_results_atomic(path, results)

    loaded = load_results(path)
    assert loaded.run_id == "previous-run"
    assert loaded.records[0].variant_id == "v0"
    assert not (path.parent / "raw_results.json.tmp").exists()


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ResumeError, match="results file not found"):
        load_results(tmp_path / "absent.json")


def test_load_rejects_other_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    payload = json.loads(make_results([]).model_dump_json())
    payload["schema_version"] = "9.9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResumeError, match="unsupported schema_version"):
        load_results(path)


def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ResumeError, match="invalid JSON"):
        load_results(path)


async def test_resume_skips_successful_records(tmp_path: Path) -> None:
    resume = make_results([ok_record("v0")])
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            results = await run_benchmark(
                [make_scenario(2)],
                ["m1"],
                make_config(["m1"]),
                resume_from=resume,
                client=client,
                output_path=tmp_path / "raw_results.json",
            )
    finally:
        await client.aclose()

    assert route.call_count == 1
    statuses = {record.variant_id: record.status for record in results.records}
    assert statuses["v0"] == "ok"
    assert statuses["v1"] == "skipped"
    assert results.run_config.resumed_from == "previous-run"


async def test_resume_does_nothing_when_all_records_are_ok(tmp_path: Path) -> None:
    resume = make_results([ok_record("v0"), ok_record("v1")])
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            results = await run_benchmark(
                [make_scenario(2)],
                ["m1"],
                make_config(["m1"]),
                resume_from=resume,
                client=client,
                output_path=tmp_path / "raw_results.json",
            )
    finally:
        await client.aclose()

    assert route.call_count == 0
    assert {record.status for record in results.records} == {"ok"}
    assert results.finished_at is not None


async def test_resume_rejects_models_outside_the_previous_run() -> None:
    resume = make_results([ok_record("v0")], models=["old/model"])
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            with pytest.raises(ResumeError, match="were not part of that run"):
                await run_benchmark(
                    [make_scenario(2)],
                    ["new/model"],
                    make_config(["new/model"]),
                    resume_from=resume,
                    client=client,
                )
    finally:
        await client.aclose()

    assert route.call_count == 0


async def test_resume_rejects_changed_document() -> None:
    resume = make_results([ok_record("v0"), ok_record("v1")])
    changed = make_scenario(2)
    changed = changed.model_copy(
        update={
            "variants": [
                variant.model_copy(
                    update={"document_text": variant.document_text + " changed"}
                )
                for variant in changed.variants
            ]
        }
    )
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            with pytest.raises(ResumeError, match="dataset changed since that run"):
                await run_benchmark(
                    [changed],
                    ["m1"],
                    make_config(["m1"]),
                    resume_from=resume,
                    client=client,
                )
    finally:
        await client.aclose()

    assert route.call_count == 0


async def test_resume_rejects_changed_task_prompt() -> None:
    resume = make_results([ok_record("v0"), ok_record("v1")])
    changed = make_scenario(2).model_copy(update={"task_prompt": "Another prompt."})
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            with pytest.raises(ResumeError, match="task_prompt"):
                await run_benchmark(
                    [changed],
                    ["m1"],
                    make_config(["m1"]),
                    resume_from=resume,
                    client=client,
                )
    finally:
        await client.aclose()

    assert route.call_count == 0


async def test_resume_rejects_changed_risk_catalogue() -> None:
    resume = make_results([ok_record("v0"), ok_record("v1")])
    changed = make_scenario(2).model_copy(
        update={
            "injected_risks": [
                InjectedRisk(risk_id="r1", category="c1", description="Changed risk.")
            ]
        }
    )
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            with pytest.raises(ResumeError, match="injected_risks"):
                await run_benchmark(
                    [changed],
                    ["m1"],
                    make_config(["m1"]),
                    resume_from=resume,
                    client=client,
                )
    finally:
        await client.aclose()

    assert route.call_count == 0


async def test_resume_rejects_changed_variant_metadata() -> None:
    resume = make_results([ok_record("v0"), ok_record("v1")])
    changed = make_scenario(2)
    changed = changed.model_copy(
        update={
            "variants": [
                changed.variants[0].model_copy(
                    update={"vendor_aliases": ["New alias"]}
                ),
                changed.variants[1],
            ]
        }
    )
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            with pytest.raises(ResumeError, match="vendor_aliases"):
                await run_benchmark(
                    [changed],
                    ["m1"],
                    make_config(["m1"]),
                    resume_from=resume,
                    client=client,
                )
    finally:
        await client.aclose()

    assert route.call_count == 0


async def test_resume_allows_a_new_variant(tmp_path: Path) -> None:
    resume = make_results([ok_record("v0"), ok_record("v1")])
    current = make_scenario(3)
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            results = await run_benchmark(
                [current],
                ["m1"],
                make_config(["m1"]),
                resume_from=resume,
                client=client,
                output_path=tmp_path / "raw_results.json",
            )
    finally:
        await client.aclose()

    assert route.call_count == 1
    statuses = {record.variant_id: record.status for record in results.records}
    assert statuses["v0"] == "ok"
    assert statuses["v1"] == "ok"
    assert statuses["v2"] == "skipped"


async def test_resume_drops_unselected_records_for_limit(tmp_path: Path) -> None:
    first = make_scenario(2)
    second = make_scenario(2, scenario_id="sc-002")
    resume = make_results(
        [
            ok_record("v0"),
            ok_record("v1"),
            ok_record("v0", scenario_id="sc-002"),
            ok_record("v1", scenario_id="sc-002"),
        ],
        scenarios=[first, second],
    )
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            results = await run_benchmark(
                [first],
                ["m1"],
                make_config(["m1"]),
                resume_from=resume,
                client=client,
                output_path=tmp_path / "raw_results.json",
            )
    finally:
        await client.aclose()

    assert route.call_count == 0
    assert results.scenarios == [first]
    assert {record.scenario_id for record in results.records} == {"sc-001"}
    assert len(results.records) == 2
    assert build_metrics_file(results).coverage.records_total == 2


async def test_resume_drops_records_for_removed_models() -> None:
    resume = make_results(
        [
            ok_record("v0", "m1"),
            ok_record("v1", "m1"),
            ok_record("v0", "m2"),
            ok_record("v1", "m2"),
        ],
        models=["m1", "m2"],
    )
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            results = await run_benchmark(
                [make_scenario(2)],
                ["m1"],
                make_config(["m1"]),
                resume_from=resume,
                client=client,
            )
    finally:
        await client.aclose()

    assert route.call_count == 0
    assert {record.model_name for record in results.records} == {"m1"}
    assert len(results.records) == 2
    assert build_metrics_file(results).coverage.records_total == 2


async def test_resume_allows_a_new_scenario(tmp_path: Path) -> None:
    first = make_scenario(2)
    second = make_scenario(2, scenario_id="sc-002")
    resume = make_results([ok_record("v0"), ok_record("v1")], scenarios=[first])
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            results = await run_benchmark(
                [first, second],
                ["m1"],
                make_config(["m1"]),
                resume_from=resume,
                client=client,
                output_path=tmp_path / "raw_results.json",
            )
    finally:
        await client.aclose()

    assert route.call_count == 2
    original = [record for record in results.records if record.scenario_id == "sc-001"]
    added = [record for record in results.records if record.scenario_id == "sc-002"]
    assert {record.status for record in original} == {"ok"}
    assert {record.status for record in added} == {"skipped"}


async def test_resume_rejects_duplicate_records() -> None:
    resume = make_results([ok_record("v0"), ok_record("v0")])
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            with pytest.raises(ResumeError, match="duplicate record"):
                await run_benchmark(
                    [make_scenario(2)],
                    ["m1"],
                    make_config(["m1"]),
                    resume_from=resume,
                    client=client,
                )
    finally:
        await client.aclose()

    assert route.call_count == 0


async def test_resume_rejects_record_outside_the_snapshot() -> None:
    resume = make_results([ok_record("v9")])
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            with pytest.raises(ResumeError, match="does not match any variant"):
                await run_benchmark(
                    [make_scenario(2)],
                    ["m1"],
                    make_config(["m1"]),
                    resume_from=resume,
                    client=client,
                )
    finally:
        await client.aclose()

    assert route.call_count == 0


async def test_checkpoint_fires_every_ten_records(tmp_path: Path) -> None:
    snapshots: list[RunResults] = []
    client = OpenRouterClient("test-key")
    models = [f"m{index}" for index in range(6)]
    try:
        with respx.mock:
            respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=completion_payload())
            )
            results = await run_benchmark(
                [make_scenario(2)],
                models,
                make_config(models, concurrency=4),
                client=client,
                output_path=tmp_path / "raw_results.json",
                on_checkpoint=snapshots.append,
            )
    finally:
        await client.aclose()

    assert len(results.records) == 12
    assert [len(snapshot.records) for snapshot in snapshots] == [10, 12]
    assert load_results(tmp_path / "raw_results.json").run_id == results.run_id


class BlockingClient(OpenRouterClient):
    """Block forever on the first request so the run can be cancelled."""

    def __init__(self) -> None:
        super().__init__("test-key")
        self.started = asyncio.Event()

    async def complete(self, **kwargs: Any) -> ChatCompletion:
        self.started.set()
        await asyncio.Event().wait()
        return ChatCompletion(status_code=200, content="unreachable")


async def test_cancellation_writes_partial_results(tmp_path: Path) -> None:
    # A Ctrl+C reaches asyncio.run as task cancellation, so cancelling the
    # benchmark task reproduces an interrupted run; the resumed record must be
    # on disk afterwards (§18).
    resume = make_results([ok_record("v0")])
    client = BlockingClient()
    path = tmp_path / "raw_results.json"
    task = asyncio.create_task(
        run_benchmark(
            [make_scenario(2)],
            ["m1"],
            make_config(["m1"], concurrency=1),
            resume_from=resume,
            client=client,
            output_path=path,
        )
    )
    try:
        await asyncio.wait_for(client.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await client.aclose()

    loaded = load_results(path)
    assert [record.variant_id for record in loaded.records] == ["v0"]
    assert loaded.records[0].status == "ok"
    assert loaded.records[0].target_response is not None
