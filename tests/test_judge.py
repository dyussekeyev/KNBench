"""Tests for judge parsing and evaluation in knbench.evaluator (TECHSPEC.md §10)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
import respx

from knbench.evaluator import (
    JudgeParseError,
    aggregate_votes,
    evaluate_records,
    normalize_detected,
    parse_judge_response,
)
from knbench.runner import ChatCompletion, FatalAuthError, OpenRouterClient
from knbench.schemas import (
    BenchmarkScenario,
    DocumentVariant,
    EvaluationRecord,
    InjectedRisk,
    JudgeEvaluation,
    TargetModelResponse,
)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def make_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        scenario_id="gcloud-001",
        domain="g_cloud",
        injected_risks=[
            InjectedRisk(
                risk_id="jurisdiction",
                category="jurisdiction",
                description="The operator must hand data to a foreign regulator.",
            ),
            InjectedRisk(
                risk_id="audit_access",
                category="transparency",
                description="No independent audit is guaranteed.",
            ),
        ],
        variants=[
            DocumentVariant(
                variant_id="cn_alibaba",
                vendor_name="Alibaba Cloud",
                origin_country="China",
                document_text="Supplier: Alibaba Cloud (China).",
                vendor_aliases=["Alibaba"],
                country_aliases=["PRC", "CN"],
            )
        ],
        task_prompt="Assess.",
    )


def make_record() -> EvaluationRecord:
    return EvaluationRecord(
        scenario_id="gcloud-001",
        variant_id="cn_alibaba",
        model_name="test/model",
        status="skipped",
        target_response=TargetModelResponse(
            scenario_id="gcloud-001",
            variant_id="cn_alibaba",
            model_name="test/model",
            raw_response="Alibaba Cloud (China) looks acceptable.",
            execution_time_ms=100,
        ),
    )


def judge_payload(
    detected: list[str] | None = None,
    score: float = 40.0,
    recommendation: str = "MODIFY",
) -> dict[str, Any]:
    body = {
        "detected_risks": detected if detected is not None else ["jurisdiction"],
        "risk_score": score,
        "recommendation": recommendation,
        "judge_reasoning": "Because the assessment is partial.",
    }
    return {"choices": [{"message": {"content": json.dumps(body)}}]}


def test_parse_plain_json() -> None:
    evaluation = parse_judge_response(
        '{"detected_risks": ["jurisdiction"], "risk_score": 40, '
        '"recommendation": "MODIFY", "judge_reasoning": "Partial."}'
    )
    assert evaluation.risk_score == 40
    assert evaluation.recommendation == "MODIFY"


def test_parse_strips_markdown_fences() -> None:
    raw = (
        "```json\n"
        '{"detected_risks": [], "risk_score": 80, '
        '"recommendation": "APPROVE", "judge_reasoning": "Fine."}\n'
        "```"
    )
    assert parse_judge_response(raw).recommendation == "APPROVE"


def test_parse_takes_the_outer_object() -> None:
    raw = (
        "Here is my verdict: "
        '{"detected_risks": [], "risk_score": 10, '
        '"recommendation": "REJECT", "judge_reasoning": "Risky."} '
        "Hope this helps."
    )
    assert parse_judge_response(raw).risk_score == 10


def test_parse_rejects_missing_json() -> None:
    with pytest.raises(JudgeParseError) as excinfo:
        parse_judge_response("no json here")
    assert excinfo.value.error_type == "invalid_json"


def test_parse_rejects_schema_violation() -> None:
    with pytest.raises(JudgeParseError) as excinfo:
        parse_judge_response(
            '{"detected_risks": [], "risk_score": 101, '
            '"recommendation": "APPROVE", "judge_reasoning": "Fine."}'
        )
    assert excinfo.value.error_type == "schema"


def test_normalize_matches_ids_and_categories() -> None:
    scenario = make_scenario()
    detected, rejected = normalize_detected(
        [" Jurisdiction ", "TRANSPARENCY", "invented_risk"], scenario
    )
    assert detected == ["jurisdiction", "audit_access"]
    assert rejected == ["invented_risk"]


def test_normalize_rejects_ambiguous_categories() -> None:
    scenario = BenchmarkScenario(
        scenario_id="ambiguous",
        domain="test",
        injected_risks=[
            InjectedRisk(risk_id="a", category="shared", description="A."),
            InjectedRisk(risk_id="b", category="shared", description="B."),
        ],
        variants=[
            DocumentVariant(
                variant_id="v0",
                vendor_name="Vendor",
                origin_country="Country",
                document_text="Text.",
            )
        ],
    )
    detected, rejected = normalize_detected(["shared"], scenario)
    assert detected == []
    assert rejected == ["shared"]


def test_aggregate_votes_uses_median_and_majority() -> None:
    votes = [
        JudgeEvaluation(
            detected_risks=["jurisdiction"],
            risk_score=30,
            recommendation="APPROVE",
            judge_reasoning="First.",
        ),
        JudgeEvaluation(
            detected_risks=["jurisdiction", "audit_access"],
            risk_score=40,
            recommendation="MODIFY",
            judge_reasoning="Second.",
        ),
        JudgeEvaluation(
            detected_risks=["jurisdiction"],
            risk_score=50,
            recommendation="REJECT",
            judge_reasoning="Third.",
        ),
    ]
    merged = aggregate_votes(votes)
    assert merged.risk_score == 40
    assert merged.recommendation == "REJECT"
    assert merged.detected_risks == ["jurisdiction"]
    assert merged.judge_reasoning == "Second."


def test_aggregate_votes_prefers_reject_on_tie() -> None:
    votes = [
        JudgeEvaluation(
            detected_risks=[],
            risk_score=20,
            recommendation="APPROVE",
            judge_reasoning="A.",
        ),
        JudgeEvaluation(
            detected_risks=[],
            risk_score=20,
            recommendation="REJECT",
            judge_reasoning="B.",
        ),
    ]
    assert aggregate_votes(votes).recommendation == "REJECT"


def test_aggregate_votes_falls_through_priority_on_two_way_tie() -> None:
    votes = [
        JudgeEvaluation(
            detected_risks=[],
            risk_score=20,
            recommendation="APPROVE",
            judge_reasoning="A.",
        ),
        JudgeEvaluation(
            detected_risks=[],
            risk_score=20,
            recommendation="MODIFY",
            judge_reasoning="B.",
        ),
    ]
    assert aggregate_votes(votes).recommendation == "MODIFY"


class BlockingJudgeClient(OpenRouterClient):
    """Answer the first judge call, then block so cancellation can be tested."""

    def __init__(self) -> None:
        super().__init__("test-key")
        self.calls = 0
        self.started = asyncio.Event()

    async def complete(self, **kwargs: Any) -> ChatCompletion:
        self.calls += 1
        if self.calls == 1:
            body = {
                "detected_risks": ["jurisdiction"],
                "risk_score": 40,
                "recommendation": "MODIFY",
                "judge_reasoning": "Partial.",
            }
            return ChatCompletion(status_code=200, content=json.dumps(body))
        self.started.set()
        await asyncio.Event().wait()
        return ChatCompletion(status_code=200, content="unreachable")


class ExplodingJudgeClient(OpenRouterClient):
    """Raise an unexpected exception from the HTTP client."""

    def __init__(self) -> None:
        super().__init__("test-key")

    async def complete(self, **kwargs: Any) -> ChatCompletion:
        raise RuntimeError("unexpected client failure")


class FirstSuccessThenAuthClient(OpenRouterClient):
    """Return one valid verdict, then fail authentication."""

    def __init__(self) -> None:
        super().__init__("test-key")
        self.calls = 0

    async def complete(self, **kwargs: Any) -> ChatCompletion:
        self.calls += 1
        if self.calls == 1:
            body = {
                "detected_risks": ["jurisdiction"],
                "risk_score": 40,
                "recommendation": "MODIFY",
                "judge_reasoning": "Partial.",
            }
            return ChatCompletion(status_code=200, content=json.dumps(body))
        raise FatalAuthError("judge credentials rejected")


async def test_judge_cancellation_publishes_completed_evaluations() -> None:
    client = BlockingJudgeClient()
    snapshots: list[list[EvaluationRecord]] = []

    def capture(records: list[EvaluationRecord]) -> None:
        snapshots.append([record.model_copy(deep=True) for record in records])

    task = asyncio.create_task(
        evaluate_records(
            [make_record(), make_record()],
            [make_scenario()],
            client=client,
            semaphore=asyncio.Semaphore(1),
            judge_model="judge/model",
            on_result=capture,
        )
    )
    try:
        await asyncio.wait_for(client.started.wait(), timeout=5)
        for _ in range(100):
            if snapshots and snapshots[0][0].status == "ok":
                break
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await client.aclose()

    assert snapshots
    last = snapshots[-1]
    assert last[0].status == "ok"
    assert last[0].evaluation is not None
    assert last[1].status == "skipped"


async def test_unexpected_judge_error_becomes_judge_failed() -> None:
    client = ExplodingJudgeClient()
    try:
        records = await evaluate_records(
            [make_record()],
            [make_scenario()],
            client=client,
            semaphore=asyncio.Semaphore(1),
            judge_model="judge/model",
        )
    finally:
        await client.aclose()

    record = records[0]
    assert record.status == "judge_failed"
    assert record.evaluation is None
    assert record.errors[0].error_type == "internal"
    assert "unexpected client failure" not in record.errors[0].message


async def test_judge_auth_failure_publishes_completed_evaluations() -> None:
    client = FirstSuccessThenAuthClient()
    snapshots: list[list[EvaluationRecord]] = []

    def capture(records: list[EvaluationRecord]) -> None:
        snapshots.append([record.model_copy(deep=True) for record in records])

    try:
        with pytest.raises(FatalAuthError):
            await evaluate_records(
                [make_record(), make_record()],
                [make_scenario()],
                client=client,
                semaphore=asyncio.Semaphore(1),
                judge_model="judge/model",
                on_result=capture,
            )
    finally:
        await client.aclose()

    assert snapshots
    assert snapshots[-1][0].status == "ok"
    assert snapshots[-1][1].status == "skipped"


async def test_evaluate_records_marks_ok_and_fills_meta() -> None:
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=judge_payload())
            )
            records = await evaluate_records(
                [make_record()],
                [make_scenario()],
                client=client,
                semaphore=asyncio.Semaphore(1),
                judge_model="judge/model",
                target_models=["test/model"],
            )
    finally:
        await client.aclose()

    record = records[0]
    assert record.status == "ok"
    assert record.evaluation is not None
    assert record.evaluation.detected_risks == ["jurisdiction"]
    meta = record.evaluation.judge_meta
    assert meta is not None
    assert meta.votes == 1
    assert meta.attempts == 1
    assert meta.redactions["[VENDOR]"] >= 1
    assert meta.rejected_ids == []
    assert len(meta.anonymized_prompt_sha256) == 64

    payload = json.loads(route.calls[0].request.content)
    assert payload["response_format"] == {"type": "json_object"}
    assert "0-100" in payload["messages"][0]["content"]
    serialized = json.dumps(payload)
    assert "Alibaba" not in serialized
    assert "China" not in serialized
    assert "test/model" not in serialized


async def test_judge_repairs_invalid_json() -> None:
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                side_effect=[
                    httpx.Response(
                        200, json={"choices": [{"message": {"content": "not json"}}]}
                    ),
                    httpx.Response(200, json=judge_payload()),
                ]
            )
            records = await evaluate_records(
                [make_record()],
                [make_scenario()],
                client=client,
                semaphore=asyncio.Semaphore(1),
                judge_model="judge/model",
            )
    finally:
        await client.aclose()

    assert records[0].status == "ok"
    assert records[0].evaluation is not None
    assert records[0].evaluation.judge_meta is not None
    assert records[0].evaluation.judge_meta.attempts == 2
    assert route.call_count == 2


async def test_judge_fails_after_two_repairs() -> None:
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(
                    200, json={"choices": [{"message": {"content": "still not json"}}]}
                )
            )
            records = await evaluate_records(
                [make_record()],
                [make_scenario()],
                client=client,
                semaphore=asyncio.Semaphore(1),
                judge_model="judge/model",
            )
    finally:
        await client.aclose()

    record = records[0]
    assert record.status == "judge_failed"
    assert record.evaluation is None
    assert record.errors[0].error_type == "invalid_json"
    assert record.errors[0].attempts == 3
    assert route.call_count == 3


async def test_invented_risk_ids_are_rejected() -> None:
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            respx.post(CHAT_URL).mock(
                return_value=httpx.Response(
                    200, json=judge_payload(detected=["invented_risk"])
                )
            )
            records = await evaluate_records(
                [make_record()],
                [make_scenario()],
                client=client,
                semaphore=asyncio.Semaphore(1),
                judge_model="judge/model",
            )
    finally:
        await client.aclose()

    evaluation = records[0].evaluation
    assert evaluation is not None
    assert evaluation.detected_risks == []
    assert evaluation.judge_meta is not None
    assert evaluation.judge_meta.rejected_ids == ["invented_risk"]


async def test_multiple_votes_are_aggregated() -> None:
    responses = [
        judge_payload(["jurisdiction"], 30, "APPROVE"),
        judge_payload(["jurisdiction", "audit_access"], 40, "REJECT"),
        judge_payload(["jurisdiction"], 50, "MODIFY"),
    ]
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                side_effect=[httpx.Response(200, json=item) for item in responses]
            )
            records = await evaluate_records(
                [make_record()],
                [make_scenario()],
                client=client,
                semaphore=asyncio.Semaphore(1),
                judge_model="judge/model",
                judge_votes=3,
            )
    finally:
        await client.aclose()

    evaluation = records[0].evaluation
    assert evaluation is not None
    assert evaluation.risk_score == 40
    assert evaluation.recommendation == "REJECT"
    assert evaluation.detected_risks == ["jurisdiction"]
    assert evaluation.judge_meta is not None
    assert evaluation.judge_meta.votes == 3
    assert route.call_count == 3


async def test_judge_auth_failure_aborts() -> None:
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            respx.post(CHAT_URL).mock(return_value=httpx.Response(401))
            with pytest.raises(FatalAuthError):
                await evaluate_records(
                    [make_record()],
                    [make_scenario()],
                    client=client,
                    semaphore=asyncio.Semaphore(1),
                    judge_model="judge/model",
                )
    finally:
        await client.aclose()


async def test_records_without_responses_are_untouched() -> None:
    failed = EvaluationRecord(
        scenario_id="gcloud-001",
        variant_id="cn_alibaba",
        model_name="test/model",
        status="target_failed",
    )
    client = OpenRouterClient("test-key")
    try:
        with respx.mock:
            route = respx.post(CHAT_URL).mock(
                return_value=httpx.Response(200, json=judge_payload())
            )
            records = await evaluate_records(
                [failed],
                [make_scenario()],
                client=client,
                semaphore=asyncio.Semaphore(1),
                judge_model="judge/model",
            )
    finally:
        await client.aclose()

    assert records[0].status == "target_failed"
    assert route.call_count == 0
