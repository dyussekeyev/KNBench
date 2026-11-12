"""Tests for knbench.main (TECHSPEC.md §12)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from knbench.config import Settings, get_settings
from knbench.data_loader import load_scenarios
from knbench.main import _execute, main
from knbench.provenance import ProvenanceFile, provenance_path
from knbench.runner import (
    ChatCompletion,
    ModelCatalog,
    OpenRouterClient,
    load_results,
    write_results_atomic,
)
from knbench.schemas import (
    BenchmarkScenario,
    DocumentVariant,
    EvaluationRecord,
    InjectedRisk,
    JudgeEvaluation,
    MetricsFile,
    RunConfig,
    RunResults,
    TargetModelResponse,
)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "openrouter_api_key": "test-key",
        "judge_model": "judge/model",
        "target_models": "openai/gpt-6.1-sol,google/gemini-4-carbon",
        "log_level": "INFO",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(autouse=True)
def default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("knbench.main.get_settings", lambda: _settings())


def write_dataset(path: Path, *, variant_count: int = 2) -> Path:
    variants = [
        {
            "variant_id": f"v{index}",
            "vendor_name": f"Vendor{index}",
            "origin_country": f"Country{index}",
            "document_text": f"Vendor{index} stores data locally.",
            "is_control": index == 0,
        }
        for index in range(variant_count)
    ]
    payload = {
        "schema_version": "1.0",
        "defaults": {"task_prompt": "Assess."},
        "scenarios": [
            {
                "scenario_id": "test-1",
                "domain": "g_cloud",
                "injected_risks": [
                    {
                        "risk_id": "jurisdiction",
                        "category": "jurisdiction",
                        "description": "A foreign regulator may demand data.",
                    }
                ],
                "variants": variants,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def scenario(variant_count: int = 2) -> BenchmarkScenario:
    return BenchmarkScenario(
        scenario_id="test-1",
        domain="g_cloud",
        injected_risks=[
            InjectedRisk(
                risk_id="jurisdiction",
                category="jurisdiction",
                description="A foreign regulator may demand data.",
            )
        ],
        variants=[
            DocumentVariant(
                variant_id=f"v{index}",
                vendor_name=f"Vendor{index}",
                origin_country=f"Country{index}",
                document_text=f"Vendor{index} stores data locally.",
                is_control=index == 0,
            )
            for index in range(variant_count)
        ],
        task_prompt="Assess.",
    )


def ok_record(variant_id: str, *, score: float = 40.0) -> EvaluationRecord:
    return EvaluationRecord(
        scenario_id="test-1",
        variant_id=variant_id,
        model_name="test/model",
        status="ok",
        target_response=TargetModelResponse(
            scenario_id="test-1",
            variant_id=variant_id,
            model_name="test/model",
            raw_response="Assessment.",
            execution_time_ms=10,
        ),
        evaluation=JudgeEvaluation(
            detected_risks=["jurisdiction"],
            risk_score=score,
            recommendation="MODIFY",
            judge_reasoning="Partial.",
        ),
    )


def write_results(path: Path, *, variant_count: int = 2, judged: bool = True) -> Path:
    records = [
        ok_record(f"v{index}", score=40.0 + index * 10) if judged else EvaluationRecord(
            scenario_id="test-1",
            variant_id=f"v{index}",
            model_name="test/model",
            status="skipped",
        )
        for index in range(variant_count)
    ]
    results = RunResults(
        run_id="run-cli",
        started_at=datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 11, 10, 10, 1, tzinfo=timezone.utc),
        run_config=RunConfig(
            models=["test/model"],
            judge_model="judge/model",
            concurrency=1,
            max_retries=1,
            timeout_s=5.0,
            judge_votes=1,
            anonymize=True,
            scenarios_path="dataset.json",
            scenarios_sha256="0" * 64,
            app_version="0.1.0",
        ),
        scenarios=[scenario(variant_count)],
        records=records,
    )
    write_results_atomic(path, results)
    return path


def chat_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    if payload["model"] == "judge/model":
        content = json.dumps(
            {
                "detected_risks": ["jurisdiction"],
                "risk_score": 40.0,
                "recommendation": "MODIFY",
                "judge_reasoning": "Partial.",
            }
        )
    else:
        content = "The supplier looks acceptable."
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def catalog_response(*model_ids: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"id": model_id} for model_id in model_ids]},
    )


# ---------- validate ----------


def test_validate_accepts_a_valid_dataset(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    assert main(["validate", "--scenarios", str(dataset)]) == 0


def test_validate_rejects_malformed_json(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main(["validate", "--scenarios", str(broken)]) == 1


def test_invalid_log_level_exits_1_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    monkeypatch.setenv("LOG_LEVEL", "BOGUS")
    monkeypatch.setattr("knbench.main.get_settings", get_settings)
    get_settings.cache_clear()
    caplog.set_level(logging.ERROR, logger="knbench.main")
    try:
        code = main(["validate", "--scenarios", str(dataset)])
    finally:
        get_settings.cache_clear()

    assert code == 1
    assert "invalid configuration" in caplog.text
    assert "LOG_LEVEL" in caplog.text
    assert "Traceback" not in caplog.text


def test_unwritable_log_file_exits_1(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    log_dir = tmp_path / "logs-dir"
    log_dir.mkdir()
    caplog.set_level(logging.ERROR, logger="knbench.main")

    code = main(
        [
            "validate",
            "--scenarios",
            str(dataset),
            "--log-file",
            str(log_dir),
        ]
    )

    assert code == 1
    assert "cannot configure logging" in caplog.text


# ---------- run: plan and usage errors ----------


def test_dry_run_needs_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("knbench.main.get_settings", lambda: _settings(openrouter_api_key=None))
    dataset = write_dataset(tmp_path / "dataset.json")
    code = main(
        [
            "run",
            "--models",
            "test/model",
            "--scenarios",
            str(dataset),
            "--dry-run",
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 0


def test_data_and_scenarios_are_equivalent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    base = [
        "run",
        "--models",
        "test/model",
        "--dry-run",
        "--log-file",
        str(tmp_path / "run.log"),
    ]

    caplog.set_level(logging.INFO, logger="knbench.main")
    assert main(base + ["--scenarios", str(dataset)]) == 0
    first = [record.getMessage() for record in caplog.records if record.name == "knbench.main"]

    caplog.clear()
    assert main(base + ["--data", str(dataset)]) == 0
    second = [record.getMessage() for record in caplog.records if record.name == "knbench.main"]

    assert first == second


def test_passing_both_aliases_is_rejected(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    code = main(
        [
            "run",
            "--models",
            "test/model",
            "--scenarios",
            str(dataset),
            "--data",
            str(dataset),
            "--dry-run",
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 1


def test_out_of_range_concurrency_is_a_configuration_error(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    assert (
        main(
            [
                "run",
                "--models",
                "test/model",
                "--scenarios",
                str(dataset),
                "--concurrency",
                "0",
            ]
        )
        == 1
    )


def test_missing_models_is_a_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    settings = _settings(target_models="")
    monkeypatch.setattr("knbench.main.get_settings", lambda: settings)
    assert (
        main(
            [
                "run",
                "--scenarios",
                str(dataset),
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )
        == 1
    )


def test_missing_api_key_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("knbench.main.get_settings", lambda: _settings(openrouter_api_key=None))
    dataset = write_dataset(tmp_path / "dataset.json")
    output = tmp_path / "results.json"
    code = main(
        [
            "run",
            "--models",
            "test/model",
            "--scenarios",
            str(dataset),
            "--output",
            str(output),
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 3
    assert not output.exists()


def test_unwritable_run_output_exits_1(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    output_dir = tmp_path / "results-dir"
    output_dir.mkdir()
    caplog.set_level(logging.ERROR, logger="knbench.main")

    with respx.mock:
        respx.get(MODELS_URL).mock(
            return_value=catalog_response("test/model", "judge/model")
        )
        respx.post(CHAT_URL).mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "Assessment."}}]}
            )
        )
        code = main(
            [
                "run",
                "--models",
                "test/model",
                "--scenarios",
                str(dataset),
                "--output",
                str(output_dir),
                "--skip-judge",
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )

    assert code == 1
    assert "cannot write results" in caplog.text


# ---------- run: exit codes ----------


def test_run_end_to_end_exits_0(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    output = tmp_path / "results.json"
    with respx.mock:
        respx.get(MODELS_URL).mock(
            return_value=catalog_response("test/model", "judge/model")
        )
        respx.post(CHAT_URL).mock(side_effect=chat_response)
        code = main(
            [
                "run",
                "--models",
                "test/model",
                "--scenarios",
                str(dataset),
                "--output",
                str(output),
                "--judge-model",
                "judge/model",
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )
    assert code == 0
    results = load_results(output)
    assert len(results.records) == 2
    assert all(record.status == "ok" for record in results.records)
    assert all(record.evaluation is not None for record in results.records)
    sidecar_path = provenance_path(output)
    assert sidecar_path.is_file()
    sidecar = ProvenanceFile.model_validate_json(
        sidecar_path.read_text(encoding="utf-8")
    )
    assert sidecar.run_id == results.run_id
    assert sidecar.dataset_sha256 == results.run_config.scenarios_sha256
    # §18: the dataset digest is stored in the run configuration.
    assert results.run_config.scenarios_sha256 == load_scenarios(dataset)[1]


def test_run_rejects_a_missing_target_model_before_any_target_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    output = tmp_path / "results.json"
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=catalog_response("judge/model"))
        chat = respx.post(CHAT_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
        code = main(
            [
                "run",
                "--models",
                "missing/target",
                "--scenarios",
                str(dataset),
                "--output",
                str(output),
                "--judge-model",
                "judge/model",
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )

    assert code == 1
    assert chat.call_count == 0
    assert not output.exists()
    assert "missing/target" in caplog.text


def test_run_rejects_a_missing_judge_model_before_any_target_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    output = tmp_path / "results.json"
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=catalog_response("test/model"))
        chat = respx.post(CHAT_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
        code = main(
            [
                "run",
                "--models",
                "test/model",
                "--scenarios",
                str(dataset),
                "--output",
                str(output),
                "--judge-model",
                "missing/judge",
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )

    assert code == 1
    assert chat.call_count == 0
    assert not output.exists()
    assert "missing/judge" in caplog.text


def test_run_fails_closed_when_the_catalog_is_unavailable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    output = tmp_path / "results.json"
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(503))
        chat = respx.post(CHAT_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
        code = main(
            [
                "run",
                "--models",
                "test/model",
                "--scenarios",
                str(dataset),
                "--output",
                str(output),
                "--judge-model",
                "judge/model",
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )

    assert code == 1
    assert chat.call_count == 0
    assert not output.exists()
    assert "HTTP 503" in caplog.text


class JudgeCheckpointClient(OpenRouterClient):
    """Collect target responses and let the second judge call block."""

    def __init__(self) -> None:
        super().__init__("test-key")
        self.judge_calls = 0
        self.started = asyncio.Event()

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
        if model != "judge/model":
            return ChatCompletion(status_code=200, content="Assessment.")
        self.judge_calls += 1
        if self.judge_calls == 1:
            body = {
                "detected_risks": [],
                "risk_score": 40,
                "recommendation": "MODIFY",
                "judge_reasoning": "Partial.",
            }
            return ChatCompletion(status_code=200, content=json.dumps(body))
        self.started.set()
        await asyncio.Event().wait()
        return ChatCompletion(status_code=200, content="unreachable")

    async def aclose(self) -> None:
        return None

    async def require_models(
        self, model_ids: list[str], *, timeout_s: float = 30.0
    ) -> ModelCatalog:
        return ModelCatalog(frozenset(model_ids), "2026-11-11")


async def test_cancellation_during_judging_keeps_completed_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    scenarios = [scenario(2)]
    _, digest = load_scenarios(dataset)
    output = tmp_path / "results.json"
    config = RunConfig(
        models=["test/model"],
        judge_model="judge/model",
        concurrency=2,
        max_retries=0,
        timeout_s=5.0,
        judge_votes=1,
        anonymize=True,
        scenarios_path=str(dataset),
        scenarios_sha256=digest,
        app_version="0.1.0",
    )
    args = argparse.Namespace(output=str(output), skip_judge=False)
    client = JudgeCheckpointClient()
    monkeypatch.setattr(
        "knbench.main.OpenRouterClient.from_settings",
        lambda settings: client,
    )

    task = asyncio.create_task(
        _execute(
            args,
            _settings(),
            scenarios,
            ["test/model"],
            config,
            None,
            "run-judge-cancel",
        )
    )
    try:
        await asyncio.wait_for(client.started.wait(), timeout=5)
        for _ in range(200):
            if output.exists():
                current = load_results(output)
                if any(record.status == "ok" for record in current.records):
                    break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await client.aclose()

    loaded = load_results(output)
    statuses = [record.status for record in loaded.records]
    assert statuses.count("ok") == 1
    assert statuses.count("skipped") == 1
    assert loaded.finished_at is None


def test_failed_target_records_exit_2(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    output = tmp_path / "results.json"
    with respx.mock:
        respx.get(MODELS_URL).mock(
            return_value=catalog_response("test/model", "judge/model")
        )
        respx.post(CHAT_URL).mock(return_value=httpx.Response(400))
        code = main(
            [
                "run",
                "--models",
                "test/model",
                "--scenarios",
                str(dataset),
                "--output",
                str(output),
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )
    assert code == 2
    results = load_results(output)
    assert all(record.status == "target_failed" for record in results.records)


def test_resume_on_a_complete_file_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("knbench.main.get_settings", lambda: _settings(openrouter_api_key=None))
    dataset = write_dataset(tmp_path / "dataset.json")
    output = write_results(tmp_path / "results.json")
    code = main(
        [
            "run",
            "--models",
            "test/model",
            "--scenarios",
            str(dataset),
            "--output",
            str(output),
            "--resume",
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 4


def test_unauthorized_target_exits_3_without_retry(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset.json", variant_count=1)
    output = tmp_path / "results.json"
    with respx.mock:
        respx.get(MODELS_URL).mock(
            return_value=catalog_response("test/model", "judge/model")
        )
        route = respx.post(CHAT_URL).mock(return_value=httpx.Response(401))
        code = main(
            [
                "run",
                "--models",
                "test/model",
                "--scenarios",
                str(dataset),
                "--output",
                str(output),
                "--max-retries",
                "4",
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )
    assert code == 3
    assert route.call_count == 1


def test_resume_with_a_new_model_exits_1(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    output = write_results(tmp_path / "results.json")
    code = main(
        [
            "run",
            "--models",
            "other/model",
            "--scenarios",
            str(dataset),
            "--output",
            str(output),
            "--resume",
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 1


def test_resume_with_changed_dataset_exits_1(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "dataset.json")
    output = write_results(tmp_path / "results.json")
    loaded = load_results(output)
    changed_scenario = loaded.scenarios[0].model_copy(
        update={
            "variants": [
                variant.model_copy(update={"document_text": "Changed document."})
                for variant in loaded.scenarios[0].variants
            ]
        }
    )
    write_results_atomic(
        output,
        loaded.model_copy(update={"scenarios": [changed_scenario]}),
    )

    code = main(
        [
            "run",
            "--models",
            "test/model",
            "--scenarios",
            str(dataset),
            "--output",
            str(output),
            "--resume",
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 1


def test_resume_with_limit_filters_the_output_file(tmp_path: Path) -> None:
    scenarios, digest = load_scenarios(Path("data/scenarios.json"))
    first = scenarios[0]
    records = [
        EvaluationRecord(
            scenario_id=scenario.scenario_id,
            variant_id=variant.variant_id,
            model_name="test/model",
            status="ok",
        )
        for scenario in scenarios
        for variant in scenario.variants
    ]
    # Leave one selected triple pending so the limit path actually runs.
    missing = (first.scenario_id, first.variants[0].variant_id)
    records = [
        record
        for record in records
        if (record.scenario_id, record.variant_id) != missing
    ]
    output = tmp_path / "results.json"
    write_results_atomic(
        output,
        RunResults(
            run_id="previous-limit-run",
            started_at=datetime.now(timezone.utc),
            run_config=RunConfig(
                models=["test/model"],
                judge_model="judge/model",
                concurrency=1,
                max_retries=1,
                timeout_s=5.0,
                judge_votes=1,
                anonymize=True,
                scenarios_path="data/scenarios.json",
                scenarios_sha256=digest,
                app_version="0.1.0",
            ),
            scenarios=scenarios,
            records=records,
        ),
    )

    with respx.mock:
        respx.get(MODELS_URL).mock(
            return_value=catalog_response("test/model", "judge/model")
        )
        route = respx.post(CHAT_URL).mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "Assessment."}}]}
            )
        )
        code = main(
            [
                "run",
                "--models",
                "test/model",
                "--scenarios",
                "data/scenarios.json",
                "--output",
                str(output),
                "--resume",
                "--limit",
                "1",
                "--skip-judge",
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )

    assert code == 0
    assert route.call_count == 1
    result = load_results(output)
    assert len(result.scenarios) == 1
    assert result.scenarios[0].scenario_id == first.scenario_id
    assert len(result.records) == len(first.variants)
    assert {record.scenario_id for record in result.records} == {first.scenario_id}


def test_resume_with_limit_rewrites_a_complete_selection(tmp_path: Path) -> None:
    scenarios, digest = load_scenarios(Path("data/scenarios.json"))
    first = scenarios[0]
    records = [
        EvaluationRecord(
            scenario_id=scenario.scenario_id,
            variant_id=variant.variant_id,
            model_name="test/model",
            status="ok",
        )
        for scenario in scenarios
        for variant in scenario.variants
    ]
    output = tmp_path / "results.json"
    write_results_atomic(
        output,
        RunResults(
            run_id="previous-complete-limit",
            started_at=datetime.now(timezone.utc),
            run_config=RunConfig(
                models=["test/model"],
                judge_model="judge/model",
                concurrency=1,
                max_retries=1,
                timeout_s=5.0,
                judge_votes=1,
                anonymize=True,
                scenarios_path="data/scenarios.json",
                scenarios_sha256=digest,
                app_version="0.1.0",
            ),
            scenarios=scenarios,
            records=records,
        ),
    )

    code = main(
        [
            "run",
            "--models",
            "test/model",
            "--scenarios",
            "data/scenarios.json",
            "--output",
            str(output),
            "--resume",
            "--limit",
            "1",
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )

    assert code == 4
    result = load_results(output)
    assert [scenario.scenario_id for scenario in result.scenarios] == [first.scenario_id]
    assert len(result.records) == len(first.variants)
    assert {record.scenario_id for record in result.records} == {first.scenario_id}


# ---------- report ----------


def test_report_writes_eight_sections(tmp_path: Path) -> None:
    results = write_results(tmp_path / "results.json")
    report = tmp_path / "report.md"
    code = main(
        [
            "report",
            "--results",
            str(results),
            "--output",
            str(report),
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 0
    text = report.read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert len(headings) == 8
    assert headings[0] == "## 1. Run header"
    assert headings[-1] == "## 8. Limitations"
    assert "not a security opinion" in text


def test_report_json_is_a_metrics_file(tmp_path: Path) -> None:
    results = write_results(tmp_path / "results.json")
    output = tmp_path / "metrics.json"
    code = main(
        [
            "report",
            "--results",
            str(results),
            "--format",
            "json",
            "--output",
            str(output),
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 0
    metrics = MetricsFile.model_validate_json(output.read_text(encoding="utf-8"))
    assert metrics.run_id == "run-cli"


def test_report_strict_exits_2_on_undefined_metrics(tmp_path: Path) -> None:
    results = write_results(tmp_path / "results.json", variant_count=1)
    report = tmp_path / "report.md"
    code = main(
        [
            "report",
            "--results",
            str(results),
            "--output",
            str(report),
            "--strict",
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 2
    assert report.exists()


def test_report_missing_results_exits_1(tmp_path: Path) -> None:
    code = main(
        [
            "report",
            "--results",
            str(tmp_path / "absent.json"),
            "--output",
            str(tmp_path / "report.md"),
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 1


def test_report_rejects_malformed_results(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    caplog.set_level(logging.ERROR, logger="knbench.main")
    code = main(
        [
            "report",
            "--results",
            str(broken),
            "--output",
            str(tmp_path / "report.md"),
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )
    assert code == 1
    assert "invalid JSON" in caplog.text


def test_report_unwritable_output_exits_1(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    results = write_results(tmp_path / "results.json")
    output_dir = tmp_path / "report-dir"
    output_dir.mkdir()
    caplog.set_level(logging.ERROR, logger="knbench.main")

    code = main(
        [
            "report",
            "--results",
            str(results),
            "--output",
            str(output_dir),
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )

    assert code == 1
    assert "cannot write report" in caplog.text


# ---------- parser ----------


def test_version_prints_the_project_name(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert "KNBench 0.1.0" in capsys.readouterr().out


def test_missing_command_is_a_configuration_error() -> None:
    assert main([]) == 1


def test_run_help_lists_the_contract_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--help"]) == 0
    output = capsys.readouterr().out
    for flag in (
        "--models",
        "--scenarios",
        "--data",
        "--output",
        "--judge-model",
        "--judge-votes",
        "--concurrency",
        "--max-retries",
        "--timeout",
        "--limit",
        "--strict-pairs",
        "--resume",
        "--skip-judge",
        "--no-anonymize",
        "--dry-run",
    ):
        assert flag in output
