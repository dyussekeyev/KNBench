"""Tests for knbench.config settings and the API key guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from knbench.config import (
    ConfigError,
    Settings,
    parse_model_list,
    require_api_key,
)


def test_defaults_do_not_require_a_dotenv_file() -> None:
    settings = Settings(_env_file=None)
    assert settings.judge_model == "anthropic/claude-opus-5.5"
    assert settings.models == [
        "openai/gpt-6.1-sol",
        "google/gemini-4-carbon",
    ]
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_app_name == "KNBench"
    assert settings.output_dir == Path("output")
    assert settings.log_level == "INFO"


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("OUTPUT_DIR", "artifacts")
    settings = Settings(_env_file=None)
    assert settings.judge_model == "openai/gpt-4o"
    assert settings.output_dir == Path("artifacts")


def test_parse_model_list_trims_and_deduplicates() -> None:
    raw = " openai/gpt-4o , deepseek/deepseek-r1,, openai/gpt-4o "
    assert parse_model_list(raw) == ["openai/gpt-4o", "deepseek/deepseek-r1"]


def test_parse_model_list_handles_empty_input() -> None:
    assert parse_model_list(None) == []
    assert parse_model_list("  ,  ") == []


def test_models_property_reads_target_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARGET_MODELS", "openai/gpt-4o,deepseek/deepseek-r1")
    settings = Settings(_env_file=None)
    assert settings.models == ["openai/gpt-4o", "deepseek/deepseek-r1"]


def test_log_file_uses_output_dir() -> None:
    settings = Settings(_env_file=None, output_dir=Path("results"))
    assert settings.log_file("run-1") == Path("results/logs/run-1.log")


def test_require_api_key_rejects_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        require_api_key(settings)


def test_require_api_key_rejects_blank_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings(_env_file=None, openrouter_api_key="   ")
    with pytest.raises(ConfigError):
        require_api_key(settings)


def test_require_api_key_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings(_env_file=None, openrouter_api_key="sk-or-v1-test")
    assert require_api_key(settings) == "sk-or-v1-test"
