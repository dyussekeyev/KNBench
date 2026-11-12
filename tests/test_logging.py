"""Tests for knbench.config.setup_logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from knbench.config import ConfigError, setup_logging


def _flush_root() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_writes_to_file_in_utf8(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "run.log"
    setup_logging("INFO", log_file)
    logging.getLogger("knbench.test").info("dataset loaded")
    _flush_root()
    assert "dataset loaded" in log_file.read_text(encoding="utf-8")


def test_httpx_loggers_are_quieted() -> None:
    setup_logging("DEBUG")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_own_handlers_are_not_duplicated(tmp_path: Path) -> None:
    setup_logging("INFO", tmp_path / "first.log")
    setup_logging("INFO", tmp_path / "second.log")
    own = [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_knbench_handler", False)
    ]
    assert len(own) == 2  # one stream handler plus the current file handler


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown log level"):
        setup_logging("LOUD")


def test_api_key_never_reaches_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-secret")
    log_file = tmp_path / "run.log"
    setup_logging("DEBUG", log_file)
    logging.getLogger("knbench.runner").debug("request prepared for openai/gpt-4o")
    _flush_root()
    assert "sk-or-v1-test-secret" not in log_file.read_text(encoding="utf-8")
