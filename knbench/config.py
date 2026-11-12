"""Configuration, secrets and logging for KNBench.

Precedence follows TECHSPEC.md §12.1: command-line values applied by the CLI
layer override environment variables, which override the ``.env`` file, which
overrides the built-in defaults defined below.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL: Final[str] = "anthropic/claude-opus-5.5"
DEFAULT_TARGET_MODELS: Final[str] = "openai/gpt-6.1-sol,google/gemini-4-carbon"
DEFAULT_APP_NAME: Final[str] = "KNBench"
DEFAULT_OUTPUT_DIR: Final[Path] = Path("output")
LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s %(message)s"

_HANDLER_MARK: Final[str] = "_knbench_handler"
_QUIET_LOGGERS: Final[tuple[str, ...]] = ("httpx", "httpcore")

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def parse_model_list(raw: str | None) -> list[str]:
    """Split a comma-separated model list, dropping blanks and duplicates."""
    if raw is None:
        return []
    seen: set[str] = set()
    models: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name not in seen:
            seen.add(name)
            models.append(name)
    return models


class Settings(BaseSettings):
    """Environment-backed configuration (TECHSPEC.md §14.1)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openrouter_api_key: str | None = None
    openrouter_base_url: str = DEFAULT_BASE_URL
    openrouter_app_name: str = DEFAULT_APP_NAME
    openrouter_app_url: str | None = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    target_models: str = DEFAULT_TARGET_MODELS
    log_level: LogLevel = "INFO"
    output_dir: Path = DEFAULT_OUTPUT_DIR

    @property
    def models(self) -> list[str]:
        """Target models from ``TARGET_MODELS`` as an ordered list."""
        return parse_model_list(self.target_models)

    def log_file(self, run_id: str) -> Path:
        """Default log path for a run: ``<output_dir>/logs/<run_id>.log``."""
        return self.output_dir / "logs" / f"{run_id}.log"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()


def require_api_key(settings: Settings | None = None) -> str:
    """Return the OpenRouter key or raise a fatal configuration error."""
    resolved = settings if settings is not None else get_settings()
    key = (resolved.openrouter_api_key or "").strip()
    if not key:
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return key


def _resolve_level(level: str) -> int:
    resolved = logging.getLevelName(level.strip().upper())
    if not isinstance(resolved, int):
        raise ConfigError(f"Unknown log level: {level!r}")
    return resolved


def _clear_own_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()


def _attach(handler: logging.Handler, formatter: logging.Formatter, level: int) -> logging.Handler:
    handler.setFormatter(formatter)
    handler.setLevel(level)
    setattr(handler, _HANDLER_MARK, True)
    logging.getLogger().addHandler(handler)
    return handler


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure logging to stdout and, optionally, to a UTF-8 file.

    Implements TECHSPEC.md §14.3: one shared format, UTF-8 file output and
    ``httpx``/``httpcore`` loggers raised to WARNING so headers cannot leak.
    Handlers installed by earlier calls are replaced; handlers owned by other
    tooling (for example pytest's capture) are left untouched.
    """
    numeric_level = _resolve_level(level)
    root = logging.getLogger()
    _clear_own_handlers(root)
    root.setLevel(numeric_level)

    formatter = logging.Formatter(LOG_FORMAT)
    _attach(logging.StreamHandler(stream=sys.stdout), formatter, numeric_level)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _attach(logging.FileHandler(log_file, encoding="utf-8"), formatter, numeric_level)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
