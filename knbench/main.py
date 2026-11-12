"""Command-line interface for KNBench (TECHSPEC.md §12).

``main`` returns the exit code instead of calling ``sys.exit`` so the whole
contract can be tested in-process.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

if __package__ in {None, ""}:
    # Support the documented `python knbench/main.py` form (§12); with
    # `python -m knbench.main` the package is importable already.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knbench import __version__
from knbench.config import ConfigError, Settings, get_settings, parse_model_list, require_api_key, setup_logging
from knbench.data_loader import DatasetError, load_scenarios
from knbench.evaluator import evaluate_records
from knbench.metrics import build_metrics_file
from knbench.provenance import write_provenance
from knbench.report import has_undefined_metrics, render_markdown, render_metrics_json
from knbench.runner import (
    CatalogError,
    FatalAuthError,
    OpenRouterClient,
    ResumeError,
    load_results,
    new_run_id,
    prepare_resume,
    run_benchmark,
    write_results_atomic,
)
from knbench.schemas import BenchmarkScenario, EvaluationRecord, RunConfig, RunResults

logger = logging.getLogger(__name__)

DEFAULT_SCENARIOS: Final = "data/scenarios.json"
DEFAULT_RESULTS: Final = "output/raw_results.json"
DEFAULT_REPORT: Final = "output/report.md"
DEFAULT_METRICS_JSON: Final = "output/metrics.json"

_LOG_LEVELS: Final = ("DEBUG", "INFO", "WARNING", "ERROR")
_FAILED_STATUSES: Final = frozenset({"target_failed", "judge_failed"})


class _ScenarioAliasAction(argparse.Action):
    """Store the value and remember which alias was used (§12.2)."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        used: list[str] = list(getattr(namespace, "_scenario_flags", None) or [])
        used.append(option_string or "")
        setattr(namespace, "_scenario_flags", used)
        setattr(namespace, self.dest, values)


def _bounded_int(low: int, high: int | None, label: str) -> Callable[[str], int]:
    """Build an argparse type that rejects out-of-range integers."""

    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
        if value < low or (high is not None and value > high):
            expected = f"between {low} and {high}" if high is not None else f">= {low}"
            raise argparse.ArgumentTypeError(f"{label} must be {expected}, got {value}")
        return value

    return parse


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"timeout must be > 0, got {raw}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser tree of §12."""
    global_options = argparse.ArgumentParser(add_help=False)
    global_options.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default=argparse.SUPPRESS,
        help="logging level (default: LOG_LEVEL or INFO)",
    )
    global_options.add_argument(
        "--log-file",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="log file (default: output/logs/<run_id>.log)",
    )

    parser = argparse.ArgumentParser(
        prog="knbench",
        description=(
            "KNBench: counterfactual testing of language models for geopolitical "
            "bias and risk blindness."
        ),
        parents=[global_options],
    )
    parser.add_argument(
        "--version", action="version", version=f"KNBench {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run_parser = subparsers.add_parser(
        "run",
        parents=[global_options],
        help="collect target responses and judge them",
        description="Run the benchmark over a dataset and write the raw results.",
    )
    run_parser.add_argument(
        "--models",
        metavar="LIST",
        default=None,
        help="comma-separated target models (default: TARGET_MODELS)",
    )
    run_parser.add_argument(
        "--scenarios",
        dest="scenarios",
        action=_ScenarioAliasAction,
        metavar="PATH",
        default=None,
        help="dataset file or directory (default: %s)" % DEFAULT_SCENARIOS,
    )
    run_parser.add_argument(
        "--data",
        dest="scenarios",
        action=_ScenarioAliasAction,
        metavar="PATH",
        default=None,
        help="alias of --scenarios",
    )
    run_parser.add_argument(
        "--output",
        metavar="PATH",
        default=DEFAULT_RESULTS,
        help="results file (default: %s)" % DEFAULT_RESULTS,
    )
    run_parser.add_argument(
        "--judge-model",
        metavar="MODEL",
        default=None,
        help="judge model (default: JUDGE_MODEL or anthropic/claude-opus-5.5)",
    )
    run_parser.add_argument(
        "--judge-votes",
        type=_bounded_int(1, None, "judge votes"),
        default=1,
        metavar="N",
        help="number of judge calls per response (default: 1)",
    )
    run_parser.add_argument(
        "--concurrency",
        type=_bounded_int(1, 32, "concurrency"),
        default=5,
        metavar="N",
        help="maximum concurrent requests, 1-32 (default: 5)",
    )
    run_parser.add_argument(
        "--max-retries",
        type=_bounded_int(0, None, "max retries"),
        default=4,
        metavar="N",
        help="retry budget per request (default: 4)",
    )
    run_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=120.0,
        metavar="SECONDS",
        help="timeout per request (default: 120)",
    )
    run_parser.add_argument(
        "--limit",
        type=_bounded_int(1, None, "limit"),
        default=None,
        metavar="N",
        help="cap the number of scenarios, keeping dataset order (default: none)",
    )
    run_parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="treat comparability failures as dataset errors",
    )
    run_parser.add_argument(
        "--resume", action="store_true", help="continue from the existing --output"
    )
    run_parser.add_argument(
        "--skip-judge", action="store_true", help="collect target responses only"
    )
    run_parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="debug only: disable anonymization and mark the run non-comparable",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without API calls",
    )

    report_parser = subparsers.add_parser(
        "report",
        parents=[global_options],
        help="render a report from raw results",
        description="Render the Markdown report or the metrics JSON.",
    )
    report_parser.add_argument(
        "--results",
        metavar="PATH",
        default=DEFAULT_RESULTS,
        help="results file (default: %s)" % DEFAULT_RESULTS,
    )
    report_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="report path (default: %s, or %s with --format json)"
        % (DEFAULT_REPORT, DEFAULT_METRICS_JSON),
    )
    report_parser.add_argument(
        "--format",
        choices=("md", "json"),
        default="md",
        help="output format (default: md)",
    )
    report_parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 when any mandatory metric is undefined",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[global_options],
        help="validate a dataset offline",
        description="Validate a dataset without network access.",
    )
    validate_parser.add_argument(
        "--scenarios",
        required=True,
        metavar="PATH",
        help="dataset file or directory (required)",
    )
    validate_parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="treat comparability failures as dataset errors",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch a command and return the §12.5 exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits for --help/--version and usage errors; the CLI contract
        # wants a return code, and usage errors are configuration errors (1).
        return 0 if exc.code == 0 else 1

    try:
        settings = get_settings()
    except ValidationError as exc:
        logger.error("invalid configuration: %s", _format_settings_error(exc))
        return 1
    except OSError as exc:
        logger.error("cannot load configuration: %s", exc)
        return 1
    level: str = getattr(args, "log_level", None) or settings.log_level
    log_file: str | None = getattr(args, "log_file", None)
    command = args.command

    if command == "validate":
        return _run_validate(args, level, log_file)
    if command == "report":
        return _run_report(args, settings, level, log_file)
    return _run_run(args, settings, level, log_file)


# ---------- Commands ----------


def _run_validate(
    args: argparse.Namespace, level: str, log_file: str | None
) -> int:
    if not _configure_logging(level, log_file):
        return 1
    try:
        scenarios, digest = load_scenarios(
            Path(args.scenarios), strict_pairs=bool(args.strict_pairs)
        )
    except DatasetError as exc:
        for problem in exc.problems:
            logger.error("%s", problem)
        logger.error("dataset validation failed with %d problem(s)", len(exc.problems))
        return 1

    variants = sum(len(scenario.variants) for scenario in scenarios)
    risks = sum(len(scenario.injected_risks) for scenario in scenarios)
    logger.info(
        "dataset is valid: %d scenario(s), %d variant(s), %d injected risk(s)",
        len(scenarios),
        variants,
        risks,
    )
    logger.info("dataset sha256: %s", digest)
    logger.info("strict pairs: %s", "enabled" if args.strict_pairs else "disabled")
    return 0


def _run_report(
    args: argparse.Namespace,
    settings: Settings,
    level: str,
    log_file: str | None,
) -> int:
    if not _configure_logging(level, log_file):
        return 1
    is_json = args.format == "json"
    output = (
        Path(args.output)
        if args.output
        else Path(DEFAULT_METRICS_JSON if is_json else DEFAULT_REPORT)
    )

    try:
        results = load_results(Path(args.results))
    except ResumeError as exc:
        logger.error("%s", exc)
        return 1

    if not log_file:
        if not _configure_logging(level, str(settings.log_file(results.run_id))):
            return 1

    metrics = build_metrics_file(results)
    text = render_metrics_json(metrics) if is_json else render_markdown(results, metrics)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.error("cannot write report to %s: %s", output, exc)
        return 1
    logger.info("report written to %s", output)

    if args.strict and has_undefined_metrics(metrics):
        logger.error(
            "strict mode: at least one mandatory metric is undefined; "
            "the report was written anyway"
        )
        return 2
    return 0


def _run_run(
    args: argparse.Namespace,
    settings: Settings,
    level: str,
    log_file: str | None,
) -> int:
    if not _configure_logging(level, log_file):
        return 1

    flags: list[str] = list(getattr(args, "_scenario_flags", None) or [])
    if len(flags) > 1:
        logger.error("--scenarios and --data are aliases; pass only one of them")
        return 1

    models = parse_model_list(args.models) if args.models else settings.models
    if not models:
        logger.error("no target models: pass --models or set TARGET_MODELS")
        return 1

    scenario_path = Path(args.scenarios or DEFAULT_SCENARIOS)
    try:
        scenarios, digest = load_scenarios(
            scenario_path, strict_pairs=bool(args.strict_pairs)
        )
    except DatasetError as exc:
        for problem in exc.problems:
            logger.error("%s", problem)
        logger.error("dataset validation failed with %d problem(s)", len(exc.problems))
        return 1

    if args.limit is not None:
        scenarios = scenarios[: args.limit]
        logger.info(
            "limit %d: keeping scenario(s) %s",
            args.limit,
            ", ".join(scenario.scenario_id for scenario in scenarios) or "none",
        )
    if not scenarios:
        logger.error("no scenarios selected")
        return 1

    judge_model = args.judge_model or settings.judge_model
    triples = sum(len(scenario.variants) for scenario in scenarios) * len(models)

    if args.dry_run:
        logger.info("dry run: no API calls will be made")
        logger.info("target models: %s", ", ".join(models))
        logger.info(
            "dataset: %d scenario(s), %d variant(s), %d triple(s)",
            len(scenarios),
            sum(len(scenario.variants) for scenario in scenarios),
            triples,
        )
        logger.info(
            "judge model: %s; votes: %d; anonymization: %s",
            judge_model,
            args.judge_votes,
            "disabled" if args.no_anonymize else "enabled",
        )
        logger.info("output: %s", args.output)
        logger.info("dataset sha256: %s", digest)
        return 0

    resume_from: RunResults | None = None
    if args.resume:
        try:
            resume_from = load_results(Path(args.output))
            if not _has_pending(resume_from, scenarios, models):
                filtered = _filtered_resume(
                    resume_from,
                    scenarios,
                    models,
                    scenario_path,
                    digest,
                )
                if (
                    filtered.scenarios != resume_from.scenarios
                    or filtered.records != resume_from.records
                ):
                    write_results_atomic(Path(args.output), filtered)
                    logger.info(
                        "resume: removed records outside the current selection"
                    )
                logger.info("nothing to do: every record is already complete")
                return 4
        except ResumeError as exc:
            logger.error("%s", exc)
            return 1
        except OSError as exc:
            logger.error("cannot write results to %s: %s", args.output, exc)
            return 1

    try:
        require_api_key(settings)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 3

    config = RunConfig(
        models=models,
        judge_model=judge_model,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        timeout_s=args.timeout,
        judge_votes=args.judge_votes,
        anonymize=not args.no_anonymize,
        scenarios_path=str(scenario_path),
        scenarios_sha256=digest,
        app_version=__version__,
        strict_pairs=bool(args.strict_pairs),
    )

    run_id = new_run_id()
    if not log_file:
        if not _configure_logging(level, str(settings.log_file(run_id))):
            return 1

    logger.info(
        "starting run %s: %d model(s), %d scenario(s), %d triple(s)",
        run_id,
        len(models),
        len(scenarios),
        triples,
    )
    try:
        results = asyncio.run(
            _execute(
                args,
                settings,
                scenarios,
                models,
                config,
                resume_from,
                run_id,
            )
        )
    except FatalAuthError as exc:
        logger.error("%s", exc)
        return 3
    except CatalogError as exc:
        logger.error("%s", exc)
        return 1
    except ResumeError as exc:
        logger.error("%s", exc)
        return 1
    except ConfigError as exc:
        logger.error("%s", exc)
        return 3
    except OSError as exc:
        logger.error("cannot write results to %s: %s", args.output, exc)
        return 1

    try:
        sidecar = write_provenance(Path(args.output), results)
    except OSError as exc:
        logger.warning("cannot write provenance sidecar for %s: %s", args.output, exc)
    else:
        logger.info("provenance written to %s", sidecar)

    failed = sum(1 for record in results.records if record.status in _FAILED_STATUSES)
    ok = sum(1 for record in results.records if record.status == "ok")
    skipped = sum(1 for record in results.records if record.status == "skipped")
    logger.info(
        "finished: %d ok, %d failed, %d skipped; results in %s",
        ok,
        failed,
        skipped,
        args.output,
    )
    if failed:
        logger.warning(
            "partial run: %d record(s) failed; see the errors section of the report",
            failed,
        )
        return 2
    return 0


async def _execute(
    args: argparse.Namespace,
    settings: Settings,
    scenarios: list[BenchmarkScenario],
    models: list[str],
    config: RunConfig,
    resume_from: RunResults | None,
    run_id: str,
) -> RunResults:
    """Run both phases and checkpoint completed judge evaluations.

    A signal or cancellation during judging must not discard paid work. The
    evaluator publishes an ordered snapshot after each result; this function
    writes it atomically with ``finished_at=None`` until the final write.
    """
    current_task = asyncio.current_task()
    previous_sigterm: Any = None
    signal_installed = False
    if threading.current_thread() is threading.main_thread():
        try:
            previous_sigterm = signal.getsignal(signal.SIGTERM)

            def _on_sigterm(signum: int, frame: Any) -> None:
                if current_task is not None:
                    current_task.cancel()

            signal.signal(signal.SIGTERM, _on_sigterm)
            signal_installed = True
        except (ValueError, OSError):
            # Documented limitation: some Windows consoles cannot deliver a
            # catchable SIGTERM to a Python process.
            logger.warning("SIGTERM handling is unavailable on this platform")

    client = OpenRouterClient.from_settings(settings)
    try:
        results = await run_benchmark(
            scenarios,
            models,
            config,
            resume_from=resume_from,
            client=client,
            output_path=Path(args.output),
            run_id=run_id,
            preflight=True,
        )
        if args.skip_judge:
            return results

        latest: list[EvaluationRecord] = list(results.records)

        def checkpoint(records: list[EvaluationRecord]) -> None:
            latest[:] = records
            write_results_atomic(
                Path(args.output),
                results.model_copy(
                    update={"records": list(records), "finished_at": None}
                ),
            )

        try:
            judged = await evaluate_records(
                results.records,
                results.scenarios,
                client=client,
                semaphore=asyncio.Semaphore(config.concurrency),
                judge_model=config.judge_model,
                judge_votes=config.judge_votes,
                anonymize=config.anonymize,
                target_models=models,
                max_retries=config.max_retries,
                timeout_s=config.timeout_s,
                on_result=checkpoint,
            )
        except FatalAuthError:
            checkpoint(latest)
            raise
        except (asyncio.CancelledError, KeyboardInterrupt):
            checkpoint(latest)
            raise

        final = results.model_copy(
            update={
                "records": judged,
                "finished_at": datetime.now(timezone.utc),
            }
        )
        write_results_atomic(Path(args.output), final)
        return final
    finally:
        if signal_installed:
            signal.signal(signal.SIGTERM, previous_sigterm)
        await client.aclose()


# ---------- Helpers ----------


def _format_settings_error(exc: ValidationError) -> str:
    """Convert settings validation details into one actionable line."""
    details: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        details.append(f"{location.upper()}: {error['msg']}")
    if not details:
        return "settings validation failed; check environment variables and .env"
    return "; ".join(details) + ". Check environment variables and .env."


def _configure_logging(level: str, log_file: str | None) -> bool:
    """Configure logging, returning False when the level or path is unusable."""
    try:
        setup_logging(level, Path(log_file) if log_file else None)
    except (ConfigError, OSError) as exc:
        logger.error("cannot configure logging: %s", exc)
        return False
    return True


def _has_pending(
    resume_from: RunResults, scenarios: list[BenchmarkScenario], models: list[str]
) -> bool:
    """Mirror the runner's resume filter to detect a fully complete run.

    Validation happens in :func:`prepare_resume`, so a changed dataset fails
    here before the CLI asks for an API key or writes a report.
    """
    resumed = prepare_resume(resume_from, scenarios, models)
    existing = {
        (record.scenario_id, record.variant_id, record.model_name): record
        for record in resumed
    }
    for scenario in scenarios:
        for variant in scenario.variants:
            for model in models:
                record: EvaluationRecord | None = existing.get(
                    (scenario.scenario_id, variant.variant_id, model)
                )
                if record is None or record.status != "ok":
                    return True
    return False


def _filtered_resume(
    resume: RunResults,
    scenarios: list[BenchmarkScenario],
    models: list[str],
    scenario_path: Path,
    digest: str,
) -> RunResults:
    """Return a view of a completed resume file for the current selection."""
    records = prepare_resume(resume, scenarios, models)
    config = resume.run_config.model_copy(
        update={
            "models": models,
            "scenarios_path": str(scenario_path),
            "scenarios_sha256": digest,
        }
    )
    return resume.model_copy(
        update={"run_config": config, "scenarios": scenarios, "records": records}
    )


if __name__ == "__main__":
    raise SystemExit(main())
