# KNBench configuration

This page lists environment variables, command-line flags, precedence, exit
codes, logging and the files a run produces. Usage steps are in
[`user-guide.md`](user-guide.md); data fields are in
[`dataset-format.md`](dataset-format.md).

## Precedence

Configuration is resolved in this order, highest first:

1. command-line flags;
2. environment variables;
3. the `.env` file in the working directory;
4. built-in defaults.

`--scenarios` and `--data` are aliases of one argument. Passing both is a
configuration error with exit code `1`, even when the values are identical.

## Installation

Runtime dependencies are pinned in `requirements.txt`; development and build
tools are pinned in `requirements-dev.txt`. For a development checkout:

```bash
python -m pip install -e ".[dev]"
```

The package exposes the `knbench` console command as well as
`python -m knbench.main`. Build a wheel with `python -m build`; the resulting
`dist/knbench-*.whl` can be installed in a clean virtual environment.

## Environment variables

| Variable | Required | Default | Meaning |
|----------|----------|---------|---------|
| `OPENROUTER_API_KEY` | yes for `run` | — | API key; never written to logs or results |
| `OPENROUTER_BASE_URL` | no | `https://openrouter.ai/api/v1` | API base URL |
| `OPENROUTER_APP_NAME` | no | `KNBench` | value of the `X-Title` header |
| `OPENROUTER_APP_URL` | no | — | value of the `HTTP-Referer` header |
| `JUDGE_MODEL` | no | `anthropic/claude-opus-5.5` | judge model |
| `TARGET_MODELS` | no | `openai/gpt-6.1-sol,google/gemini-4-carbon` | comma-separated target models |
| `LOG_LEVEL` | no | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `OUTPUT_DIR` | no | `output` | root for results, reports and logs |

Copy `.env.example` to `.env` and fill in the key. `.env` is git-ignored and
must never be committed. A missing key stops `run` with exit code `3`, except
under `--dry-run`, which makes no API calls.

`TARGET_MODELS` has a default, and `--models` overrides it; at least one model
is required in both cases. `--dry-run` needs models because it enumerates the
planned triples. `--judge-model` overrides `JUDGE_MODEL`; its default is
`anthropic/claude-opus-5.5`.

## Global options

These options are accepted before or after the subcommand.

| Option | Default | Meaning |
|--------|---------|---------|
| `--log-level` | `LOG_LEVEL` or `INFO` | logging threshold |
| `--log-file` | for `run` and `report`: `OUTPUT_DIR/logs/<run_id>.log` | explicit log path |
| `--version` | — | print `KNBench <version>` and exit `0` |

## `run`

```bash
python -m knbench.main run --models openai/gpt-6.1-sol,google/gemini-4-carbon \
    --scenarios data/scenarios.json \
    --output output/raw_results.json
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--models` | `TARGET_MODELS` | comma-separated target models; at least one |
| `--scenarios` / `--data` | `data/scenarios.json` | dataset file or directory |
| `--output` | `output/raw_results.json` | results file |
| `--judge-model` | `JUDGE_MODEL` or `anthropic/claude-opus-5.5` | judge model |
| `--judge-votes` | `1` | independent judge calls per response; at least 1 |
| `--concurrency` | `5` | concurrent requests, 1–32 |
| `--max-retries` | `4` | retries per request, at least 0; attempts = retries + 1 |
| `--timeout` | `120` | seconds per HTTP request; must be positive |
| `--limit` | — | keep the first N scenarios in dataset order |
| `--strict-pairs` | off | comparability failures become dataset errors |
| `--resume` | off | continue from the existing `--output` |
| `--skip-judge` | off | collect target responses only |
| `--no-anonymize` | off | debug only; marks the run non-comparable |
| `--dry-run` | off | validate and log the plan without API calls |

Without `--skip-judge`, the run judges every collected response and rewrites the
results file with `status="ok"` or `judge_failed`. With `--skip-judge`, records
keep `status="skipped"` and the report shows coverage only.

Before the first target request, `run` calls
`GET {OPENROUTER_BASE_URL}/models` and checks every target id and the judge id.
A missing id exits `1` with the missing ids and the catalog date. An unavailable,
malformed or incomplete catalog also exits `1` and no target request is sent;
check `OPENROUTER_BASE_URL` and the model ids. `--dry-run` skips this network
check and remains fully offline.

`--resume` keeps records whose status is `ok` and retries every other triple.
Before reusing a response, it compares the previous scenario snapshot with the
selected dataset: if an existing scenario or variant changed its document,
prompt, risks, vendor, country, aliases or control flag, the command exits `1`
with the changed field. New scenarios and variants are allowed. Records for
scenarios, variants or models outside the current selection are dropped, so
`--limit` and a model subset cannot inflate coverage. A new `run_id` is
generated and `run_config.resumed_from` records the previous run. Models that
were not part of the previous run cannot be resumed. When no selected triple is
pending, the command exits `4` without calling the API; if the file contains
records outside the current selection, it is rewritten to the filtered view.

`--limit` keeps dataset order: file order for a directory, then array order
inside a file, and logs the selected `scenario_id` values.

## `report`

```bash
python -m knbench.main report --results output/raw_results.json --output output/report.md
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--results` | `output/raw_results.json` | results file |
| `--output` | `output/report.md`, or `output/metrics.json` with `--format json` | report path |
| `--format` | `md` | `md` or `json` |
| `--strict` | off | exit `2` when a mandatory metric is `null` |

A missing or invalid results file exits `1`. `--strict` writes the report and
then exits `2`; a scenario flagged `partial` keeps its values and does not by
itself trigger strict mode.

## `validate`

```bash
python -m knbench.main validate --scenarios data/scenarios.json --strict-pairs
```

| Option | Required | Meaning |
|--------|----------|---------|
| `--scenarios` | yes | dataset file or directory |
| `--strict-pairs` | no | treat comparability warnings as errors |

`validate` makes no network calls and needs no API key. It prints the scenario,
variant and risk counts plus the dataset SHA-256, and exits `0` when the dataset
is valid.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | success |
| `1` | configuration, dataset, model-catalog or results-file error |
| `2` | partial run: failed records, or undefined metrics under `--strict` |
| `3` | fatal access error: HTTP 401/403 or a missing `OPENROUTER_API_KEY` |
| `4` | nothing to do: `--resume` found no pending triple |

Usage errors reported by the parser also exit `1`, so scripts can treat every
bad invocation the same way.

Invalid settings, an unreadable or malformed results file, and an unwritable
log, report or results path are reported as one English error and exit `1`;
the CLI does not print a traceback. `LOG_LEVEL` accepts `DEBUG`, `INFO`,
`WARNING`, `ERROR` and `CRITICAL`. A log or output path that is a directory is
rejected with exit `1` instead of raising a traceback.

## Logging

The format is `%(asctime)s %(levelname)s %(name)s %(message)s`. Messages go to
stdout and to `OUTPUT_DIR/logs/<run_id>.log` in UTF-8. The `httpx` and
`httpcore` loggers are pinned to `WARNING`, so request headers cannot reach the
log. Request bodies and the `Authorization` header are never logged.

`run` generates the run id before the first request so the log file can be named
after it. `report` reads the run id from the results file.

## Output layout

```text
output/
├── raw_results.json      # collected responses and evaluations
├── raw_results.provenance.json  # run metadata, hashes and versions
├── report.md             # eight-section Markdown report
├── metrics.json          # metrics document, written with --format json
└── logs/
    └── <run_id>.log      # one log per run
```

`raw_results.json` is written atomically through a temporary file after every
ten target records, after every completed judge evaluation, at the end of a
run, and when a run is interrupted. During judging, intermediate files keep
`finished_at` as `null`; unjudged records remain `skipped` with the target
responses already collected. On POSIX, `SIGTERM` triggers the same checkpoint
as `Ctrl+C`; some Windows console sessions cannot deliver a catchable `SIGTERM`,
so `Ctrl+C` is the reliable path there. `output/` is git-ignored because
results may contain sensitive context.

The provenance sidecar is written after a CLI `run` completes. It records the
run id, Python and dependency versions, an optional git revision, the dataset
hash, model ids and prompt hashes. It contains no API key, document text,
prompts or model responses. A sidecar write failure is logged as a warning and
does not invalidate the results file.

## Examples

```bash
# Validate a directory of datasets, with strict comparisons
python -m knbench.main validate --scenarios data/examples --strict-pairs

# Show the plan without spending API credits
python -m knbench.main run --models openai/gpt-6.1-sol --scenarios data/scenarios.json --dry-run

# Collect responses only; the report then shows coverage without metrics
python -m knbench.main run --models openai/gpt-6.1-sol --skip-judge

# Render the JSON metrics document with strict undefined-value checking
python -m knbench.main report --format json --strict

# Continue an interrupted run
python -m knbench.main run --models openai/gpt-6.1-sol --resume
```
