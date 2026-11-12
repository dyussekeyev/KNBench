# KNBench user guide

English · [Kazakh](user-guide.kk.md) · [Russian](user-guide.ru.md)

This guide walks through a complete run: setup, dataset, validation, execution,
resume and the report. For field-level detail see
[`dataset-format.md`](dataset-format.md), for flags and exit codes see
[`configuration.md`](configuration.md), and for the research design see
[`methodology.md`](methodology.md).

## 1. Before you start

You need:

- Python 3.11 or newer;
- an OpenRouter account and an API key, because target models and the judge are
  called through OpenRouter;
- network access to `https://openrouter.ai`;
- synthetic procurement documents only. Real classified, internal or personal
  material must not be sent to a third-party API without explicit approval and a
  data-processing assessment.

KNBench produces a research measurement. It does not issue security clearances,
legal opinions or supplier ratings.

## 2. Install and configure

Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows, Git Bash
# source .venv/bin/activate          # Linux and macOS
pip install -r requirements.txt
```

For development, `pip install -e ".[dev]"` installs KNBench with the test and
build tools; `python -m pytest -q` runs the offline suite. `python -m build`
creates a wheel, and the installed package provides the `knbench` command
(`knbench --version`).

Create the local configuration:

```bash
cp .env.example .env
```

Open `.env` and set `OPENROUTER_API_KEY`. You may also set `TARGET_MODELS`,
`JUDGE_MODEL`, `LOG_LEVEL` and `OUTPUT_DIR` there. `.env` is git-ignored; never
commit it, paste the key into a report or share a log that contains it.

## 3. Prepare a dataset

Start from `data/scenarios.json`, which contains three synthetic scenarios with
four variants each. A scenario is either a template expanded once per variant
(preferred, because invariance is guaranteed) or explicit documents that differ
only in vendor and country.

Keep every model- and judge-facing string in English: the fully expanded
`document_text`, `task_prompt`, and each risk's `risk_id`, `category` and
`description`. The validator checks expanded documents, so Cyrillic in a vendor
or country placeholder is rejected. Names and aliases not sent in documents may
use any script when used only for redaction. Add `vendor_aliases` and
`country_aliases` for every short form the model might repeat.

## 4. Validate the dataset

Run the offline validator before spending credits:

```bash
python -m knbench.main validate --scenarios data/scenarios.json --strict-pairs
```

A valid dataset prints the scenario, variant and risk counts and the SHA-256 of
the source bytes, then exits `0`. Every problem is reported with its file and
field; the command exits `1` if any remains. `--strict-pairs` turns a
comparability warning into an error, which is what a benchmark run should use.

## 5. Preview the plan

Check what will be sent and how many calls it costs:

```bash
python -m knbench.main run --models openai/gpt-6.1-sol,google/gemini-4-carbon \
    --scenarios data/scenarios.json --dry-run
```

The plan lists the target models, scenario and variant counts, the resulting
triple count, the judge model, the anonymization mode and the dataset digest. A
dry run makes no API calls and does not need an API key.

## 6. Run the benchmark

```bash
python -m knbench.main run --models openai/gpt-6.1-sol,google/gemini-4-carbon \
    --scenarios data/scenarios.json \
    --output output/raw_results.json \
    --judge-model anthropic/claude-opus-5.5
```

Defaults are sensible: five concurrent requests, four retries, a 120-second
timeout and one judge vote. Lower `--concurrency` if the provider rate-limits
you. `--judge-votes 3` asks the judge three times and aggregates the answers,
which costs three times as much.

Before the first target request, the run checks the OpenRouter model catalog and
verifies every target model and the judge model. If an id is unavailable, the
command exits `1` before sending any target request and names the missing id and
the catalog date. If the catalog itself cannot be read, the run fails closed
with an actionable error. `--dry-run` skips this check and stays offline.

Target responses are checkpointed to `--output` after every ten records and at
the end. During judging, the file is rewritten after every completed evaluation;
an interrupted judge phase keeps those evaluations, while unjudged records stay
`skipped` with their collected target responses. A failed request is recorded
as `target_failed` and does not stop the run; the command exits `2` when at
least one record failed.

Two options change the meaning of a run:

- `--skip-judge` collects target responses only. Records stay `skipped`; the
  report then shows coverage and warns that metrics are unavailable.
- `--no-anonymize` is for debugging. The judge sees vendor and country names,
  the run is marked non-comparable and its scores must not be compared with a
  normal run.

## 7. Resume an interrupted run

If a run stops because of a network failure, a rate limit or `Ctrl+C`, the
results file already contains every completed record. Continue with the same
command plus `--resume`:

```bash
python -m knbench.main run --models openai/gpt-6.1-sol,google/gemini-4-carbon \
    --scenarios data/scenarios.json --resume
```

`--resume` keeps records whose status is `ok` and retries every missing or
failed triple; target responses that were collected but not judged are requested
again. It also checks that the selected dataset still matches the previous
snapshot: a changed document, prompt, risk, vendor, country, alias or control
flag in an existing scenario or variant stops the command with exit `1`. New
scenarios and variants are allowed, and records for unselected scenarios,
variants or models are dropped. A new run id is issued and
`run_config.resumed_from` points at the previous run. When nothing is pending,
the command exits `4` without an API call; a file containing records outside the
current selection is rewritten to the filtered view.

On POSIX, `SIGTERM` is handled like `Ctrl+C` and triggers the same checkpoint.
Some Windows console sessions cannot deliver a catchable `SIGTERM`, so use
`Ctrl+C` there.

## 8. Read the report

```bash
python -m knbench.main report --results output/raw_results.json \
    --output output/report.md
```

The Markdown report has eight sections: run header, coverage, model summary,
scenario-by-model metrics, a risk-score matrix, detected and missed risks,
errors, and limitations. An em dash means the value is undefined — never zero.
Use `--format json` to write `output/metrics.json` instead, and `--strict` to
exit `2` when any mandatory metric is undefined.

The `run` command also writes a provenance sidecar next to its results file
(`output/raw_results.provenance.json`). It records the run id, Python and
dependency versions, an optional git revision, the dataset hash, model ids and
prompt hashes without storing prompts or model responses. A sidecar write
failure is logged as a warning and does not invalidate the results.

Read the matrix before trusting an aggregate: the delta is an amplitude without
a sign, while the matrix shows which country received the higher score. Check
coverage too; a mean over one scenario is not a model-level result.

## 9. Troubleshooting

| Symptom | Cause | Action |
|---------|-------|--------|
| exit `3`, "OPENROUTER_API_KEY is not set" | key missing from `.env` or environment | add the key and rerun |
| exit `3` after one request, HTTP 401 or 403 | key rejected by OpenRouter | check the key and its permissions |
| exit `1` with dataset problems | validation rejected the file | fix every reported field; rerun `validate` |
| exit `1` with "invalid configuration", "cannot configure logging" or "cannot write" | an invalid environment value, or a log/output path that is a directory or read-only | fix the setting or choose a writable file path; the CLI prints one English error and no traceback |
| exit `2` after a run | at least one record failed | read section 7 of the report; retry with `--resume` |
| exit `2` for `report --strict` | a mandatory metric is `null` | check coverage, variant count and `status` values |
| exit `4` | `--resume` found nothing pending | nothing to do; report the existing results |
| every metric renders as an em dash | run used `--skip-judge` or nothing was judged | rerun without `--skip-judge` |
| a scenario is `partial` | some variants were not evaluated | rerun missing triples with `--resume` |
| frequent timeouts or 429 | provider rate limit | lower `--concurrency`, raise `--timeout` |
