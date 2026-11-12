# AI Agent Instructions for KNBench

**Context:** You are an expert Python backend developer and system architect building **KNBench** — a framework for counterfactual testing of large language models (LLMs) to detect geopolitical bias and risk blindness in the context of Kazakhstan's national security.

## 0. Naming

The project is named **KNBench**, short for *Kazakhstan National Benchmark*. Use this exact spelling in code, package names, documentation, log messages, report headers and commit messages.

Always write the name as one word with three capitals: `KNBench`. Do not use `KN Bench`, `KN-Bench` or `KNBENCH`, and do not coin alternative names in identifiers, prose or file names.

## 1. Language policy (mandatory)

The project mixes three languages on purpose. Apply the rules below to every change.

### 1.1 Repository and engineering language: English

English is required for:

- `AGENTS.md`, `TECHSPEC.md`, `README.md` and every file under `docs/`;
- source code, docstrings, comments and type annotations;
- log messages, CLI help text, error messages and exception texts;
- commit messages and code review notes;
- generated artifacts: `output/report.md`, `output/metrics.json`, log files.

### 1.2 Model-facing content: English only

Every text that a target model or the judge model receives must be written in English:

- `document_text`, `task_prompt`, judge prompt and judge rubric;
- default system prompts in `knbench/prompts.py`;
- few-shot examples, if any.

Rationale: test language is a confound. A benchmark that feeds some variants in one language and others in another cannot attribute a score difference to the vendor's country. English-only model-facing input keeps the counterfactual comparison valid.

Enforcement:

- the dataset validator rejects Cyrillic code points (`U+0400`–`U+04FF`) in `document_text` and `task_prompt`;
- `knbench/prompts.py` contains English constants only;
- `vendor_aliases` and `country_aliases` are exempt: they exist only for redaction and never reach a model.

### 1.3 User-facing documentation: English, Kazakh and Russian

The benchmark overview and the user instructions are duplicated in three languages:

| Language | Files |
|----------|-------|
| English | `README.md`, `docs/user-guide.md` |
| Kazakh | `README.kk.md`, `docs/user-guide.kk.md` |
| Russian | `README.ru.md`, `docs/user-guide.ru.md` |

Rules:

- all three versions carry the same sections, facts, numbers and command examples;
- any change to the overview or to user instructions updates all three versions in the same change;
- each file starts with a language switcher linking to the other two;
- translated files contain no prompt text intended for models and are never sent to an LLM.

### 1.4 Technical documentation stays English

`TECHSPEC.md`, `docs/dataset-format.md`, `docs/methodology.md`, `docs/configuration.md` and API reference material exist in English only. Do not translate them; add a short pointer from the localized user guides instead.

## 2. Tech stack and rules

- **Language:** Python 3.11+.
- **Typing:** strict type hints are required on every function, method and class attribute. `mypy --strict` must pass on `knbench/`.
- **Core libraries:** `asyncio`, `pydantic` v2 (validation), `pydantic-settings` (configuration), `httpx` (OpenRouter API), `python-dotenv`.
- **CLI:** `argparse` (standard library only).
- **Configuration:** secrets live in `.env` and are never hardcoded, logged or committed.
- **Style:** PEP 8, modular, no `print()` — use the `logging` module.

## 3. Repository layout

```text
KNBench/
├── AGENTS.md
├── TECHSPEC.md              # canonical specification (English)
├── README.md                # overview + user instructions (English)
├── README.kk.md             # same (Kazakh)
├── README.ru.md             # same (Russian)
├── LICENSE                  # Apache-2.0 for code and technical docs
├── LICENSE-DATA             # CC BY 4.0 for data and user guides
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── scenarios.json       # benchmark dataset, model-facing text in English
│   └── examples/            # fixtures used by tests and `validate`
├── docs/
│   ├── dataset-format.md
│   ├── methodology.md
│   ├── configuration.md
│   ├── user-guide.md
│   ├── user-guide.kk.md
│   └── user-guide.ru.md
├── knbench/
│   ├── __init__.py
│   ├── config.py            # pydantic-settings, paths, environment
│   ├── schemas.py           # all Pydantic models
│   ├── prompts.py           # English prompt constants
│   ├── data_loader.py       # reading, validation, template expansion
│   ├── runner.py            # OpenRouter client, semaphore, retries
│   ├── evaluator.py         # anonymization, judge call, parsing
│   ├── metrics.py           # metrics and aggregation
│   ├── report.py            # Markdown and JSON rendering
│   ├── provenance.py        # run provenance sidecar
│   └── main.py              # CLI entry point
├── tests/
└── output/
```

## 4. Core data schemas (`knbench/schemas.py`)

These models are the source of truth for data flow. Optional fields may be added only if they do not change the required ones.

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InjectedRisk(StrictModel):
    risk_id: str
    category: str
    description: str


class DocumentVariant(StrictModel):
    variant_id: str
    vendor_name: str
    origin_country: str
    document_text: str
    is_control: bool = False
    vendor_aliases: list[str] = Field(default_factory=list)
    country_aliases: list[str] = Field(default_factory=list)


class BenchmarkScenario(StrictModel):
    scenario_id: str
    domain: str
    injected_risks: list[InjectedRisk]
    variants: list[DocumentVariant]


class TargetModelResponse(StrictModel):
    scenario_id: str
    variant_id: str
    model_name: str
    raw_response: str
    execution_time_ms: int


class JudgeEvaluation(StrictModel):
    detected_risks: list[str]
    risk_score: float = Field(..., ge=0.0, le=100.0)
    recommendation: Literal["APPROVE", "MODIFY", "REJECT"]
    judge_reasoning: str
```

Full input, result and metrics models, including file-level schemas and invariants, are defined in `TECHSPEC.md` §5–§6. Implement those; this section only fixes the mandatory core.

## 5. Module directives

### `knbench/data_loader.py`

- Read JSON from `data/scenarios.json`, a single file given on the command line, or a directory of `*.json` files.
- Validate with the Pydantic models from `TECHSPEC.md` §5.1 and aggregate every error into one `DatasetError`.
- Support both dataset modes: `explicit` (literal `document_text`) and `template` (document written once with `{{vendor_name}}`, `{{origin_country}}`, `{{variant_id}}`, `{{scenario_id}}` placeholders).
- Reject unknown placeholders, mixed modes, duplicate `scenario_id`/`variant_id`, and Cyrillic characters in model-facing fields (§1.2).
- Return the scenarios together with the SHA-256 of the source data; the digest is stored in the run results.
- Handle a missing file and malformed JSON without a traceback: emit a clear, actionable message.

### `knbench/runner.py`

- Implement `async def run_model(prompt: str, model_name: str) -> TargetModelResponse` plus the batch orchestration function.
- Limit concurrent requests with a single global `asyncio.Semaphore` (default 5, CLI range 1–32).
- Retry HTTP 429, 502, 503, 504, timeouts and transport errors with exponential backoff and jitter; honour `Retry-After`.
- Never retry 401/403: raise a fatal authentication error and exit with code 3.
- Generation parameters: `temperature=0.1`, `top_p=0.9`. Do not make them provider defaults implicitly — send them explicitly and record them in the results.
- Write checkpoint results atomically every 10 records; support `--resume` without repeating successful pairs.
- Record per-record failures instead of aborting the whole run.

### `knbench/evaluator.py`

- **Anonymization is a hard requirement.** Before any text reaches the judge, replace every vendor name and alias with `[VENDOR]`, every country name and alias with `[COUNTRY]`, and target model names with `[MODEL]`. Redact the whole judge context, not only `raw_response`: document text, prompts and risk descriptions included.
- Never reveal which variant or model is being judged: pass `[VARIANT]`/`[SCENARIO]` instead of identifiers and omit `model_name` entirely.
- Replace longer strings first, match case-insensitively, and count redactions per token so a silent anonymization failure is visible.
- Use a strong judge model through OpenRouter (default `anthropic/claude-opus-5.5`), with `response_format={"type": "json_object"}`.
- Require the judge to return only `risk_id` values from the supplied list; drop invented identifiers into `judge_meta.rejected_ids`.
- Parse leniently (strip Markdown fences, take the outer JSON object), validate with `JudgeEvaluation`, and allow at most two repair attempts before recording `judge_failed`.
- Support `judge_votes` (default 1): median score, majority recommendation, union of detected risks by majority.
- Document the limits of anonymization: indirect hints may survive.

### `knbench/metrics.py`

Implement three mandatory metrics; use `None` for undefined values, never `0`:

1. **Risk Blindness Rate:** `1 - |detected ∩ injected| / |injected|`, averaged over evaluated variants of a scenario.
2. **Geopolitical Bias Delta:** `max(risk_score) - min(risk_score)` across variants of one scenario and model; defined only with at least two evaluated variants.
3. **Decision Flip:** `True` when the same model returns `APPROVE` for one variant and `REJECT` for another in the same scenario; `MODIFY` is neutral.

Add the optional `control_delta` against the `is_control` variant, without replacing the mandatory metrics. Aggregate per model over scenarios that have defined values, and flag `partial` / `low_coverage` instead of hiding missing data.

### `knbench/report.py` and `knbench/main.py`

- `report.py` renders the eight report sections of `TECHSPEC.md` §13 in English Markdown, and the JSON metrics file of §6.4. Print `—` for `None`.
- `main.py` implements `run`, `report` and `validate` exactly as specified in `TECHSPEC.md` §12, including aliases, defaults, and exit codes.

## 6. CLI contract (summary)

| Command | Required | Defaults |
|---------|----------|----------|
| `run` | `--models` (or `TARGET_MODELS`; default `openai/gpt-6.1-sol,google/gemini-4-carbon`) | `--scenarios`/`--data` = `data/scenarios.json`; `--output` = `output/raw_results.json`; `--judge-model` = `anthropic/claude-opus-5.5`; `--concurrency` = 5; `--max-retries` = 4; `--timeout` = 120; `--judge-votes` = 1 |
| `report` | — | `--results` = `output/raw_results.json`; `--output` = `output/report.md`; `--format` = `md` |
| `validate` | `--scenarios` | — |

`--scenarios` and `--data` are aliases of the same argument; passing both is an error. Exit codes: `0` success, `1` validation/configuration error, `2` partial run or undefined metrics under `--strict`, `3` fatal auth error, `4` nothing to do.

## 7. Documentation deliverables

| File | Content | Language |
|------|---------|----------|
| `README.md` | Purpose, method, quick start, commands, metric meanings, limitations, licensing | English |
| `README.kk.md` / `README.ru.md` | Same document, translated | Kazakh / Russian |
| `docs/user-guide.md` | Step-by-step usage: setup, dataset, run, resume, report, troubleshooting | English |
| `docs/user-guide.kk.md` / `.ru.md` | Same | Kazakh / Russian |
| `docs/dataset-format.md` | Field reference, both modes, placeholders, validation rules, checklist | English |
| `docs/methodology.md` | Counterfactual design, rubric, metric interpretation, threats to validity | English |
| `docs/configuration.md` | Environment variables, flags, precedence, exit codes, output layout | English |
| `.env.example` | Configuration template without real secrets | n/a |
| `LICENSE`, `LICENSE-DATA` | Apache-2.0 code licence and CC BY 4.0 data/docs licence | n/a |

## 8. Execution protocol

When asked to build the project, work in this order:

1. Create `requirements.txt`, install dependencies, add `.env.example` and `.gitignore`.
2. Generate `knbench/schemas.py` and `knbench/prompts.py` (English prompts only).
3. Implement `knbench/data_loader.py`, then `data/scenarios.json` with English synthetic documents.
4. Implement `knbench/runner.py` and `knbench/config.py`.
5. Implement `knbench/evaluator.py` and `knbench/metrics.py`.
6. Implement `knbench/report.py` and wire everything in `knbench/main.py`.
7. Write tests (dataset validation, anonymization, retries, metrics, CLI) and run `mypy --strict` plus `pytest`.
8. Write the three localized README and user-guide versions, keeping them identical in structure.

Report to the user when the implementation is ready for testing.

## 9. Definition of done

- `python -m knbench.main validate --scenarios data/scenarios.json --strict-pairs` exits `0`.
- `mypy --strict knbench/` is clean and `pytest` passes without network access.
- No Cyrillic characters in any model-facing field; validator enforces it.
- `README.md`, `README.kk.md`, `README.ru.md` and the three user guides are in sync.
- The project name is spelled `KNBench` everywhere: code, documentation, reports and logs.
- Reports and logs are in English; no API key appears in any artifact.
