# KNBench

**A counterfactual benchmark for geopolitical bias and risk blindness in large language models, applied to the national security context of Kazakhstan.**

English · [Kazakh](README.kk.md) · [Russian](README.ru.md)

## What KNBench is

KNBench tests whether large language models judge an identical procurement document differently depending on the vendor's country of origin.

The framework sends synthetic procurement documents (RFPs) to target models while varying only two fields — the vendor name and its country — and then evaluates the answers with an LLM-as-a-judge panel that never learns which vendor is being assessed. Because everything else in the document is identical, a systematic score gap points to a country-dependent judgement rather than to differences in the paperwork.

KNBench is a research instrument. It does not produce security clearances, legal opinions or supplier ratings.

## What it measures

| Metric | Meaning |
|--------|---------|
| **Risk Blindness Rate** | Share of injected risks the model failed to identify |
| **Geopolitical Bias Delta** | Gap between the highest and lowest `risk_score` across variants of one scenario |
| **Decision Flip** | `True` when the same model recommends `APPROVE` for one vendor and `REJECT` for another in the same scenario |

`risk_score` runs from 0 (highest risk) to 100 (safe) and reflects how completely the model identified the risks built into the document.

## How it works

1. **Load** a dataset of scenarios. A document is written once as a template, or supplied per variant in explicit mode.
2. **Expand** each scenario into variants that differ only in vendor and country.
3. **Query** the target models through OpenRouter with deterministic sampling (`temperature=0.1`, `top_p=0.9`) and bounded concurrency.
4. **Anonymize** every response before scoring: vendor names become `[VENDOR]`, countries `[COUNTRY]`, model names `[MODEL]`.
5. **Judge** the anonymized response against the list of injected risks and obtain a structured evaluation.
6. **Report** the metrics as a Markdown summary with per-scenario tables, a risk-score matrix and an explicit limitations section.

## Quick start

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -e ".[dev]"
cp .env.example .env
# Add OPENROUTER_API_KEY to .env before a live run.

python -m knbench.main validate --scenarios data/scenarios.json --strict-pairs
python -m knbench.main run --limit 1
python -m knbench.main report --results output/raw_results.json
```

The console entry point is `knbench`, so the same commands can be run as
`knbench validate ...`, `knbench run ...` and `knbench report ...`.

The default judge is `anthropic/claude-opus-5.5`; the default targets are
`openai/gpt-6.1-sol` and `google/gemini-4-carbon`. Pass `--models` or
`--judge-model` to use other OpenRouter ids. `--dry-run` validates the plan
without an API key or network calls. Start with `--limit 1` to control spend.

The offline test suite uses mocked HTTP:

```bash
python -m pytest -q
python -m mypy --strict knbench
```

## Language policy

- **Model-facing content is English only.** Documents, task prompts, system prompts and the judge rubric are never sent in another language, because the test language would otherwise compete with the vendor's country as an explanation for a score difference.
- **Technical reference documentation is English** and is published under `docs/`.
- **The overview and user instructions are published in three languages**: English, Kazakh and Russian, kept structurally identical.

## Project status

The framework is implemented and runnable offline: the dataset loader, OpenRouter client with catalog preflight and retries, anonymizing judge, metrics, eight-section report, CLI and provenance sidecar are in place. The shipped dataset contains three synthetic scenarios and twelve variants.

A live OpenRouter smoke test passed on 2026-11-12 using one target (`openai/gpt-6.1-sol`), one judge (`anthropic/claude-opus-5.5`), one synthetic scenario and one judge vote. The default model ids are pinned in `.env.example` and checked against the live catalog before any target request. Results are descriptive and are not a security or legal assessment.

User instructions are in [`docs/user-guide.md`](docs/user-guide.md), with the dataset, methodology and configuration references under `docs/`.

## Limitations

- Sample sizes are small; results are descriptive and no statistical significance is claimed.
- Anonymization removes explicit identifiers but not indirect hints such as distinctive phrasing or rare technologies.
- Scores depend on the prompt wording and on the judge model; runs are comparable only when both are held fixed.
- Only synthetic documents may be used: they are transmitted to a third-party API.
- A live model run consumes OpenRouter credits; use `--limit`, `--judge-votes` and `--skip-judge` while developing.

## Documentation

| Document | Content |
|----------|---------|
| [`docs/user-guide.md`](docs/user-guide.md) | Step-by-step usage guide |
| [`docs/dataset-format.md`](docs/dataset-format.md) | Dataset reference |
| [`docs/methodology.md`](docs/methodology.md) | Metric definitions and threats to validity |
| [`docs/configuration.md`](docs/configuration.md) | Environment variables and flags |

Each run writes a `*.provenance.json` sidecar next to its results file. It records the run id, Python and dependency versions, optional git revision, dataset hash, model ids and prompt hashes without storing prompts or responses.

## License

The source code and technical documentation are licensed under the [Apache License 2.0](LICENSE). The synthetic dataset and user-facing documentation are licensed under [CC BY 4.0](LICENSE-DATA). See both files for the full terms.
