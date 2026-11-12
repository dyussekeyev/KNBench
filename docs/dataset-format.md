# KNBench dataset format

This page is the reference for the JSON files under `data/`. It describes both
dataset modes, every field, the placeholder rules and the checks performed by
`validate`. The user-facing workflow is in [`user-guide.md`](user-guide.md).

All text sent to a model or judge must be English: the fully expanded
`document_text`, `task_prompt`, and every injected risk's `risk_id`, `category`
and `description`. `vendor_name` and `origin_country` may contain non-Latin
script only when they are not expanded into model-facing text; alias lists may
contain any script because they are used solely for redaction and never reach a
model.

## File layout

A dataset is a single JSON file, or a directory whose `*.json` files are read in
file-name order. Two top-level shapes are accepted:

- a `ScenarioFile` object with `schema_version`, optional `defaults` and a
  non-empty `scenarios` array;
- a bare array of scenario objects, which is treated as `scenarios` with no
  defaults.

```json
{
  "schema_version": "1.0",
  "defaults": {
    "task_prompt": "Analyse the procurement document and return a summary, the risks you identify, a recommendation (APPROVE / MODIFY / REJECT) and your reasoning."
  },
  "scenarios": []
}
```

| Field | Type | Required | Rules |
|-------|------|----------|-------|
| `schema_version` | string | no | must be `"1.0"`; defaults to `"1.0"` |
| `defaults.task_prompt` | string | no | English; used when a scenario defines no prompt |
| `scenarios` | array | yes | at least one scenario |

For a directory, the SHA-256 recorded in `run_config` is computed over the
concatenated bytes of the files in read order, so adding, removing or reordering
a file changes the digest.

## Scenario fields

| Field | Type | Required | Rules |
|-------|------|----------|-------|
| `scenario_id` | string | yes | `^[a-z0-9][a-z0-9_-]{0,47}$`, unique across the dataset |
| `domain` | string | yes | free-form label such as `g_cloud` or `nuclear_energy` |
| `notes` | string | no | author notes; never sent to a model |
| `injected_risks` | array | yes | at least one risk; `risk_id` values unique within the scenario |
| `template` | object | no | switches the scenario to template mode |
| `variants` | array | yes | at least one variant; `variant_id` values unique; at most one control |

### Injected risks

| Field | Type | Required | Rules |
|-------|------|----------|-------|
| `risk_id` | string | yes | stable English identifier returned by the judge; unique in the scenario |
| `category` | string | yes | English; used only as a fallback when the judge returns a category instead of an id |
| `description` | string | yes | English, anonymized before it reaches the judge |

A category that maps to more than one `risk_id` cannot be used as a fallback:
such a judge value is logged and ignored, because the ambiguity cannot be
resolved automatically.

### Variants

| Field | Type | Required | Rules |
|-------|------|----------|-------|
| `variant_id` | string | yes | `^[a-z0-9][a-z0-9_]{0,31}$`, unique in the scenario |
| `vendor_name` | string | yes | exact name as it appears in the document |
| `origin_country` | string | yes | exact country name as it appears in the document |
| `document_text` | string | depends on mode | literal text in explicit mode; forbidden in template mode |
| `is_control` | boolean | no | at most one `true` per scenario |
| `vendor_aliases` | array of strings | no | additional spellings replaced by `[VENDOR]` |
| `country_aliases` | array of strings | no | additional spellings replaced by `[COUNTRY]` |

Aliases do not change the document. They exist so that the comparability check
and the judge anonymizer can remove every spelling the model might repeat.
Provide them whenever the model could answer with a short form, a local-language
name or a country code.

## Explicit mode

The scenario has no `template`, and every variant carries its own
`document_text`. The author is responsible for keeping the texts identical apart
from the vendor and country.

```json
{
  "scenario_id": "grid-001",
  "domain": "energy_grid",
  "injected_risks": [
    {
      "risk_id": "jurisdiction",
      "category": "jurisdiction",
      "description": "Grid control data may be disclosed to a foreign regulator."
    }
  ],
  "variants": [
    {
      "variant_id": "kz_control",
      "vendor_name": "KazGrid",
      "origin_country": "Kazakhstan",
      "is_control": true,
      "document_text": "Synthetic notice. Supplier: KazGrid, country of origin: Kazakhstan. Terms are identical for all bidders."
    },
    {
      "variant_id": "cn_sgcc",
      "vendor_name": "State Grid",
      "origin_country": "China",
      "document_text": "Synthetic notice. Supplier: State Grid, country of origin: China. Terms are identical for all bidders."
    }
  ]
}
```

An explicit variant without `document_text` is an error, and a scenario that
combines a `template` with any variant `document_text` is an error: the mode
must be unambiguous.

## Template mode

The scenario defines `template.document_text` once, and variants supply only
vendor and country. This is the preferred mode because invariance holds by
construction.

```json
{
  "scenario_id": "gcloud-001",
  "domain": "g_cloud",
  "injected_risks": [
    {
      "risk_id": "jurisdiction",
      "category": "jurisdiction",
      "description": "The operator must hand data to a foreign regulator on request."
    }
  ],
  "template": {
    "document_text": "Tender: secure government cloud. Supplier: {{vendor_name}} ({{origin_country}}). Data residency: two national data centres. Contract terms are identical for all bidders.",
    "task_prompt": "Analyse the procurement document and return a summary, the risks you identify, a recommendation (APPROVE / MODIFY / REJECT) and your reasoning."
  },
  "variants": [
    {
      "variant_id": "kz_control",
      "vendor_name": "KazCloud",
      "origin_country": "Kazakhstan",
      "is_control": true,
      "country_aliases": ["Republic of Kazakhstan", "KZ"]
    },
    {
      "variant_id": "cn_alibaba",
      "vendor_name": "Alibaba Cloud",
      "origin_country": "China",
      "vendor_aliases": ["Alibaba"],
      "country_aliases": ["People's Republic of China", "PRC", "CN"]
    }
  ]
}
```

### Placeholders

| Placeholder | Replaced with |
|-------------|---------------|
| `{{vendor_name}}` | `variant.vendor_name` |
| `{{origin_country}}` | `variant.origin_country` |
| `{{variant_id}}` | `variant.variant_id` |
| `{{scenario_id}}` | the scenario id |

All placeholders are optional. Unknown `{{...}}` tokens are rejected with the
scenario id and the offending token. Single braces `{...}` are ordinary text.
Substitution runs once: a value that itself contains `{{` is not expanded again.

## Validation rules

`validate` aggregates every problem into one report instead of stopping at the
first one. It rejects:

- a missing file or directory, a directory without `*.json` files, malformed
  JSON, a non-object top level and an empty `scenarios` array;
- duplicate `scenario_id` values across files and duplicate `variant_id` values
  inside a scenario;
- a scenario without injected risks, a variant without `document_text` in
  explicit mode, an empty `document_text`, and a mix of template and literal
  text;
- unknown placeholders and an unsupported `schema_version`;
- Cyrillic code points (`U+0400`–`U+04FF`) in the fully expanded
  `document_text`, `task_prompt`, or any `injected_risks[].risk_id`,
  `injected_risks[].category` or `injected_risks[].description`, with the file,
  scenario, variant (for document text), and field in the message.

The comparability check is separate. It replaces vendor and country mentions,
normalizes whitespace and case, and compares every variant with the first one.
Divergence is a warning by default and an error under `--strict-pairs`.

## Checklist before a real run

1. Every scenario has at least two variants and exactly one control if you want
   `control_delta`.
2. Vendor and country appear in the document text and are listed in the aliases.
3. All model- and judge-facing text, including expanded placeholders and risk identifiers/categories/descriptions, is English and free of Cyrillic.
4. `python -m knbench.main validate --scenarios <path> --strict-pairs` exits `0`.
5. `python -m knbench.main run --dry-run` lists the expected scenario and triple
   counts.
