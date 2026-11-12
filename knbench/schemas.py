"""Pydantic models for KNBench (TECHSPEC.md §5, §5.1, §5.2).

All models are strict so that a typo in JSON fails loudly instead of passing
silently. Format versions are pinned by ``schema_version``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION: Final = "1.0"

VARIANT_ID_PATTERN: Final = r"^[a-z0-9][a-z0-9_]{0,31}$"
SCENARIO_ID_PATTERN: Final = r"^[a-z0-9][a-z0-9_-]{0,47}$"


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


# ---------- Canonical domain models (source of truth) ----------


class InjectedRisk(StrictModel):
    risk_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    description: str = Field(min_length=1)


class DocumentVariant(StrictModel):
    variant_id: str = Field(pattern=VARIANT_ID_PATTERN)
    vendor_name: str = Field(min_length=1)
    origin_country: str = Field(min_length=1)
    document_text: str = Field(min_length=1)
    is_control: bool = False
    vendor_aliases: list[str] = Field(default_factory=list)
    country_aliases: list[str] = Field(default_factory=list)


class BenchmarkScenario(StrictModel):
    scenario_id: str = Field(pattern=SCENARIO_ID_PATTERN)
    domain: str = Field(min_length=1)
    injected_risks: list[InjectedRisk] = Field(min_length=1)
    variants: list[DocumentVariant] = Field(min_length=1)
    # Effective task prompt for this scenario (EXECPLAN S8). Optional so that
    # results written before the field existed still validate; the loader
    # always populates it with the prompt actually sent to the models.
    task_prompt: str | None = None

    @field_validator("injected_risks")
    @classmethod
    def _unique_risks(cls, v: list[InjectedRisk]) -> list[InjectedRisk]:
        ids = [risk.risk_id for risk in v]
        if len(ids) != len(set(ids)):
            raise ValueError("risk_id values must be unique within a scenario")
        return v

    @model_validator(mode="after")
    def _unique_variants(self) -> BenchmarkScenario:
        ids = [variant.variant_id for variant in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("variant_id values must be unique within a scenario")
        if sum(variant.is_control for variant in self.variants) > 1:
            raise ValueError("at most one control variant is allowed")
        return self


# ---------- Response and evaluation models ----------


class RequestMeta(StrictModel):
    temperature: float
    top_p: float
    attempts: int = Field(ge=1)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class TargetModelResponse(StrictModel):
    scenario_id: str
    variant_id: str
    model_name: str
    raw_response: str
    execution_time_ms: int = Field(ge=0)
    request_meta: RequestMeta | None = None


class JudgeMeta(StrictModel):
    model_name: str
    votes: int = Field(ge=1)
    attempts: int = Field(ge=1)
    anonymized_prompt_sha256: str
    anonymization_version: str = "1"
    redactions: dict[str, int] = Field(default_factory=dict)
    rejected_ids: list[str] = Field(default_factory=list)


class JudgeEvaluation(StrictModel):
    detected_risks: list[str]
    risk_score: float = Field(ge=0.0, le=100.0)
    recommendation: Literal["APPROVE", "MODIFY", "REJECT"]
    judge_reasoning: str = Field(min_length=1)
    judge_meta: JudgeMeta | None = None

    @field_validator("recommendation", mode="before")
    @classmethod
    def _normalize_recommendation(cls, v: object) -> object:
        return v.strip().upper() if isinstance(v, str) else v


# ---------- Dataset input models (§5.1) ----------


class VariantEntry(StrictModel):
    variant_id: str = Field(pattern=VARIANT_ID_PATTERN)
    vendor_name: str = Field(min_length=1)
    origin_country: str = Field(min_length=1)
    document_text: str | None = None
    is_control: bool = False
    vendor_aliases: list[str] = Field(default_factory=list)
    country_aliases: list[str] = Field(default_factory=list)


class PromptTemplate(StrictModel):
    document_text: str = Field(min_length=1)
    task_prompt: str | None = None


class ScenarioEntry(StrictModel):
    scenario_id: str = Field(pattern=SCENARIO_ID_PATTERN)
    domain: str = Field(min_length=1)
    notes: str | None = None
    injected_risks: list[InjectedRisk] = Field(min_length=1)
    template: PromptTemplate | None = None
    variants: list[VariantEntry] = Field(min_length=1)


class ScenarioDefaults(StrictModel):
    task_prompt: str | None = None


class ScenarioFile(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    defaults: ScenarioDefaults | None = None
    scenarios: list[ScenarioEntry] = Field(min_length=1)


# ---------- Result and metrics models (§5.2) ----------


class RecordError(StrictModel):
    stage: Literal["target", "judge"]
    error_type: str
    message: str
    attempts: int = Field(ge=1)


class EvaluationRecord(StrictModel):
    scenario_id: str
    variant_id: str
    model_name: str
    status: Literal["ok", "target_failed", "judge_failed", "skipped"]
    target_response: TargetModelResponse | None = None
    evaluation: JudgeEvaluation | None = None
    errors: list[RecordError] = Field(default_factory=list)


class RunConfig(StrictModel):
    models: list[str] = Field(min_length=1)
    judge_model: str
    temperature: float = 0.1
    top_p: float = 0.9
    concurrency: int = Field(ge=1, le=32)
    max_retries: int = Field(ge=0)
    timeout_s: float = Field(gt=0)
    judge_votes: int = Field(ge=1)
    anonymize: bool
    scenarios_path: str
    scenarios_sha256: str
    app_version: str
    strict_pairs: bool = False
    resumed_from: str | None = None


class RunResults(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    run_config: RunConfig
    scenarios: list[BenchmarkScenario]
    records: list[EvaluationRecord]


class ScenarioMetrics(StrictModel):
    scenario_id: str
    model_name: str
    variants_expected: int
    variants_evaluated: int
    risk_blindness_rate: float | None
    bias_delta: float | None
    decision_flip: bool | None
    control_delta: float | None = None
    partial: bool = False


class ModelMetrics(StrictModel):
    model_name: str
    scenarios_evaluated: int
    mean_risk_blindness_rate: float | None
    mean_bias_delta: float | None
    max_bias_delta: float | None
    decision_flip_scenarios: int
    low_coverage: bool = False


class Coverage(StrictModel):
    records_total: int
    records_ok: int
    records_failed: int
    # Records collected under --skip-judge: they have a target response but no
    # evaluation, so they are neither ok nor failed (EXECPLAN S1).
    records_skipped: int = 0


class MetricsFile(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: str
    generated_at: datetime
    coverage: Coverage
    by_scenario: list[ScenarioMetrics]
    by_model: list[ModelMetrics]
