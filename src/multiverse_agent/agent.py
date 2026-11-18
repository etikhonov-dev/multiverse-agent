"""Outcome-blind Stage-1 LLM architect and pluggable LLMBackend providers."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from multiverse_agent.cohort import OutcomeBlindSchema, VariableRole
from multiverse_agent.spec import (
    FORBIDDEN_ADJUSTMENT_ROLES,
    FORBIDDEN_ADJUSTMENT_TIMINGS,
    AdjustmentSet,
    CohortDefinition,
    EstimatorFamily,
    MissingnessStrategy,
    SpecificationSpace,
)

LOGGER = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
DEFAULT_ARCHITECT_FIXTURE = _FIXTURES_DIR / "replay_architect_plans.json"
DEFAULT_SYNTHESIZER_FIXTURE = _FIXTURES_DIR / "replay_synthesizer_reports.json"

_FENCE_RE = re.compile(r"^```(?:json|markdown|md)?\s*|\s*```$", re.IGNORECASE)


class LLMBackend(Protocol):
    """Minimal synchronous protocol for Stage-1 architect and Stage-2 synthesizer calls."""

    @property
    def name(self) -> str: ...

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: Literal["architect", "synthesizer"],
        context_key: str,
    ) -> str: ...


class ReplayBackend:
    """Offline deterministic backend reading strictly separated Stage-1/Stage-2 fixtures."""

    def __init__(
        self,
        *,
        architect_fixture_path: Path = DEFAULT_ARCHITECT_FIXTURE,
        synthesizer_fixture_path: Path = DEFAULT_SYNTHESIZER_FIXTURE,
    ) -> None:
        self._architect_path = architect_fixture_path
        self._synthesizer_path = synthesizer_fixture_path

    @property
    def name(self) -> str:
        return "replay"

    @staticmethod
    def _resolve_entry(data: dict[str, object], context_key: str) -> object:
        if context_key in data:
            return data[context_key]
        lowered = context_key.lower()
        for prefix in ("nhanes", "synthetic"):
            if lowered.startswith(prefix) and prefix in data:
                return data[prefix]
        raise KeyError(f"No replay fixture entry matching context_key '{context_key}'.")

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: Literal["architect", "synthesizer"],
        context_key: str,
    ) -> str:
        _ = (system_prompt, user_prompt)
        target = self._architect_path if stage == "architect" else self._synthesizer_path
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Replay fixture {target} must be a JSON object.")
        entry = self._resolve_entry(raw, context_key)
        if stage == "architect":
            if not isinstance(entry, dict):
                raise ValueError("Stage-1 architect replay entry must be a JSON object.")
            return json.dumps(entry, sort_keys=True)
        if not isinstance(entry, str):
            raise ValueError("Stage-2 synthesizer replay entry must be a markdown string.")
        return entry


class OllamaBackend:
    """Local Ollama HTTP chat adapter using httpx."""

    def __init__(
        self,
        *,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: Literal["architect", "synthesizer"],
        context_key: str,
    ) -> str:
        _ = (stage, context_key)
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        with httpx.Client(timeout=self._timeout_s, transport=self._transport) as client:
            resp = client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return str(data["message"]["content"])


class OpenAICompatBackend:
    """OpenAI-compatible /chat/completions HTTP adapter using httpx."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self._timeout_s = timeout_s
        self._transport = transport

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: Literal["architect", "synthesizer"],
        context_key: str,
    ) -> str:
        _ = (stage, context_key)
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAICompatBackend.")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        with httpx.Client(timeout=self._timeout_s, transport=self._transport) as client:
            resp = client.post(f"{self._base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return str(data["choices"][0]["message"]["content"])


class AnthropicBackend:
    """Anthropic /messages HTTP adapter using httpx."""

    def __init__(
        self,
        *,
        model: str = "claude-3-5-sonnet-latest",
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self._timeout_s = timeout_s
        self._transport = transport

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: Literal["architect", "synthesizer"],
        context_key: str,
    ) -> str:
        _ = (stage, context_key)
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicBackend.")
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self._model,
            "max_tokens": 2048,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        with httpx.Client(timeout=self._timeout_s, transport=self._transport) as client:
            resp = client.post(f"{self._base_url}/messages", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return str(data["content"][0]["text"])


def build_backend(
    backend_name: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMBackend:
    """Construct a configured LLMBackend by provider name."""
    key = backend_name.strip().lower()
    if key == "replay":
        return ReplayBackend()
    if key == "ollama":
        return OllamaBackend(
            model=model or "llama3.1:8b",
            base_url=base_url or "http://localhost:11434",
        )
    if key == "openai":
        return OpenAICompatBackend(
            model=model or "gpt-4o-mini",
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
        )
    if key == "anthropic":
        return AnthropicBackend(
            model=model or "claude-3-5-sonnet-latest",
            base_url=base_url or "https://api.anthropic.com/v1",
            api_key=api_key,
        )
    raise ValueError(f"Unknown backend '{backend_name}'.")


class ArchitectProposal(BaseModel):
    """Raw JSON structure expected from Stage-1 LLM backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    declared_roles: dict[str, VariableRole]
    age_floors: tuple[int, ...] = Field(min_length=1)
    cohort_definitions: tuple[CohortDefinition, ...] = Field(min_length=1)
    washout_months: tuple[int, ...] = Field(min_length=1)
    followup_horizons_months: tuple[int | None, ...] = Field(min_length=1)
    missingness_strategies: tuple[MissingnessStrategy, ...] = Field(min_length=1)
    adjustment_sets: dict[str, tuple[str, ...]]
    estimators: tuple[EstimatorFamily, ...] = Field(min_length=1)
    rationale: str = ""


class FirewallAuditEvent(BaseModel):
    """Structured record of an adjustment-set covariate or set blocked pre-execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adjustment_set: str
    covariate: str
    reason: str


class ArchitectPlan(BaseModel):
    """Validated, firewall-sanitized Stage-1 specification plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort_name: str
    cohort_sha256: str
    hypothesis: str
    declared_roles: dict[str, VariableRole]
    space: SpecificationSpace
    blocked_covariates: tuple[FirewallAuditEvent, ...]
    rationale: str
    prompt_payload: str


def build_architect_prompts(
    blind_schema: OutcomeBlindSchema,
    hypothesis: str,
) -> tuple[str, str]:
    """Serialize strictly outcome-blind schema metadata into Stage-1 prompts."""
    system_prompt = (
        "You are an outcome-blind epidemiological study architect. "
        "Given solely baseline covariate summaries and a causal hypothesis, "
        "classify each variable role and propose a SpecificationSpace JSON object. "
        "Never include the exposure, comorbidity flags, post-baseline follow-up mediators, "
        "or healthcare utilization colliders inside adjustment_sets. "
        "Respond with strictly valid JSON matching ArchitectProposal."
    )
    user_payload = {
        "hypothesis": hypothesis,
        "schema": blind_schema.model_dump(mode="json"),
        "allowed_cohort_definitions": [item.value for item in CohortDefinition],
        "allowed_missingness_strategies": [item.value for item in MissingnessStrategy],
        "allowed_estimators": [item.value for item in EstimatorFamily],
    }
    user_prompt = json.dumps(user_payload, sort_keys=True, indent=2)
    return system_prompt, user_prompt


def _strip_code_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned).strip()
    return cleaned


class OutcomeBlindArchitect:
    """Stage-1 outcome-blind specification architect with deterministic timing firewall."""

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    def plan(self, blind_schema: OutcomeBlindSchema, hypothesis: str) -> ArchitectPlan:
        """Propose and firewall-sanitize a SpecificationSpace from OutcomeBlindSchema only."""
        system_prompt, user_prompt = build_architect_prompts(blind_schema, hypothesis)
        raw_reply = self._backend.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage="architect",
            context_key=blind_schema.cohort_name,
        )
        proposal = ArchitectProposal.model_validate_json(_strip_code_fences(raw_reply))

        schema_cols = {col.name: col for col in blind_schema.columns}
        blocked: list[FirewallAuditEvent] = []
        sanitized_sets: dict[str, tuple[str, ...]] = {}
        seen_tuples: dict[tuple[str, ...], str] = {}

        for set_name, covariates in proposal.adjustment_sets.items():
            clean_covs: list[str] = []
            for cov in covariates:
                reason: str | None = None
                if cov not in schema_cols:
                    reason = f"unknown column '{cov}' absent from OutcomeBlindSchema"
                elif cov == blind_schema.exposure_col:
                    reason = f"primary exposure '{cov}' cannot be adjusted as a covariate"
                else:
                    col_meta = schema_cols[cov]
                    declared = proposal.declared_roles.get(cov)
                    if col_meta.timing in FORBIDDEN_ADJUSTMENT_TIMINGS:
                        reason = f"forbidden CollectionTiming '{col_meta.timing}'"
                    elif declared is not None and declared in FORBIDDEN_ADJUSTMENT_ROLES:
                        reason = f"forbidden declared VariableRole '{declared}'"

                if reason is not None:
                    ev = FirewallAuditEvent(adjustment_set=set_name, covariate=cov, reason=reason)
                    blocked.append(ev)
                    LOGGER.warning(
                        "Stage-1 firewall blocked covariate '%s' in set '%s': %s",
                        cov,
                        set_name,
                        reason,
                    )
                else:
                    clean_covs.append(cov)

            canonical = AdjustmentSet(name=set_name, covariates=tuple(clean_covs)).covariates
            if not canonical:
                ev = FirewallAuditEvent(
                    adjustment_set=set_name,
                    covariate="*",
                    reason="adjustment set became empty after firewall stripping",
                )
                blocked.append(ev)
                continue
            if canonical in seen_tuples:
                ev = FirewallAuditEvent(
                    adjustment_set=set_name,
                    covariate="*",
                    reason=(
                        f"collapsed into duplicate of '{seen_tuples[canonical]}' after stripping"
                    ),
                )
                blocked.append(ev)
                continue
            seen_tuples[canonical] = set_name
            sanitized_sets[set_name] = canonical

        if not sanitized_sets:
            raise ValueError("All proposed adjustment sets were rejected by the Stage-1 firewall.")

        space = SpecificationSpace(
            age_floors=proposal.age_floors,
            cohort_definitions=proposal.cohort_definitions,
            washout_months=proposal.washout_months,
            followup_horizons_months=proposal.followup_horizons_months,
            missingness_strategies=proposal.missingness_strategies,
            adjustment_sets=sanitized_sets,
            estimators=proposal.estimators,
        )
        return ArchitectPlan(
            cohort_name=blind_schema.cohort_name,
            cohort_sha256=blind_schema.cohort_sha256,
            hypothesis=hypothesis,
            declared_roles=dict(proposal.declared_roles),
            space=space,
            blocked_covariates=tuple(blocked),
            rationale=proposal.rationale,
            prompt_payload=f"{system_prompt}\n\n{user_prompt}",
        )


__all__ = [
    "AnthropicBackend",
    "ArchitectPlan",
    "ArchitectProposal",
    "DEFAULT_ARCHITECT_FIXTURE",
    "DEFAULT_SYNTHESIZER_FIXTURE",
    "FirewallAuditEvent",
    "LLMBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "OutcomeBlindArchitect",
    "ReplayBackend",
    "build_architect_prompts",
    "build_backend",
]
