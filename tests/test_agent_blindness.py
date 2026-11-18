"""Tests for OutcomeBlindArchitect, LLMBackend providers, and Stage-1 outcome blindness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import httpx
import pytest

from multiverse_agent.agent import (
    DEFAULT_ARCHITECT_FIXTURE,
    AnthropicBackend,
    OllamaBackend,
    OpenAICompatBackend,
    OutcomeBlindArchitect,
    ReplayBackend,
    build_architect_prompts,
    build_backend,
)
from multiverse_agent.cohort import (
    OutcomeBlindSchema,
    load_nhanes_mortality_cohort,
    make_synthetic_bias_cohort,
)

FORBIDDEN_OUTCOME_SUBSTRINGS = (
    "mortstat",
    "permth",
    "ucod",
    "event",
    "duration",
    "run_id",
)


class _SpyCapturingBackend:
    def __init__(self, response_payload: dict[str, object]) -> None:
        self._response = json.dumps(response_payload)
        self.captured_prompts: list[tuple[str, str, str, str]] = []

    @property
    def name(self) -> str:
        return "spy"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: Literal["architect", "synthesizer"],
        context_key: str,
    ) -> str:
        self.captured_prompts.append((system_prompt, user_prompt, stage, context_key))
        return self._response


def test_stage1_fixture_and_prompts_contain_zero_outcome_substrings() -> None:
    fixture_text = DEFAULT_ARCHITECT_FIXTURE.read_text(encoding="utf-8").lower()
    for token in FORBIDDEN_OUTCOME_SUBSTRINGS:
        assert token not in fixture_text

    for cohort in (
        make_synthetic_bias_cohort(n=400, seed=42),
        load_nhanes_mortality_cohort(),
    ):
        blind_schema = OutcomeBlindSchema.from_cohort(cohort)
        system_prompt, user_prompt = build_architect_prompts(
            blind_schema,
            hypothesis="Association between baseline log CRP and all-cause mortality.",
        )
        combined = f"{system_prompt}\n{user_prompt}".lower()
        for token in FORBIDDEN_OUTCOME_SUBSTRINGS:
            assert token not in combined
        assert "duration_months" not in combined
        assert blind_schema.cohort_name in combined


def test_replay_backend_compiles_validated_spaces_for_nhanes_and_synthetic() -> None:
    backend = ReplayBackend()
    architect = OutcomeBlindArchitect(backend)

    nhanes = load_nhanes_mortality_cohort()
    nhanes_schema = OutcomeBlindSchema.from_cohort(nhanes)
    nhanes_plan = architect.plan(
        nhanes_schema,
        hypothesis="Baseline systemic inflammation (log CRP) and long-term mortality.",
    )
    nhanes_specs = nhanes_plan.space.compile(nhanes, declared_roles=nhanes_plan.declared_roles)
    assert len(nhanes_specs) == 108
    assert nhanes_plan.blocked_covariates == ()

    synth = make_synthetic_bias_cohort(n=400, seed=42)
    synth_schema = OutcomeBlindSchema.from_cohort(synth)
    synth_plan = architect.plan(
        synth_schema,
        hypothesis="Baseline log CRP and mortality under acute reverse causation.",
    )
    synth_specs = synth_plan.space.compile(synth, declared_roles=synth_plan.declared_roles)
    assert len(synth_specs) == 108
    assert synth_plan.blocked_covariates == ()


def test_adversarial_llm_collider_and_mediator_are_blocked_and_logged() -> None:
    nhanes = load_nhanes_mortality_cohort()
    nhanes_schema = OutcomeBlindSchema.from_cohort(nhanes)

    # Adversarial LLM deliberately mislabels collider 'hospitalized_past_year' as 'confounder'
    # and injects it into the adjustment set.
    adv_nhanes_payload: dict[str, object] = {
        "declared_roles": {
            "log_crp_z": "exposure",
            "age_years": "confounder",
            "sex_male": "confounder",
            "hospitalized_past_year": "confounder",
        },
        "age_floors": [40],
        "cohort_definitions": ["all_eligible"],
        "washout_months": [0, 12],
        "followup_horizons_months": [None],
        "missingness_strategies": ["complete_case"],
        "adjustment_sets": {
            "minimal": ["age_years", "sex_male"],
            "with_collider": ["age_years", "sex_male", "hospitalized_past_year"],
        },
        "estimators": ["piecewise_exponential"],
        "rationale": "Adversarial test attempting to leak utilization collider.",
    }
    spy_nhanes = _SpyCapturingBackend(adv_nhanes_payload)
    plan_nhanes = OutcomeBlindArchitect(spy_nhanes).plan(nhanes_schema, "CRP test")

    assert len(spy_nhanes.captured_prompts) == 1
    sys_p, usr_p, stage, _ = spy_nhanes.captured_prompts[0]
    assert stage == "architect"
    for token in FORBIDDEN_OUTCOME_SUBSTRINGS:
        assert token not in f"{sys_p}\n{usr_p}".lower()

    for covs in plan_nhanes.space.adjustment_sets.values():
        assert "hospitalized_past_year" not in covs
    assert any(ev.covariate == "hospitalized_past_year" for ev in plan_nhanes.blocked_covariates)
    # Stripping hospitalized_past_year collapses 'with_collider' into 'minimal', so it is dropped
    assert list(plan_nhanes.space.adjustment_sets.keys()) == ["minimal"]

    # Adversarial LLM on Tier 1 synthetic cohort mislabeling mediator 'post_tx_statin'
    synth = make_synthetic_bias_cohort(n=400, seed=42)
    synth_schema = OutcomeBlindSchema.from_cohort(synth)
    adv_synth_payload: dict[str, object] = {
        "declared_roles": {
            "log_crp_z": "exposure",
            "age_years": "confounder",
            "sex_male": "confounder",
            "bmi": "confounder",
            "post_tx_statin": "confounder",
        },
        "age_floors": [20],
        "cohort_definitions": ["all_eligible"],
        "washout_months": [0],
        "followup_horizons_months": [None],
        "missingness_strategies": ["complete_case"],
        "adjustment_sets": {
            "minimal": ["age_years", "sex_male"],
            "leaky": ["age_years", "sex_male", "bmi", "post_tx_statin"],
        },
        "estimators": ["cox_breslow"],
        "rationale": "Adversarial test attempting to adjust for post-baseline mediator.",
    }
    plan_synth = OutcomeBlindArchitect(_SpyCapturingBackend(adv_synth_payload)).plan(
        synth_schema, "CRP test"
    )
    assert plan_synth.space.adjustment_sets["leaky"] == ("age_years", "bmi", "sex_male")
    assert any(ev.covariate == "post_tx_statin" for ev in plan_synth.blocked_covariates)
    # Stripped space still compiles cleanly
    assert len(plan_synth.space.compile(synth, declared_roles=plan_synth.declared_roles)) == 2


def test_architect_rejects_when_all_adjustment_sets_are_illegal() -> None:
    synth = make_synthetic_bias_cohort(n=400, seed=42)
    synth_schema = OutcomeBlindSchema.from_cohort(synth)
    all_illegal: dict[str, object] = {
        "declared_roles": {"log_crp_z": "exposure", "post_tx_statin": "confounder"},
        "age_floors": [20],
        "cohort_definitions": ["all_eligible"],
        "washout_months": [0],
        "followup_horizons_months": [None],
        "missingness_strategies": ["complete_case"],
        "adjustment_sets": {"only_mediator": ["post_tx_statin"]},
        "estimators": ["cox_breslow"],
    }
    with pytest.raises(ValueError, match="rejected"):
        OutcomeBlindArchitect(_SpyCapturingBackend(all_illegal)).plan(synth_schema, "CRP test")


def test_http_backends_parse_mocked_responses_without_network() -> None:
    def _ollama_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/chat")
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "llama3.1:8b"
        return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})

    ollama = OllamaBackend(transport=httpx.MockTransport(_ollama_handler))
    assert (
        ollama.complete(
            system_prompt="sys", user_prompt="usr", stage="architect", context_key="nhanes"
        )
        == '{"ok": true}'
    )

    def _openai_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "openai-ok"}}]},
        )

    openai = OpenAICompatBackend(
        api_key="test-key",
        transport=httpx.MockTransport(_openai_handler),
    )
    assert (
        openai.complete(
            system_prompt="sys", user_prompt="usr", stage="synthesizer", context_key="nhanes"
        )
        == "openai-ok"
    )

    def _anthropic_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/messages")
        assert request.headers["x-api-key"] == "ant-key"
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "anthropic-ok"}]},
        )

    anthropic = AnthropicBackend(
        api_key="ant-key",
        transport=httpx.MockTransport(_anthropic_handler),
    )
    assert (
        anthropic.complete(
            system_prompt="sys", user_prompt="usr", stage="synthesizer", context_key="nhanes"
        )
        == "anthropic-ok"
    )

    assert isinstance(build_backend("replay"), ReplayBackend)
    assert isinstance(build_backend("ollama"), OllamaBackend)
    assert isinstance(build_backend("openai", api_key="k"), OpenAICompatBackend)
    assert isinstance(build_backend("anthropic", api_key="k"), AnthropicBackend)
    with pytest.raises(ValueError, match="Unknown backend"):
        build_backend("unknown")


def test_replay_backend_keeps_stage1_and_stage2_physically_separate(tmp_path: Path) -> None:
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    s1.write_text('{"nhanes": {"stage": 1}}', encoding="utf-8")
    s2.write_text('{"nhanes": "stage-2-report"}', encoding="utf-8")
    backend = ReplayBackend(architect_fixture_path=s1, synthesizer_fixture_path=s2)

    assert (
        backend.complete(
            system_prompt="s",
            user_prompt="u",
            stage="architect",
            context_key="nhanes_2007_2010_mort",
        )
        == '{"stage": 1}'
    )
    assert (
        backend.complete(
            system_prompt="s",
            user_prompt="u",
            stage="synthesizer",
            context_key="nhanes_2007_2010_mort",
        )
        == "stage-2-report"
    )


def test_architect_strips_code_fences_blocks_exposure_and_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FencedBackend:
        @property
        def name(self) -> str:
            return "fenced"

        def complete(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            stage: Literal["architect", "synthesizer"],
            context_key: str,
        ) -> str:
            _ = (system_prompt, user_prompt, stage, context_key)
            payload = {
                "declared_roles": {"log_crp_z": "exposure", "age_years": "confounder"},
                "age_floors": [40],
                "cohort_definitions": ["all_eligible"],
                "washout_months": [0],
                "followup_horizons_months": [None],
                "missingness_strategies": ["complete_case"],
                "adjustment_sets": {
                    "clean": ["age_years", "log_crp_z", "nonexistent_col"],
                },
                "estimators": ["cox_breslow"],
            }
            return f"```json\n{json.dumps(payload)}\n```"

    nhanes = load_nhanes_mortality_cohort()
    plan = OutcomeBlindArchitect(_FencedBackend()).plan(
        OutcomeBlindSchema.from_cohort(nhanes),
        "CRP test",
    )
    assert plan.space.adjustment_sets == {"clean": ("age_years",)}
    assert {ev.covariate for ev in plan.blocked_covariates} == {"log_crp_z", "nonexistent_col"}

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAICompatBackend(api_key="").complete(
            system_prompt="s", user_prompt="u", stage="architect", context_key="nhanes"
        )
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend(api_key="").complete(
            system_prompt="s", user_prompt="u", stage="architect", context_key="nhanes"
        )
