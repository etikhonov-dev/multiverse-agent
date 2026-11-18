# ADR-0002: Outcome-Blind Stage-1 Agent & Deterministic Timing Firewall

- **Status:** Accepted
- **Date:** 2026-11-18

## Context

Recent evaluations of autonomous LLM data-science agents ([arXiv:2602.18710](https://arxiv.org/abs/2602.18710)) document two systematic failures when agents inspect full datasets and iteratively refine analyses:
1. **Confirmation steering (34–66 percentage points):** Agents peeking at outcome distributions or early p-values selectively tune washout windows, subgroup filters, or covariate sets toward a desired result.
2. **Causal leakage:** Agents adjust for post-baseline mediators (e.g., `post_tx_statin`), healthcare-utilization colliders (e.g., `hospitalized_past_year`), or prevalent comorbidity flags that induce collider stratification or over-adjustment bias.

## Decision

1. **Physical Outcome-Blind Schema (`OutcomeBlindSchema`):** `OutcomeBlindSchema.from_cohort(bundle)` physically drops all `VariableRole.DURATION`, `VariableRole.EVENT`, and `CollectionTiming.OUTCOME_FOLLOWUP` columns before computing summary quantiles and missingness rates. CDC `UCOD_LEADING` is stripped at ingestion (`fetch_nhanes_cohort.py`).
2. **Two-Stage Separation:**
   - Stage 1 (`OutcomeBlindArchitect`) receives *only* `OutcomeBlindSchema` and the study hypothesis, never `CohortBundle` or `RunRecord`s. Stage-1 and Stage-2 offline replay fixtures (`fixtures/replay_architect_plans.json` and `fixtures/replay_synthesizer_reports.json`) are physically separated.
   - Stage 2 (`GroundedSynthesizer`) executes strictly *after* `MultiverseRunner` seals `runs.jsonl` and cannot trigger additional model runs.
3. **Dual `CollectionTiming` + `VariableRole` Firewall:** Both `OutcomeBlindArchitect` and `SpecificationSpace.compile()` enforce ground-truth `CollectionTiming` (`POST_BASELINE_FOLLOWUP`, `HEALTHCARE_UTILIZATION_SELECTION`, `OUTCOME_FOLLOWUP`) and `VariableRole` (`MEDIATOR`, `COLLIDER`, `COMORBIDITY_FLAG`, `EXPOSURE`, `DURATION`, `EVENT`). Even if an adversarial LLM mislabels a collider or mediator as `"confounder"`, it is deterministically stripped/blocked and logged as a `FirewallAuditEvent`.

## Rejected Alternatives

- **Prompt-only instructions ("Please do not look at outcomes or use colliders"):** Rejected because prompt-only guardrails fail silently under LLM misclassification or adversarial steering.
- **Executing LLM-generated Python code (`exec`/`eval`):** Rejected for security, reproducibility, and leakage risk.

## Consequences

- Serialized Stage-1 prompts and fixtures are provably free of outcome/survival fields (`mortstat`, `permth`, `ucod`, `event`, `duration`, `run_id`).
- Pre-execution specification design is 100% invariant to outcome permutations and reproducible across `replay`, `ollama`, `openai`, and `anthropic` backends.
