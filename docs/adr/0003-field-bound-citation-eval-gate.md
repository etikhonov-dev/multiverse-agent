# ADR-0003: Field-Bound Mechanical Citation Gate (`eval_gate.py`)

- **Status:** Accepted
- **Date:** 2026-11-18

## Context

Even when statistical execution is deterministic and outcome-blind, an LLM summarizing multiverse outputs can still hallucinate numbers, swap hazard ratios across runs in the same sentence ("musical chairs"), flip a borderline $p$-value across $\alpha = 0.05$ (`0.052` reported as `0.048`), emit bare citation tags without bound values, or cherry-pick a secondary Sobol-Shapley driver while omitting the primary source of variance.

A naive verifier that checks only whether cited numbers exist *somewhere* in `runs.jsonl` with a uniform `abs_tol = 0.01` fails to catch every one of these bypasses.

## Decision

Implement `verify_grounded_report(markdown_text, run_records, curve_summary)` in `src/multiverse_agent/eval_gate.py` enforcing five mechanical checks:

1. **Field-Bound Syntax:** Every numeric claim must immediately precede `[run:<hex_prefix_8+> :<field>]` (no spaces inside tag, i.e. `[run:<id8>:<field>]`) or `[curve:<key>]`. Bare tags without a preceding numeric literal fail with `unbound_citation_tag`.
2. **Unambiguous Converged Resolution:** Every `run:<id8>` prefix must uniquely resolve to a single `converged=True` `RunRecord` (`unknown_run_id`, `ambiguous_run_id`, `non_converged_run_cited`).
3. **Metric-Specific Tolerances & Significance-Side Guard:**
   - Counts (`n_subjects`, `n_events`, `n_specs_*`): exact integer equality (`abs_tol = 0`).
   - `p_value`: `< 0.0001` threshold allowed iff `logged < 1e-4`; otherwise `rel_tol <= 2e-2` **and** `(claimed < 0.05) == (logged < 0.05)` (`significance_flip` on boundary crossing).
   - Proportions & Shapley shares: `%` normalized by `/100`, matched within `abs_tol = 2e-3`.
   - Effect/index floats (`hr`, `ci_lower_95`, `ci_upper_95`, `log_hr`, `concordance_index`, `person_years`): matched within `(rel_tol=5e-3, abs_tol=5e-3)`.
4. **Uncited Prose Scanner:** After stripping fenced code blocks, ATX headers, backticked code spans, matched `<literal> [tag]` pairs, and a narrow structural allowlist (`95% CI`, `1-SD`, `SHA-256`, `alpha = 0.05`), any remaining decimal or `%` token raises `ungrounded_number`.
5. **Anti-Cherry-Picking Completeness:** Requires bound citations for all 7 mandatory curve keys (`n_specs_converged`, `median_hr`, `min_hr`, `max_hr`, `share_ci_excluding_one`, `sign_consistency_share`, and `shapley.<true_top_driver>`).

## Rejected Alternatives

- **Uniform `abs_tol` across all fields:** Rejected because `0.005` tolerance allows `p = 0.052 -> 0.048` significance flips and `50%` relative errors on small `p_value`s.
- **LLM-as-a-judge verification:** Rejected because probabilistic judges cannot guarantee deterministic, zero-cost mechanical auditability in CI.

## Consequences

- `multiverse-agent verify` provides a deterministic exit-code `2` gate suitable for CI and tamper detection.
- Both deterministic templates and live LLM backends (`ollama`, `openai`, `anthropic`) are held to the identical mechanical ground-truth contract.
