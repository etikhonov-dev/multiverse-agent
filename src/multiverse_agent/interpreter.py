"""Stage-2 grounded markdown synthesizer bound to immutable RunRecord and curve outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from multiverse_agent.agent import LLMBackend
from multiverse_agent.curve import SPEC_DIMENSIONS, SpecCurveSummary
from multiverse_agent.eval_gate import AuditResult, verify_grounded_report
from multiverse_agent.runner import RunRecord
from multiverse_agent.spec import CohortDefinition


def select_notable_runs(records: Sequence[RunRecord]) -> dict[str, RunRecord]:
    """Select deterministic representative converged runs without post-hoc peeking."""
    converged = sorted(
        [
            rec
            for rec in records
            if rec.converged
            and rec.hr is not None
            and rec.ci_lower_95 is not None
            and rec.ci_upper_95 is not None
            and rec.p_value is not None
        ],
        key=lambda rec: (float(rec.hr or 0.0), rec.spec_id),
    )
    if not converged:
        raise ValueError("Cannot select notable runs from zero converged RunRecords.")

    hrs = [float(rec.hr or 0.0) for rec in converged]
    median_hr = hrs[len(hrs) // 2]
    min_run = converged[0]
    max_run = converged[-1]
    median_run = min(
        converged,
        key=lambda rec: (abs(float(rec.hr or 0.0) - median_hr), rec.spec_id),
    )

    selected: dict[str, RunRecord] = {
        "min_hr_run": min_run,
        "median_hr_run": median_run,
        "max_hr_run": max_run,
    }

    naive_candidates = [
        rec
        for rec in converged
        if rec.spec.washout_months == 0
        and rec.spec.cohort_definition == CohortDefinition.ALL_ELIGIBLE
    ]
    if naive_candidates:
        selected["naive_run"] = naive_candidates[-1]

    clean_candidates = [
        rec
        for rec in converged
        if rec.spec.washout_months >= 12
        and rec.spec.cohort_definition == CohortDefinition.SUPER_CONTROLS_CLEAN
    ]
    if clean_candidates:
        selected["clean_run"] = clean_candidates[len(clean_candidates) // 2]

    return selected


def _format_p_value_claim(p_value: float, id8: str) -> str:
    if p_value < 1e-4:
        return f"p < 0.0001 [run:{id8}:p_value]"
    return f"p = {p_value:.4g} [run:{id8}:p_value]"


def render_deterministic_report(
    curve_summary: SpecCurveSummary,
    notable_runs: Mapping[str, RunRecord],
    *,
    cohort_name: str,
    hypothesis: str,
) -> str:
    """Render a deterministic, audit-compliant markdown report from sealed results."""
    if curve_summary.n_specs_converged == 0 or curve_summary.median_hr is None:
        raise ValueError("Cannot synthesize report with zero converged specifications.")

    min_hr = float(curve_summary.min_hr or 0.0)
    max_hr = float(curve_summary.max_hr or 0.0)
    median_hr = float(curve_summary.median_hr)
    ci_pct = 100.0 * curve_summary.share_ci_excluding_one
    sign_pct = 100.0 * curve_summary.sign_consistency_share

    ranked_dims = sorted(
        SPEC_DIMENSIONS,
        key=lambda dim: (-curve_summary.shapley_shares[dim], dim),
    )
    top_dim = curve_summary.top_driver

    lines: list[str] = [
        f"# Multiverse Survival Report: {cohort_name}",
        "",
        f"**Hypothesis:** {hypothesis}",
        "",
        "## Specification Curve Summary",
        "",
        (
            f"Across {curve_summary.n_specs_converged} [curve:n_specs_converged] converged "
            f"specifications (out of {curve_summary.n_specs_total} [curve:n_specs_total] "
            f"planned), the median hazard ratio is {median_hr:.4f} [curve:median_hr] per 1-SD "
            f"increment in `log_crp_z`, spanning from {min_hr:.4f} [curve:min_hr] to "
            f"{max_hr:.4f} [curve:max_hr]. Across the multiverse, "
            f"{ci_pct:.2f}% [curve:share_ci_excluding_one] of specifications have 95% CI bounds "
            f"excluding unity and {sign_pct:.2f}% [curve:sign_consistency_share] exhibit "
            "consistent effect direction."
        ),
        "",
        "## Exact Owen Sobol-Shapley Variance Attribution",
        "",
    ]

    if top_dim is not None:
        top_pct = 100.0 * curve_summary.shapley_shares[top_dim]
        lines.append(
            f"The top driver of cross-specification log-HR variance is `{top_dim}` at "
            f"{top_pct:.2f}% [curve:shapley.{top_dim}]."
        )
    else:
        lines.append("Cross-specification log-HR variance is zero across converged specifications.")

    lines.append("")
    for dim in ranked_dims:
        share_pct = 100.0 * curve_summary.shapley_shares[dim]
        lines.append(f"- `{dim}`: {share_pct:.2f}% [curve:shapley.{dim}]")

    lines.extend(["", "## Representative Specifications", ""])

    seen_ids: set[str] = set()
    for label, rec in notable_runs.items():
        if rec.run_id in seen_ids:
            continue
        seen_ids.add(rec.run_id)
        id8 = rec.run_id[:8]
        hr = float(rec.hr or 0.0)
        ci_lo = float(rec.ci_lower_95 or 0.0)
        ci_hi = float(rec.ci_upper_95 or 0.0)
        p_val = float(rec.p_value or 1.0)
        c_idx = float(rec.concordance_index or 0.5)
        p_str = _format_p_value_claim(p_val, id8)
        spec_desc = (
            f"`cohort_definition={rec.spec.cohort_definition.value}`, "
            f"`washout_months={rec.spec.washout_months}`, "
            f"`missingness={rec.spec.missingness.value}`, "
            f"`adjustment_set={rec.spec.adjustment_set}`, "
            f"`estimator={rec.spec.estimator.value}`"
        )
        lines.append(
            f"- **{label}** ({spec_desc}): "
            f"N = {rec.n_subjects} [run:{id8}:n_subjects], "
            f"deaths = {rec.n_events} [run:{id8}:n_events], "
            f"HR = {hr:.4f} [run:{id8}:hr] "
            f"(95% CI {ci_lo:.4f} [run:{id8}:ci_lower_95] to {ci_hi:.4f} [run:{id8}:ci_upper_95], "
            f"{p_str}, C-index = {c_idx:.4f} [run:{id8}:concordance_index])."
        )

    lines.append("")
    return "\n".join(lines)


def build_synthesizer_prompts(
    curve_summary: SpecCurveSummary,
    notable_runs: Mapping[str, RunRecord],
    *,
    cohort_name: str,
    hypothesis: str,
) -> tuple[str, str]:
    """Serialize sealed curve summary and notable RunRecords for Stage-2 synthesis."""
    system_prompt = (
        "You are a grounded scientific report synthesizer. "
        "Every numeric claim in your markdown report MUST be immediately followed by an inline "
        "field-bound citation tag: '<number> [run:<id8>:<field>]' or '<number>[%] [curve:<key>]'. "
        "Include mandatory citations for [curve:n_specs_converged], [curve:median_hr], "
        "[curve:min_hr], [curve:max_hr], [curve:share_ci_excluding_one], "
        "[curve:sign_consistency_share], and [curve:shapley.<top_driver>]. "
        "Do not emit any uncited decimal numbers or percentages."
    )
    notable_payload = {
        label: {
            "id8": rec.run_id[:8],
            "spec": rec.spec.model_dump(mode="json"),
            "n_subjects": rec.n_subjects,
            "n_events": rec.n_events,
            "person_years": rec.person_years,
            "hr": rec.hr,
            "ci_lower_95": rec.ci_lower_95,
            "ci_upper_95": rec.ci_upper_95,
            "p_value": rec.p_value,
            "concordance_index": rec.concordance_index,
        }
        for label, rec in notable_runs.items()
    }
    user_payload = {
        "cohort_name": cohort_name,
        "hypothesis": hypothesis,
        "curve_summary": curve_summary.model_dump(mode="json"),
        "notable_runs": notable_payload,
    }
    return system_prompt, json.dumps(user_payload, sort_keys=True, indent=2)


class GroundedSynthesizer:
    """Stage-2 report synthesizer wired directly into `verify_grounded_report`."""

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self._backend = backend

    def synthesize(
        self,
        records: Sequence[RunRecord],
        curve_summary: SpecCurveSummary,
        *,
        cohort_name: str,
        hypothesis: str,
    ) -> tuple[str, AuditResult]:
        """Generate grounded markdown and mechanically verify all citations."""
        notable_runs = select_notable_runs(records)
        if self._backend is None:
            markdown = render_deterministic_report(
                curve_summary,
                notable_runs,
                cohort_name=cohort_name,
                hypothesis=hypothesis,
            )
        else:
            system_prompt, user_prompt = build_synthesizer_prompts(
                curve_summary,
                notable_runs,
                cohort_name=cohort_name,
                hypothesis=hypothesis,
            )
            markdown = self._backend.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage="synthesizer",
                context_key=cohort_name,
            )

        audit = verify_grounded_report(markdown, records, curve_summary)
        return markdown, audit


__all__ = [
    "GroundedSynthesizer",
    "build_synthesizer_prompts",
    "render_deterministic_report",
    "select_notable_runs",
]
