"""Tests for GroundedSynthesizer and field-bound mechanical eval_gate."""

from __future__ import annotations

import math

from multiverse_agent.curve import summarize_specification_curve
from multiverse_agent.eval_gate import AuditResult, verify_grounded_report
from multiverse_agent.interpreter import GroundedSynthesizer, select_notable_runs
from multiverse_agent.runner import RunRecord
from multiverse_agent.spec import (
    AnalyticalSpec,
    CohortDefinition,
    EstimatorFamily,
    MissingnessStrategy,
)


def _make_run(
    *,
    run_id: str,
    washout_months: int = 0,
    cohort_definition: CohortDefinition = CohortDefinition.ALL_ELIGIBLE,
    log_hr: float = 0.20,
    se_log_hr: float = 0.05,
    p_value: float = 0.002,
    n_subjects: int = 2500,
    n_events: int = 310,
    converged: bool = True,
) -> RunRecord:
    spec = AnalyticalSpec(
        age_floor=40,
        cohort_definition=cohort_definition,
        washout_months=washout_months,
        followup_horizon_months=None,
        missingness=MissingnessStrategy.COMPLETE_CASE,
        adjustment_set="minimal",
        estimator=EstimatorFamily.COX_BRESLOW,
    )
    if not converged:
        return RunRecord(
            run_id=run_id,
            spec_id=spec.spec_id,
            spec=spec,
            n_subjects=n_subjects,
            n_events=n_events,
            person_years=15000.0,
            converged=False,
            failure_reason="insufficient_events",
        )
    return RunRecord(
        run_id=run_id,
        spec_id=spec.spec_id,
        spec=spec,
        n_subjects=n_subjects,
        n_events=n_events,
        person_years=15000.0,
        converged=True,
        failure_reason=None,
        log_hr=log_hr,
        se_log_hr=se_log_hr,
        hr=math.exp(log_hr),
        ci_lower_95=math.exp(log_hr - 1.96 * se_log_hr),
        ci_upper_95=math.exp(log_hr + 1.96 * se_log_hr),
        p_value=p_value,
        concordance_index=0.71,
    )


def _sample_multiverse() -> tuple[tuple[RunRecord, ...], str]:
    records = (
        _make_run(
            run_id="aaaa1111" + "0" * 56,
            washout_months=0,
            cohort_definition=CohortDefinition.ALL_ELIGIBLE,
            log_hr=0.35,
            se_log_hr=0.05,
            p_value=0.00003,
            n_subjects=3200,
            n_events=420,
        ),
        _make_run(
            run_id="bbbb2222" + "0" * 56,
            washout_months=12,
            cohort_definition=CohortDefinition.ALL_ELIGIBLE,
            log_hr=0.22,
            se_log_hr=0.06,
            p_value=0.00025,
            n_subjects=3000,
            n_events=340,
        ),
        _make_run(
            run_id="cccc3333" + "0" * 56,
            washout_months=0,
            cohort_definition=CohortDefinition.SUPER_CONTROLS_CLEAN,
            log_hr=0.26,
            se_log_hr=0.06,
            p_value=0.00001,
            n_subjects=2100,
            n_events=210,
        ),
        _make_run(
            run_id="dddd4444" + "0" * 56,
            washout_months=12,
            cohort_definition=CohortDefinition.SUPER_CONTROLS_CLEAN,
            log_hr=0.14,
            se_log_hr=0.072,
            p_value=0.052,
            n_subjects=1950,
            n_events=160,
        ),
    )
    summary = summarize_specification_curve(records)
    report, audit = GroundedSynthesizer(backend=None).synthesize(
        records,
        summary,
        cohort_name="test_cohort",
        hypothesis="Baseline log CRP and all-cause mortality.",
    )
    assert audit.passed is True, audit.violations
    return records, report


def test_valid_synthesized_report_passes_with_structural_allowlist() -> None:
    records, report = _sample_multiverse()
    summary = summarize_specification_curve(records)
    augmented = (
        "# 1. Empirical Summary (2007-2010)\n\n"
        + report
        + "\n```python\n# Code example should not trigger ungrounded_number: hr = 99.9\n```\n"
        + "\nAll 95% CI intervals per 1-SD increment (`washout_months=12`, SHA-256 verified) "
        + "use alpha = 0.05.\n"
    )
    audit = verify_grounded_report(augmented, records, summary)
    assert isinstance(audit, AuditResult)
    assert audit.passed is True
    assert audit.violations == ()
    assert audit.n_curve_citations >= 7
    assert audit.n_run_citations >= 2


def test_catches_unknown_ambiguous_and_nonconverged_run_ids_and_unknown_fields() -> None:
    records, report = _sample_multiverse()
    summary = summarize_specification_curve(records)

    # Unknown run_id
    bad_unknown = report + "\nExtra HR = 1.42 [run:deadbeef:hr].\n"
    res_unknown = verify_grounded_report(bad_unknown, records, summary)
    assert res_unknown.passed is False
    assert any(v.code == "unknown_run_id" for v in res_unknown.violations)

    # Unknown field on valid run_id
    bad_field = report + "\nExtra metric = 1.42 [run:aaaa1111:nonexistent_metric].\n"
    res_field = verify_grounded_report(bad_field, records, summary)
    assert res_field.passed is False
    assert any(v.code == "unknown_field_or_key" for v in res_field.violations)

    # Ambiguous prefix matching two converged records
    ambig_records = (
        records[0],
        _make_run(run_id="aaaa1111" + "f" * 56, washout_months=6),
        *records[1:],
    )
    res_ambig = verify_grounded_report(
        report, ambig_records, summarize_specification_curve(ambig_records)
    )
    assert res_ambig.passed is False
    assert any(v.code == "ambiguous_run_id" for v in res_ambig.violations)

    # Non-converged run cited
    failed_rec = _make_run(run_id="eeee5555" + "0" * 56, washout_months=24, converged=False)
    res_nonconv = verify_grounded_report(
        report + "\nFailed N = 2500 [run:eeee5555:n_subjects].\n",
        (*records, failed_rec),
        summarize_specification_curve((*records, failed_rec)),
    )
    assert res_nonconv.passed is False
    assert any(v.code == "non_converged_run_cited" for v in res_nonconv.violations)


def test_catches_unbound_citation_tag() -> None:
    records, report = _sample_multiverse()
    summary = summarize_specification_curve(records)

    bare = report + "\nThe minimum estimate [curve:min_hr] was lower.\n"
    res = verify_grounded_report(bare, records, summary)
    assert res.passed is False
    assert any(v.code == "unbound_citation_tag" for v in res.violations)


def test_catches_musical_chairs_hr_swap_between_two_runs() -> None:
    records, report = _sample_multiverse()
    summary = summarize_specification_curve(records)

    hr_a = f"{records[0].hr:.3f}"
    hr_d = f"{records[3].hr:.3f}"
    # Swap the HRs between aaaa1111 and dddd4444 in the same sentence
    swapped = (
        report + f"\nComparing HR = {hr_d} [run:aaaa1111:hr] vs HR = {hr_a} [run:dddd4444:hr].\n"
    )
    res = verify_grounded_report(swapped, records, summary)
    assert res.passed is False
    assert any(v.code == "value_mismatch" for v in res.violations)


def test_catches_p_value_significance_flip_and_validates_threshold_form() -> None:
    records, report = _sample_multiverse()
    summary = summarize_specification_curve(records)

    # dddd4444 has p_value = 0.052; rewriting as 0.048 crosses alpha=0.05
    flipped = report + "\nBorderline result p = 0.048 [run:dddd4444:p_value].\n"
    res_flip = verify_grounded_report(flipped, records, summary)
    assert res_flip.passed is False
    assert any(v.code == "significance_flip" for v in res_flip.violations)

    # Threshold form < 0.0001 valid when logged < 1e-4 (aaaa1111 has p=0.00003)
    valid_thresh = report + "\nAcute p < 0.0001 [run:aaaa1111:p_value].\n"
    assert verify_grounded_report(valid_thresh, records, summary).passed is True

    # Threshold form < 0.0001 invalid when logged >= 1e-4 (bbbb2222 has p=0.00025)
    invalid_thresh = report + "\nInvalid threshold p < 0.0001 [run:bbbb2222:p_value].\n"
    res_thresh = verify_grounded_report(invalid_thresh, records, summary)
    assert res_thresh.passed is False
    assert any(v.code == "value_mismatch" for v in res_thresh.violations)


def test_catches_ungrounded_prose_numbers_and_integer_mismatches() -> None:
    records, report = _sample_multiverse()
    summary = summarize_specification_curve(records)

    uncited = report + "\nOverall mortality increased by 24.5% across specifications.\n"
    res_uncited = verify_grounded_report(uncited, records, summary)
    assert res_uncited.passed is False
    assert any(v.code == "ungrounded_number" for v in res_uncited.violations)

    bad_count = report + "\nSample size N = 3199 [run:aaaa1111:n_subjects].\n"
    res_count = verify_grounded_report(bad_count, records, summary)
    assert res_count.passed is False
    assert any(v.code == "value_mismatch" for v in res_count.violations)


def test_catches_missing_mandatory_summary_and_false_top_shapley_driver() -> None:
    records, report = _sample_multiverse()
    summary = summarize_specification_curve(records)
    assert summary.top_driver is not None

    # Remove the mandatory top-driver Shapley citation and replace with 2nd-largest driver
    other_dim = (
        "cohort_definition" if summary.top_driver != "cohort_definition" else "washout_months"
    )
    other_pct = f"{100.0 * summary.shapley_shares[other_dim]:.1f}%"
    top_tag = f"[curve:shapley.{summary.top_driver}]"
    lines = [line for line in report.splitlines() if top_tag not in line]
    cherry_picked = (
        "\n".join(lines)
        + f"\nThe top driver was `{other_dim}` at {other_pct} [curve:shapley.{other_dim}].\n"
    )
    res = verify_grounded_report(cherry_picked, records, summary)
    assert res.passed is False
    codes = {v.code for v in res.violations}
    assert "missing_mandatory_summary_citation" in codes
    assert "wrong_top_driver" in codes

    # Also omitting median_hr alone triggers missing_mandatory_summary_citation
    no_median = "\n".join(line for line in report.splitlines() if "[curve:median_hr]" not in line)
    res_med = verify_grounded_report(no_median, records, summary)
    assert res_med.passed is False
    assert any(v.code == "missing_mandatory_summary_citation" for v in res_med.violations)


def test_select_notable_runs_handles_distinct_and_single_run() -> None:
    records, _ = _sample_multiverse()
    notable = select_notable_runs(records)
    assert "min_hr_run" in notable
    assert "median_hr_run" in notable
    assert "max_hr_run" in notable
    # Verify exact single-run edge case does not crash
    single_summary = summarize_specification_curve([records[0]])
    rep, audit = GroundedSynthesizer(backend=None).synthesize(
        [records[0]],
        single_summary,
        cohort_name="single",
        hypothesis="Single run test.",
    )
    assert audit.passed is True
    assert "[curve:n_specs_converged]" in rep
