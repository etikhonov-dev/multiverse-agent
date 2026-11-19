"""Tests for SpecCurveSummary, exact Owen Sobol-Shapley attribution, and strike plot."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest

from multiverse_agent.cohort import load_nhanes_mortality_cohort
from multiverse_agent.curve import (
    SPEC_DIMENSIONS,
    SpecCurveSummary,
    render_specification_curve,
    summarize_specification_curve,
)
from multiverse_agent.runner import MultiverseRunner, RunRecord
from multiverse_agent.spec import (
    AnalyticalSpec,
    CohortDefinition,
    EstimatorFamily,
    MissingnessStrategy,
    SpecificationSpace,
)


def _make_record(
    *,
    age_floor: int = 40,
    cohort_definition: CohortDefinition = CohortDefinition.ALL_ELIGIBLE,
    washout_months: int = 0,
    followup_horizon_months: int | None = None,
    missingness: MissingnessStrategy = MissingnessStrategy.COMPLETE_CASE,
    adjustment_set: str = "minimal",
    estimator: EstimatorFamily = EstimatorFamily.COX_BRESLOW,
    log_hr: float | None = 0.2,
    se_log_hr: float | None = 0.05,
    converged: bool = True,
) -> RunRecord:
    spec = AnalyticalSpec(
        age_floor=age_floor,
        cohort_definition=cohort_definition,
        washout_months=washout_months,
        followup_horizon_months=followup_horizon_months,
        missingness=missingness,
        adjustment_set=adjustment_set,
        estimator=estimator,
    )
    if not converged or log_hr is None or se_log_hr is None:
        return RunRecord(
            run_id=spec.spec_id,
            spec_id=spec.spec_id,
            spec=spec,
            n_subjects=500,
            n_events=50,
            person_years=2500.0,
            converged=False,
            failure_reason="failed",
        )
    return RunRecord(
        run_id=spec.spec_id,
        spec_id=spec.spec_id,
        spec=spec,
        n_subjects=500,
        n_events=50,
        person_years=2500.0,
        converged=True,
        failure_reason=None,
        log_hr=log_hr,
        se_log_hr=se_log_hr,
        hr=math.exp(log_hr),
        ci_lower_95=math.exp(log_hr - 1.96 * se_log_hr),
        ci_upper_95=math.exp(log_hr + 1.96 * se_log_hr),
        p_value=0.01,
        concordance_index=0.70,
    )


def test_sobol_shapley_exact_xor_and_additive_decomposition() -> None:
    # Pure interaction (XOR) between washout_months and cohort_definition -> exact 50/50 split
    xor_records: list[RunRecord] = []
    for w, cdef in itertools.product(
        (0, 12), (CohortDefinition.ALL_ELIGIBLE, CohortDefinition.SUPER_CONTROLS_CLEAN)
    ):
        val = 0.3 if (w == 12) ^ (cdef == CohortDefinition.SUPER_CONTROLS_CLEAN) else 0.1
        xor_records.append(_make_record(washout_months=w, cohort_definition=cdef, log_hr=val))

    xor_summary = summarize_specification_curve(xor_records)
    assert xor_summary.zero_variance is False
    assert xor_summary.shapley_shares["washout_months"] == pytest.approx(0.5, abs=1e-12)
    assert xor_summary.shapley_shares["cohort_definition"] == pytest.approx(0.5, abs=1e-12)
    for dim in SPEC_DIMENSIONS:
        if dim not in {"washout_months", "cohort_definition"}:
            assert xor_summary.shapley_shares[dim] == pytest.approx(0.0, abs=1e-12)


def test_sobol_shapley_balanced_and_pruned_7d_grids_sum_to_one() -> None:
    grid_records: list[RunRecord] = []
    for age, cdef, w, h, miss, adj, est in itertools.product(
        (40, 55),
        (CohortDefinition.ALL_ELIGIBLE, CohortDefinition.SUPER_CONTROLS_CLEAN),
        (0, 12),
        (60, None),
        (MissingnessStrategy.COMPLETE_CASE, MissingnessStrategy.MEDIAN_INDICATOR),
        ("minimal", "clinical"),
        (EstimatorFamily.COX_BRESLOW, EstimatorFamily.PIECEWISE_EXPONENTIAL),
    ):
        y = (
            0.05 * (1 if age == 55 else 0)
            + 0.12 * (1 if cdef == CohortDefinition.SUPER_CONTROLS_CLEAN else 0)
            - 0.25 * (1 if w == 12 else 0)
            + 0.04 * (1 if h == 60 else 0)
            + 0.03 * (1 if miss == MissingnessStrategy.MEDIAN_INDICATOR else 0)
            - 0.08 * (1 if adj == "clinical" else 0)
            + 0.02 * (1 if est == EstimatorFamily.PIECEWISE_EXPONENTIAL else 0)
            + 0.10 * ((1 if w == 12 else 0) * (1 if adj == "clinical" else 0))
        )
        grid_records.append(
            _make_record(
                age_floor=age,
                cohort_definition=cdef,
                washout_months=w,
                followup_horizon_months=h,
                missingness=miss,
                adjustment_set=adj,
                estimator=est,
                log_hr=y,
            )
        )

    full_summary = summarize_specification_curve(grid_records)
    assert full_summary.n_specs_total == 128
    assert full_summary.n_specs_converged == 128
    for dim in SPEC_DIMENSIONS:
        assert full_summary.shapley_shares[dim] > 0.0
    assert abs(sum(full_summary.shapley_shares.values()) - 1.0) < 1e-12
    assert full_summary.top_driver == "washout_months"

    # Heavily pruned unbalanced grid (keep ~57% of branches non-uniformly)
    pruned = [rec for idx, rec in enumerate(grid_records) if idx % 7 not in {1, 3, 5}]
    pruned_summary = summarize_specification_curve(pruned)
    assert pruned_summary.n_specs_converged == len(pruned)
    for dim in SPEC_DIMENSIONS:
        assert pruned_summary.shapley_shares[dim] >= 0.0
    assert abs(sum(pruned_summary.shapley_shares.values()) - 1.0) < 1e-12


def test_failed_runs_isolated_and_zero_variance_edge_cases() -> None:
    conv_a = _make_record(washout_months=0, log_hr=0.20, se_log_hr=0.04)
    conv_b = _make_record(washout_months=12, log_hr=0.20, se_log_hr=0.04)
    failed = _make_record(washout_months=24, converged=False)

    zero_var_summary = summarize_specification_curve([conv_a, conv_b, failed])
    assert zero_var_summary.n_specs_total == 3
    assert zero_var_summary.n_specs_converged == 2
    assert zero_var_summary.n_specs_failed == 1
    assert zero_var_summary.zero_variance is True
    assert zero_var_summary.top_driver is None
    assert zero_var_summary.median_hr == pytest.approx(math.exp(0.20))
    assert zero_var_summary.min_hr == pytest.approx(math.exp(0.20))
    assert zero_var_summary.max_hr == pytest.approx(math.exp(0.20))
    assert zero_var_summary.share_ci_excluding_one == pytest.approx(1.0)
    assert zero_var_summary.sign_consistency_share == pytest.approx(1.0)
    for val in zero_var_summary.shapley_shares.values():
        assert val == 0.0
        assert not math.isnan(val)

    empty_summary = summarize_specification_curve([failed])
    assert empty_summary.n_specs_total == 1
    assert empty_summary.n_specs_converged == 0
    assert empty_summary.n_specs_failed == 1
    assert empty_summary.zero_variance is True
    assert empty_summary.median_hr is None
    assert empty_summary.top_driver is None
    assert sum(empty_summary.shapley_shares.values()) == 0.0


def test_nhanes_108_branch_curve_and_svg_png_render(tmp_path: Path) -> None:
    nhanes = load_nhanes_mortality_cohort()
    space = SpecificationSpace.default_nhanes_space(nhanes)

    records = MultiverseRunner(nhanes, space, seed=42).run_all()
    summary = summarize_specification_curve(records)
    svg_path = tmp_path / "spec_curve.svg"
    png_path = tmp_path / "spec_curve.png"
    render_specification_curve(records, summary, svg_path, png_path=png_path)

    assert isinstance(summary, SpecCurveSummary)
    assert summary.n_specs_total == 108
    assert summary.n_specs_converged == 108
    assert summary.zero_variance is False
    assert abs(sum(summary.shapley_shares.values()) - 1.0) < 1e-12
    assert summary.top_driver in SPEC_DIMENSIONS
    assert summary.min_hr is not None
    assert summary.median_hr is not None
    assert summary.max_hr is not None
    assert summary.min_hr <= summary.median_hr <= summary.max_hr
    assert 0.0 <= summary.share_ci_excluding_one <= 1.0
    assert 0.5 <= summary.sign_consistency_share <= 1.0

    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert len(svg_text) > 5_000
    assert png_path.exists()
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    svg_path_2 = tmp_path / "spec_curve_2.svg"
    render_specification_curve(records, summary, svg_path_2)
    assert svg_path.read_bytes() == svg_path_2.read_bytes()


def test_render_empty_converged_records(tmp_path: Path) -> None:
    failed = _make_record(converged=False)
    summary = summarize_specification_curve([failed])
    svg_path = tmp_path / "empty.svg"
    png_path = tmp_path / "empty.png"
    render_specification_curve([failed], summary, svg_path, png_path=png_path)
    assert "No converged specifications" in svg_path.read_text(encoding="utf-8")
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
