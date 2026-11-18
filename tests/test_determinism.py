"""Determinism and RunRecord contract tests for MultiverseRunner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multiverse_agent.cohort import CohortBundle, make_synthetic_bias_cohort
from multiverse_agent.runner import MultiverseRunner, RunRecord
from multiverse_agent.spec import (
    AnalyticalSpec,
    CohortDefinition,
    EstimatorFamily,
    MissingnessStrategy,
    SpecificationSpace,
)


def _small_space() -> SpecificationSpace:
    return SpecificationSpace(
        age_floors=(40,),
        cohort_definitions=(
            CohortDefinition.ALL_ELIGIBLE,
            CohortDefinition.SUPER_CONTROLS_CLEAN,
        ),
        washout_months=(0, 12),
        followup_horizons_months=(None,),
        missingness_strategies=(
            MissingnessStrategy.COMPLETE_CASE,
            MissingnessStrategy.MEDIAN_INDICATOR,
            MissingnessStrategy.REG_IMPUTATION,
        ),
        adjustment_sets={
            "minimal": ("age_years", "sex_male"),
            "clinical": ("age_years", "sex_male", "bmi", "sbp_mmhg", "total_chol_mg_dl"),
        },
        estimators=(
            EstimatorFamily.COX_BRESLOW,
            EstimatorFamily.COX_SPLINE,
            EstimatorFamily.PIECEWISE_EXPONENTIAL,
        ),
    )


def test_two_runs_produce_byte_identical_jsonl_and_run_ids(tmp_path: Path) -> None:
    cohort = make_synthetic_bias_cohort(n=600, seed=42)
    space = _small_space()

    out1 = tmp_path / "run1.jsonl"
    out2 = tmp_path / "run2.jsonl"

    r1 = MultiverseRunner(cohort, space, seed=42).run_all(jsonl_path=out1)
    r2 = MultiverseRunner(cohort, space, seed=42).run_all(jsonl_path=out2)

    assert len(r1) == len(r2) == 72
    assert out1.read_bytes() == out2.read_bytes()

    lines = out1.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 72
    for rec1, rec2, line in zip(r1, r2, lines, strict=True):
        assert rec1.run_id == rec2.run_id
        assert rec1.spec_id == rec2.spec_id
        assert len(rec1.run_id) == 64
        parsed = RunRecord.model_validate(json.loads(line))
        assert parsed == rec1


def test_run_id_sensitivity_to_seed_cohort_and_spec() -> None:
    cohort_a = make_synthetic_bias_cohort(n=400, seed=42)
    cohort_b = make_synthetic_bias_cohort(n=400, seed=43)
    space = SpecificationSpace(
        age_floors=(40,),
        cohort_definitions=(CohortDefinition.ALL_ELIGIBLE,),
        washout_months=(0,),
        followup_horizons_months=(None,),
        missingness_strategies=(MissingnessStrategy.COMPLETE_CASE,),
        adjustment_sets={"minimal": ("age_years", "sex_male")},
        estimators=(EstimatorFamily.PIECEWISE_EXPONENTIAL,),
    )

    rec_a1 = MultiverseRunner(cohort_a, space, seed=42).run_all()[0]
    rec_a2 = MultiverseRunner(cohort_a, space, seed=99).run_all()[0]
    rec_b1 = MultiverseRunner(cohort_b, space, seed=42).run_all()[0]

    assert rec_a1.spec_id == rec_a2.spec_id == rec_b1.spec_id
    assert rec_a1.run_id != rec_a2.run_id
    assert rec_a1.run_id != rec_b1.run_id


def test_underpowered_branch_isolates_failure_with_null_metrics() -> None:
    base = make_synthetic_bias_cohort(n=400, seed=42)
    low_event_df = base.df.copy()
    low_event_df["event"] = 0
    low_event_df.loc[:4, "event"] = 1
    low_cohort = CohortBundle(
        name="low_events",
        sha256="d" * 64,
        df=low_event_df,
        columns=base.columns,
        id_col=base.id_col,
    )
    spec = AnalyticalSpec(
        age_floor=40,
        cohort_definition=CohortDefinition.ALL_ELIGIBLE,
        washout_months=0,
        followup_horizon_months=None,
        missingness=MissingnessStrategy.COMPLETE_CASE,
        adjustment_set="minimal",
        estimator=EstimatorFamily.COX_BRESLOW,
    )
    space = SpecificationSpace(
        age_floors=(40,),
        cohort_definitions=(CohortDefinition.ALL_ELIGIBLE,),
        washout_months=(0,),
        followup_horizons_months=(None,),
        missingness_strategies=(MissingnessStrategy.COMPLETE_CASE,),
        adjustment_sets={"minimal": ("age_years", "sex_male")},
        estimators=(EstimatorFamily.COX_BRESLOW,),
    )

    record = MultiverseRunner(low_cohort, space, seed=42).run_spec(spec)
    assert record.converged is False
    assert record.n_events == 5
    assert record.failure_reason is not None
    assert record.log_hr is None
    assert record.se_log_hr is None
    assert record.hr is None
    assert record.ci_lower_95 is None
    assert record.ci_upper_95 is None
    assert record.p_value is None
    assert record.concordance_index is None
    assert np.isfinite(record.person_years)
    eligible_dur = low_event_df.loc[low_event_df["age_years"] >= 40.0, "duration_months"]
    assert pytest.approx(record.person_years) == float(eligible_dur.sum()) / 12.0


def test_horizon_censoring_empty_adjustment_and_post_washout_constant_indicator() -> None:
    base = make_synthetic_bias_cohort(n=500, seed=42)
    df = base.df.copy()
    # Make bmi missing ONLY among early subjects so post-washout miss_bmi has zero variance
    early = df["duration_months"] <= 12.0
    df.loc[early, "bmi"] = np.nan
    df.loc[~early, "bmi"] = df.loc[~early, "bmi"].fillna(27.0)

    cohort = CohortBundle(
        name="horizon_edge",
        sha256="h" * 64,
        df=df,
        columns=base.columns,
        id_col=base.id_col,
    )
    space = SpecificationSpace(
        age_floors=(20,),
        cohort_definitions=(CohortDefinition.ALL_ELIGIBLE,),
        washout_months=(0, 12),
        followup_horizons_months=(48,),
        missingness_strategies=(MissingnessStrategy.MEDIAN_INDICATOR,),
        adjustment_sets={
            "unadjusted": (),
            "bmi_only": ("bmi",),
        },
        estimators=(EstimatorFamily.PIECEWISE_EXPONENTIAL,),
    )
    records = MultiverseRunner(cohort, space, seed=42).run_all()
    assert len(records) == 4
    assert all(rec.converged for rec in records)
    # Horizon=48 caps max person-years per subject at 4.0 years (3.0 years after 12m washout)
    for rec in records:
        max_py = rec.n_subjects * ((48.0 - rec.spec.washout_months) / 12.0)
        assert rec.person_years <= max_py + 1e-9
