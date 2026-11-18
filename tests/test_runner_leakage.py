"""Zero-leakage, reverse-causation recovery, and PH equivalence tests."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from multiverse_agent.cohort import (
    CohortBundle,
    CollectionTiming,
    ColumnProvenance,
    VariableRole,
    load_nhanes_mortality_cohort,
    make_synthetic_bias_cohort,
)
from multiverse_agent.runner import MultiverseRunner
from multiverse_agent.spec import (
    AnalyticalSpec,
    CohortDefinition,
    EstimatorFamily,
    MissingnessStrategy,
    SpecificationSpace,
)


@pytest.mark.parametrize("missingness", list(MissingnessStrategy))
@pytest.mark.parametrize("washout", [0, 12, 24])
def test_outcome_permutation_invariance_of_baseline_design(
    missingness: MissingnessStrategy,
    washout: int,
) -> None:
    orig = make_synthetic_bias_cohort(n=600, seed=42)
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(orig.df))

    perm_df = orig.df.copy()
    perm_df["duration_months"] = orig.df["duration_months"].to_numpy()[perm]
    perm_df["event"] = orig.df["event"].to_numpy()[perm]
    perm_cohort = CohortBundle(
        name=orig.name,
        sha256="p" * 64,
        df=perm_df,
        columns=orig.columns,
        id_col=orig.id_col,
    )

    space = SpecificationSpace(
        age_floors=(40,),
        cohort_definitions=(CohortDefinition.ALL_ELIGIBLE,),
        washout_months=(washout,),
        followup_horizons_months=(None,),
        missingness_strategies=(missingness,),
        adjustment_sets={
            "clinical": ("age_years", "sex_male", "bmi", "total_chol_mg_dl", "hdl_chol_mg_dl"),
        },
        estimators=(EstimatorFamily.PIECEWISE_EXPONENTIAL,),
    )
    spec = space.compile(orig)[0]

    runner_orig = MultiverseRunner(orig, space, seed=42)
    runner_perm = MultiverseRunner(perm_cohort, space, seed=42)

    design_orig = runner_orig.prepare_baseline_design(spec)
    design_perm = runner_perm.prepare_baseline_design(spec)

    assert design_orig.feature_names == design_perm.feature_names
    assert np.array_equal(design_orig.row_indices, design_perm.row_indices)
    assert np.array_equal(design_orig.matrix, design_perm.matrix)


def test_median_indicator_deduplicates_co_missing_bitmasks_and_excludes_nan_comorbidity() -> None:
    base = make_synthetic_bias_cohort(n=500, seed=42)
    df = base.df.copy()
    df.loc[df.index[:10], "prevalent_cvd"] = np.nan
    cohort = CohortBundle(
        name="nan_cvd",
        sha256="e" * 64,
        df=df,
        columns=base.columns,
        id_col=base.id_col,
    )
    space = SpecificationSpace(
        age_floors=(40,),
        cohort_definitions=(
            CohortDefinition.ALL_ELIGIBLE,
            CohortDefinition.EXCLUDE_PREVALENT_CVD,
        ),
        washout_months=(0,),
        followup_horizons_months=(None,),
        missingness_strategies=(MissingnessStrategy.MEDIAN_INDICATOR,),
        adjustment_sets={
            "lipids": ("age_years", "total_chol_mg_dl", "hdl_chol_mg_dl"),
        },
        estimators=(EstimatorFamily.PIECEWISE_EXPONENTIAL,),
    )
    specs = space.compile(cohort)
    runner = MultiverseRunner(cohort, space, seed=42)

    design_all = runner.prepare_baseline_design(specs[0])
    design_no_cvd = runner.prepare_baseline_design(specs[1])

    missing_ind_cols = [name for name in design_all.feature_names if name.startswith("miss_")]
    assert len(missing_ind_cols) == 1
    assert len(design_no_cvd.row_indices) < len(design_all.row_indices)
    excluded_ids = set(df.loc[:9, "subject_id"].tolist())
    kept_ids = set(df.iloc[design_no_cvd.row_indices]["subject_id"].tolist())
    assert excluded_ids.isdisjoint(kept_ids)


@pytest.mark.parametrize("estimator", list(EstimatorFamily))
def test_tier1_reverse_causation_bias_and_structural_hr_recovery(
    estimator: EstimatorFamily,
) -> None:
    cohort = make_synthetic_bias_cohort(n=4_000, seed=42)
    space = SpecificationSpace(
        age_floors=(20,),
        cohort_definitions=(
            CohortDefinition.ALL_ELIGIBLE,
            CohortDefinition.SUPER_CONTROLS_CLEAN,
        ),
        washout_months=(0, 12),
        followup_horizons_months=(None,),
        missingness_strategies=(MissingnessStrategy.COMPLETE_CASE,),
        adjustment_sets={
            "minimal": ("age_years", "sex_male"),
            "confounders": (
                "age_years",
                "sex_male",
                "smoking_ever",
                "bmi",
                "sbp_mmhg",
                "total_chol_mg_dl",
                "hdl_chol_mg_dl",
            ),
        },
        estimators=(estimator,),
    )
    runner = MultiverseRunner(cohort, space, seed=42)

    naive = runner.run_spec(
        AnalyticalSpec(
            age_floor=20,
            cohort_definition=CohortDefinition.ALL_ELIGIBLE,
            washout_months=0,
            followup_horizon_months=None,
            missingness=MissingnessStrategy.COMPLETE_CASE,
            adjustment_set="minimal",
            estimator=estimator,
        )
    )
    clean = runner.run_spec(
        AnalyticalSpec(
            age_floor=20,
            cohort_definition=CohortDefinition.SUPER_CONTROLS_CLEAN,
            washout_months=12,
            followup_horizon_months=None,
            missingness=MissingnessStrategy.COMPLETE_CASE,
            adjustment_set="confounders",
            estimator=estimator,
        )
    )

    assert naive.converged is True and naive.hr is not None
    assert clean.converged is True and clean.hr is not None
    assert naive.hr > 1.38
    assert 1.12 <= clean.hr <= 1.28


def test_three_ph_estimators_agree_under_mid_interval_censoring_and_zero_event_tail() -> None:
    rng = np.random.default_rng(77)
    n = 3_000
    x_exp = rng.normal(0.0, 1.0, size=n)
    x_age = np.clip(rng.normal(55.0, 10.0, size=n), 20.0, 80.0)
    x_sex = rng.binomial(1, 0.5, size=n).astype(np.float64)

    rate = 0.012 * np.exp(np.log(1.30) * x_exp + 0.04 * (x_age - 55.0) + 0.2 * x_sex)
    u = rng.uniform(size=n)
    t_event = (-np.log(u) / rate) ** (1.0 / 1.15)
    t_cens = rng.uniform(18.0, 66.0, size=n)

    duration = np.maximum(0.5, np.minimum(t_event, t_cens))
    event = (t_event <= t_cens).astype(np.int64)
    # Create a guaranteed zero-event tail interval (72m - 96m) with censored survivors
    tail_idx = np.where(event == 0)[0][:25]
    duration[tail_idx] = rng.uniform(76.0, 95.0, size=len(tail_idx))

    df = pd.DataFrame(
        {
            "subject_id": np.arange(1, n + 1, dtype=np.int64),
            "log_crp_z": x_exp,
            "age_years": x_age,
            "sex_male": x_sex,
            "duration_months": duration,
            "event": event,
        }
    )
    cols = {
        "log_crp_z": ColumnProvenance(
            name="log_crp_z",
            description="Exposure",
            role=VariableRole.EXPOSURE,
            timing=CollectionTiming.BASELINE_CONCURRENT,
        ),
        "age_years": ColumnProvenance(
            name="age_years",
            description="Age",
            role=VariableRole.CONFOUNDER,
            timing=CollectionTiming.BASELINE_PRE_EXPOSURE,
        ),
        "sex_male": ColumnProvenance(
            name="sex_male",
            description="Sex",
            role=VariableRole.CONFOUNDER,
            timing=CollectionTiming.BASELINE_PRE_EXPOSURE,
            is_binary=True,
        ),
        "duration_months": ColumnProvenance(
            name="duration_months",
            description="Duration",
            role=VariableRole.DURATION,
            timing=CollectionTiming.OUTCOME_FOLLOWUP,
        ),
        "event": ColumnProvenance(
            name="event",
            description="Event",
            role=VariableRole.EVENT,
            timing=CollectionTiming.OUTCOME_FOLLOWUP,
            is_binary=True,
        ),
    }
    cohort = CohortBundle(
        name="ph_equiv", sha256="f" * 64, df=df, columns=cols, id_col="subject_id"
    )
    space = SpecificationSpace(
        age_floors=(20,),
        cohort_definitions=(CohortDefinition.ALL_ELIGIBLE,),
        washout_months=(0,),
        followup_horizons_months=(None,),
        missingness_strategies=(MissingnessStrategy.COMPLETE_CASE,),
        adjustment_sets={"adj": ("age_years", "sex_male")},
        estimators=tuple(EstimatorFamily),
    )
    records = MultiverseRunner(cohort, space, seed=77).run_all()
    assert len(records) == 3
    for rec in records:
        assert rec.converged is True
        assert rec.log_hr is not None

    log_hrs = [float(rec.log_hr) for rec in records if rec.log_hr is not None]
    assert max(log_hrs) - min(log_hrs) < 0.04


def test_nhanes_108_branch_benchmark_runs_under_10s() -> None:
    nhanes = load_nhanes_mortality_cohort()
    space = SpecificationSpace.default_nhanes_space(nhanes)

    t0 = time.perf_counter()
    records = MultiverseRunner(nhanes, space, seed=42).run_all()
    elapsed = time.perf_counter() - t0

    assert len(records) == 108
    assert all(rec.converged for rec in records)
    assert elapsed < 10.0
