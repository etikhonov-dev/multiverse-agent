"""Tests for AnalyticalSpec and SpecificationSpace validation firewalls."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multiverse_agent.cohort import (
    CohortBundle,
    CollectionTiming,
    ColumnProvenance,
    VariableRole,
    load_nhanes_mortality_cohort,
    make_synthetic_bias_cohort,
)
from multiverse_agent.spec import (
    AdjustmentSet,
    AnalyticalSpec,
    CohortDefinition,
    EstimatorFamily,
    MissingnessStrategy,
    SpecificationSpace,
)


def _base_space(
    *,
    age_floors: tuple[int, ...] = (40, 55),
    cohort_definitions: tuple[CohortDefinition, ...] = (
        CohortDefinition.ALL_ELIGIBLE,
        CohortDefinition.EXCLUDE_PREVALENT_CVD,
    ),
    washout_months: tuple[int, ...] = (0, 12),
    followup_horizons_months: tuple[int | None, ...] = (60, None),
    missingness_strategies: tuple[MissingnessStrategy, ...] = (
        MissingnessStrategy.COMPLETE_CASE,
        MissingnessStrategy.MEDIAN_INDICATOR,
    ),
    adjustment_sets: dict[str, tuple[str, ...]] | None = None,
    estimators: tuple[EstimatorFamily, ...] = (
        EstimatorFamily.COX_BRESLOW,
        EstimatorFamily.PIECEWISE_EXPONENTIAL,
    ),
) -> SpecificationSpace:
    sets = (
        adjustment_sets
        if adjustment_sets is not None
        else {
            "minimal": ("age_years", "sex_male"),
            "clinical": ("age_years", "sex_male", "bmi", "sbp_mmhg"),
        }
    )
    return SpecificationSpace(
        age_floors=age_floors,
        cohort_definitions=cohort_definitions,
        washout_months=washout_months,
        followup_horizons_months=followup_horizons_months,
        missingness_strategies=missingness_strategies,
        adjustment_sets=sets,
        estimators=estimators,
    )


def test_analytical_spec_rejects_invalid_temporal_ordering_and_age_floor() -> None:
    with pytest.raises(ValidationError, match="washout_months"):
        AnalyticalSpec(
            age_floor=40,
            cohort_definition=CohortDefinition.ALL_ELIGIBLE,
            washout_months=24,
            followup_horizon_months=24,
            missingness=MissingnessStrategy.COMPLETE_CASE,
            adjustment_set="minimal",
            estimator=EstimatorFamily.COX_BRESLOW,
        )

    with pytest.raises(ValidationError, match="washout_months"):
        AnalyticalSpec(
            age_floor=40,
            cohort_definition=CohortDefinition.ALL_ELIGIBLE,
            washout_months=36,
            followup_horizon_months=24,
            missingness=MissingnessStrategy.COMPLETE_CASE,
            adjustment_set="minimal",
            estimator=EstimatorFamily.COX_BRESLOW,
        )

    with pytest.raises(ValidationError):
        AnalyticalSpec(
            age_floor=71,
            cohort_definition=CohortDefinition.ALL_ELIGIBLE,
            washout_months=0,
            followup_horizon_months=None,
            missingness=MissingnessStrategy.COMPLETE_CASE,
            adjustment_set="minimal",
            estimator=EstimatorFamily.COX_BRESLOW,
        )


def test_analytical_spec_id_is_deterministic_and_frozen() -> None:
    spec1 = AnalyticalSpec(
        age_floor=45,
        cohort_definition=CohortDefinition.SUPER_CONTROLS_CLEAN,
        washout_months=12,
        followup_horizon_months=60,
        missingness=MissingnessStrategy.MEDIAN_INDICATOR,
        adjustment_set="clinical",
        estimator=EstimatorFamily.COX_SPLINE,
    )
    spec2 = AnalyticalSpec(**spec1.model_dump())
    assert spec1.spec_id == spec2.spec_id
    assert len(spec1.spec_id) == 64

    with pytest.raises(ValidationError):
        spec1.__setattr__("washout_months", 24)


def test_adjustment_set_canonicalizes_sorted_tuple_and_rejects_duplicates() -> None:
    adj = AdjustmentSet(name="clinical", covariates=("sex_male", "age_years", "bmi"))
    assert adj.covariates == ("age_years", "bmi", "sex_male")

    with pytest.raises(ValidationError, match="duplicate"):
        AdjustmentSet(name="bad", covariates=("age_years", "age_years"))


def test_specification_space_enforces_bijection_on_adjustment_sets() -> None:
    with pytest.raises(ValidationError, match="bijection|identical"):
        _base_space(
            adjustment_sets={
                "minimal_a": ("age_years", "sex_male"),
                "minimal_b": ("sex_male", "age_years"),
            }
        )


def test_specification_space_rejects_redundant_horizon_ge_max_duration() -> None:
    cohort = make_synthetic_bias_cohort(n=400, seed=42)
    short_df = cohort.df.copy()
    short_df["duration_months"] = short_df["duration_months"].clip(upper=60.0)
    short_cohort = CohortBundle(
        name="short_cohort",
        sha256="b" * 64,
        df=short_df,
        columns=cohort.columns,
        id_col=cohort.id_col,
    )

    space_equal = _base_space(followup_horizons_months=(60, None))
    with pytest.raises(ValueError, match="max_duration_months"):
        space_equal.compile(short_cohort)

    space_greater = _base_space(followup_horizons_months=(72, None))
    with pytest.raises(ValueError, match="max_duration_months"):
        space_greater.compile(short_cohort)


@pytest.mark.parametrize(
    "forbidden_col",
    [
        "post_tx_statin",
        "hospitalized_past_year",
        "prevalent_cvd",
        "prevalent_cancer",
        "diabetes_dx",
        "log_crp_z",
        "duration_months",
        "event",
    ],
)
def test_specification_space_blocks_forbidden_roles_in_adjustment_sets(forbidden_col: str) -> None:
    cohort = make_synthetic_bias_cohort(n=400, seed=42)
    space = _base_space(
        adjustment_sets={
            "minimal": ("age_years", "sex_male"),
            "tainted": ("age_years", forbidden_col),
        }
    )
    with pytest.raises(ValueError, match="Forbidden"):
        space.compile(cohort)


def test_specification_space_blocks_post_baseline_and_utilization_timing_even_if_mislabeled() -> (
    None
):
    cohort = make_synthetic_bias_cohort(n=400, seed=42)

    space_med = _base_space(adjustment_sets={"bad_med": ("age_years", "post_tx_statin")})
    with pytest.raises(ValueError, match="CollectionTiming"):
        space_med.compile(cohort, declared_roles={"post_tx_statin": VariableRole.CONFOUNDER})

    space_col = _base_space(adjustment_sets={"bad_col": ("age_years", "hospitalized_past_year")})
    with pytest.raises(ValueError, match="CollectionTiming"):
        space_col.compile(
            cohort, declared_roles={"hospitalized_past_year": VariableRole.CONFOUNDER}
        )

    mislabeled_cols = dict(cohort.columns)
    mislabeled_cols["hospitalized_past_year"] = ColumnProvenance(
        name="hospitalized_past_year",
        description="Mislabeled collider",
        role=VariableRole.CONFOUNDER,
        timing=CollectionTiming.HEALTHCARE_UTILIZATION_SELECTION,
        is_binary=True,
    )
    mislabeled_cohort = CohortBundle(
        name="mislabeled",
        sha256="c" * 64,
        df=cohort.df,
        columns=mislabeled_cols,
        id_col=cohort.id_col,
    )
    with pytest.raises(ValueError, match="CollectionTiming"):
        space_col.compile(mislabeled_cohort)


def test_specification_space_rejects_unknown_column_and_empty_temporal_grid() -> None:
    cohort = make_synthetic_bias_cohort(n=400, seed=42)
    space_unknown = _base_space(adjustment_sets={"bad": ("age_years", "SEQN")})
    with pytest.raises(ValueError, match="unknown covariate"):
        space_unknown.compile(cohort)

    with pytest.raises(ValidationError, match="valid \\(washout, horizon\\)"):
        _base_space(washout_months=(24, 36), followup_horizons_months=(24,))


def test_default_nhanes_space_compiles_cleanly() -> None:
    nhanes = load_nhanes_mortality_cohort()
    space = SpecificationSpace.default_nhanes_space(nhanes)
    specs = space.compile(nhanes)

    assert len(specs) == 108
    assert len({spec.spec_id for spec in specs}) == 108
    for spec in specs:
        assert spec.age_floor <= 70
        if spec.followup_horizon_months is not None:
            assert spec.washout_months < spec.followup_horizon_months < nhanes.max_duration_months
