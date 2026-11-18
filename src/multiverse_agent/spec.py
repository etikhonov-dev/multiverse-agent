"""Pydantic v2 analytical specification contracts and epidemiological firewalls."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from multiverse_agent.cohort import CohortBundle, CollectionTiming, VariableRole

FORBIDDEN_ADJUSTMENT_ROLES = frozenset(
    {
        VariableRole.EXPOSURE,
        VariableRole.DURATION,
        VariableRole.EVENT,
        VariableRole.MEDIATOR,
        VariableRole.COLLIDER,
        VariableRole.COMORBIDITY_FLAG,
    }
)

FORBIDDEN_ADJUSTMENT_TIMINGS = frozenset(
    {
        CollectionTiming.POST_BASELINE_FOLLOWUP,
        CollectionTiming.HEALTHCARE_UTILIZATION_SELECTION,
        CollectionTiming.OUTCOME_FOLLOWUP,
    }
)


class CohortDefinition(StrEnum):
    ALL_ELIGIBLE = "all_eligible"
    EXCLUDE_PREVALENT_CVD = "exclude_prevalent_cvd"
    SUPER_CONTROLS_CLEAN = "super_controls_clean"


class MissingnessStrategy(StrEnum):
    COMPLETE_CASE = "complete_case"
    MEDIAN_INDICATOR = "median_indicator"
    REG_IMPUTATION = "reg_imputation"


class EstimatorFamily(StrEnum):
    COX_BRESLOW = "cox_breslow"
    COX_SPLINE = "cox_spline"
    PIECEWISE_EXPONENTIAL = "piecewise_exponential"


class AdjustmentSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    covariates: tuple[str, ...]

    @field_validator("covariates")
    @classmethod
    def _canonicalize_covariates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("AdjustmentSet contains duplicate covariate names.")
        return tuple(sorted(value))


class AnalyticalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    age_floor: int = Field(ge=20, le=70)
    cohort_definition: CohortDefinition
    washout_months: int = Field(ge=0, le=36)
    followup_horizon_months: int | None = Field(default=None, ge=24, le=144)
    missingness: MissingnessStrategy
    adjustment_set: str = Field(min_length=1)
    estimator: EstimatorFamily

    @model_validator(mode="after")
    def _validate_temporal_ordering(self) -> AnalyticalSpec:
        if (
            self.followup_horizon_months is not None
            and self.washout_months >= self.followup_horizon_months
        ):
            raise ValueError(
                f"washout_months ({self.washout_months}) must be strictly less than "
                f"followup_horizon_months ({self.followup_horizon_months})."
            )
        return self

    @property
    def spec_id(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SpecificationSpace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    age_floors: tuple[int, ...] = Field(min_length=1)
    cohort_definitions: tuple[CohortDefinition, ...] = Field(min_length=1)
    washout_months: tuple[int, ...] = Field(min_length=1)
    followup_horizons_months: tuple[int | None, ...] = Field(min_length=1)
    missingness_strategies: tuple[MissingnessStrategy, ...] = Field(min_length=1)
    adjustment_sets: dict[str, tuple[str, ...]]
    estimators: tuple[EstimatorFamily, ...] = Field(min_length=1)

    @field_validator("age_floors")
    @classmethod
    def _validate_age_floors(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        for age in value:
            if age < 20 or age > 70:
                raise ValueError(f"age_floor {age} must be in [20, 70].")
        if len(set(value)) != len(value):
            raise ValueError("Duplicate entries in age_floors.")
        return value

    @field_validator("washout_months")
    @classmethod
    def _validate_washout_months(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        for months in value:
            if months < 0 or months > 36:
                raise ValueError(f"washout_months {months} must be in [0, 36].")
        if len(set(value)) != len(value):
            raise ValueError("Duplicate entries in washout_months.")
        return value

    @field_validator("followup_horizons_months")
    @classmethod
    def _validate_horizons(cls, value: tuple[int | None, ...]) -> tuple[int | None, ...]:
        for horizon in value:
            if horizon is not None and (horizon < 24 or horizon > 144):
                raise ValueError(f"followup_horizon_months {horizon} must be in [24, 144].")
        if len(set(value)) != len(value):
            raise ValueError("Duplicate entries in followup_horizons_months.")
        return value

    @field_validator("adjustment_sets")
    @classmethod
    def _validate_adjustment_sets(
        cls, value: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        if not value:
            raise ValueError("adjustment_sets must contain at least one named adjustment set.")
        canonical: dict[str, tuple[str, ...]] = {}
        seen_tuples: dict[tuple[str, ...], str] = {}
        for name, covariates in value.items():
            adj = AdjustmentSet(name=name, covariates=covariates)
            if adj.covariates in seen_tuples:
                other = seen_tuples[adj.covariates]
                raise ValueError(
                    f"Adjustment set bijection violated: '{name}' and '{other}' map to "
                    f"identical covariates {adj.covariates}."
                )
            seen_tuples[adj.covariates] = name
            canonical[name] = adj.covariates
        return canonical

    @model_validator(mode="after")
    def _validate_non_empty_temporal_grid(self) -> SpecificationSpace:
        has_valid_pair = any(
            horizon is None or washout < horizon
            for washout in self.washout_months
            for horizon in self.followup_horizons_months
        )
        if not has_valid_pair:
            raise ValueError(
                "SpecificationSpace has zero valid (washout, horizon) pairs with washout < horizon."
            )
        return self

    def validate_against_cohort(
        self,
        cohort: CohortBundle,
        *,
        declared_roles: Mapping[str, VariableRole] | None = None,
    ) -> None:
        """Enforce cohort horizon bounds, column existence, CollectionTiming, and VariableRole."""
        max_dur = cohort.max_duration_months
        for horizon in self.followup_horizons_months:
            if horizon is not None and float(horizon) >= max_dur:
                raise ValueError(
                    f"followup_horizon_months ({horizon}) must be strictly less than "
                    f"cohort.max_duration_months ({max_dur:.1f}) to prevent redundant censoring."
                )

        for set_name, covariates in self.adjustment_sets.items():
            for cov in covariates:
                if cov not in cohort.columns:
                    raise ValueError(
                        f"Adjustment set '{set_name}' references unknown covariate '{cov}'."
                    )
                prov = cohort.columns[cov]
                if prov.timing in FORBIDDEN_ADJUSTMENT_TIMINGS:
                    raise ValueError(
                        f"Forbidden CollectionTiming '{prov.timing}' for covariate '{cov}' "
                        f"in adjustment set '{set_name}'."
                    )
                if prov.role in FORBIDDEN_ADJUSTMENT_ROLES:
                    raise ValueError(
                        f"Forbidden VariableRole '{prov.role}' for covariate '{cov}' "
                        f"in adjustment set '{set_name}'."
                    )
                if declared_roles is not None and cov in declared_roles:
                    declared = declared_roles[cov]
                    if declared in FORBIDDEN_ADJUSTMENT_ROLES:
                        raise ValueError(
                            f"Forbidden declared VariableRole '{declared}' for covariate '{cov}' "
                            f"in adjustment set '{set_name}'."
                        )

    def compile(
        self,
        cohort: CohortBundle,
        *,
        declared_roles: Mapping[str, VariableRole] | None = None,
    ) -> tuple[AnalyticalSpec, ...]:
        """Validate against cohort and enumerate all valid AnalyticalSpec combinations."""
        self.validate_against_cohort(cohort, declared_roles=declared_roles)
        specs: list[AnalyticalSpec] = []
        for (
            age_floor,
            cohort_def,
            washout,
            horizon,
            missingness,
            adj_name,
            estimator,
        ) in itertools.product(
            self.age_floors,
            self.cohort_definitions,
            self.washout_months,
            self.followup_horizons_months,
            self.missingness_strategies,
            self.adjustment_sets.keys(),
            self.estimators,
        ):
            if horizon is not None and washout >= horizon:
                continue
            specs.append(
                AnalyticalSpec(
                    age_floor=age_floor,
                    cohort_definition=cohort_def,
                    washout_months=washout,
                    followup_horizon_months=horizon,
                    missingness=missingness,
                    adjustment_set=adj_name,
                    estimator=estimator,
                )
            )
        return tuple(specs)

    @classmethod
    def default_nhanes_space(cls, cohort: CohortBundle | None = None) -> SpecificationSpace:
        """Return the canonical 108-branch multiverse specification space for NHANES."""
        space = cls(
            age_floors=(40,),
            cohort_definitions=(
                CohortDefinition.ALL_ELIGIBLE,
                CohortDefinition.EXCLUDE_PREVALENT_CVD,
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
                "clinical": (
                    "age_years",
                    "bmi",
                    "hdl_chol_mg_dl",
                    "sbp_mmhg",
                    "sex_male",
                    "smoking_ever",
                    "total_chol_mg_dl",
                ),
            },
            estimators=(
                EstimatorFamily.COX_BRESLOW,
                EstimatorFamily.COX_SPLINE,
                EstimatorFamily.PIECEWISE_EXPONENTIAL,
            ),
        )
        if cohort is not None:
            space.validate_against_cohort(cohort)
        return space


__all__ = [
    "AdjustmentSet",
    "AnalyticalSpec",
    "CohortDefinition",
    "CollectionTiming",
    "EstimatorFamily",
    "MissingnessStrategy",
    "SpecificationSpace",
    "VariableRole",
]
