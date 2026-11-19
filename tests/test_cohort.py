"""Tests for CohortBundle, Tier 1 DGP, Tier 2 NHANES loader, and OutcomeBlindSchema."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multiverse_agent.cohort import (
    NHANES_PARQUET_SHA256,
    SYNTHETIC_PARQUET_SHA256,
    CohortBundle,
    CollectionTiming,
    OutcomeBlindSchema,
    VariableRole,
    load_nhanes_mortality_cohort,
    load_synthetic_bias_cohort,
    make_synthetic_bias_cohort,
)

FORBIDDEN_BLIND_SUBSTRINGS = (
    "duration",
    "event",
    "mortstat",
    "permth",
    "ucod",
    "outcome_followup",
    "deceased",
    "death",
)


def test_tier1_synthetic_exposure_is_normalized_and_observed() -> None:
    bundle = make_synthetic_bias_cohort(n=4_000, seed=42)
    exp = bundle.df[bundle.exposure_col]
    assert len(bundle.df) == 4_000
    assert exp.notna().all()
    assert float(exp.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(exp.std(ddof=0)) == pytest.approx(1.0, abs=1e-6)


def test_tier1_synthetic_has_mediator_collider_and_acute_reverse_causation() -> None:
    bundle = make_synthetic_bias_cohort(n=4_000, seed=42)
    assert bundle.columns["post_tx_statin"].role == VariableRole.MEDIATOR
    assert bundle.columns["post_tx_statin"].timing == CollectionTiming.POST_BASELINE_FOLLOWUP
    assert bundle.columns["hospitalized_past_year"].role == VariableRole.COLLIDER
    assert (
        bundle.columns["hospitalized_past_year"].timing
        == CollectionTiming.HEALTHCARE_UTILIZATION_SELECTION
    )
    assert bundle.columns["prevalent_cvd"].role == VariableRole.COMORBIDITY_FLAG

    acute = bundle.df[(bundle.df["duration_months"] <= 12.0) & (bundle.df["event"] == 1)]
    survivors_12m = bundle.df[bundle.df["duration_months"] > 12.0]
    assert len(acute) >= 30
    assert float(acute["log_crp_z"].mean()) > float(survivors_12m["log_crp_z"].mean()) + 0.35

    for col in ("bmi", "sbp_mmhg", "total_chol_mg_dl", "hdl_chol_mg_dl"):
        assert bundle.df[col].isna().sum() > 0
    assert bundle.df["duration_months"].isna().sum() == 0
    assert bundle.df["event"].isna().sum() == 0


def test_tier1_synthetic_is_deterministic_and_validates_size() -> None:
    b1 = make_synthetic_bias_cohort(n=1_000, seed=7)
    b2 = make_synthetic_bias_cohort(n=1_000, seed=7)
    assert b1.sha256 == b2.sha256
    assert np.array_equal(b1.df.to_numpy(), b2.df.to_numpy(), equal_nan=True)

    with pytest.raises(ValueError, match="n must be >= 200"):
        make_synthetic_bias_cohort(n=50, seed=42)


def test_tier1_sealed_synthetic_loads_verified_cohort_fast() -> None:
    """CLI / CI path must load the golden parquet, not regenerate the DGP."""
    t0 = time.perf_counter()
    bundle = load_synthetic_bias_cohort()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 200.0
    assert bundle.sha256 == SYNTHETIC_PARQUET_SHA256
    assert bundle.name == "synthetic_bias_n4000_seed42"
    assert len(bundle.df) == 4_000
    assert int(bundle.df["event"].sum()) == 1_533
    assert not bundle.df["subject_id"].duplicated().any()
    assert (bundle.df["duration_months"] >= 0.5).all()
    assert bundle.df["log_crp_z"].notna().all()
    assert bundle.columns["post_tx_statin"].role == VariableRole.MEDIATOR
    exp = bundle.df[bundle.exposure_col]
    assert float(exp.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(exp.std(ddof=0)) == pytest.approx(1.0, abs=1e-6)


def test_tier1_sealed_synthetic_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_synthetic_bias_cohort(parquet_path=tmp_path / "missing.parquet")

    source = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "synthetic_bias_n4000_seed42.parquet"
    )
    copy = tmp_path / "tampered.parquet"
    copy.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="SHA-256"):
        load_synthetic_bias_cohort(parquet_path=copy, expected_sha256="0" * 64)

    bad_df = pd.read_parquet(source).head(5).copy()
    bad_df["subject_id"] = 1
    dup_path = tmp_path / "dup.parquet"
    bad_df.to_parquet(dup_path, index=False)
    with pytest.raises(ValueError, match="unique subject_id"):
        load_synthetic_bias_cohort(parquet_path=dup_path, expected_sha256=None)


def test_tier2_nhanes_loads_verified_real_cohort_fast() -> None:
    t0 = time.perf_counter()
    bundle = load_nhanes_mortality_cohort()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 200.0
    assert bundle.sha256 == NHANES_PARQUET_SHA256
    assert bundle.max_duration_months == pytest.approx(159.0)
    df = bundle.df
    assert len(df) == 11_055
    assert int(df["event"].sum()) == 1_697
    assert not df["SEQN"].duplicated().any()
    assert (df["duration_months"] >= 0.5).all()
    assert df["age_years"].min() >= 20.0
    assert df["age_years"].max() <= 80.0
    assert df["log_crp_z"].notna().all()
    assert float(df["log_crp_z"].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(df["log_crp_z"].std(ddof=0)) == pytest.approx(1.0, abs=1e-6)
    assert "UCOD_LEADING" not in df.columns
    assert "ucod_leading" not in df.columns

    binary_flags = (
        "sex_male",
        "smoking_ever",
        "prevalent_cvd",
        "prevalent_cancer",
        "diabetes_dx",
        "hospitalized_past_year",
    )
    for col in binary_flags:
        observed = set(df[col].dropna().unique().tolist())
        assert observed <= {0.0, 1.0}


def test_tier2_nhanes_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_nhanes_mortality_cohort(parquet_path=tmp_path / "missing.parquet")

    source = Path(__file__).resolve().parent.parent / "data" / "nhanes_2007_2010_mort.parquet"
    copy = tmp_path / "tampered.parquet"
    copy.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="SHA-256"):
        load_nhanes_mortality_cohort(parquet_path=copy, expected_sha256="0" * 64)

    bad_df = pd.read_parquet(source).head(5).copy()
    bad_df["SEQN"] = 1
    dup_path = tmp_path / "dup.parquet"
    bad_df.to_parquet(dup_path, index=False)
    with pytest.raises(ValueError, match="unique SEQN"):
        load_nhanes_mortality_cohort(parquet_path=dup_path, expected_sha256=None)

    ucod_df = pd.read_parquet(source).head(5).copy()
    ucod_df["UCOD_LEADING"] = 1.0
    ucod_path = tmp_path / "ucod.parquet"
    ucod_df.to_parquet(ucod_path, index=False)
    with pytest.raises(ValueError, match="UCOD_LEADING"):
        load_nhanes_mortality_cohort(parquet_path=ucod_path, expected_sha256=None)

    refusal_df = pd.read_parquet(source).head(5).copy()
    refusal_df.loc[0, "smoking_ever"] = 9.0
    refusal_path = tmp_path / "refusal.parquet"
    refusal_df.to_parquet(refusal_path, index=False)
    with pytest.raises(ValueError, match="invalid or unrecoded"):
        load_nhanes_mortality_cohort(parquet_path=refusal_path, expected_sha256=None)


def test_cohort_bundle_invariants() -> None:
    base = make_synthetic_bias_cohort(n=250, seed=1)

    df_nan_exp = base.df.copy()
    df_nan_exp.loc[0, "log_crp_z"] = np.nan
    with pytest.raises(ValueError, match="never imputed"):
        CohortBundle(name="bad_exp", sha256="a", df=df_nan_exp, columns=base.columns)

    df_zero_dur = base.df.copy()
    df_zero_dur.loc[0, "duration_months"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        CohortBundle(name="bad_dur", sha256="a", df=df_zero_dur, columns=base.columns)

    df_bad_age = base.df.copy()
    df_bad_age.loc[0, "age_years"] = 85.0
    with pytest.raises(ValueError, match="age_years out of"):
        CohortBundle(name="bad_age", sha256="a", df=df_bad_age, columns=base.columns)

    df_missing_col = base.df.drop(columns=["event"])
    with pytest.raises(ValueError, match="missing required columns"):
        CohortBundle(name="bad_col", sha256="a", df=df_missing_col, columns=base.columns)


@pytest.mark.parametrize("cohort_kind", ["synthetic", "nhanes"])
def test_outcome_blind_schema_strips_all_outcome_and_duration_fields(cohort_kind: str) -> None:
    bundle = (
        make_synthetic_bias_cohort(n=500, seed=42)
        if cohort_kind == "synthetic"
        else load_nhanes_mortality_cohort()
    )
    blind = OutcomeBlindSchema.from_cohort(bundle)
    payload_json = json.dumps(blind.model_dump(mode="json")).lower()

    assert blind.n_baseline_rows == len(bundle.df)
    assert blind.exposure_col == "log_crp_z"
    col_names = {col.name for col in blind.columns}
    assert "duration_months" not in col_names
    assert "event" not in col_names
    assert "SEQN" not in col_names
    assert "subject_id" not in col_names

    for forbidden in FORBIDDEN_BLIND_SUBSTRINGS:
        assert forbidden not in payload_json

    for col_summary in blind.columns:
        assert 0.0 <= col_summary.missing_rate <= 1.0
        assert col_summary.timing != CollectionTiming.OUTCOME_FOLLOWUP
