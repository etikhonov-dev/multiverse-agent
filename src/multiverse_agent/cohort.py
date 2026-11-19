"""Cohort bundles, outcome-blind schema firewall, Tier 1 DGP, and Tier 2 NHANES loader."""

from __future__ import annotations

import decimal
import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

NHANES_PARQUET_SHA256 = "e968d9a9752de372b27f1870cf325593ff07fbc81fab4a78a9f93a73a21f3865"
DEFAULT_NHANES_PARQUET = (
    Path(__file__).resolve().parents[2] / "data" / "nhanes_2007_2010_mort.parquet"
)
REFUSAL_CODES = frozenset({7.0, 9.0, 77.0, 99.0, 777.0, 999.0})


class CollectionTiming(StrEnum):
    BASELINE_PRE_EXPOSURE = "baseline_pre_exposure"
    BASELINE_CONCURRENT = "baseline_concurrent"
    POST_BASELINE_FOLLOWUP = "post_baseline_followup"
    HEALTHCARE_UTILIZATION_SELECTION = "utilization_selection"
    OUTCOME_FOLLOWUP = "outcome_followup"


class VariableRole(StrEnum):
    EXPOSURE = "exposure"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    COLLIDER = "collider"
    COMORBIDITY_FLAG = "comorbidity_flag"
    DURATION = "duration"
    EVENT = "event"


class ColumnProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    role: VariableRole
    timing: CollectionTiming
    is_binary: bool = False


class BlindColumnSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    timing: CollectionTiming
    missing_rate: float
    is_binary: bool
    p25: float | None
    p50: float | None
    p75: float | None


class OutcomeBlindSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort_name: str
    cohort_sha256: str
    n_baseline_rows: int
    exposure_col: str
    columns: tuple[BlindColumnSummary, ...]

    @classmethod
    def from_cohort(cls, bundle: CohortBundle) -> OutcomeBlindSchema:
        """Physically drop all outcome/duration/ID columns before computing metadata."""
        forbidden_roles = {VariableRole.DURATION, VariableRole.EVENT}
        safe_columns = [
            prov
            for prov in bundle.columns.values()
            if prov.role not in forbidden_roles and prov.timing != CollectionTiming.OUTCOME_FOLLOWUP
        ]
        safe_df = bundle.df[[prov.name for prov in safe_columns]].copy()

        summaries: list[BlindColumnSummary] = []
        for prov in safe_columns:
            series = pd.to_numeric(safe_df[prov.name], errors="coerce")
            missing_rate = float(series.isna().mean())
            non_null = series.dropna().to_numpy(dtype=np.float64)
            if non_null.size == 0:
                p25 = p50 = p75 = None
            else:
                q25, q50, q75 = np.quantile(non_null, [0.25, 0.50, 0.75])
                p25, p50, p75 = float(q25), float(q50), float(q75)
            summaries.append(
                BlindColumnSummary(
                    name=prov.name,
                    description=prov.description,
                    timing=prov.timing,
                    missing_rate=round(missing_rate, 6),
                    is_binary=prov.is_binary,
                    p25=round(p25, 6) if p25 is not None else None,
                    p50=round(p50, 6) if p50 is not None else None,
                    p75=round(p75, 6) if p75 is not None else None,
                )
            )

        return cls(
            cohort_name=bundle.name,
            cohort_sha256=bundle.sha256,
            n_baseline_rows=int(len(safe_df)),
            exposure_col=bundle.exposure_col,
            columns=tuple(summaries),
        )


@dataclass(frozen=True)
class CohortBundle:
    name: str
    sha256: str
    df: pd.DataFrame
    columns: dict[str, ColumnProvenance]
    exposure_col: str = "log_crp_z"
    duration_col: str = "duration_months"
    event_col: str = "event"
    id_col: str = "SEQN"

    def __post_init__(self) -> None:
        required_cols = {self.exposure_col, self.duration_col, self.event_col}
        missing_df = required_cols - set(self.df.columns)
        if missing_df:
            raise ValueError(f"Cohort DataFrame missing required columns: {sorted(missing_df)}")
        missing_prov = required_cols - set(self.columns)
        if missing_prov:
            raise ValueError(f"Cohort provenance missing required columns: {sorted(missing_prov)}")
        if self.df[self.exposure_col].isna().any():
            raise ValueError("Primary exposure column must be fully observed (never imputed).")
        if (self.df[self.duration_col] <= 0.0).any():
            raise ValueError("Follow-up duration_months must be strictly positive for all rows.")
        if "age_years" in self.df.columns:
            min_age = float(self.df["age_years"].min())
            max_age = float(self.df["age_years"].max())
            if min_age < 20.0 or max_age > 80.0:
                raise ValueError(f"age_years out of [20, 80] range: [{min_age}, {max_age}]")

    @property
    def max_duration_months(self) -> float:
        return float(self.df[self.duration_col].max())


# libmpdec evaluates exp/sqrt identically on macOS and Linux. np.exp and np.sum do not:
# libm and SIMD reduction width change float bytes, which desynchronizes later binomial draws.
_PORTABLE_DECIMAL = decimal.Context(prec=40, rounding=decimal.ROUND_HALF_EVEN)


def _portable_exp(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate exp elementwise with a platform-independent decimal implementation."""
    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape(-1)
    out = np.empty(flat.shape[0], dtype=np.float64)
    ctx = _PORTABLE_DECIMAL
    for index, value in enumerate(flat.tolist()):
        if value > 709.0:
            out[index] = math.inf
        elif value < -745.0:
            out[index] = 0.0
        else:
            out[index] = float(ctx.exp(decimal.Decimal(value)))
    return out.reshape(array.shape)


def _portable_mean_std(values: NDArray[np.float64]) -> tuple[float, float]:
    """Population mean and std via math.fsum, independent of SIMD reduction width."""
    flat = [float(item) for item in np.asarray(values, dtype=np.float64).reshape(-1)]
    count = len(flat)
    mean = math.fsum(flat) / count
    variance = math.fsum((item - mean) * (item - mean) for item in flat) / count
    std = float(_PORTABLE_DECIMAL.sqrt(decimal.Decimal(variance)))
    return mean, std


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    exponent = _portable_exp(-np.asarray(x, dtype=np.float64))
    return np.asarray(1.0 / (1.0 + exponent), dtype=np.float64)


def _base_provenance(*, include_mediator: bool) -> dict[str, ColumnProvenance]:
    entries = [
        ColumnProvenance(
            name="log_crp_z",
            description="Baseline standardized log C-reactive protein (1-SD units)",
            role=VariableRole.EXPOSURE,
            timing=CollectionTiming.BASELINE_CONCURRENT,
        ),
        ColumnProvenance(
            name="age_years",
            description="Age at baseline examination in years (20-80)",
            role=VariableRole.CONFOUNDER,
            timing=CollectionTiming.BASELINE_PRE_EXPOSURE,
        ),
        ColumnProvenance(
            name="sex_male",
            description="Biological sex indicator (1=male, 0=female)",
            role=VariableRole.CONFOUNDER,
            timing=CollectionTiming.BASELINE_PRE_EXPOSURE,
            is_binary=True,
        ),
        ColumnProvenance(
            name="bmi",
            description="Body mass index at baseline examination (kg/m^2)",
            role=VariableRole.CONFOUNDER,
            timing=CollectionTiming.BASELINE_CONCURRENT,
        ),
        ColumnProvenance(
            name="sbp_mmhg",
            description="Mean baseline systolic blood pressure (mmHg)",
            role=VariableRole.CONFOUNDER,
            timing=CollectionTiming.BASELINE_CONCURRENT,
        ),
        ColumnProvenance(
            name="total_chol_mg_dl",
            description="Baseline serum total cholesterol (mg/dL)",
            role=VariableRole.CONFOUNDER,
            timing=CollectionTiming.BASELINE_CONCURRENT,
        ),
        ColumnProvenance(
            name="hdl_chol_mg_dl",
            description="Baseline serum HDL cholesterol (mg/dL)",
            role=VariableRole.CONFOUNDER,
            timing=CollectionTiming.BASELINE_CONCURRENT,
        ),
        ColumnProvenance(
            name="smoking_ever",
            description="Ever smoked >=100 cigarettes at baseline (1=yes, 0=no)",
            role=VariableRole.CONFOUNDER,
            timing=CollectionTiming.BASELINE_PRE_EXPOSURE,
            is_binary=True,
        ),
        ColumnProvenance(
            name="prevalent_cvd",
            description="Prevalent baseline cardiovascular disease diagnosis flag",
            role=VariableRole.COMORBIDITY_FLAG,
            timing=CollectionTiming.BASELINE_PRE_EXPOSURE,
            is_binary=True,
        ),
        ColumnProvenance(
            name="prevalent_cancer",
            description="Prevalent baseline cancer/malignancy diagnosis flag",
            role=VariableRole.COMORBIDITY_FLAG,
            timing=CollectionTiming.BASELINE_PRE_EXPOSURE,
            is_binary=True,
        ),
        ColumnProvenance(
            name="diabetes_dx",
            description="Baseline diagnosed diabetes mellitus flag",
            role=VariableRole.COMORBIDITY_FLAG,
            timing=CollectionTiming.BASELINE_PRE_EXPOSURE,
            is_binary=True,
        ),
        ColumnProvenance(
            name="hospitalized_past_year",
            description="Overnight hospital stay in prior 12 months (utilization collider)",
            role=VariableRole.COLLIDER,
            timing=CollectionTiming.HEALTHCARE_UTILIZATION_SELECTION,
            is_binary=True,
        ),
    ]
    if include_mediator:
        entries.append(
            ColumnProvenance(
                name="post_tx_statin",
                description="Post-baseline statin initiation during follow-up (mediator)",
                role=VariableRole.MEDIATOR,
                timing=CollectionTiming.POST_BASELINE_FOLLOWUP,
                is_binary=True,
            )
        )
    entries.extend(
        [
            ColumnProvenance(
                name="duration_months",
                description="Follow-up time in months from baseline",
                role=VariableRole.DURATION,
                timing=CollectionTiming.OUTCOME_FOLLOWUP,
            ),
            ColumnProvenance(
                name="event",
                description="All-cause mortality indicator (1=deceased, 0=censored)",
                role=VariableRole.EVENT,
                timing=CollectionTiming.OUTCOME_FOLLOWUP,
                is_binary=True,
            ),
        ]
    )
    return {item.name: item for item in entries}


def _hash_dataframe(name: str, df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    for col in df.columns:
        digest.update(str(col).encode("utf-8"))
        arr = df[col].to_numpy()
        digest.update(arr.tobytes())
    return digest.hexdigest()


def make_synthetic_bias_cohort(n: int = 4_000, seed: int = 42) -> CohortBundle:
    """Generate Tier 1 synthetic survival cohort with calibrated acute reverse causation.

    Structural proportional hazard after t > 12m has exact beta_true = log(1.20)
    per 1.0 SD of log_crp_z (HR_true = 1.20), whereas acute (0, 12]m mortality has
    a reverse-causation spike driven by latent subclinical illness U_subclinical.
    """
    if n < 200:
        raise ValueError(f"Synthetic cohort size n must be >= 200, got {n}.")
    rng = np.random.default_rng(seed)

    u_subclinical = rng.normal(0.0, 1.0, size=n)
    age_years = np.clip(rng.normal(52.0, 14.0, size=n), 20.0, 80.0)
    sex_male = rng.binomial(1, 0.48, size=n).astype(np.float64)
    smoking_ever = rng.binomial(
        1, _sigmoid(-0.4 + 0.3 * sex_male + 0.01 * (age_years - 50.0))
    ).astype(np.float64)

    bmi_true = np.clip(rng.normal(27.5 + 0.04 * (age_years - 50.0), 5.0, size=n), 16.0, 55.0)
    sbp_true = np.clip(
        rng.normal(124.0 + 0.55 * (age_years - 50.0) + 0.8 * (bmi_true - 27.5), 16.0, size=n),
        85.0,
        220.0,
    )
    total_chol_true = np.clip(
        rng.normal(198.0 + 0.3 * (age_years - 50.0), 36.0, size=n), 100.0, 360.0
    )
    hdl_chol_true = np.clip(
        rng.normal(54.0 - 4.0 * sex_male - 0.5 * (bmi_true - 27.5), 14.0, size=n), 20.0, 120.0
    )

    prevalent_cvd = rng.binomial(
        1, _sigmoid(-2.4 + 1.45 * u_subclinical + 0.045 * (age_years - 50.0))
    ).astype(np.float64)
    prevalent_cancer = rng.binomial(
        1, _sigmoid(-2.6 + 1.35 * u_subclinical + 0.035 * (age_years - 50.0))
    ).astype(np.float64)
    diabetes_dx = rng.binomial(
        1, _sigmoid(-2.2 + 0.65 * u_subclinical + 0.06 * (bmi_true - 27.5))
    ).astype(np.float64)

    raw_crp = (
        0.025 * (age_years - 50.0)
        + 0.08 * (bmi_true - 27.5)
        + 0.25 * smoking_ever
        + 0.85 * u_subclinical
        + rng.normal(0.0, 0.75, size=n)
    )
    crp_mean, crp_std = _portable_mean_std(raw_crp)
    log_crp_z = (raw_crp - crp_mean) / crp_std

    beta_true = math.log(1.20)
    eta_conf = (
        0.055 * (age_years - 55.0)
        + 0.25 * sex_male
        + 0.22 * smoking_ever
        + 0.015 * (bmi_true - 27.5)
        + 0.012 * (sbp_true - 125.0)
        + 0.002 * (total_chol_true - 200.0)
        - 0.008 * (hdl_chol_true - 52.0)
    )
    rate_acute = 0.0022 * _portable_exp(beta_true * log_crp_z + eta_conf + 1.50 * u_subclinical)
    rate_chronic = 0.0034 * _portable_exp(beta_true * log_crp_z + eta_conf)

    exp_rv = rng.exponential(1.0, size=n)
    h1 = 12.0 * rate_acute
    t_event = np.where(exp_rv <= h1, exp_rv / rate_acute, 12.0 + (exp_rv - h1) / rate_chronic)
    censor_t = rng.uniform(72.0, 144.0, size=n)
    duration_months = np.maximum(0.5, np.minimum(t_event, censor_t))
    event = (t_event <= censor_t).astype(np.int64)

    post_tx_statin = rng.binomial(
        1,
        _sigmoid(-1.5 + 0.75 * log_crp_z + 0.015 * (total_chol_true - 200.0) + 0.9 * prevalent_cvd),
    ).astype(np.float64)
    hospitalized_past_year = rng.binomial(
        1, _sigmoid(-1.8 + 0.65 * log_crp_z + 1.2 * u_subclinical + 0.9 * prevalent_cvd)
    ).astype(np.float64)

    bmi_obs = bmi_true.copy()
    sbp_obs = sbp_true.copy()
    tchol_obs = total_chol_true.copy()
    hdl_obs = hdl_chol_true.copy()

    miss_bmi = rng.binomial(1, _sigmoid(-2.4 + 0.015 * (age_years - 50.0))) == 1
    miss_sbp = rng.binomial(1, _sigmoid(-2.3 + 0.2 * sex_male)) == 1
    miss_lipids = rng.binomial(1, _sigmoid(-2.5 + 0.01 * (age_years - 50.0))) == 1
    bmi_obs[miss_bmi] = np.nan
    sbp_obs[miss_sbp] = np.nan
    tchol_obs[miss_lipids] = np.nan
    hdl_obs[miss_lipids] = np.nan

    df = pd.DataFrame(
        {
            "subject_id": np.arange(1, n + 1, dtype=np.int64),
            "log_crp_z": log_crp_z.astype(np.float64),
            "age_years": age_years.astype(np.float64),
            "sex_male": sex_male,
            "bmi": bmi_obs.astype(np.float64),
            "sbp_mmhg": sbp_obs.astype(np.float64),
            "total_chol_mg_dl": tchol_obs.astype(np.float64),
            "hdl_chol_mg_dl": hdl_obs.astype(np.float64),
            "smoking_ever": smoking_ever,
            "prevalent_cvd": prevalent_cvd,
            "prevalent_cancer": prevalent_cancer,
            "diabetes_dx": diabetes_dx,
            "post_tx_statin": post_tx_statin,
            "hospitalized_past_year": hospitalized_past_year,
            "duration_months": duration_months.astype(np.float64),
            "event": event,
        }
    )
    name = f"synthetic_bias_n{n}_seed{seed}"
    return CohortBundle(
        name=name,
        sha256=_hash_dataframe(name, df),
        df=df,
        columns=_base_provenance(include_mediator=True),
        id_col="subject_id",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_nhanes_mortality_cohort(
    parquet_path: Path | None = None,
    *,
    expected_sha256: str | None = NHANES_PARQUET_SHA256,
) -> CohortBundle:
    """Load and verify the bundled CDC NHANES 2007-2010 + 2019 Linked Mortality cohort."""
    path = parquet_path if parquet_path is not None else DEFAULT_NHANES_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"NHANES cohort parquet not found at {path}")

    actual_sha256 = _sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"NHANES parquet SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    df = pq.read_table(path).to_pandas()
    if "SEQN" not in df.columns or df["SEQN"].duplicated().any():
        raise ValueError("NHANES parquet must contain unique SEQN identifiers.")
    if "UCOD_LEADING" in df.columns or "ucod_leading" in df.columns:
        raise ValueError("UCOD_LEADING must not be present in the bundled NHANES cohort.")
    if (df["duration_months"] < 0.5).any():
        raise ValueError("All NHANES duration_months entries must be >= 0.5.")

    binary_cols = (
        "sex_male",
        "smoking_ever",
        "prevalent_cvd",
        "prevalent_cancer",
        "diabetes_dx",
        "hospitalized_past_year",
    )
    for col in binary_cols:
        vals = set(df[col].dropna().unique().tolist())
        if vals & REFUSAL_CODES or not vals <= {0.0, 1.0}:
            raise ValueError(f"Column {col} contains invalid or unrecoded values: {sorted(vals)}")

    return CohortBundle(
        name="nhanes_2007_2010_mort",
        sha256=actual_sha256,
        df=df,
        columns=_base_provenance(include_mediator=False),
        id_col="SEQN",
    )
