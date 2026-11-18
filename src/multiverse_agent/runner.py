"""Leak-free survival runner with baseline-cached imputation and three PH estimators."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm

from multiverse_agent.cohort import CohortBundle, VariableRole
from multiverse_agent.spec import (
    AnalyticalSpec,
    CohortDefinition,
    EstimatorFamily,
    MissingnessStrategy,
    SpecificationSpace,
)

Z_95 = float(norm.ppf(0.975))
PIECEWISE_INTERVAL_MONTHS = 12.0
MIN_SUBJECTS = 20
MIN_EVENTS = 10


class RunRecord(BaseModel):
    """Immutable result record for one AnalyticalSpec execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    spec_id: str
    spec: AnalyticalSpec
    n_subjects: int
    n_events: int
    person_years: float
    converged: bool
    failure_reason: str | None = None
    log_hr: float | None = None
    se_log_hr: float | None = None
    hr: float | None = None
    ci_lower_95: float | None = None
    ci_upper_95: float | None = None
    p_value: float | None = None
    concordance_index: float | None = None


@dataclass(frozen=True)
class BaselineDesign:
    """Outcome-blind baseline cohort subset and imputed design matrix."""

    row_indices: NDArray[np.int64]
    matrix: NDArray[np.float64]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class _FitResult:
    log_hr: float
    se_log_hr: float
    concordance_index: float


def _compute_run_id(cohort_sha256: str, spec: AnalyticalSpec, seed: int) -> str:
    payload = json.dumps(
        {
            "cohort_sha256": cohort_sha256,
            "spec": spec.model_dump(mode="json"),
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _harrell_concordance_index(
    event_times: NDArray[np.float64],
    predicted_survival_scores: NDArray[np.float64],
    events: NDArray[np.int64],
) -> float:
    """Compute exact Harrell's C-index in O(N log N) via a Fenwick tree."""
    n = len(event_times)
    if n < 2 or int(np.sum(events)) == 0:
        return 0.5

    _, ranks_zero = np.unique(predicted_survival_scores, return_inverse=True)
    ranks = ranks_zero + 1
    max_rank = int(ranks.max())
    tree = np.zeros(max_rank + 1, dtype=np.int64)

    def _add(idx: int) -> None:
        while idx <= max_rank:
            tree[idx] += 1
            idx += idx & -idx

    def _prefix_sum(idx: int) -> int:
        total = 0
        while idx > 0:
            total += int(tree[idx])
            idx -= idx & -idx
        return total

    # Process descending by time; at tied times process censored (0) before events (1)
    order = np.lexsort((events, -event_times))
    total_inserted = 0
    total_pairs = 0
    concordant_twice = 0

    pos = 0
    while pos < n:
        t_curr = float(event_times[order[pos]])
        end = pos + 1
        while end < n and float(event_times[order[end]]) == t_curr:
            end += 1

        # Insert all censored subjects at t_curr first
        event_positions: list[int] = []
        for idx_pos in range(pos, end):
            row = int(order[idx_pos])
            if events[row] == 0:
                _add(int(ranks[row]))
                total_inserted += 1
            else:
                event_positions.append(row)

        # Query all events at t_curr against strictly later times or tied censored
        for row in event_positions:
            r = int(ranks[row])
            le_count = _prefix_sum(r)
            lt_count = _prefix_sum(r - 1)
            gt_count = total_inserted - le_count
            eq_count = le_count - lt_count
            total_pairs += total_inserted
            concordant_twice += 2 * gt_count + eq_count

        # Insert event subjects at t_curr after querying so tied events aren't paired
        for row in event_positions:
            _add(int(ranks[row]))
            total_inserted += 1

        pos = end

    if total_pairs == 0:
        return 0.5
    return float(concordant_twice) / float(2 * total_pairs)


def _standardize_covariates(x_risk: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center all columns and unit-scale covariate columns j >= 1; preserve exposure scale."""
    x_work = x_risk.copy()
    x_work[:, 0] = x_work[:, 0] - float(np.mean(x_work[:, 0]))
    for col_idx in range(1, x_work.shape[1]):
        col = x_work[:, col_idx]
        col_mean = float(np.mean(col))
        col_sd = float(np.std(col, ddof=0))
        if col_sd > 1e-12:
            x_work[:, col_idx] = (col - col_mean) / col_sd
        else:
            x_work[:, col_idx] = col - col_mean
    return x_work


def _solve_piecewise_beta(
    x_work: NDArray[np.float64],
    t_risk: NDArray[np.float64],
    e_risk: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve profiled Holford (1980) piecewise-exponential Newton-Raphson on scaled X."""
    p = x_work.shape[1]
    max_t = float(np.max(t_risk))
    n_intervals = max(1, int(math.ceil(max_t / PIECEWISE_INTERVAL_MONTHS)))
    edges = np.arange(0, n_intervals + 1, dtype=np.float64) * PIECEWISE_INTERVAL_MONTHS
    tau = np.clip(
        np.minimum(t_risk[:, None], edges[1:][None, :]) - edges[:-1][None, :],
        0.0,
        PIECEWISE_INTERVAL_MONTHS,
    )

    event_idx = np.clip(
        np.ceil(t_risk / PIECEWISE_INTERVAL_MONTHS).astype(np.int64) - 1,
        0,
        n_intervals - 1,
    )
    e_float = e_risk.astype(np.float64)
    d_k = np.bincount(event_idx, weights=e_float, minlength=n_intervals)
    active = d_k > 0.0
    if not np.any(active):
        raise ValueError("No active intervals with positive event counts.")
    d_active = d_k[active]
    tau_active = tau[:, active]

    x_e = x_work.T @ e_float
    beta = np.zeros(p, dtype=np.float64)

    def _eval(b: NDArray[np.float64]) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        eta = x_work @ b
        eta_shift = eta - float(np.max(eta))
        w = np.exp(eta_shift)
        e_k = tau_active.T @ w
        if np.any(e_k <= 0.0) or not np.all(np.isfinite(e_k)):
            raise ValueError("Non-positive interval exposure sum.")
        loglik = float(np.dot(e_float, eta_shift) - np.dot(d_active, np.log(e_k)))
        p_mat = (tau_active * w[:, None]) / e_k[None, :]
        x_bar = x_work.T @ p_mat
        grad = x_e - x_bar @ d_active
        subj_w = p_mat @ d_active
        fisher = (x_work.T * subj_w[None, :]) @ x_work - (x_bar * d_active[None, :]) @ x_bar.T
        return loglik, grad, fisher

    loglik, grad, fisher = _eval(beta)
    converged = False
    for _ in range(60):
        try:
            step = np.linalg.solve(fisher, grad)
        except np.linalg.LinAlgError:
            fisher_reg = fisher + 1e-8 * np.eye(p, dtype=np.float64)
            step = np.linalg.solve(fisher_reg, grad)

        if not np.all(np.isfinite(step)):
            raise ValueError("Non-finite Newton step in piecewise exponential fit.")

        decrement = 0.5 * float(np.dot(grad, step))
        if decrement <= 1e-10 and float(np.max(np.abs(grad))) < 1e-6:
            converged = True
            break

        alpha = 1.0
        accepted = False
        directional = float(np.dot(grad, step))
        for _ in range(25):
            beta_cand = beta + alpha * step
            try:
                ll_cand, grad_cand, fisher_cand = _eval(beta_cand)
            except (ValueError, FloatingPointError):
                alpha *= 0.5
                continue
            if np.isfinite(ll_cand) and ll_cand >= loglik + 1e-4 * alpha * directional:
                beta = beta_cand
                loglik = ll_cand
                grad = grad_cand
                fisher = fisher_cand
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            raise ValueError("Line search failed in piecewise exponential fit.")
        if float(np.max(np.abs(alpha * step))) < 1e-8 and float(np.max(np.abs(grad))) < 1e-6:
            converged = True
            break

    if not converged:
        raise ValueError("Newton-Raphson exceeded maximum iterations.")
    return beta, fisher


def _fit_piecewise_exponential(
    x_risk: NDArray[np.float64],
    t_risk: NDArray[np.float64],
    e_risk: NDArray[np.int64],
) -> _FitResult:
    """Fit profiled Holford (1980) 12-month piecewise-exponential PH model."""
    x_work = _standardize_covariates(x_risk)
    beta, fisher = _solve_piecewise_beta(x_work, t_risk, e_risk)

    cov = np.linalg.inv(fisher)
    var_0 = float(cov[0, 0])
    if not math.isfinite(var_0) or var_0 <= 0.0:
        raise ValueError("Invalid exposure variance estimate.")
    beta_0 = float(beta[0])
    if not math.isfinite(beta_0):
        raise ValueError("Non-finite exposure log-HR.")

    eta_hat = x_work @ beta
    c_index = _harrell_concordance_index(t_risk, -eta_hat, e_risk)
    return _FitResult(
        log_hr=beta_0,
        se_log_hr=math.sqrt(var_0),
        concordance_index=c_index,
    )


def _fit_lifelines_cox(
    x_risk: NDArray[np.float64],
    t_risk: NDArray[np.float64],
    e_risk: NDArray[np.int64],
    *,
    feature_names: tuple[str, ...],
    method: str,
) -> _FitResult:
    x_work = _standardize_covariates(x_risk)
    fit_df = pd.DataFrame(x_work, columns=list(feature_names))
    if method == "spline":
        t_scale = float(np.median(t_risk))
        t_scaled = t_risk / (t_scale if t_scale > 0.0 else 1.0)
        fit_df["_duration"] = t_scaled
    else:
        t_scaled = t_risk
        fit_df["_duration"] = t_risk
    fit_df["_event"] = e_risk

    kwargs: dict[str, Any] = {"baseline_estimation_method": method}
    fit_kwargs: dict[str, Any] = {}
    if method == "spline":
        kwargs["n_baseline_knots"] = 3
        fit_kwargs["fit_options"] = {"ftol": 1e-6}
        try:
            beta_init, _ = _solve_piecewise_beta(x_work, t_risk, e_risk)
            rate0 = math.log(max(1e-6, float(np.sum(e_risk)) / float(np.sum(t_scaled))))
            fit_kwargs["initial_point"] = {
                "beta_": np.append(beta_init, rate0),
                "phi1_": np.array([1.0]),
                "phi2_": np.array([0.0]),
                "phi3_": np.array([0.0]),
            }
        except (ValueError, np.linalg.LinAlgError):
            fit_kwargs.pop("initial_point", None)

    cph = CoxPHFitter(**kwargs)
    cph.fit(fit_df, duration_col="_duration", event_col="_event", **fit_kwargs)

    exp_col = feature_names[0]
    if method == "spline":
        beta_0 = float(cph.params_[("beta_", exp_col)])
        se_0 = float(cph.standard_errors_[("beta_", exp_col)])
        beta_vec = np.array(
            [float(cph.params_[("beta_", name)]) for name in feature_names],
            dtype=np.float64,
        )
        eta_hat = x_work @ beta_vec
        c_index = _harrell_concordance_index(t_risk, -eta_hat, e_risk)
    else:
        beta_0 = float(cph.params_[exp_col])
        se_0 = float(cph.standard_errors_[exp_col])
        c_index = float(cph.concordance_index_)

    if not (
        math.isfinite(beta_0) and math.isfinite(se_0) and 0.0 < se_0 < 50.0 and abs(beta_0) < 20.0
    ):
        raise ValueError("Non-finite or diverged parameter/standard error from lifelines.")
    return _FitResult(log_hr=beta_0, se_log_hr=se_0, concordance_index=c_index)


class MultiverseRunner:
    """Deterministic, outcome-leak-free survival executor over a SpecificationSpace."""

    def __init__(
        self,
        cohort: CohortBundle,
        space: SpecificationSpace,
        *,
        seed: int = 42,
        declared_roles: Mapping[str, VariableRole] | None = None,
    ) -> None:
        self._cohort = cohort
        self._space = space
        self._seed = seed
        self._specs = space.compile(cohort, declared_roles=declared_roles)

        raw_exp = cohort.df[cohort.exposure_col].to_numpy(dtype=np.float64)
        mean_exp = float(np.mean(raw_exp))
        sd_exp = float(np.std(raw_exp, ddof=0))
        if not math.isfinite(sd_exp) or sd_exp <= 0.0:
            raise ValueError("Cohort exposure column must have strictly positive variance.")
        self._exposure_z_full = (raw_exp - mean_exp) / sd_exp

        self._baseline_cache: dict[
            tuple[int, CohortDefinition, MissingnessStrategy, str], BaselineDesign
        ] = {}

    @property
    def specs(self) -> tuple[AnalyticalSpec, ...]:
        return self._specs

    def _eligible_indices(self, age_floor: int, cohort_def: CohortDefinition) -> NDArray[np.int64]:
        df = self._cohort.df
        n = len(df)
        mask = np.ones(n, dtype=bool)
        if "age_years" in df.columns:
            ages = df["age_years"].to_numpy(dtype=np.float64)
            mask &= ages >= float(age_floor)

        if cohort_def == CohortDefinition.EXCLUDE_PREVALENT_CVD:
            cvd = df["prevalent_cvd"].to_numpy(dtype=np.float64)
            mask &= cvd == 0.0
        elif cohort_def == CohortDefinition.SUPER_CONTROLS_CLEAN:
            for col in ("prevalent_cvd", "prevalent_cancer", "diabetes_dx"):
                flag = df[col].to_numpy(dtype=np.float64)
                mask &= flag == 0.0

        return np.flatnonzero(mask).astype(np.int64)

    def prepare_baseline_design(self, spec: AnalyticalSpec) -> BaselineDesign:
        """Build or retrieve cached outcome-blind baseline imputed design matrix."""
        cache_key = (
            spec.age_floor,
            spec.cohort_definition,
            spec.missingness,
            spec.adjustment_set,
        )
        cached = self._baseline_cache.get(cache_key)
        if cached is not None:
            return cached

        if spec.adjustment_set not in self._space.adjustment_sets:
            raise ValueError(f"Unknown adjustment_set '{spec.adjustment_set}'.")

        cov_names = self._space.adjustment_sets[spec.adjustment_set]
        row_idx = self._eligible_indices(spec.age_floor, spec.cohort_definition)
        exp_sub = self._exposure_z_full[row_idx]

        if not cov_names:
            design = BaselineDesign(
                row_indices=row_idx,
                matrix=exp_sub[:, None].copy(),
                feature_names=(self._cohort.exposure_col,),
            )
            self._baseline_cache[cache_key] = design
            return design

        cov_raw = self._cohort.df.iloc[row_idx][list(cov_names)].to_numpy(dtype=np.float64)

        if spec.missingness == MissingnessStrategy.COMPLETE_CASE:
            keep = ~np.isnan(cov_raw).any(axis=1)
            kept_rows = row_idx[keep]
            mat = np.column_stack([exp_sub[keep], cov_raw[keep]])
            names = (self._cohort.exposure_col, *cov_names)
        elif spec.missingness == MissingnessStrategy.MEDIAN_INDICATOR:
            n_sub = len(row_idx)
            filled = cov_raw.copy()
            indicator_cols: list[NDArray[np.float64]] = []
            indicator_names: list[str] = []

            for j, cov_name in enumerate(cov_names):
                col = filled[:, j]
                is_nan = np.isnan(col)
                n_miss = int(np.sum(is_nan))
                if n_miss == n_sub:
                    raise ValueError(f"Covariate '{cov_name}' is 100% missing in baseline subset.")
                if n_miss > 0:
                    col_median = float(np.nanmedian(col))
                    filled[is_nan, j] = col_median
                    indicator_cols.append(is_nan.astype(np.float64))
                    indicator_names.append(f"miss_{cov_name}")

            kept_rows = row_idx
            if indicator_cols:
                ind_mat = np.column_stack(indicator_cols)
                _, unique_idx = np.unique(ind_mat, axis=1, return_index=True)
                unique_idx_sorted = np.sort(unique_idx)
                dedup_ind = ind_mat[:, unique_idx_sorted]
                dedup_names = tuple(indicator_names[int(idx)] for idx in unique_idx_sorted)
                mat = np.column_stack([exp_sub, filled, dedup_ind])
                names = (self._cohort.exposure_col, *cov_names, *dedup_names)
            else:
                mat = np.column_stack([exp_sub, filled])
                names = (self._cohort.exposure_col, *cov_names)
        else:
            # REG_IMPUTATION: outcome-blind deterministic single conditional-mean imputation
            raw_full = np.column_stack([exp_sub, cov_raw])
            if np.isnan(cov_raw).any():
                importlib.import_module("sklearn.experimental.enable_iterative_imputer")
                from sklearn.impute import IterativeImputer

                imputer = IterativeImputer(
                    max_iter=10,
                    sample_posterior=False,
                    random_state=self._seed,
                )
                mat = np.asarray(imputer.fit_transform(raw_full), dtype=np.float64)
            else:
                mat = raw_full
            kept_rows = row_idx
            names = (self._cohort.exposure_col, *cov_names)

        design = BaselineDesign(row_indices=kept_rows, matrix=mat, feature_names=names)
        self._baseline_cache[cache_key] = design
        return design

    def run_spec(self, spec: AnalyticalSpec) -> RunRecord:
        """Execute one validated AnalyticalSpec cleanly without outcome leakage."""
        baseline = self.prepare_baseline_design(spec)

        duration = self._cohort.df[self._cohort.duration_col].to_numpy(dtype=np.float64)[
            baseline.row_indices
        ]
        event = self._cohort.df[self._cohort.event_col].to_numpy(dtype=np.int64)[
            baseline.row_indices
        ]

        if spec.followup_horizon_months is not None:
            horizon = float(spec.followup_horizon_months)
            t_cens = np.minimum(duration, horizon)
            e_cens = np.where(duration <= horizon, event, 0)
        else:
            t_cens = duration
            e_cens = event

        washout = float(spec.washout_months)
        at_risk = t_cens > washout
        t_risk = t_cens[at_risk] - washout
        e_risk = e_cens[at_risk]
        x_raw = baseline.matrix[at_risk]

        # Drop post-washout zero-variance covariate/indicator columns (never exposure col 0)
        keep_cols = [0]
        for j in range(1, x_raw.shape[1]):
            if float(np.std(x_raw[:, j], ddof=0)) > 1e-12:
                keep_cols.append(j)
        x_risk = x_raw[:, keep_cols]
        feature_names = tuple(baseline.feature_names[j] for j in keep_cols)

        n_subjects = int(len(t_risk))
        n_events = int(np.sum(e_risk))
        person_years = float(np.sum(t_risk)) / 12.0
        run_id = _compute_run_id(self._cohort.sha256, spec, self._seed)

        if (
            n_subjects < MIN_SUBJECTS
            or n_events < MIN_EVENTS
            or float(np.std(x_risk[:, 0], ddof=0)) <= 1e-12
        ):
            return RunRecord(
                run_id=run_id,
                spec_id=spec.spec_id,
                spec=spec,
                n_subjects=n_subjects,
                n_events=n_events,
                person_years=person_years,
                converged=False,
                failure_reason="insufficient_events_or_variation",
            )

        try:
            if spec.estimator == EstimatorFamily.PIECEWISE_EXPONENTIAL:
                fit = _fit_piecewise_exponential(x_risk, t_risk, e_risk)
            elif spec.estimator == EstimatorFamily.COX_BRESLOW:
                fit = _fit_lifelines_cox(
                    x_risk,
                    t_risk,
                    e_risk,
                    feature_names=feature_names,
                    method="breslow",
                )
            else:
                fit = _fit_lifelines_cox(
                    x_risk,
                    t_risk,
                    e_risk,
                    feature_names=feature_names,
                    method="spline",
                )
        except (ConvergenceError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return RunRecord(
                run_id=run_id,
                spec_id=spec.spec_id,
                spec=spec,
                n_subjects=n_subjects,
                n_events=n_events,
                person_years=person_years,
                converged=False,
                failure_reason=f"estimator_failed: {exc}",
            )

        hr = math.exp(fit.log_hr)
        ci_lower = math.exp(fit.log_hr - Z_95 * fit.se_log_hr)
        ci_upper = math.exp(fit.log_hr + Z_95 * fit.se_log_hr)
        p_value = float(2.0 * norm.sf(abs(fit.log_hr / fit.se_log_hr)))

        return RunRecord(
            run_id=run_id,
            spec_id=spec.spec_id,
            spec=spec,
            n_subjects=n_subjects,
            n_events=n_events,
            person_years=person_years,
            converged=True,
            failure_reason=None,
            log_hr=fit.log_hr,
            se_log_hr=fit.se_log_hr,
            hr=hr,
            ci_lower_95=ci_lower,
            ci_upper_95=ci_upper,
            p_value=p_value,
            concordance_index=fit.concordance_index,
        )

    def run_all(self, *, jsonl_path: Path | None = None) -> tuple[RunRecord, ...]:
        """Execute every compiled specification deterministically and optionally write JSONL."""
        records = tuple(self.run_spec(spec) for spec in self._specs)
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                json.dumps(rec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
                for rec in records
            ]
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return records


__all__ = ["BaselineDesign", "MultiverseRunner", "RunRecord"]
