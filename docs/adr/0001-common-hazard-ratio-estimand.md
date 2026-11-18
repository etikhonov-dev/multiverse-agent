# ADR-0001: Common Conditional Hazard Ratio Estimand & Pre-Washout Imputation

- **Status:** Accepted
- **Date:** 2026-11-18

## Context

Multiverse sensitivity analyses in observational survival cohorts frequently mix incompatible statistical estimands under a single "model choice" dimension—comparing `LogisticRegression` odds ratios over fixed horizons, `LightGBM` tree probabilities, and Cox proportional-hazards hazard ratios. Because odds ratios diverge from hazard ratios under non-rare events and fixed horizons discard censoring times, variance attributed to "estimator" conflates *estimand* mismatch with genuine parametric sensitivity.

Additionally, fitting missing-value imputers (`IterativeImputer`, medians) *after* landmark washout truncation (`T > W`) leaks post-baseline survival status into baseline covariate imputations, creating subtle outcome-permutation dependence across washout branches.

## Decision

1. **Common Conditional PH Estimand:** Restrict `EstimatorFamily` to three proportional-hazards models estimating the conditional log hazard ratio per 1-SD increment in `log_crp_z`:
   - `cox_breslow`: semiparametric Cox PH with Breslow ties (`lifelines.CoxPHFitter`)
   - `cox_spline`: Royston-Parmar flexible parametric PH with 3 cubic spline knots on median-scaled follow-up times (`n_baseline_knots=3`), warm-started from piecewise-exponential estimates
   - `piecewise_exponential`: profiled Holford (1980) 12-month piecewise-exponential PH model using exact intra-interval exposure $\tau_{ik}$ and closed-form $\hat{\lambda}_k(\beta) = 0$ elimination of zero-event intervals
2. **Pre-Washout Outcome-Blind Imputation Caching:** Standardize primary exposure once on observed baseline values (never imputed), filter baseline eligibility (`age_floor <= 70`, `cohort_definition`), and fit missingness (`complete_case`, bitmask-deduplicated `median_indicator`, deterministic `reg_imputation`) strictly before landmark survival truncation `t_risk = min(T, H) - W > 0`.
3. **Non-Collapsibility Disclosure:** Because conditional Cox hazard ratios are non-collapsible across distinct covariate sets, `adjustment_set` shifts between `minimal` and `clinical` reflect both confounding control and conditional-scale non-collapsibility rather than pure bias alone.

## Rejected Alternatives

- **Mixing `LogisticRegression` / `LightGBM` / `CoxPHFitter`:** Rejected because odds ratios, risk differences, and hazard ratios are mathematically non-comparable on the same specification curve.
- **Post-washout imputation:** Rejected because conditioning baseline imputation on `T > W` violates outcome-permutation invariance and introduces survival-into-feature leakage.

## Consequences

- All three estimators agree within `atol = 0.04` on `log_hr` under mid-interval censoring and zero-event tail intervals.
- Baseline design caching yields a ~6x speedup, keeping 108-branch real NHANES execution under `10s` single-core CPU.
- Exact Owen (2014) Sobol-Shapley attribution cleanly separates genuine estimator sensitivity (`0.2%` on NHANES) from epidemiological design drivers (`cohort_definition` `56.9%`, `washout_months` `34.1%`).
