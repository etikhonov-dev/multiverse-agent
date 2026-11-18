"""Specification curve aggregation, exact Owen Sobol-Shapley attribution, and strike plot."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(tempfile.gettempdir()) / "multiverse_agent_mpl"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from multiverse_agent.runner import RunRecord
from multiverse_agent.spec import AnalyticalSpec

SPEC_DIMENSIONS: tuple[str, ...] = (
    "age_floor",
    "cohort_definition",
    "washout_months",
    "followup_horizon_months",
    "missingness",
    "adjustment_set",
    "estimator",
)

_DIMENSION_LABELS: dict[str, str] = {
    "age_floor": "Age floor",
    "cohort_definition": "Cohort def.",
    "washout_months": "Washout",
    "followup_horizon_months": "Horizon",
    "missingness": "Missingness",
    "adjustment_set": "Adjust set",
    "estimator": "Estimator",
}


class SpecCurveSummary(BaseModel):
    """Summary statistics and exact D=7 Owen Sobol-Shapley variance shares."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_specs_total: int
    n_specs_converged: int
    n_specs_failed: int
    median_hr: float | None
    min_hr: float | None
    max_hr: float | None
    q25_hr: float | None
    q75_hr: float | None
    median_log_hr: float | None
    share_ci_excluding_one: float
    sign_consistency_share: float
    shapley_shares: dict[str, float]
    top_driver: str | None
    zero_variance: bool


def _spec_dim_value(spec: AnalyticalSpec, dim: str) -> str:
    if dim == "age_floor":
        return f"{spec.age_floor}y"
    if dim == "cohort_definition":
        return spec.cohort_definition.value
    if dim == "washout_months":
        return f"{spec.washout_months}m"
    if dim == "followup_horizon_months":
        if spec.followup_horizon_months is None:
            return "full"
        return f"{spec.followup_horizon_months}m"
    if dim == "missingness":
        return spec.missingness.value
    if dim == "adjustment_set":
        return spec.adjustment_set
    if dim == "estimator":
        return spec.estimator.value
    raise ValueError(f"Unknown specification dimension: {dim}")


def _compute_exact_owen_shapley(
    specs: Sequence[AnalyticalSpec],
    y: NDArray[np.float64],
) -> tuple[dict[str, float], bool]:
    """Compute exact cell-means Owen (2014) Sobol-Shapley variance attribution for D=7."""
    m = len(y)
    if m <= 1:
        return {dim: 0.0 for dim in SPEC_DIMENSIONS}, True

    y_centered = y - float(np.mean(y))
    tss = float(np.dot(y_centered, y_centered))
    if not math.isfinite(tss) or tss <= 1e-15:
        return {dim: 0.0 for dim in SPEC_DIMENSIONS}, True

    d_count = len(SPEC_DIMENSIONS)
    codes = np.empty((m, d_count), dtype=np.int64)
    for d_idx, dim in enumerate(SPEC_DIMENSIONS):
        vals = [_spec_dim_value(spec, dim) for spec in specs]
        _, inv = np.unique(vals, return_inverse=True)
        codes[:, d_idx] = inv

    n_masks = 1 << d_count
    v_coalition = np.zeros(n_masks, dtype=np.float64)
    for mask in range(1, n_masks):
        cols = [d_idx for d_idx in range(d_count) if (mask >> d_idx) & 1]
        sub = codes[:, cols]
        _, inv = np.unique(sub, axis=0, return_inverse=True)
        n_groups = int(inv.max()) + 1
        counts = np.bincount(inv, minlength=n_groups)
        sums = np.bincount(inv, weights=y_centered, minlength=n_groups)
        bss = float(np.sum((sums * sums) / counts))
        v_coalition[mask] = bss / tss

    fact = [math.factorial(k) for k in range(d_count + 1)]
    d_fact = float(fact[d_count])
    shares: dict[str, float] = {}

    for d_idx, dim in enumerate(SPEC_DIMENSIONS):
        bit = 1 << d_idx
        phi = 0.0
        for mask in range(n_masks):
            if mask & bit:
                continue
            s_size = mask.bit_count()
            weight = (fact[s_size] * fact[d_count - s_size - 1]) / d_fact
            marginal = float(v_coalition[mask | bit] - v_coalition[mask])
            phi += weight * marginal
        shares[dim] = max(0.0, phi)

    total_share = sum(shares.values())
    if total_share > 0.0:
        shares = {dim: val / total_share for dim, val in shares.items()}
    return shares, False


def summarize_specification_curve(records: Sequence[RunRecord]) -> SpecCurveSummary:
    """Aggregate converged RunRecords and compute exact D=7 Sobol-Shapley shares."""
    n_total = len(records)
    converged_recs = [
        rec
        for rec in records
        if rec.converged
        and rec.log_hr is not None
        and rec.hr is not None
        and rec.ci_lower_95 is not None
        and rec.ci_upper_95 is not None
    ]
    n_converged = len(converged_recs)
    n_failed = n_total - n_converged

    if n_converged == 0:
        return SpecCurveSummary(
            n_specs_total=n_total,
            n_specs_converged=0,
            n_specs_failed=n_failed,
            median_hr=None,
            min_hr=None,
            max_hr=None,
            q25_hr=None,
            q75_hr=None,
            median_log_hr=None,
            share_ci_excluding_one=0.0,
            sign_consistency_share=0.0,
            shapley_shares={dim: 0.0 for dim in SPEC_DIMENSIONS},
            top_driver=None,
            zero_variance=True,
        )

    log_hrs = np.array([float(rec.log_hr) for rec in converged_recs if rec.log_hr is not None])
    hrs = np.array([float(rec.hr) for rec in converged_recs if rec.hr is not None])

    ci_excl = sum(
        1
        for rec in converged_recs
        if rec.ci_lower_95 is not None
        and rec.ci_upper_95 is not None
        and (rec.ci_lower_95 > 1.0 or rec.ci_upper_95 < 1.0)
    )
    n_pos = int(np.sum(log_hrs > 0.0))
    n_neg = int(np.sum(log_hrs < 0.0))
    sign_consistency = max(n_pos, n_neg) / float(n_converged)

    shapley_shares, zero_variance = _compute_exact_owen_shapley(
        [rec.spec for rec in converged_recs],
        log_hrs,
    )
    top_driver = (
        None if zero_variance else max(SPEC_DIMENSIONS, key=lambda dim: (shapley_shares[dim], dim))
    )

    q25, q50, q75 = np.quantile(hrs, [0.25, 0.50, 0.75])
    return SpecCurveSummary(
        n_specs_total=n_total,
        n_specs_converged=n_converged,
        n_specs_failed=n_failed,
        median_hr=float(q50),
        min_hr=float(np.min(hrs)),
        max_hr=float(np.max(hrs)),
        q25_hr=float(q25),
        q75_hr=float(q75),
        median_log_hr=float(np.median(log_hrs)),
        share_ci_excluding_one=float(ci_excl) / float(n_converged),
        sign_consistency_share=float(sign_consistency),
        shapley_shares=shapley_shares,
        top_driver=top_driver,
        zero_variance=zero_variance,
    )


def render_specification_curve(
    records: Sequence[RunRecord],
    summary: SpecCurveSummary,
    svg_path: Path,
    *,
    png_path: Path | None = None,
    title: str = "Specification Curve & Sobol-Shapley Decision Attribution",
) -> Path:
    """Render a deterministic 2-panel specification curve + decision strike plot."""
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "multiverse-agent"

    converged = sorted(
        [
            rec
            for rec in records
            if rec.converged
            and rec.hr is not None
            and rec.ci_lower_95 is not None
            and rec.ci_upper_95 is not None
        ],
        key=lambda rec: (float(rec.hr or 0.0), rec.spec_id),
    )

    if not converged:
        fig = Figure(figsize=(10, 4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No converged specifications to display.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        fig.savefig(svg_path, format="svg", metadata={"Creator": "multiverse-agent", "Date": None})
        if png_path is not None:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, format="png", dpi=200)
        return svg_path

    # Build strike-plot rows grouped by dimension
    row_specs: list[tuple[str, str, str]] = []
    for dim in SPEC_DIMENSIONS:
        levels = sorted({_spec_dim_value(rec.spec, dim) for rec in converged})
        for lvl in levels:
            row_specs.append((dim, lvl, f"{_DIMENSION_LABELS[dim]}: {lvl}"))

    n_rows = len(row_specs)
    fig_height = max(6.5, 3.6 + 0.24 * n_rows)
    fig = Figure(figsize=(11.5, fig_height))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.35, max(1.2, 0.14 * n_rows)],
        hspace=0.10,
    )
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    x_idx = np.arange(len(converged), dtype=np.float64)
    hrs = np.array([float(rec.hr or 1.0) for rec in converged], dtype=np.float64)
    ci_lo = np.array([float(rec.ci_lower_95 or 1.0) for rec in converged], dtype=np.float64)
    ci_hi = np.array([float(rec.ci_upper_95 or 1.0) for rec in converged], dtype=np.float64)
    sig_mask = (ci_lo > 1.0) | (ci_hi < 1.0)

    ax_top.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2, label="Null (HR = 1.0)")
    if summary.median_hr is not None:
        ax_top.axhline(
            summary.median_hr,
            color="#0f4c81",
            linestyle=":",
            linewidth=1.1,
            label=f"Median HR = {summary.median_hr:.3f}",
        )

    for mask, color, label in (
        (~sig_mask, "#9aa5b1", "95% CI includes 1.0"),
        (sig_mask, "#1b6ca8", "95% CI excludes 1.0"),
    ):
        if np.any(mask):
            yerr = np.vstack([hrs[mask] - ci_lo[mask], ci_hi[mask] - hrs[mask]])
            ax_top.errorbar(
                x_idx[mask],
                hrs[mask],
                yerr=yerr,
                fmt="o",
                markersize=3.2,
                elinewidth=0.9,
                capsize=0,
                color=color,
                ecolor=color,
                alpha=0.85,
                label=label,
            )

    ax_top.set_yscale("log")
    hr_formatter = FormatStrFormatter("%.2f")
    ax_top.yaxis.set_major_formatter(hr_formatter)
    ax_top.yaxis.set_minor_formatter(hr_formatter)
    ax_top.set_ylabel("Hazard Ratio (95% CI, log scale)", fontsize=10)
    ax_top.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)
    ax_top.legend(loc="upper left", frameon=True, fontsize=8.5)

    top_drv_txt = (
        f"{summary.top_driver} ({100.0 * summary.shapley_shares[summary.top_driver]:.1f}%)"
        if summary.top_driver is not None
        else "n/a"
    )
    info_text = (
        f"Converged: {summary.n_specs_converged}/{summary.n_specs_total}\n"
        f"Range HR: [{summary.min_hr:.3f}, {summary.max_hr:.3f}]\n"
        f"CI excl. 1.0: {100.0 * summary.share_ci_excluding_one:.1f}%\n"
        f"Top driver: {top_drv_txt}"
    )
    ax_top.text(
        0.99,
        0.04,
        info_text,
        transform=ax_top.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f9fb", "edgecolor": "#cbd5e1"},
    )

    # Bottom strike plot
    dim_band_colors = ["#f8fafc", "#ffffff"]
    current_dim = ""
    band_idx = -1
    y_positions = np.arange(n_rows, dtype=np.float64)

    for row_i, (dim, lvl, _) in enumerate(row_specs):
        if dim != current_dim:
            if row_i > 0:
                ax_bot.axhline(row_i - 0.5, color="#cbd5e1", linewidth=0.6, zorder=1)
            current_dim = dim
            band_idx += 1
        ax_bot.axhspan(
            row_i - 0.5,
            row_i + 0.5,
            facecolor=dim_band_colors[band_idx % 2],
            zorder=0,
        )
        matched_x = [
            float(i) for i, rec in enumerate(converged) if _spec_dim_value(rec.spec, dim) == lvl
        ]
        if matched_x:
            matched_arr = np.array(matched_x, dtype=np.float64)
            sig_here = np.array([bool(sig_mask[int(i)]) for i in matched_x], dtype=bool)
            if np.any(~sig_here):
                ax_bot.scatter(
                    matched_arr[~sig_here],
                    np.full(int(np.sum(~sig_here)), float(row_i)),
                    marker="|",
                    s=45,
                    linewidths=1.4,
                    color="#7b8794",
                    zorder=2,
                )
            if np.any(sig_here):
                ax_bot.scatter(
                    matched_arr[sig_here],
                    np.full(int(np.sum(sig_here)), float(row_i)),
                    marker="|",
                    s=45,
                    linewidths=1.4,
                    color="#1b6ca8",
                    zorder=3,
                )

    # Annotate Shapley % share on the right edge per dimension group
    dim_rows: dict[str, list[int]] = {dim: [] for dim in SPEC_DIMENSIONS}
    for row_i, (dim, _, _) in enumerate(row_specs):
        dim_rows[dim].append(row_i)
    ax_right = ax_bot.twinx()
    ax_right.set_ylim(n_rows - 0.5, -0.5)
    right_ticks: list[float] = []
    right_labels: list[str] = []
    for dim in SPEC_DIMENSIONS:
        rows_for_dim = dim_rows[dim]
        if rows_for_dim:
            right_ticks.append(float(sum(rows_for_dim)) / float(len(rows_for_dim)))
            share_pct = 100.0 * summary.shapley_shares[dim]
            right_labels.append(f"{share_pct:.1f}%")
    ax_right.set_yticks(right_ticks)
    ax_right.set_yticklabels(right_labels, fontsize=8.5, fontweight="bold", color="#1f2933")
    ax_right.set_ylabel("Sobol-Shapley Share", fontsize=9.5, labelpad=8)
    ax_right.tick_params(length=0)

    ax_bot.set_yticks(y_positions)
    ax_bot.set_yticklabels([label for _, _, label in row_specs], fontsize=8.2)
    ax_bot.set_ylim(n_rows - 0.5, -0.5)
    ax_bot.set_xlim(-1.0, float(len(converged)))
    ax_bot.set_xlabel("Specification Rank (sorted ascending by Hazard Ratio)", fontsize=10)
    ax_bot.spines["top"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.22, right=0.89, top=0.92, bottom=0.08)
    fig.savefig(svg_path, format="svg", metadata={"Creator": "multiverse-agent", "Date": None})
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, format="png", dpi=200)
    return svg_path


__all__ = [
    "SPEC_DIMENSIONS",
    "SpecCurveSummary",
    "render_specification_curve",
    "summarize_specification_curve",
]
