"""Mechanical field-bound citation verifier for grounded multiverse reports."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from multiverse_agent.curve import SPEC_DIMENSIONS, SpecCurveSummary
from multiverse_agent.runner import RunRecord

_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")
_ATX_HEADER_RE = re.compile(r"(?m)^\s*#{1,6}\s+.*$")
_BACKTICK_SPAN_RE = re.compile(r"`[^`\n]+`")
_MARKDOWN_URL_RE = re.compile(r"\]\([^)\n]+\)")
_ALLOWLIST_RE = re.compile(
    r"\b95%\s*(?:CI|confidence\s+interval)\b"
    r"|\b1[- ]SD\b"
    r"|\bSHA-256\b"
    r"|(?:alpha|α)\s*=\s*0\.05\b",
    re.IGNORECASE,
)

_TAG_RE = re.compile(r"\[(?P<kind>run|curve):(?P<body>[^\]\s]+)\]")
_RUN_BODY_RE = re.compile(r"^(?P<prefix>[0-9a-fA-F]{8,}):(?P<field>[a-zA-Z0-9_]+)$")
_CURVE_BODY_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_.]+)$")
_PRECEDING_LITERAL_RE = re.compile(
    r"(?:(?P<lt><\s*)|(?<![a-zA-Z0-9_.]))"
    r"(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?P<pct>%)?\s*$"
)
_UNGROUNDED_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+(?:[eE][-+]?\d+)?"
    r"|[-+]?\d+(?:\.\d+)?[eE][-+]?\d+"
    r"|[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%"
)
_TOP_DRIVER_CLAIM_RE = re.compile(
    r"(?i)\b(?:top|largest|primary|dominant)\s+(?:variance\s+)?driver\b[^\n.]{0,120}"
)

_RUN_INT_FIELDS = frozenset({"n_subjects", "n_events"})
_RUN_FLOAT_FIELDS = frozenset(
    {
        "person_years",
        "log_hr",
        "se_log_hr",
        "hr",
        "ci_lower_95",
        "ci_upper_95",
        "concordance_index",
    }
)
_CURVE_INT_KEYS = frozenset({"n_specs_total", "n_specs_converged", "n_specs_failed"})
_CURVE_FLOAT_KEYS = frozenset(
    {"median_hr", "min_hr", "max_hr", "q25_hr", "q75_hr", "median_log_hr"}
)
_CURVE_SHARE_KEYS = frozenset({"share_ci_excluding_one", "sign_consistency_share"})

_MANDATORY_CURVE_KEYS: tuple[str, ...] = (
    "n_specs_converged",
    "median_hr",
    "min_hr",
    "max_hr",
    "share_ci_excluding_one",
    "sign_consistency_share",
)


class GateViolation(BaseModel):
    """Single mechanical citation gate failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    span: str | None = None


class AuditResult(BaseModel):
    """Complete verification outcome from `verify_grounded_report`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    violations: tuple[GateViolation, ...]
    n_run_citations: int
    n_curve_citations: int


def _parse_numeric_literal(raw_num: str) -> float:
    return float(raw_num.replace(",", ""))


def _verify_integer_match(
    *,
    claimed_num: str,
    pct: str | None,
    lt: str | None,
    logged_val: int,
    target_label: str,
) -> GateViolation | None:
    if pct is not None or lt is not None or "." in claimed_num or "e" in claimed_num.lower():
        return GateViolation(
            code="value_mismatch",
            message=f"Expected exact integer for {target_label}, got '{claimed_num}'.",
            span=claimed_num,
        )
    claimed_int = int(claimed_num.replace(",", ""))
    if claimed_int != logged_val:
        return GateViolation(
            code="value_mismatch",
            message=(
                f"Integer mismatch for {target_label}: claimed {claimed_int}, logged {logged_val}."
            ),
            span=claimed_num,
        )
    return None


def _verify_p_value(
    *,
    claimed_num: str,
    pct: str | None,
    lt: str | None,
    logged_p: float,
    run_prefix: str,
) -> GateViolation | None:
    if pct is not None:
        return GateViolation(
            code="value_mismatch",
            message=f"p_value for run '{run_prefix}' must not carry a '%' suffix.",
            span=f"{claimed_num}{pct}",
        )
    claimed = _parse_numeric_literal(claimed_num)
    if lt is not None:
        if not math.isclose(claimed, 1e-4, rel_tol=0.0, abs_tol=1e-12):
            return GateViolation(
                code="value_mismatch",
                message=(
                    f"Only '< 0.0001' threshold form is allowed for p_value, got '< {claimed_num}'."
                ),
                span=f"< {claimed_num}",
            )
        if not (logged_p < 1e-4):
            return GateViolation(
                code="value_mismatch",
                message=(
                    f"Threshold '< 0.0001' invalid for run '{run_prefix}': "
                    f"logged p_value = {logged_p:.6g} >= 1e-4."
                ),
                span=f"< {claimed_num}",
            )
        return None

    if (claimed < 0.05) != (logged_p < 0.05):
        return GateViolation(
            code="significance_flip",
            message=(
                f"Alpha=0.05 significance boundary flipped for run '{run_prefix}': "
                f"claimed p={claimed}, logged p={logged_p:.6g}."
            ),
            span=claimed_num,
        )
    rel_err = abs(claimed - logged_p) / max(abs(logged_p), 1e-12)
    if rel_err > 2e-2:
        return GateViolation(
            code="value_mismatch",
            message=(
                f"p_value mismatch for run '{run_prefix}': "
                f"claimed {claimed}, logged {logged_p:.6g} (rel_err={rel_err:.4f} > 0.02)."
            ),
            span=claimed_num,
        )
    return None


def _verify_share(
    *,
    claimed_num: str,
    pct: str | None,
    lt: str | None,
    logged_share: float,
    key: str,
) -> GateViolation | None:
    if lt is not None:
        return GateViolation(
            code="value_mismatch",
            message=f"Threshold '<' not valid for proportion '{key}'.",
            span=f"{lt}{claimed_num}",
        )
    claimed = _parse_numeric_literal(claimed_num)
    claimed_prop = claimed / 100.0 if pct == "%" else claimed
    if not (0.0 <= claimed_prop <= 1.0 + 1e-9):
        return GateViolation(
            code="value_mismatch",
            message=f"Proportion '{key}' out of [0, 1] range: {claimed_prop}.",
            span=f"{claimed_num}{pct or ''}",
        )
    if abs(claimed_prop - logged_share) > 2e-3:
        return GateViolation(
            code="value_mismatch",
            message=(
                f"Share mismatch for '{key}': claimed {claimed_prop:.4f}, "
                f"logged {logged_share:.4f} (abs_diff > 2e-3)."
            ),
            span=f"{claimed_num}{pct or ''}",
        )
    return None


def verify_grounded_report(
    markdown_text: str,
    run_records: Sequence[RunRecord],
    curve_summary: SpecCurveSummary,
) -> AuditResult:
    """Mechanically verify every citation, numeric literal, and mandatory disclosure."""
    text_no_code = _FENCED_CODE_RE.sub(lambda m: " " * len(m.group(0)), markdown_text)
    violations: list[GateViolation] = []
    matched_spans: list[tuple[int, int]] = []
    cited_curve_keys: set[str] = set()
    cited_shapley_dims: set[str] = set()
    n_run_citations = 0
    n_curve_citations = 0

    for tag_match in _TAG_RE.finditer(text_no_code):
        kind = tag_match.group("kind")
        body = tag_match.group("body")
        prefix_slice = text_no_code[: tag_match.start()]
        lit_match = _PRECEDING_LITERAL_RE.search(prefix_slice)
        if lit_match is None:
            violations.append(
                GateViolation(
                    code="unbound_citation_tag",
                    message=(
                        f"Citation tag '{tag_match.group(0)}' lacks an immediately "
                        "preceding numeric literal."
                    ),
                    span=tag_match.group(0),
                )
            )
            continue

        lt = lit_match.group("lt")
        claimed_num = lit_match.group("num")
        pct = lit_match.group("pct")
        lit_start = lit_match.start("lt") if lt is not None else lit_match.start("num")
        matched_spans.append((lit_start, tag_match.end()))

        if kind == "run":
            n_run_citations += 1
            parsed_run = _RUN_BODY_RE.match(body)
            if parsed_run is None:
                violations.append(
                    GateViolation(
                        code="unknown_field_or_key",
                        message=f"Malformed run citation tag '{tag_match.group(0)}'.",
                        span=tag_match.group(0),
                    )
                )
                continue
            prefix = parsed_run.group("prefix").lower()
            field = parsed_run.group("field")

            matches = [rec for rec in run_records if rec.run_id.lower().startswith(prefix)]
            if not matches:
                violations.append(
                    GateViolation(
                        code="unknown_run_id",
                        message=f"Run prefix '{prefix}' does not match any logged RunRecord.",
                        span=tag_match.group(0),
                    )
                )
                continue
            if len(matches) > 1:
                violations.append(
                    GateViolation(
                        code="ambiguous_run_id",
                        message=f"Run prefix '{prefix}' matches {len(matches)} RunRecords.",
                        span=tag_match.group(0),
                    )
                )
                continue

            rec = matches[0]
            if not rec.converged:
                violations.append(
                    GateViolation(
                        code="non_converged_run_cited",
                        message=(
                            f"Run '{prefix}' has converged=False and cannot be cited "
                            "as an effect estimate."
                        ),
                        span=tag_match.group(0),
                    )
                )
                continue

            if field in _RUN_INT_FIELDS:
                logged_int = int(getattr(rec, field))
                err = _verify_integer_match(
                    claimed_num=claimed_num,
                    pct=pct,
                    lt=lt,
                    logged_val=logged_int,
                    target_label=f"run:{prefix}:{field}",
                )
                if err is not None:
                    violations.append(err)
            elif field == "p_value":
                if rec.p_value is None:
                    violations.append(
                        GateViolation(
                            code="value_mismatch",
                            message=f"Run '{prefix}' has null p_value.",
                            span=tag_match.group(0),
                        )
                    )
                else:
                    err = _verify_p_value(
                        claimed_num=claimed_num,
                        pct=pct,
                        lt=lt,
                        logged_p=float(rec.p_value),
                        run_prefix=prefix,
                    )
                    if err is not None:
                        violations.append(err)
            elif field in _RUN_FLOAT_FIELDS:
                raw_logged = getattr(rec, field)
                if raw_logged is None or pct is not None or lt is not None:
                    violations.append(
                        GateViolation(
                            code="value_mismatch",
                            message=f"Invalid or null float claim for run:{prefix}:{field}.",
                            span=tag_match.group(0),
                        )
                    )
                else:
                    claimed_f = _parse_numeric_literal(claimed_num)
                    logged_f = float(raw_logged)
                    if not math.isclose(claimed_f, logged_f, rel_tol=5e-3, abs_tol=5e-3):
                        violations.append(
                            GateViolation(
                                code="value_mismatch",
                                message=(
                                    f"Value mismatch for run:{prefix}:{field}: "
                                    f"claimed {claimed_f}, logged {logged_f:.6g}."
                                ),
                                span=claimed_num,
                            )
                        )
            else:
                violations.append(
                    GateViolation(
                        code="unknown_field_or_key",
                        message=f"Unknown RunRecord field '{field}' in '{tag_match.group(0)}'.",
                        span=tag_match.group(0),
                    )
                )
        else:
            n_curve_citations += 1
            parsed_curve = _CURVE_BODY_RE.match(body)
            if parsed_curve is None:
                violations.append(
                    GateViolation(
                        code="unknown_field_or_key",
                        message=f"Malformed curve citation tag '{tag_match.group(0)}'.",
                        span=tag_match.group(0),
                    )
                )
                continue
            key = parsed_curve.group("key")
            cited_curve_keys.add(key)

            if key in _CURVE_INT_KEYS:
                logged_int = int(getattr(curve_summary, key))
                err = _verify_integer_match(
                    claimed_num=claimed_num,
                    pct=pct,
                    lt=lt,
                    logged_val=logged_int,
                    target_label=f"curve:{key}",
                )
                if err is not None:
                    violations.append(err)
            elif key in _CURVE_FLOAT_KEYS:
                raw_logged = getattr(curve_summary, key)
                if raw_logged is None or pct is not None or lt is not None:
                    violations.append(
                        GateViolation(
                            code="value_mismatch",
                            message=f"Invalid or null float claim for curve:{key}.",
                            span=tag_match.group(0),
                        )
                    )
                else:
                    claimed_f = _parse_numeric_literal(claimed_num)
                    logged_f = float(raw_logged)
                    if not math.isclose(claimed_f, logged_f, rel_tol=5e-3, abs_tol=5e-3):
                        violations.append(
                            GateViolation(
                                code="value_mismatch",
                                message=(
                                    f"Value mismatch for curve:{key}: "
                                    f"claimed {claimed_f}, logged {logged_f:.6g}."
                                ),
                                span=claimed_num,
                            )
                        )
            elif key in _CURVE_SHARE_KEYS:
                err = _verify_share(
                    claimed_num=claimed_num,
                    pct=pct,
                    lt=lt,
                    logged_share=float(getattr(curve_summary, key)),
                    key=key,
                )
                if err is not None:
                    violations.append(err)
            elif key.startswith("shapley."):
                dim = key.split(".", 1)[1]
                if dim not in curve_summary.shapley_shares:
                    violations.append(
                        GateViolation(
                            code="unknown_field_or_key",
                            message=f"Unknown Shapley dimension '{dim}' in '{tag_match.group(0)}'.",
                            span=tag_match.group(0),
                        )
                    )
                else:
                    cited_shapley_dims.add(dim)
                    err = _verify_share(
                        claimed_num=claimed_num,
                        pct=pct,
                        lt=lt,
                        logged_share=float(curve_summary.shapley_shares[dim]),
                        key=key,
                    )
                    if err is not None:
                        violations.append(err)
            else:
                violations.append(
                    GateViolation(
                        code="unknown_field_or_key",
                        message=f"Unknown SpecCurveSummary key '{key}' in '{tag_match.group(0)}'.",
                        span=tag_match.group(0),
                    )
                )

    # Check 4: Zero uncited empirical numbers
    chars = list(text_no_code)
    for start, end in matched_spans:
        for idx in range(start, end):
            chars[idx] = " "
    residual = "".join(chars)
    residual = _ATX_HEADER_RE.sub("", residual)
    residual = _BACKTICK_SPAN_RE.sub(" ", residual)
    residual = _MARKDOWN_URL_RE.sub("]()", residual)
    residual = _ALLOWLIST_RE.sub(" ", residual)

    for num_match in _UNGROUNDED_NUMBER_RE.finditer(residual):
        violations.append(
            GateViolation(
                code="ungrounded_number",
                message=f"Uncited empirical numeric token '{num_match.group(0)}' in report prose.",
                span=num_match.group(0),
            )
        )

    # Check 5: Anti-cherry-picking completeness & top-driver truthfulness
    for req_key in _MANDATORY_CURVE_KEYS:
        if req_key not in cited_curve_keys:
            violations.append(
                GateViolation(
                    code="missing_mandatory_summary_citation",
                    message=f"Missing mandatory curve summary citation '[curve:{req_key}]'.",
                    span=req_key,
                )
            )

    if curve_summary.top_driver is not None:
        top_key = f"shapley.{curve_summary.top_driver}"
        if top_key not in cited_curve_keys:
            violations.append(
                GateViolation(
                    code="missing_mandatory_summary_citation",
                    message=(f"Missing mandatory top Shapley driver citation '[curve:{top_key}]'."),
                    span=top_key,
                )
            )
            if cited_shapley_dims - {curve_summary.top_driver}:
                violations.append(
                    GateViolation(
                        code="wrong_top_driver",
                        message=(
                            f"Cited secondary Shapley driver(s) {sorted(cited_shapley_dims)} "
                            f"while omitting true top driver '{curve_summary.top_driver}'."
                        ),
                        span=",".join(sorted(cited_shapley_dims)),
                    )
                )
        for claim_match in _TOP_DRIVER_CLAIM_RE.finditer(text_no_code):
            snippet = claim_match.group(0)
            mentioned = [dim for dim in SPEC_DIMENSIONS if dim in snippet]
            if mentioned and curve_summary.top_driver not in mentioned:
                violations.append(
                    GateViolation(
                        code="wrong_top_driver",
                        message=(
                            f"Prose claims {mentioned} as top driver, "
                            f"but true argmax is '{curve_summary.top_driver}'."
                        ),
                        span=snippet.strip(),
                    )
                )

    return AuditResult(
        passed=len(violations) == 0,
        violations=tuple(violations),
        n_run_citations=n_run_citations,
        n_curve_citations=n_curve_citations,
    )


__all__ = ["AuditResult", "GateViolation", "verify_grounded_report"]
