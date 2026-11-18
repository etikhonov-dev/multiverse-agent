"""Command-line interface for running and verifying outcome-blind multiverse analyses."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from multiverse_agent.agent import OutcomeBlindArchitect, build_backend
from multiverse_agent.cohort import (
    CohortBundle,
    OutcomeBlindSchema,
    load_nhanes_mortality_cohort,
    make_synthetic_bias_cohort,
)
from multiverse_agent.curve import (
    SpecCurveSummary,
    render_specification_curve,
    summarize_specification_curve,
)
from multiverse_agent.eval_gate import verify_grounded_report
from multiverse_agent.interpreter import GroundedSynthesizer
from multiverse_agent.runner import MultiverseRunner, RunRecord

DEFAULT_HYPOTHESIS = (
    "Association between baseline standardized log C-reactive protein (1-SD) "
    "and all-cause mortality across landmark washouts, cohort definitions, "
    "missingness strategies, adjustment sets, and proportional-hazards estimators."
)


def _load_cohort(cohort_name: str, *, seed: int) -> CohortBundle:
    key = cohort_name.strip().lower()
    if key == "nhanes":
        return load_nhanes_mortality_cohort()
    if key == "synthetic":
        return make_synthetic_bias_cohort(n=4_000, seed=seed)
    raise ValueError(f"Unknown cohort '{cohort_name}'. Expected 'nhanes' or 'synthetic'.")


def _load_runs_jsonl(runs_path: Path) -> tuple[RunRecord, ...]:
    lines = [line for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(RunRecord.model_validate_json(line) for line in lines)


def _cmd_run(args: argparse.Namespace) -> int:
    cohort = _load_cohort(args.cohort, seed=args.seed)
    blind_schema = OutcomeBlindSchema.from_cohort(cohort)
    backend = build_backend(
        args.backend,
        model=args.model,
        base_url=args.base_url,
    )

    architect = OutcomeBlindArchitect(backend)
    plan = architect.plan(blind_schema, args.hypothesis)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    summary_path = out_dir / "curve_summary.json"
    svg_path = out_dir / "spec_curve.svg"
    png_path = out_dir / "spec_curve.png" if args.png else None
    report_path = out_dir / "report.md"
    audit_path = out_dir / "audit.json"

    runner = MultiverseRunner(
        cohort,
        plan.space,
        seed=args.seed,
        declared_roles=plan.declared_roles,
    )
    records = runner.run_all(jsonl_path=runs_path)

    summary = summarize_specification_curve(records)
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    render_specification_curve(
        records,
        summary,
        svg_path,
        png_path=png_path,
        title=f"Specification Curve: {cohort.name}",
    )

    synthesizer = GroundedSynthesizer(backend)
    report_md, audit = synthesizer.synthesize(
        records,
        summary,
        cohort_name=cohort.name,
        hypothesis=args.hypothesis,
    )
    report_path.write_text(report_md, encoding="utf-8")

    audit_payload = {
        "cohort_name": cohort.name,
        "cohort_sha256": cohort.sha256,
        "backend": backend.name,
        "passed": audit.passed,
        "n_run_citations": audit.n_run_citations,
        "n_curve_citations": audit.n_curve_citations,
        "blocked_covariates": [ev.model_dump(mode="json") for ev in plan.blocked_covariates],
        "violations": [v.model_dump(mode="json") for v in audit.violations],
    }
    audit_path.write_text(
        json.dumps(audit_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    if not audit.passed:
        for violation in audit.violations:
            print(f"[{violation.code}] {violation.message}", file=sys.stderr)
        return 2

    print(
        f"run: {summary.n_specs_converged}/{summary.n_specs_total} converged, "
        f"median HR={summary.median_hr:.3f}, top_driver={summary.top_driver}, "
        f"citations=({audit.n_run_citations} run, {audit.n_curve_citations} curve)"
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    records = _load_runs_jsonl(Path(args.runs))
    if args.summary is not None:
        summary = SpecCurveSummary.model_validate_json(
            Path(args.summary).read_text(encoding="utf-8")
        )
    else:
        summary = summarize_specification_curve(records)

    report_text = Path(args.report).read_text(encoding="utf-8")
    audit = verify_grounded_report(report_text, records, summary)

    if args.audit_out is not None:
        Path(args.audit_out).write_text(
            json.dumps(audit.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    if not audit.passed:
        for violation in audit.violations:
            print(f"[{violation.code}] {violation.message}", file=sys.stderr)
        return 2

    print(
        f"verify: PASSED ({audit.n_run_citations} run citations, "
        f"{audit.n_curve_citations} curve citations)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser for `multiverse-agent`."""
    parser = argparse.ArgumentParser(
        prog="multiverse-agent",
        description="Outcome-blind epidemiological multiverse survival runner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Plan, execute, plot, synthesize, and audit a multiverse.")
    run_p.add_argument("--cohort", choices=("nhanes", "synthetic"), default="nhanes")
    run_p.add_argument(
        "--backend",
        choices=("replay", "ollama", "openai", "anthropic"),
        default="replay",
    )
    run_p.add_argument("--model", default=None)
    run_p.add_argument("--base-url", default=None)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--hypothesis", default=DEFAULT_HYPOTHESIS)
    run_p.add_argument("--out", default="artifacts/run")
    run_p.add_argument("--png", action="store_true", help="Also write spec_curve.png.")
    run_p.set_defaults(func=_cmd_run)

    ver_p = sub.add_parser("verify", help="Verify a markdown report against sealed RunRecords.")
    ver_p.add_argument("--runs", required=True, help="Path to runs.jsonl.")
    ver_p.add_argument("--summary", default=None, help="Optional path to curve_summary.json.")
    ver_p.add_argument("--report", required=True, help="Path to report.md.")
    ver_p.add_argument("--audit-out", default=None, help="Optional output path for audit.json.")
    ver_p.set_defaults(func=_cmd_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = args.func
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
