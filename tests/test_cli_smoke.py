"""End-to-end CLI smoke tests for `multiverse-agent run` and `multiverse-agent verify`."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from multiverse_agent.cli import main as cli_main


def test_cli_nhanes_end_to_end_and_tamper_detection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "nhanes_artifacts"

    t0 = time.perf_counter()
    exit_code = cli_main(
        [
            "run",
            "--cohort",
            "nhanes",
            "--backend",
            "replay",
            "--out",
            str(out_dir),
        ]
    )
    elapsed = time.perf_counter() - t0
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    assert "108/108 converged" in captured.out
    assert elapsed < 12.0

    expected_files = (
        "runs.jsonl",
        "curve_summary.json",
        "spec_curve.svg",
        "report.md",
        "audit.json",
    )
    for fname in expected_files:
        assert (out_dir / fname).exists(), f"Missing artifact {fname}"

    audit = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["violations"] == []

    # Clean verify passes with exit code 0
    verify_ok = cli_main(
        [
            "verify",
            "--runs",
            str(out_dir / "runs.jsonl"),
            "--summary",
            str(out_dir / "curve_summary.json"),
            "--report",
            str(out_dir / "report.md"),
        ]
    )
    ok_captured = capsys.readouterr()
    assert verify_ok == 0, ok_captured.err
    assert "verify: PASSED" in ok_captured.out

    # Mutate a single digit in report.md -> verify must exit with code 2 and print GateViolation
    report_path = out_dir / "report.md"
    orig_text = report_path.read_text(encoding="utf-8")
    mutated_text = orig_text.replace(
        "108 [curve:n_specs_converged]", "107 [curve:n_specs_converged]"
    )
    assert mutated_text != orig_text
    report_path.write_text(mutated_text, encoding="utf-8")

    audit_out = out_dir / "verify_tampered_audit.json"
    verify_tampered = cli_main(
        [
            "verify",
            "--runs",
            str(out_dir / "runs.jsonl"),
            "--report",
            str(report_path),
            "--audit-out",
            str(audit_out),
        ]
    )
    tampered_captured = capsys.readouterr()
    assert verify_tampered == 2
    assert "value_mismatch" in tampered_captured.err
    assert json.loads(audit_out.read_text(encoding="utf-8"))["passed"] is False


def test_cli_synthetic_replay_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "synth_artifacts"
    exit_code = cli_main(
        [
            "run",
            "--cohort",
            "synthetic",
            "--backend",
            "replay",
            "--out",
            str(out_dir),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    audit = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
