# Agent Guide: `multiverse-agent`

Read `CONSTRAINTS.md` before writing code. Do not weaken it to make a change pass.

## Repository Rules

- Follow test-driven development (`RED -> GREEN -> REFACTOR`): write failing tests before implementation.
- Never impute primary exposure (`log_crp_z` / `crp_mg_l`).
- Never include `DURATION`, `EVENT`, `OUTCOME_FOLLOWUP`, or `UCOD_LEADING` in `OutcomeBlindSchema`.
- Never allow `MEDIATOR`, `COLLIDER`, `COMORBIDITY_FLAG`, `POST_BASELINE_FOLLOWUP`, or `HEALTHCARE_UTILIZATION_SELECTION` variables in `adjustment_sets`.
- Keep all changes on short-lived feature branches and merge cleanly to `main` after all quality gates pass.

## Standard Verification

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy --strict src
uv run pytest --cov=multiverse_agent --cov-fail-under=90
uv run pip-audit
python3 scripts/floor_guard.py --base main
```
