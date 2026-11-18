# Constraints

Last reviewed: 2026-11-18 by @etikhonov

## Floor (always enforced, no setup required)

- No new suppression comments: `# noqa`, `# type: ignore`, `pyright: ignore`
- No unimplemented stubs: `raise NotImplementedError`, `TODO`, `pass # stub`, silent `except ...: pass`
- No skipped or deleted tests without a reason in the commit message
- No secrets in source
- This file does not get weakened to make a change pass

## Enforced with numbers

| Dimension | Rule | Checked by | Runs at |
|-----------|------|-----------|---------|
| Types | Zero type errors (`0` errors) | `uv run mypy --strict src` | task end, CI |
| Lint & Static Security | Zero ruff or bandit errors (`0` errors) | `uv run ruff check src tests scripts` | every edit, CI |
| Format | Zero formatting diffs (`0` diffs) | `uv run ruff format --check src tests scripts` | every edit, CI |
| Coverage | Line coverage >= 90% on `src/multiverse_agent` | `uv run pytest --cov=multiverse_agent --cov-fail-under=90` | task end, CI |
| Security: deps | Zero known vulnerabilities (`0` vulnerabilities) | `uv run pip-audit` | task end, CI |
| Floor guard | Zero floor violations (`0` violations) | `python3 scripts/floor_guard.py --base main` | task end, CI |
| Performance: cohorts | Tier 1 and Tier 2 cohort load <= 200 ms | `uv run pytest tests/test_cohort.py` | task end, CI |

## Measured, not yet enforced

| Metric | Today | Direction |
|--------|-------|-----------|
| Package line coverage | 100.0% | must not fall below 90% |

## Exceptions

| ID | Rule | Path | Reason | Owner | Expires |
|----|------|------|--------|-------|---------|
