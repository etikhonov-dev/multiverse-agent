FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data
COPY fixtures ./fixtures

RUN uv sync --frozen --no-dev --no-editable \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["uv", "run", "--no-sync", "multiverse-agent"]
CMD ["run", "--cohort", "nhanes", "--backend", "replay", "--out", "/tmp/nhanes-artifacts"]
