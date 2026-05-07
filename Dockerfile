# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install uv and resolve/install all dependencies into a venv.
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml .
# uv needs a README to exist for hatchling metadata
RUN touch README.md

# Install all production dependencies into /app/.venv
# --no-install-project: skip installing the project itself (done below after COPY src)
RUN uv sync --no-install-project --no-dev

# Copy application source
COPY src/ src/

# Install the project itself (editable install into the venv)
RUN uv sync --no-dev


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# git is required by GitPython for repo cloning
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the fully resolved venv from the builder — no pip in runtime image
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Make the venv the active Python
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default: run the API server.
# Override CMD in docker-compose for the worker service.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]