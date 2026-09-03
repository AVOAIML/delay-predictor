# ---------------------------------------------------------------------------
# ONE image family (plan §3 "Dockerfile.base/train/serve" collapsed into multi-
# stage), built from a SINGLE pinned env (pyproject.toml + uv.lock). The exact
# same image is what gets pushed to ACR in Phase 2 — train and serve share the
# base so the AML Environment digest == ACR image digest (parity CI guard §12).
# ---------------------------------------------------------------------------

# ---- web: build the Configurator SPA -------------------------------------
# VITE_API_BASE is EMPTY on purpose. Vite inlines it at build time, so a value
# here would pin the bundle to one API hostname; empty makes every call in
# api.ts relative ("/api/..."), which resolves against whatever origin serves
# the page. One image, any hostname, and no CORS in the path.
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
# `npm ci` not `npm install`: the lock file is the point of having one, and it
# fails loudly on drift instead of silently resolving something else on CI.
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ENV VITE_API_BASE=""
RUN npm run build

# ---- base: pinned env, no source -----------------------------------------
FROM python:3.10-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
# uv for fast, reproducible installs from the single lock file. Installed from
# PyPI (not `COPY --from=ghcr.io/astral-sh/uv`), because that ghcr.io pull can
# TLS-timeout on some networks; same version + --frozen sync => identical env.
RUN pip install "uv==0.11.19"
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# Resolve the locked env ONCE; baked into every downstream stage.
RUN uv venv /opt/venv && uv sync --frozen --no-install-project --extra serve --extra ui --extra aml

# ---- src: add the monorepo source + install the project ------------------
FROM base AS src
COPY libs ./libs
COPY modules ./modules
COPY config ./config
COPY db ./db
COPY docker ./docker
COPY services ./services
COPY README.md dataset* ./dataset/
RUN uv pip install --no-deps -e .

# ---- train: default target for AML training jobs / `make train` ----------
# ML logic is plain Python functions + CLI; the future AML wrapper calls these
# unchanged (plan: "Keep ML logic in plain Python functions + Makefile").
FROM src AS train
ENV APP_ENV=local
ENTRYPOINT ["maxxflow"]
CMD ["--help"]

# ---- serve: BYOC multi-model router under azmlinfsrv ----------------------
# Identical server + routes + request envelope locally and on the AML endpoint.
FROM src AS serve
ENV APP_ENV=local \
    AZUREML_MODEL_DIR=/var/azureml-app \
    AZUREML_ENTRY_SCRIPT=score.py \
    AML_APP_ROOT=/var/azureml-app
# AML convention: the entry script lives in AZUREML_MODEL_DIR.
RUN mkdir -p /var/azureml-app && cp docker/score.py /var/azureml-app/score.py
WORKDIR /var/azureml-app
EXPOSE 5001
# azureml-inference-server-http == azmlinfsrv (installed via the `serve` extra).
ENTRYPOINT ["azmlinfsrv", "--entry_script", "/var/azureml-app/score.py", "--port", "5001"]

# ---- ui: Configurator API (FastAPI) + dashboard (Streamlit) ---------------
# Same image; compose sets the command (uvicorn / streamlit) per service.
FROM src AS ui
ENV APP_ENV=local
WORKDIR /app
EXPOSE 8000 8501
CMD ["python", "-c", "print('set command in compose: uvicorn services.configurator.app:app  OR  streamlit run services/dashboard/app.py')"]

# ---- api: Configurator API with the command baked in ----------------------
# Azure Container Apps mangles --command/--args string splitting, so the
# entrypoint lives in the image. Same stage contents as `ui`.
FROM src AS api
ENV APP_ENV=azure
WORKDIR /app
# The SPA ships INSIDE the API image and is served by FastAPI at / (see the
# bottom of services/configurator/app.py). Same origin as the API it calls, so
# there is one URL to publish and one to lock down.
COPY --from=web /web/dist ./frontend/dist
EXPOSE 8000
CMD ["uvicorn", "services.configurator.app:app", "--host", "0.0.0.0", "--port", "8000"]
