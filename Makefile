# MaXXFlow EPIC 10 — local inner loop (plan §11). Each target == one AML job.
# Flip APP_ENV=dev-azure (Phase 2) and the SAME targets run against Azure.

TENANT ?= demo
MODULE ?= all
COMPOSE = docker compose -f compose.local.yml
UV = uv run

.PHONY: help start bootstrap up down ps logs seed validate fe train serve score drift test lock fmt clean ui train-csv

help:
	@echo "Targets: bootstrap seed validate fe train serve score drift test  (TENANT=$(TENANT) MODULE=$(MODULE))"

## one-time: sync env, bring up postgres+minio+mlflow, provision tenant schema
bootstrap: lock
	uv sync --extra dev --extra serve
	$(COMPOSE) up -d --wait postgres minio minio-init mlflow
	$(UV) maxxflow db-provision --tenant $(TENANT)
	@echo "bootstrap complete — stack is up, tenant schema provisioned"

lock:
	@test -f uv.lock || uv lock

up:
	$(COMPOSE) up -d --wait
down:
	$(COMPOSE) down
ps:
	$(COMPOSE) ps
logs:
	$(COMPOSE) logs -f --tail=100

## ONE command: build + start the ENTIRE stack (db, minio, mlflow, model-server, API, dashboard, UI)
start:
	$(COMPOSE) up -d --build --wait
	@echo ""
	@echo "MaXXFlow stack is up:"
	@echo "  React UI    -> http://localhost:5173"
	@echo "  API (docs)  -> http://localhost:8000/docs"
	@echo "  Dashboard   -> http://localhost:8501"
	@echo "  MLflow      -> http://localhost:8085"
	@echo "  Model server-> http://localhost:5001"
	@echo "  MinIO console-> http://localhost:9001"

## synthetic ground-truth into tenant Postgres  == ADF ingest job
seed:
	$(UV) maxxflow seed --tenant $(TENANT) --module $(MODULE)

## 3 CI validation gates (schema / realism / leakage+learnability)
validate:
	$(UV) maxxflow validate --tenant $(TENANT) --module $(MODULE)

## bronze->silver->gold features on minio  == AML feature step
fe:
	$(UV) maxxflow fe --tenant $(TENANT) --module $(MODULE)

## train + isotonic calibrate + register to local MLflow  == AML training job
train:
	$(UV) maxxflow train --tenant $(TENANT) --module $(MODULE)

## BYOC scoring server under azmlinfsrv (same image as ACR/endpoint) == AML endpoint
serve:
	$(COMPOSE) up -d --wait model-server
	@echo "model-server (azmlinfsrv) on http://localhost:5001  (M1/M3/M4; M2 is batch)"

## Configurator API (:8000) + Streamlit dashboard (:8501) + React UI (:5173)
ui:
	$(COMPOSE) up -d --build configurator-api dashboard frontend
	@echo "React UI -> http://localhost:5173   ·   API -> http://localhost:8000/docs   ·   Dashboard -> http://localhost:8501"

## train the two M1 CSV models (win + price) from dataset/ for a tenant (TENANT=demo)
train-csv:
	$(UV) python services/train_csv.py --tenant $(TENANT)

## write advisory scores back to Postgres customElements  == AML batch/endpoint score
score:
	$(UV) maxxflow score --tenant $(TENANT) --module $(MODULE)

## Evidently drift report  == AML model monitor
drift:
	$(UV) maxxflow drift --tenant $(TENANT) --module $(MODULE)

## unit + contract + data-quality + smoke-train + parity
test:
	$(UV) --extra dev pytest -q

clean:
	$(COMPOSE) down -v || true
	rm -rf lake reports mlflow.db mlruns artifacts
