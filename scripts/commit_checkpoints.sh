#!/usr/bin/env bash
# Materialise the build as four small, reviewable checkpoint commits (plan §10).
# The build session could not write to git (the host held .git/index.lock), so run
# this from a clean checkout to create the intended history. Safe to adapt.
set -euo pipefail
cd "$(dirname "$0")/.."

git add pyproject.toml uv.lock .gitignore .env.local.example compose.local.yml Makefile \
        README.md config docker libs/maxxflow_core libs/maxxflow_mlops libs/maxxflow_providers \
        libs/maxxflow_cli tests/parity 2>/dev/null || true
git commit -m "checkpoint 0: parity skeleton (config/profiles, ports/adapters, compose, multi-stage Dockerfile from uv.lock, local MLflow, stub train+serve parity, tests/parity green)" || true

git add libs/maxxflow_data libs/maxxflow_synth libs/maxxflow_features db \
        tests/unit/test_money_decimal.py tests/unit/test_clock_tz_invariance.py \
        tests/unit/test_transforms.py tests/unit/test_hashing_masterdata.py \
        tests/data_quality tests/contract/test_schema_contract.py 2>/dev/null || true
git commit -m "checkpoint 1: DAL + synthetic generator + 3 validation gates (schema/realism/leakage+learnability; AUC~1.0 fails) green" || true

git add modules/m1_quote tests/unit/test_m1_guardrails.py tests/smoke/test_m1_smoke_train.py 2>/dev/null || true
git commit -m "checkpoint 2: M1 Quote full vertical slice (features->train->isotonic->register@champion->guardrailed score->writeback->drift)" || true

git add libs/maxxflow_events modules/m2_inventory modules/m3_delay modules/m4_bom modules/m5_scheduling \
        tests/unit/test_m2_guardrails.py tests/unit/test_m3_guardrails.py tests/unit/test_m3_leakage.py \
        tests/unit/test_m4_guardrails.py tests/unit/test_m5_stub.py tests/smoke tests/contract NOTES.md scripts 2>/dev/null || true
git commit -m "checkpoint 3: fan out M2 (batch) / M3 (event, leakage-safe, 2 heads) / M4 (rules+classical anomaly, micro-batch) + M5 stub; NOTES" || true

echo "Done. Review with: git log --oneline"
