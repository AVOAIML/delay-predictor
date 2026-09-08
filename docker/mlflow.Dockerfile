# The official ghcr.io/mlflow/mlflow image only `pip install`s mlflow itself —
# no Postgres driver — so --backend-store-uri postgresql+psycopg2://... fails
# at runtime with ModuleNotFoundError. This adds the one missing dependency.
FROM ghcr.io/mlflow/mlflow:v2.22.4
RUN pip install --no-cache-dir psycopg2-binary
