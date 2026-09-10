# The official ghcr.io/mlflow/mlflow image only `pip install`s mlflow itself —
# no Postgres driver, no Azure Storage SDKs — so --backend-store-uri
# postgresql+psycopg2://... and an abfss://.../wasbs://... --default-artifact-root
# both fail at runtime with ModuleNotFoundError. This adds the missing deps:
# psycopg2-binary for Postgres, azure-storage-blob (wasbs://) and
# azure-storage-file-datalake (abfss://) for the artifact store.
FROM ghcr.io/mlflow/mlflow:v2.22.4
RUN pip install --no-cache-dir psycopg2-binary azure-storage-blob azure-storage-file-datalake
