"""maxxflow_mlops — MLflow tracking/registry helpers, promotion gates, serving
router, drift. The MLflow API here is identical local (self-hosted) and on Azure
ML (its tracking/registry IS MLflow-compatible), so train/serve code is unchanged
across environments (plan §1, §2, §6, §7)."""

from maxxflow_mlops.naming import registered_model_name, parse_model_name
from maxxflow_mlops.registry import MLflowRegistry, get_registry

__all__ = ["MLflowRegistry", "get_registry", "registered_model_name", "parse_model_name"]
