"""Trivial stub model proving train+serve are identical local↔Azure-shaped
(plan §10 checkpoint 0). It uses the SAME uniform ``pyfunc.PythonModel`` flavor
that every real module uses, so the registry round-trip and the BYOC router are
exercised before any real ML exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel
from sklearn.linear_model import LogisticRegression

from maxxflow_mlops.naming import registered_model_name
from maxxflow_mlops.registry import MLflowRegistry

FEATURES = ["x0", "x1"]


class ParityStubModel(PythonModel):
    """Holds a fitted LogisticRegression; returns a calibrated-probability frame.

    The estimator is stored on the instance and cloudpickled by MLflow — the same
    pattern the module wrappers use, so serving code is uniform.
    """

    def __init__(self, estimator: LogisticRegression):
        self.estimator = estimator

    def predict(self, context, model_input, params=None):  # mlflow pyfunc contract
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        proba = self.estimator.predict_proba(df[FEATURES].to_numpy())[:, 1]
        return pd.DataFrame({"probability": proba})


def train_parity_stub(tenant: str = "demo", module: str = "parity", seed: int = 0) -> str:
    """Fit on tiny synthetic data, log+register a pyfunc, set the champion alias.
    Returns the registered version. The SAME call shape runs as an AML job later."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=200) > 0).astype(int)
    est = LogisticRegression().fit(X, y)

    Xdf = pd.DataFrame(X, columns=FEATURES)
    model = ParityStubModel(est)
    signature = infer_signature(Xdf, model.predict(None, Xdf))

    reg = MLflowRegistry()
    name = registered_model_name(tenant, module)
    version = reg.log_and_register(
        model,
        name=name,
        params={"algo": "logreg", "seed": seed},
        metrics={"train_accuracy": float(est.score(X, y))},
        tags={"tenant": tenant, "module": module, "data_provenance": "synthetic"},
        signature=signature,
        input_example=Xdf.head(3),
    )
    reg.set_alias(name=name, alias="champion", version=version)
    return version
