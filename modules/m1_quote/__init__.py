"""M1 — Smart Quote Optimizer (reference module / template for M2–M4).

Exposes the standard module interface the CLI dispatches to: ``fe`` / ``train`` /
``score`` / ``drift`` (added in checkpoint 2), plus ``build_features`` used by the
synthetic gate and by training/scoring alike.
"""

from maxxflow_core.errors import get_logger
from m1_quote.features import FEATURE_COLUMNS, build_features, training_frame

log = get_logger("m1_quote")
__all__ = ["build_features", "training_frame", "FEATURE_COLUMNS"]

# fe/train/score/drift are attached by submodules when imported (checkpoint 2).
try:  # pragma: no cover - available once checkpoint 2 lands
    from m1_quote.pipeline import drift, fe, score, train  # noqa: F401
    __all__ += ["fe", "train", "score", "drift"]
except Exception:
   
    log.exception("m1_quote.pipeline failed to import; fe/train/score/drift unavailable")
