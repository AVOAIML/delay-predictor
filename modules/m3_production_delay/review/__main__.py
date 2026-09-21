"""``python -m m3_production_delay.review`` -> the batch review CLI.

A separate entry module rather than running ``pipeline.py`` with ``-m``
directly: the package's ``__init__`` imports ``pipeline``, so ``-m
...review.pipeline`` would execute a module Python has already imported and
warn about it. Nothing lives here but the delegation.
"""

from m3_production_delay.review.pipeline import _run_cli

if __name__ == "__main__":
    raise SystemExit(_run_cli())
