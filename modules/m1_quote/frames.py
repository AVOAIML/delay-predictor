"""Raw combined export -> the training frame each M1 model card expects.

WHY THIS IS A LIBRARY MODULE AND NOT PART OF THE WEB SERVICE
------------------------------------------------------------
These two functions used to live inside services/configurator/jobs.py, in the
closure of the background training thread. Two things pulled them out here.

First, correctness: the CLI (`maxxflow train-csv`, and therefore every scheduled
retrain and every Azure ML job) has to run the SAME conversion as the
Configurator upload. Two copies would drift the first time a builder changed,
and an unattended job would quietly train on a different frame shape from the
one a human reviewed in the UI.

Second, packaging: pyproject installs `libs` and `modules` as packages —
`services` is copied into the image but is not importable from an arbitrary
working directory. A CLI importing from services.configurator failed in the
Azure ML training container with ModuleNotFoundError: No module named 'services'
the first time a job ran there. The dependency direction was backwards anyway: a
training command should not need the web service to exist.

So the conversion lives here, with the ingest code it wraps, and both the API and
the CLI import it.
"""

from __future__ import annotations

import json

import pandas as pd

from m1_quote.raw_ingest import (
    OPTION_GRAPH_PATH,
    RATE_LOOKUP_PATH,
    build_option_graph,
    build as build_gold_win_frame,
    build_line_frame as build_gold_line_frame,
    build_mil_frame as build_gold_mil_frame,
    build_price_frame as build_gold_price_frame,
)


def write_option_graph(raw: pd.DataFrame, logger) -> None:
    """Persist industry -> productID -> materialSpec for the Test-Predictions
    dropdowns. Written on EVERY raw upload regardless of card, because all four
    cards share one export and one form vocabulary; failure is logged, never
    fatal — a stale graph degrades to unfiltered dropdowns, which is a far better
    outcome than a failed retrain."""
    try:
        graph = build_option_graph(raw)
        OPTION_GRAPH_PATH.write_text(json.dumps(graph, indent=2))
        logger.log("Test-Predictions option graph refreshed: "
                   + ", ".join(f"{c} narrowed by {g['parent']} "
                               f"({len(g['options_by'])} values)" for c, g in graph.items()))
    except Exception as e:
        logger.log(f"WARNING: could not write the option graph ({type(e).__name__}: {e}) — "
                   f"Test-Predictions dropdowns will not cascade")


def build_training_frame(model_key: str, raw: pd.DataFrame, tenant: str, logger) -> pd.DataFrame:
    """Raw combined export -> the training frame a given model card expects.

    Lifted out of the background thread so the CLI (`maxxflow train-csv`, and
    therefore any scheduled retrain) runs the SAME conversion as the Configurator
    upload. Left inside the closure, the two would drift the first time a builder
    changed, and a monthly job would quietly train on a different frame shape
    from the one anyone reviewed in the UI."""
    if model_key == "m1_quote_win":
        logger.log("Converting the raw upload into training features "
                   "(collapsing revisions, deriving price ratios and win-rates)…")
        frame, rate_lookup = build_gold_win_frame(raw, tenant)
        logger.log(f"Built {len(frame)} quote rows from {len(raw)} raw line rows "
                   f"(families collapsed, open quotes dropped)")
        RATE_LOOKUP_PATH.write_text(json.dumps(rate_lookup, indent=2))
        return frame
    if model_key == "m1_quote_price":
        logger.log("Converting the raw upload into training features "
                   "(won lines only, deriving price ratios)…")
        frame = build_gold_price_frame(raw, tenant)
        logger.log(f"Built {len(frame)} won line rows from {len(raw)} raw line rows")
        return frame
    if model_key == "m1_quote_line_win":
        logger.log("Converting the raw upload into per-product training features "
                   "(won + lost lines, quote outcome broadcast to each line)…")
        frame = build_gold_line_frame(raw, tenant)
        extras = [c for c in ("quoteDate", "product_type", "payment_terms", "quote_total")
                  if c in frame.columns]
        logger.log(f"Built {len(frame)} line rows across {frame['quotationID'].nunique()} "
                   f"quotes from {len(raw)} raw line rows"
                   + (f"; this export also supplied {', '.join(extras)}" if extras else ""))
        return frame
    if model_key == "m1_quote_mil":
        logger.log("Converting the raw upload into MIL training features "
                   "(one row per line, grouped into quote-level bags)…")
        frame = build_gold_mil_frame(raw, tenant)
        logger.log(f"Built {len(frame)} line rows across "
                   f"{frame['quotationID'].nunique()} quotes from {len(raw)} raw line rows")
        return frame
    raise KeyError(model_key)
