"""CLI: train the two M1 Smart Quote Optimiser models from the prepared gold CSVs.

    python services/train_csv.py                       # GLOBAL base model (default)
    python services/train_csv.py --publish             # ...and move @champion via the gate
    python services/train_csv.py --tenant furniched     # a specific tenant's own model

The `tenant` value ONLY names the registered model (t_<tenant>__m_...); it is never a
feature and never filters rows — training always uses the whole CSV. The default
tenant is `global`: the shared base model that every newly onboarded tenant sees and
is served until they train + publish their own. Trains + registers a CANDIDATE; use
--publish (or the Configurator) to move the @champion alias through the gate."""

from __future__ import annotations

import argparse

from maxxflow_mlops.naming import GLOBAL_TENANT
from m1_quote import csv_price, csv_win


def run(tenant: str, dataset: str, publish: bool) -> None:
    label = f"{tenant}  (SHARED BASE MODEL)" if tenant == GLOBAL_TENANT else tenant
    print(label)
    print(f"\n=== {label} ===")

    w = csv_win.train(f"{dataset}/gold_quote_win.csv", tenant)
    print(w)
    print(f"WIN   v{w['version']}  accuracy={w['metrics']['accuracy']:.3f} AUC={w['metrics']['auc']:.3f}")
    # p = csv_price.train(f"{dataset}/gold_price_band.csv", tenant)
    # print(f"PRICE v{p['version']}  served={p['served_mode']}  coverage={p['metrics']['coverage']:.3f}")
    # if publish:
    #     print("  win publish   :", csv_win.publish(tenant, w["version"], w["metrics"]))
    #     print("  price publish :", csv_price.publish(tenant, p["version"], p["metrics"]))


def main():
    ap = argparse.ArgumentParser(description="Train the M1 Smart Quote Optimiser models.")
    
    ap.add_argument("--tenant", default=GLOBAL_TENANT,
                    help=f"model owner slug (default '{GLOBAL_TENANT}' = shared base model)")
    
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--publish", action="store_true", help="also move @champion through the gate")
    ap.add_argument("--all", action="store_true",
                    help="(deprecated) training is tenant-agnostic now; trains the global base model")
    a = ap.parse_args()
    
    tenant = GLOBAL_TENANT if a.all else a.tenant
    if a.all:
        print("note: --all is deprecated — rows are no longer split by tenant; "
              "training the shared global base model instead.")
    
    run(tenant, a.dataset, a.publish)


if __name__ == "__main__":
    main()
