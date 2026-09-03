"""MaXXflow — interactive system & model dashboard (Streamlit).

Local read-only view over the running stack:
  * Current models (per tenant) + champion versions
  * Pipeline / dataflow status: seed -> fe -> train -> serve -> drift
  * Model performance (win: AUC/Brier/ECE; price: coverage/pinball/MAE + fitted-vs-empirical)
  * Training (last run params, champion version, gate/served mode)
  * Inference (model-server health)
  * Model storage (registered models, versions, @champion, MinIO artifact path)
  * Drift (latest Evidently report)

Sources: MLflow REST/registry, model-server /health, local reports/ dir.
Run:  streamlit run services/dashboard/app.py   (port 8501)
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8085")
MODEL_SERVER = os.environ.get("MODEL_SERVER_URL", "http://localhost:5001")
REPORTS_DIR = os.environ.get("DRIFT_OUT", "reports")

st.set_page_config(page_title="MaXXflow ML Dashboard", layout="wide")


@st.cache_resource
def _client():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    from mlflow.tracking import MlflowClient
    return MlflowClient(MLFLOW_URI, MLFLOW_URI)


def _parse(name: str):
    # t_<tenant>__m_<module>
    if not name.startswith("t_") or "__m_" not in name:
        return None, None
    body = name[2:]; tenant, module = body.split("__m_", 1)
    return tenant, module


def _champion(client, name):
    try:
        mv = client.get_model_version_by_alias(name, "champion")
        return mv
    except Exception:
        return None


def load_registry():
    client = _client()
    rows = []
    try:
        models = client.search_registered_models(max_results=1000)
    except Exception as e:
        st.error(f"Cannot reach MLflow at {MLFLOW_URI}: {e}")
        return pd.DataFrame()
    for rm in models:
        tenant, module = _parse(rm.name)
        if tenant is None:
            continue
        mv = _champion(client, rm.name)
        tags = dict(mv.tags) if mv else {}
        rows.append({
            "tenant": tenant, "module": module, "name": rm.name,
            "champion_version": mv.version if mv else "—",
            "status": "Published" if mv else "Not trained",
            "provenance": tags.get("data_provenance", "—"),
            "served_mode": tags.get("served_mode", "—"),
            "accuracy": tags.get("accuracy"), "auc": tags.get("auc"),
            "brier": tags.get("brier"), "ece": tags.get("ece"),
            "coverage": tags.get("coverage"), "pinball_mean": tags.get("pinball_mean"),
            "mae_p50": tags.get("mae_p50"),
            "artifact": mv.source if mv else "—",
        })
    return pd.DataFrame(rows)


def server_health():
    for path in ("/", "/health"):
        try:
            r = requests.get(MODEL_SERVER + path, timeout=2)
            if r.status_code < 500:
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
st.title("MaXXflow — AI/ML System Dashboard")
df = load_registry()
tenants = sorted(df["tenant"].unique()) if not df.empty else []
tenant = st.sidebar.selectbox("Tenant", tenants or ["(none)"])
st.sidebar.caption(f"MLflow: {MLFLOW_URI}")
st.sidebar.caption(f"Model server: {MODEL_SERVER}")

tdf = df[df["tenant"] == tenant] if not df.empty else df
have_models = not tdf.empty
serve_ok = server_health()
drift_files = []
try:
    drift_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".html")]
except Exception:
    pass

# --- pipeline / dataflow ---
st.subheader("Pipeline / dataflow")
stages = {
    "seed": "ok" if have_models else "—",
    "fe": "ok" if have_models else "—",
    "train": "ok" if have_models else "—",
    "serve": "ok" if serve_ok else "down",
    "drift": "ok" if drift_files else "—",
}
dot = "digraph{rankdir=LR;node[shape=box style=rounded];"
prev = None
for s, statev in stages.items():
    color = {"ok": "#e6f4ea", "down": "#fde8e6", "—": "#f2f2f2"}[statev]
    dot += f'"{s}"[label="{s}\\n{statev}" style="rounded,filled" fillcolor="{color}"];'
    if prev:
        dot += f'"{prev}"->"{s}";'
    prev = s
dot += "}"
st.graphviz_chart(dot)

# --- current models + performance ---
st.subheader("Current models")
if have_models:
    st.dataframe(tdf[["module", "status", "champion_version", "provenance", "served_mode"]],
                 use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    win = tdf[tdf["module"] == "m1_quote_win"]
    price = tdf[tdf["module"] == "m1_quote_price"]
    with c1:
        st.markdown("**Win model (classification)**")
        if not win.empty:
            r = win.iloc[0]
            st.metric("Accuracy", r["accuracy"] or "—")
            st.write({"AUC": r["auc"], "Brier": r["brier"], "ECE": r["ece"],
                      "champion": r["champion_version"]})
        else:
            st.info("Not trained yet.")
    with c2:
        st.markdown("**Price-band model (regression)**")
        if not price.empty:
            r = price.iloc[0]
            st.metric("P25–P75 coverage", r["coverage"] or "—")
            st.write({"pinball_mean": r["pinball_mean"], "P50 MAE": r["mae_p50"],
                      "served": r["served_mode"], "champion": r["champion_version"]})
            st.caption("Served band is the fitted regressor only if it beats the empirical "
                       "baseline on coverage AND pinball; otherwise empirical is retained.")
        else:
            st.info("Not trained yet.")
else:
    st.info("No registered models for this tenant yet. Train via the Configurator.")

# --- inference + storage + drift ---
c3, c4 = st.columns(2)
with c3:
    st.subheader("Inference")
    st.write(f"Model server ({MODEL_SERVER}): " + ("🟢 healthy" if serve_ok else "🔴 unreachable"))
with c4:
    st.subheader("Drift")
    if drift_files:
        for f in drift_files:
            st.write("📄 " + f)
    else:
        st.write("No Evidently reports in ./reports yet — run `make drift`.")

st.subheader("Model storage (MLflow registry)")
if have_models:
    st.dataframe(tdf[["name", "champion_version", "artifact"]], use_container_width=True, hide_index=True)
