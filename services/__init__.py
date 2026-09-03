"""MaXXflow local service apps (run from the docker-compose stack).

* configurator — FastAPI backing the ML Model Configurator UI
* dashboard     — Streamlit interactive system/model dashboard
These are run as apps (uvicorn / streamlit), not imported as a library.
"""
