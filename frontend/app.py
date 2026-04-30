"""
frontend/app.py — HawkWatch Streamlit entry point.

Run from project root:
    streamlit run frontend/app.py

Requires the FastAPI backend to be running:
    uvicorn backend.main:app --reload --port 8000
"""

import sys
from pathlib import Path

import streamlit as st

# Make pages/ importable as a package
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="HawkWatch — AI Surveillance",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("## 🦅 HawkWatch")
    st.markdown("*AI Surveillance & Disaster Response*")
    st.divider()
    page = st.radio(
        "Navigation",
        ["📤 Analyze Video", "📚 Incident Library", "🔍 Search Incidents"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Powered by finetuned Gemma 4 · Unsloth LoRA")

if page == "📤 Analyze Video":
    from pages import upload
    upload.show()
elif page == "📚 Incident Library":
    from pages import library
    library.show()
elif page == "🔍 Search Incidents":
    from pages import query
    query.show()
