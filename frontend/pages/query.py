"""
frontend/pages/query.py — Natural language search across all incidents.
"""

import os
from pathlib import Path

import httpx
import streamlit as st

# API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_BASE = os.getenv("API_BASE_URL", "https://arnette-permutable-bridger.ngrok-free.dev")

_SEVERITY_ICON = {"CRITICAL": "🔴", "WARNING": "🟡", "CLEAR": "🟢"}

_EXAMPLES = [
    "person waving for help near debris",
    "suspicious activity at night near entrance",
    "fight or assault in parking lot",
    "medical emergency or person collapsed",
    "weapon visible",
]


def _search(query: str) -> dict:
    try:
        r = httpx.post(f"{API_BASE}/query", json={"query": query}, timeout=60.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Search failed: {e.response.status_code} — {e.response.text}")
        return {}
    except Exception as e:
        st.error(f"Error: {e}")
        return {}


def _fetch_incident(incident_id: str) -> dict:
    try:
        r = httpx.get(f"{API_BASE}/incidents/{incident_id}", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _render_match(match: dict, idx: int):
    score      = match.get("relevance_score", 0)
    reason     = match.get("reason", "")
    ts         = match.get("timestamp", "—")
    frame_path = match.get("frame_path", "")

    inc      = _fetch_incident(match.get("incident_id", ""))
    severity = inc.get("severity", "CLEAR")
    icon     = _SEVERITY_ICON.get(severity, "⚪")
    activity = inc.get("activity_detected", "—")

    label = f"#{idx + 1}  {icon} {severity}  |  {ts}  |  Score: {score}%  |  {activity[:50]}"

    with st.expander(label, expanded=(idx == 0)):
        col_img, col_info = st.columns([1, 2])

        with col_img:
            if frame_path:
                frame_url = f"{API_BASE}/frames/{Path(frame_path).name}"
                st.image(frame_url, use_column_width=True)

        with col_info:
            st.markdown(f"**Relevance:** {score}%")
            st.markdown(f"**Why matched:** {reason}")
            st.markdown(f"**Severity:** {icon} {severity}")
            st.markdown(f"**Category:** {inc.get('category', '—')}")
            st.markdown(f"**Activity:** {activity}")
            st.markdown(f"**Confidence:** {inc.get('confidence', 0)}%")
            if inc.get("alert_sent"):
                st.success("📱 SMS alert was sent for this incident")

        report = inc.get("report", "")
        if report:
            st.markdown("---")
            st.code(report, language=None)


def show():
    st.title("🔍 Search Incidents")
    st.markdown(
        "Describe what you're looking for in plain English. "
        "HawkWatch searches all stored incidents and returns the most relevant matches."
    )

    # Example query buttons
    st.markdown("**Try an example:**")
    cols = st.columns(len(_EXAMPLES))
    chosen = None
    for i, ex in enumerate(_EXAMPLES):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            chosen = ex
            st.session_state["_query_input"] = ex

    st.divider()

    default = st.session_state.get("_query_input", chosen or "")
    query = st.text_input(
        "Search query",
        value=default,
        placeholder="e.g. person waving for help near debris",
        label_visibility="collapsed",
        key="_query_input",
    )

    if st.button("🔎 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query.", icon="⚠️")
            return

        with st.spinner(f"Searching for: *{query}*"):
            result = _search(query)

        matches = result.get("matches", [])

        st.markdown(f"### Results for: *{query}*")

        if not matches:
            st.info(
                "No matching incidents found. Try a different query or upload more videos.",
                icon="ℹ️",
            )
        else:
            st.success(f"Found **{len(matches)}** matching incident(s)")
            for idx, match in enumerate(matches):
                _render_match(match, idx)
