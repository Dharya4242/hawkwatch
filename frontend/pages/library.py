"""
frontend/pages/library.py — Incident library with filters and expandable report cards.
"""

import os
from pathlib import Path

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
# API_BASE = os.getenv("API_BASE_URL", "https://arnette-permutable-bridger.ngrok-free.dev)

_SEVERITY_ICON = {"CRITICAL": "🔴", "WARNING": "🟡", "CLEAR": "🟢"}
_SEVERITIES    = ["All", "CRITICAL", "WARNING", "CLEAR"]
_CATEGORIES    = ["All", "Crime", "Medical Emergency", "Suspicious Activity", "Disaster", "Normal"]


def _fetch_incidents(severity=None, category=None) -> list:
    params = {"limit": 200}
    if severity:
        params["severity"] = severity
    if category:
        params["category"] = category
    try:
        r = httpx.get(f"{API_BASE}/incidents", params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to load incidents: {e}")
        return []


def _render_incident_row(inc: dict):
    severity = inc.get("severity", "CLEAR")
    icon     = _SEVERITY_ICON.get(severity, "⚪")
    category = inc.get("category", "—")
    ts       = inc.get("timestamp", "—")
    conf     = inc.get("confidence", 0)
    activity = inc.get("activity_detected", "—")

    short_activity = activity[:60] + ("…" if len(activity) > 60 else "")
    label = f"{icon} {ts}  |  {category}  |  {short_activity}  ({conf}%)"

    with st.expander(label):
        col_img, col_info = st.columns([1, 2])

        with col_img:
            frame_path = inc.get("frame_path", "")
            if frame_path:
                frame_url = f"{API_BASE}/frames/{Path(frame_path).name}"
                st.image(frame_url, use_column_width=True)

        with col_info:
            st.markdown(f"**ID:** `{inc.get('id', '—')[:8]}…`")
            st.markdown(f"**Source:** {inc.get('video_source', '—')}")
            st.markdown(f"**Severity:** {icon} {severity}")
            st.markdown(f"**Category:** {category}")
            st.markdown(f"**Confidence:** {conf}%")
            st.markdown(f"**Persons:** {inc.get('persons_count', 0)}")
            objects = inc.get("objects_of_interest", [])
            if objects:
                st.markdown(f"**Objects:** {', '.join(objects)}")
            st.markdown(f"**Action:** {inc.get('recommended_action', '—')}")
            if inc.get("alert_sent"):
                st.success("📱 SMS alert was sent")

        report = inc.get("report", "")
        if report:
            st.markdown("---")
            st.code(report, language=None)


def show():
    st.title("📚 Incident Library")
    st.markdown("All detected incidents, newest first. Filter by severity or category.")

    col_sev, col_cat, col_btn = st.columns([2, 2, 1])
    with col_sev:
        severity_filter = st.selectbox("Severity", _SEVERITIES, index=0)
    with col_cat:
        category_filter = st.selectbox("Category", _CATEGORIES, index=0)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔄 Refresh", use_container_width=True)

    st.divider()

    incidents = _fetch_incidents(
        severity=severity_filter if severity_filter != "All" else None,
        category=category_filter if category_filter != "All" else None,
    )

    if not incidents:
        st.info(
            "No incidents yet. Upload a video on **Analyze Video** to get started.",
            icon="ℹ️",
        )
        return

    critical = sum(1 for i in incidents if i["severity"] == "CRITICAL")
    warning  = sum(1 for i in incidents if i["severity"] == "WARNING")
    clear    = sum(1 for i in incidents if i["severity"] == "CLEAR")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",       len(incidents))
    c2.metric("🔴 Critical", critical)
    c3.metric("🟡 Warning",  warning)
    c4.metric("🟢 Clear",    clear)

    st.markdown(f"### {len(incidents)} Incident(s)")

    order = {"CRITICAL": 0, "WARNING": 1, "CLEAR": 2}
    for inc in sorted(incidents, key=lambda x: order.get(x.get("severity", "CLEAR"), 3)):
        _render_incident_row(inc)
