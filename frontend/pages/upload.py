"""
frontend/pages/upload.py — Upload an MP4 and run the SecureSight pipeline.
"""

import os
from pathlib import Path

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
# API_BASE = os.getenv("API_BASE_URL", "https://arnette-permutable-bridger.ngrok-free.dev")

_SEVERITY_ICON = {"CRITICAL": "🔴", "WARNING": "🟡", "CLEAR": "🟢"}


def _render_incident_card(inc: dict):
    severity = inc.get("severity", "CLEAR")
    category = inc.get("category", "Normal")
    confidence = inc.get("confidence", 0)
    icon = _SEVERITY_ICON.get(severity, "⚪")

    label = f"{icon} {severity} — {category} ({confidence}% confidence)"
    with st.expander(label, expanded=(severity == "CRITICAL")):
        col_img, col_info = st.columns([1, 2])

        with col_img:
            frame_path = inc.get("frame_path", "")
            if frame_path:
                frame_url = f"{API_BASE}/frames/{Path(frame_path).name}"
                st.image(frame_url, caption=Path(frame_path).name, use_column_width=True)
            else:
                st.info("No frame available")

        with col_info:
            st.markdown(f"**Timestamp:** {inc.get('timestamp', '—')}")
            st.markdown(f"**Activity:** {inc.get('activity_detected', '—')}")
            st.markdown(f"**Persons detected:** {inc.get('persons_count', 0)}")
            objects = inc.get("objects_of_interest", [])
            if objects:
                st.markdown(f"**Objects:** {', '.join(objects)}")
            if inc.get("alert_sent"):
                st.success("📱 SMS alert sent")

        report = inc.get("report", "")
        if report:
            st.markdown("---")
            st.markdown("**Full Report:**")
            st.code(report, language=None)


def show():
    st.title("📤 Analyze Video")
    st.markdown(
        "Upload a surveillance video (MP4 / MOV / AVI). "
        "SecureSight extracts frames, analyzes each with your finetuned Gemma 4 model, "
        "and generates structured incident reports."
    )

    # Backend status
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3.0)
        # if r.status_code == 200:
        #     st.success(f"Backend connected — {API_BASE}", icon="✅")
        # else:
        #     st.warning(f"Backend returned {r.status_code}", icon="⚠️")
    except Exception:
        st.error(
            f"Cannot reach backend at **{API_BASE}**. "
            "Run: `uvicorn backend.main:app --reload --port 8000`",
            icon="🚫",
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi"],
        help="Max recommended: 10 min. Each frame takes ~35s on Kaggle T4.",
    )

    max_frames = st.slider(
        "Max frames to analyze",
        min_value=1, max_value=30, value=5,
        help="Each frame ≈ 35s inference. 5 frames ≈ 3 minutes total.",
    )

    if uploaded_file is not None:
        st.info(
            f"**{uploaded_file.name}** "
            f"({uploaded_file.size / 1024 / 1024:.1f} MB) — "
            f"up to {max_frames} frames will be analyzed.",
            icon="ℹ️",
        )

        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner(
                f"Running pipeline — this takes ~{max_frames * 35}s "
                f"({max_frames} frames × 35s on T4)..."
            ):
                try:
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")
                    }
                    resp = httpx.post(
                        f"{API_BASE}/upload",
                        files=files,
                        params={"max_frames": max_frames},
                        timeout=600.0,
                    )
                    resp.raise_for_status()
                    incidents = resp.json()
                except httpx.HTTPStatusError as e:
                    st.error(f"Upload failed: {e.response.status_code} — {e.response.text}")
                    incidents = []
                except Exception as e:
                    st.error(f"Error: {e}")
                    incidents = []

            if incidents:
                critical = sum(1 for i in incidents if i["severity"] == "CRITICAL")
                warning  = sum(1 for i in incidents if i["severity"] == "WARNING")
                clear    = sum(1 for i in incidents if i["severity"] == "CLEAR")

                st.success(f"Analysis complete — {len(incidents)} incident(s) saved to library.")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", len(incidents))
                c2.metric("🔴 Critical", critical)
                c3.metric("🟡 Warning",  warning)
                c4.metric("🟢 Clear",    clear)

                st.markdown("### Incident Reports")
                for inc in incidents:
                    _render_incident_card(inc)

            elif incidents is not None:
                st.warning("No incidents found — video may have no detectable motion.")
