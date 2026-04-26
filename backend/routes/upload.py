"""
backend/routes/upload.py — POST /upload

Accepts an MP4 (or MOV/AVI) file upload, runs the full HawkWatch pipeline
on it, saves every incident to the database, sends SMS alerts for CRITICAL
incidents, and returns the list of incidents as JSON.
"""

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

import frame_extractor
import gemma_client
import incident_detector
import report_generator
from backend.alerts import send_sms_alert
from backend.database import get_db, incident_to_db
from backend.models import IncidentRead, incident_to_read

router = APIRouter()

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}
_UPLOAD_DIR = Path("data/uploads")


@router.post("/upload", response_model=List[IncidentRead])
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload an MP4 video, run the HawkWatch pipeline, and return all incidents found.

    - Extracts frames (motion-filtered, every FRAME_INTERVAL_SECONDS seconds)
    - Analyzes each frame with Gemma 4
    - Generates a structured incident report per frame
    - Saves every incident to SQLite
    - Sends Twilio SMS for CRITICAL incidents (if configured)
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Upload MP4, MOV, or AVI.")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(_UPLOAD_DIR / f"{uuid.uuid4()}{suffix}")

    print(f"\n[Upload] Receiving: {file.filename} → {save_path}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    max_frames = int(os.getenv("MAX_FRAMES_PER_UPLOAD", "20"))
    interval = float(os.getenv("FRAME_INTERVAL_SECONDS", "3"))
    motion_threshold = int(os.getenv("MOTION_THRESHOLD", "500"))

    frames = frame_extractor.extract_frames(
        video_source=save_path,
        output_dir="data/frames",
        interval_seconds=interval,
        motion_threshold=motion_threshold,
        use_motion_detection=True,
        max_frames=max_frames,
    )

    if not frames:
        raise HTTPException(422, "No frames could be extracted from the uploaded video.")

    incidents = []
    for frame in frames:
        raw_json = gemma_client.analyze_frame(frame.path)
        incident = incident_detector.parse_gemma_output(
            raw=raw_json,
            frame_path=frame.path,
            video_source=file.filename,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        incident = report_generator.generate_report(incident)

        if send_sms_alert(incident):
            incident.alert_sent = True

        db.add(incident_to_db(incident))
        incidents.append(incident)

    db.commit()
    print(f"[Upload] Done — {len(incidents)} incident(s) saved.")

    return [incident_to_read(inc) for inc in incidents]
