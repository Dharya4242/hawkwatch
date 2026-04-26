"""
backend/routes/stream.py — POST /stream

Accepts a live stream or remote video URL (RTSP / YouTube / direct MP4 link),
runs the HawkWatch pipeline on it, and saves incidents to the database.

Behaviour is identical to /upload — the only difference is the video source
is a URL instead of an uploaded file.
"""

import os
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

import frame_extractor
import gemma_client
import incident_detector
import report_generator
from backend.alerts import send_sms_alert
from backend.database import get_db, incident_to_db
from backend.models import IncidentRead, incident_to_read

router = APIRouter()


class StreamRequest(BaseModel):
    url: str
    max_frames: int = 10


@router.post("/stream", response_model=List[IncidentRead])
async def analyze_stream(
    request: StreamRequest,
    db: Session = Depends(get_db),
):
    """
    Analyze a video stream or remote URL and return all incidents found.

    Accepts RTSP streams, YouTube URLs (via OpenCV), or direct MP4 links.
    max_frames caps the number of frames analyzed (default 10, lower = faster).
    """
    if not request.url:
        raise HTTPException(400, "url is required.")

    interval = float(os.getenv("FRAME_INTERVAL_SECONDS", "3"))
    motion_threshold = int(os.getenv("MOTION_THRESHOLD", "500"))

    print(f"\n[Stream] Opening: {request.url}")
    frames = frame_extractor.extract_frames(
        video_source=request.url,
        output_dir="data/frames",
        interval_seconds=interval,
        motion_threshold=motion_threshold,
        use_motion_detection=True,
        max_frames=request.max_frames,
    )

    if not frames:
        raise HTTPException(422, "No frames could be extracted from the stream URL.")

    incidents = []
    for frame in frames:
        raw_json = gemma_client.analyze_frame(frame.path)
        incident = incident_detector.parse_gemma_output(
            raw=raw_json,
            frame_path=frame.path,
            video_source=request.url,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        incident = report_generator.generate_report(incident)

        if send_sms_alert(incident):
            incident.alert_sent = True

        db.add(incident_to_db(incident))
        incidents.append(incident)

    db.commit()
    print(f"[Stream] Done — {len(incidents)} incident(s) saved.")

    return [incident_to_read(inc) for inc in incidents]
