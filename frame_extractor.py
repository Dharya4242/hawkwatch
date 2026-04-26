"""
frame_extractor.py — Extracts frames from video files or streams.

What this does:
- Takes an MP4 file path (or stream URL) as input
- Extracts one frame every N seconds (configurable)
- Uses motion detection to skip boring frames (nothing changed)
- Saves frames as JPEG files to the output folder
- Returns a list of saved frame paths with timestamps

No GPU needed. No API keys needed. Pure OpenCV.
"""

import cv2
import os
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExtractedFrame:
    """Represents one extracted frame from the video."""
    path: str              # path to saved JPEG file
    timestamp_seconds: float  # when in the video this frame is from
    timestamp_label: str   # human readable e.g. "00:02:34"
    frame_number: int      # frame index in video


def seconds_to_label(seconds: float) -> str:
    """Convert seconds to HH:MM:SS label."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def detect_motion(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: int = 500) -> bool:
    """
    Returns True if significant motion detected between two frames.
    Compares grayscale versions to find pixel differences.
    threshold = minimum number of changed pixels to count as motion.
    """
    if prev_frame is None:
        return True  # always process first frame

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed_pixels = np.sum(thresh > 0)

    return changed_pixels > threshold


def extract_frames(
    video_source: str,
    output_dir: str = "data/frames",
    interval_seconds: float = 3.0,
    motion_threshold: int = 500,
    use_motion_detection: bool = True,
    max_frames: int = None,
) -> list[ExtractedFrame]:
    """
    Extract frames from a video file or stream URL.

    Args:
        video_source:         Path to MP4 file or stream URL (RTSP / YouTube)
        output_dir:           Where to save extracted JPEG frames
        interval_seconds:     Extract one frame every N seconds
        motion_threshold:     Min changed pixels to count as motion (lower = more sensitive)
        use_motion_detection: If True, skip frames with no motion
        max_frames:           Stop after extracting this many frames (None = no limit)

    Returns:
        List of ExtractedFrame objects for each saved frame
    """

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video opened: {video_source}")
    print(f"FPS: {fps:.1f} | Total frames: {total_frames} | Duration: {seconds_to_label(duration)}")
    print(f"Extracting 1 frame every {interval_seconds}s | Motion detection: {use_motion_detection}")
    print("-" * 60)

    frames_interval = int(fps * interval_seconds)  # how many frames to skip between extractions
    extracted = []
    prev_frame = None
    frame_number = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process at our interval
        if frame_number % frames_interval == 0:
            timestamp_seconds = frame_number / fps
            timestamp_label = seconds_to_label(timestamp_seconds)

            # Motion detection check
            should_save = True
            if use_motion_detection and prev_frame is not None:
                if not detect_motion(prev_frame, frame, motion_threshold):
                    should_save = False
                    print(f"  [{timestamp_label}] Skipped — no motion detected")

            if should_save:
                # Save frame as JPEG
                # filename includes timestamp for easy sorting
                filename = f"frame_{int(timestamp_seconds * 1000):010d}ms.jpg"
                filepath = str(output_path / filename)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                extracted_frame = ExtractedFrame(
                    path=filepath,
                    timestamp_seconds=timestamp_seconds,
                    timestamp_label=timestamp_label,
                    frame_number=frame_number,
                )
                extracted.append(extracted_frame)
                extracted_count += 1
                print(f"  [{timestamp_label}] Frame saved → {filename}")

                prev_frame = frame

                # Stop if we hit max_frames limit
                if max_frames and extracted_count >= max_frames:
                    print(f"\nReached max_frames limit ({max_frames}). Stopping.")
                    break

        frame_number += 1

    cap.release()
    print("-" * 60)
    print(f"Done. Extracted {len(extracted)} frames from {seconds_to_label(duration)} of video.")
    return extracted


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run this file directly to test frame extraction.
    Put any MP4 in data/sample_videos/ and update the path below.

    Usage:
        python pipeline/frame_extractor.py
    """
    import sys

    # Check if a video path was passed as argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default test path — put any MP4 here
        video_path = "data/sample_videos/test.mp4"

    if not os.path.exists(video_path):
        print(f"No video found at {video_path}")
        print("Usage: python pipeline/frame_extractor.py path/to/video.mp4")
        print("\nTo test without a video, we'll use your webcam for 10 seconds...")

        # Webcam fallback for testing
        print("Starting webcam capture... press Ctrl+C to stop")
        cap = cv2.VideoCapture(0)
        frames_saved = 0
        output_path = Path("data/frames")
        output_path.mkdir(parents=True, exist_ok=True)
        start = time.time()

        try:
            while time.time() - start < 10:
                ret, frame = cap.read()
                if ret:
                    filename = f"webcam_{int(time.time() * 1000)}.jpg"
                    cv2.imwrite(str(output_path / filename), frame)
                    frames_saved += 1
                    print(f"Saved: {filename}")
                    time.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            print(f"Done. {frames_saved} frames saved to data/frames/")
    else:
        frames = extract_frames(
            video_source=video_path,
            output_dir="data/frames",
            interval_seconds=3.0,
            motion_threshold=500,
            use_motion_detection=True,
        )
        print(f"\nExtracted frames:")
        for f in frames:
            print(f"  {f.timestamp_label} → {f.path}")
