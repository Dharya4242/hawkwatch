"""
test_pipeline.py — Full end-to-end pipeline test: video → frames → Gemma → incidents → reports.

What this does:
  1. Extracts frames from an MP4 file (frame_extractor.py)
  2. Sends each frame to Gemma for analysis (gemma_client.py)
  3. Parses the JSON response into a typed Incident (incident_detector.py)
  4. Generates a formatted report for each incident (report_generator.py)
  5. Prints everything to console and saves a summary to data/test_report.txt

No database, no UI, no server — pure pipeline smoke test.
Requires GEMMA_API_KEY in .env.

Usage:
    python test_pipeline.py path/to/video.mp4
    python test_pipeline.py                     # uses testing/videos/10.mp4 if present
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import frame_extractor
import gemma_client
import incident_detector
import report_generator

load_dotenv()


def run_pipeline(video_path: str, max_frames: int = 5) -> list[incident_detector.Incident]:
    """
    Run the full pipeline on a video file.

    Args:
        video_path:  Path to an MP4 file.
        max_frames:  Cap frames to keep API costs low during testing.

    Returns:
        List of completed Incident objects (with reports attached).
    """
    print("\n" + "=" * 60)
    print("SecureSight — end-to-end pipeline test")
    print(f"Video : {video_path}")
    print(f"Limit : {max_frames} frames")
    print(f"Time  : {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)

    # ── Step 1: Extract frames ─────────────────────────────────────────────────
    print("\n[Step 1] Extracting frames...")
    frames = frame_extractor.extract_frames(
        video_source=video_path,
        output_dir="data/frames",
        interval_seconds=3.0,
        motion_threshold=500,
        use_motion_detection=True,
        max_frames=max_frames,
    )

    if not frames:
        print("No frames extracted. Check the video path and try again.")
        return []

    print(f"\n  {len(frames)} frame(s) ready for analysis.")

    # ── Steps 2–4: Analyze → Parse → Report ───────────────────────────────────
    incidents: list[incident_detector.Incident] = []
    video_source = Path(video_path).name

    for i, frame in enumerate(frames, start=1):
        print(f"\n{'─' * 60}")
        print(f"[Frame {i}/{len(frames)}] {frame.timestamp_label} → {Path(frame.path).name}")

        # Step 2: Gemma vision call
        raw_json = gemma_client.analyze_frame(frame.path)

        # Step 3: Parse into Incident
        incident = incident_detector.parse_gemma_output(
            raw=raw_json,
            frame_path=frame.path,
            video_source=video_source,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        # Step 4: Generate formatted report
        incident = report_generator.generate_report(incident)

        incidents.append(incident)

    return incidents


def save_summary(incidents: list[incident_detector.Incident], output_path: str) -> None:
    """Write all incident reports to a text file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "SecureSight — PIPELINE TEST REPORT",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Incidents: {len(incidents)}",
        "=" * 60,
        "",
    ]

    for i, inc in enumerate(incidents, start=1):
        lines += [
            f"INCIDENT {i} of {len(incidents)}",
            f"ID         : {inc.id}",
            f"Timestamp  : {inc.timestamp}",
            f"Source     : {inc.video_source}",
            f"Frame      : {inc.frame_path}",
            f"Severity   : {inc.severity}",
            f"Category   : {inc.category}",
            f"Confidence : {inc.confidence}%",
            f"Persons    : {inc.persons_count}",
            f"Activity   : {inc.activity_detected}",
            f"Objects    : {', '.join(inc.objects_of_interest) or 'None'}",
            "",
            inc.report,
            "",
            "─" * 60,
            "",
        ]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary saved → {output_path}")


def print_summary(incidents: list[incident_detector.Incident]) -> None:
    """Print a concise table of all incidents to console."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"{'#':<4} {'Time':<12} {'Severity':<10} {'Category':<25} {'Conf':>5}")
    print("─" * 60)
    for i, inc in enumerate(incidents, start=1):
        ts_short = inc.timestamp[11:19]  # HH:MM:SS portion
        print(f"{i:<4} {ts_short:<12} {inc.severity:<10} {inc.category:<25} {inc.confidence:>4}%")
    print("─" * 60)

    critical = sum(1 for inc in incidents if inc.severity == "CRITICAL")
    warning = sum(1 for inc in incidents if inc.severity == "WARNING")
    clear = sum(1 for inc in incidents if inc.severity == "CLEAR")
    print(f"CRITICAL: {critical}  |  WARNING: {warning}  |  CLEAR: {clear}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve video path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        default = "testing/videos/10.mp4"
        if Path(default).exists():
            video_path = default
        else:
            print("Usage: python test_pipeline.py path/to/video.mp4")
            print(f"\nDefault path '{default}' not found.")
            sys.exit(1)

    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    if not os.getenv("GEMMA_API_KEY"):
        print("ERROR: GEMMA_API_KEY is not set.")
        print("Copy .env.example to .env and add your Google AI Studio key.")
        sys.exit(1)

    incidents = run_pipeline(video_path, max_frames=5)

    if incidents:
        print_summary(incidents)
        save_summary(incidents, "data/test_report.txt")

    print("\nDone.")
