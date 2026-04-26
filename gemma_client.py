"""
gemma_client.py — Wraps Gemma 4 inference for the HawkWatch pipeline.

Phase 1 (testing):  calls Google AI Studio REST API with base Gemma 4.
Phase 3 (final):    calls finetuned Gemma 4 served on Kaggle GPU via ngrok.

Switch phases by setting GEMMA_ENDPOINT in .env:
  GEMMA_ENDPOINT=aistudio               -> Phase 1 (AI Studio API)
  GEMMA_ENDPOINT=https://xxxx.ngrok.io  -> Phase 3 (finetuned model)

Only this file + .env change between phases. Everything else stays the same.

Public API:
  analyze_frame(image_path)  -> raw JSON string  (parsed by incident_detector.py)
  generate_report(...)       -> formatted report string
"""

import os
import base64
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv

from prompts import FRAME_ANALYSIS_PROMPT, NL_QUERY_PROMPT, REPORT_GENERATION_PROMPT

load_dotenv()

GEMMA_API_KEY: str = os.getenv("GEMMA_API_KEY", "")
GEMMA_ENDPOINT: str = os.getenv("GEMMA_ENDPOINT", "aistudio")
GEMMA_MODEL: str = os.getenv("GEMMA_MODEL", "gemma-3-27b-it")

_AISTUDIO_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Timeouts: vision calls are heavier than text calls
_VISION_TIMEOUT = 90.0
_TEXT_TIMEOUT = 60.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type) for an image file."""
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime_type = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime_type


def _extract_text(response_json: dict) -> str:
    """Pull generated text out of an AI Studio response envelope."""
    return response_json["candidates"][0]["content"]["parts"][0]["text"]


# ── AI Studio calls (Phase 1) ─────────────────────────────────────────────────

def _aistudio_vision(image_path: str, prompt: str) -> str:
    """Vision call: image + prompt -> text via AI Studio REST API."""
    if not GEMMA_API_KEY:
        raise ValueError("GEMMA_API_KEY is not set in .env")

    image_data, mime_type = _encode_image(image_path)
    url = f"{_AISTUDIO_BASE}/{GEMMA_MODEL}:generateContent?key={GEMMA_API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": image_data}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,   # low temp = consistent structured JSON output
            "maxOutputTokens": 1024,
        },
    }
    with httpx.Client(timeout=_VISION_TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
    return _extract_text(resp.json())


def _aistudio_text(prompt: str) -> str:
    """Text-only call via AI Studio REST API."""
    if not GEMMA_API_KEY:
        raise ValueError("GEMMA_API_KEY is not set in .env")

    url = f"{_AISTUDIO_BASE}/{GEMMA_MODEL}:generateContent?key={GEMMA_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }
    with httpx.Client(timeout=_TEXT_TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
    return _extract_text(resp.json())


# ── Ngrok calls (Phase 3 — finetuned model on Kaggle GPU) ─────────────────────

def _ngrok_vision(image_path: str, prompt: str) -> str:
    """Vision call via finetuned model endpoint on ngrok."""
    image_data, _ = _encode_image(image_path)
    payload = {"image_base64": image_data, "prompt": prompt}
    with httpx.Client(timeout=_VISION_TIMEOUT) as client:
        resp = client.post(f"{GEMMA_ENDPOINT}/analyze_frame", json=payload)
        resp.raise_for_status()
    return resp.json()["output"]


def _ngrok_text(prompt: str) -> str:
    """Text call via finetuned model endpoint on ngrok."""
    with httpx.Client(timeout=_TEXT_TIMEOUT) as client:
        resp = client.post(f"{GEMMA_ENDPOINT}/generate", json={"prompt": prompt})
        resp.raise_for_status()
    return resp.json()["output"]


# ── Fallback JSON when Gemma call fails ────────────────────────────────────────

def _fallback_analysis(error: str) -> str:
    return json.dumps({
        "scene_description": f"Analysis failed: {error}",
        "activity_detected": "Unknown — analysis error",
        "persons_count": 0,
        "severity": "CLEAR",
        "category": "Normal",
        "confidence": 0,
        "recommended_action": "Manual review required",
        "objects_of_interest": [],
    })


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_frame(image_path: str) -> str:
    """
    Send a surveillance frame to Gemma for analysis.

    Args:
        image_path: Path to a JPEG frame extracted by frame_extractor.py.

    Returns:
        Raw JSON string matching the FRAME_ANALYSIS_PROMPT schema.
        On failure, returns a safe fallback JSON string (never raises).
    """
    print(f"  [Gemma] Analyzing frame: {Path(image_path).name} | endpoint={GEMMA_ENDPOINT}")
    try:
        if GEMMA_ENDPOINT == "aistudio":
            raw = _aistudio_vision(image_path, FRAME_ANALYSIS_PROMPT)
        else:
            raw = _ngrok_vision(image_path, FRAME_ANALYSIS_PROMPT)

        print(f"  [Gemma] Vision response received ({len(raw)} chars)")
        return raw

    except Exception as e:
        print(f"  [Gemma] ERROR in analyze_frame: {e}")
        return _fallback_analysis(str(e))


def generate_report(
    timestamp: str,
    source: str,
    scene_description: str,
    activity: str,
    severity: str,
    category: str,
    confidence: int,
) -> str:
    """
    Generate a formatted incident report from structured incident data.

    Args:
        timestamp:         ISO timestamp string e.g. "2026-04-15T14:32:15"
        source:            Video filename or stream URL
        scene_description: Raw scene text from analyze_frame()
        activity:          Specific activity string from analyze_frame()
        severity:          CRITICAL / WARNING / CLEAR
        category:          Crime / Medical Emergency / Suspicious Activity / Disaster / Normal
        confidence:        Integer 0-100

    Returns:
        Formatted INCIDENT REPORT string.
        On failure, returns a minimal error report (never raises).
    """
    print(f"  [Gemma] Generating report | severity={severity} | endpoint={GEMMA_ENDPOINT}")
    prompt = REPORT_GENERATION_PROMPT.format(
        timestamp=timestamp,
        source=source,
        scene_description=scene_description,
        activity=activity,
        severity=severity,
        category=category,
        confidence=confidence,
    )
    try:
        if GEMMA_ENDPOINT == "aistudio":
            report = _aistudio_text(prompt)
        else:
            report = _ngrok_text(prompt)

        print(f"  [Gemma] Report generated ({len(report)} chars)")
        return report

    except Exception as e:
        print(f"  [Gemma] ERROR in generate_report: {e}")
        return (
            f"INCIDENT REPORT\n"
            f"===============\n"
            f"Timestamp:   {timestamp}\n"
            f"Source:      {source}\n"
            f"Severity:    {severity}\n"
            f"ERROR: Report generation failed — {e}\n"
            f"Recommended Action: Manual review required."
        )


def search_incidents(query: str, incidents_json: str) -> str:
    """
    Use Gemma to rank incidents against a natural language query.

    Args:
        query:          Free-text search string from the user.
        incidents_json: JSON string array of incident records to search through.

    Returns:
        Raw JSON array string of matching incidents (as per NL_QUERY_PROMPT schema).
        On failure, returns "[]" so the caller always gets a parseable result.
    """
    print(f"  [Gemma] NL search | query='{query[:60]}' | endpoint={GEMMA_ENDPOINT}")
    prompt = NL_QUERY_PROMPT.format(query=query, incidents_json=incidents_json)

    try:
        if GEMMA_ENDPOINT == "aistudio":
            raw = _aistudio_text(prompt)
        else:
            raw = _ngrok_text(prompt)

        print(f"  [Gemma] Search response received ({len(raw)} chars)")
        return raw

    except Exception as e:
        print(f"  [Gemma] ERROR in search_incidents: {e}")
        return "[]"


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Smoke-test both Gemma calls.

    Usage:
        python gemma_client.py                          # uses first frame in data/frames/
        python gemma_client.py path/to/frame.jpg        # use a specific frame
    """
    import sys

    print("=" * 60)
    print("HawkWatch — gemma_client.py smoke test")
    print(f"  GEMMA_ENDPOINT : {GEMMA_ENDPOINT}")
    print(f"  GEMMA_MODEL    : {GEMMA_MODEL}")
    print(f"  GEMMA_API_KEY  : {'set' if GEMMA_API_KEY else 'NOT SET — check .env'}")
    print("=" * 60)

    # Pick a test frame
    if len(sys.argv) > 1:
        test_frame = sys.argv[1]
    else:
        frames_dir = Path("data/frames")
        candidates = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
        if not candidates:
            print("\nNo frames found in data/frames/. Run frame_extractor.py first.")
            print("Or pass a JPEG path as an argument: python gemma_client.py frame.jpg")
            sys.exit(1)
        test_frame = str(candidates[0])

    print(f"\nTest frame: {test_frame}")

    # --- Test 1: analyze_frame ---
    print("\n--- Test 1: analyze_frame() ---")
    raw_json = analyze_frame(test_frame)
    print("Raw output:")
    print(raw_json)

    # Try to parse it so we can use it in test 2
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        print("\nWarning: response is not valid JSON — Gemma returned free text.")
        print("This is OK for Phase 1 — incident_detector.py handles this.")
        parsed = {
            "scene_description": raw_json[:200],
            "activity_detected": "See raw output",
            "severity": "CLEAR",
            "category": "Normal",
            "confidence": 0,
        }

    # --- Test 2: generate_report ---
    print("\n--- Test 2: generate_report() ---")
    from datetime import datetime
    report = generate_report(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        source=Path(test_frame).name,
        scene_description=parsed.get("scene_description", ""),
        activity=parsed.get("activity_detected", ""),
        severity=parsed.get("severity", "CLEAR"),
        category=parsed.get("category", "Normal"),
        confidence=int(parsed.get("confidence", 0)),
    )
    print("Report output:")
    print(report)

    print("\n" + "=" * 60)
    print("Smoke test complete.")
