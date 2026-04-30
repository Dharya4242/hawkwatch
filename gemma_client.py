"""
gemma_client.py — Wraps Gemma 4 inference for the HawkWatch pipeline.

Phase 1 (testing):  calls Google AI Studio REST API with base Gemma 4.
Phase 3 (final):    calls finetuned Gemma 4 served on Kaggle GPU via ngrok.
                    NO AI Studio API key required in Phase 3.

Switch phases by setting GEMMA_ENDPOINT in .env:
  GEMMA_ENDPOINT=aistudio                               -> Phase 1
  GEMMA_ENDPOINT=https://paddle-probing-march.ngrok-free.dev  -> Phase 3

Phase 3 endpoints used (all on the ngrok server):
  POST /analyze_vision  — base64 image -> incident JSON (vision + finetuned)
  (report formatting and incident search are handled locally — no API needed)

Only this file + .env change between phases. Everything else stays the same.

Public API:
  analyze_frame(image_path)  -> raw JSON string  (parsed by incident_detector.py)
  generate_report(...)       -> formatted report string
  search_incidents(...)      -> raw JSON array string
"""

import os
import base64
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv

from prompts import (
    FRAME_ANALYSIS_PROMPT,
    NL_QUERY_PROMPT,
    REPORT_GENERATION_PROMPT,
    SCENE_DESCRIPTION_PROMPT,
)

load_dotenv()

GEMMA_API_KEY: str = os.getenv("GEMMA_API_KEY", "")
GEMMA_ENDPOINT: str = os.getenv("GEMMA_ENDPOINT", "aistudio")
GEMMA_MODEL: str = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")

_AISTUDIO_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Ngrok inference is slow (~35s per call on T4) — give it plenty of room
_VISION_TIMEOUT = 90.0
_TEXT_TIMEOUT = 60.0
_NGROK_TIMEOUT = 180.0  # /analyze_vision does two model passes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type) for an image file."""
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime_type = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime_type


def _extract_text(response_json: dict) -> str:
    """
    Pull the final answer out of an AI Studio response envelope.

    Thinking models return multiple parts: thinking part(s) with "thought": true
    followed by the actual answer. Always return the last non-thinking part.
    Falls back to the last part if no distinction is present (non-thinking models).
    """
    parts = response_json["candidates"][0]["content"]["parts"]
    for part in reversed(parts):
        if not part.get("thought", False):
            return part["text"]
    return parts[-1]["text"]


# ── AI Studio calls (Phase 1 only) ────────────────────────────────────────────

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
            "temperature": 0.1,
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
            "thinkingConfig": {"thinkingBudget": 0},  # disable thinking for formatting/search tasks
        },
    }
    with httpx.Client(timeout=_TEXT_TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
    return _extract_text(resp.json())


# ── Ngrok calls (Phase 3 — finetuned model on Kaggle T4 via ngrok) ─────────────

def _ngrok_analyze(scene_description: str) -> str:
    """
    POST to /analyze — text-only endpoint.
    Input: plain text scene description.
    Output: structured incident JSON dict.
    """
    payload = {"scene_description": scene_description, "max_new_tokens": 400}
    with httpx.Client(timeout=_NGROK_TIMEOUT) as client:
        resp = client.post(f"{GEMMA_ENDPOINT}/analyze", json=payload)
        resp.raise_for_status()
    result = resp.json()
    result.pop("_inference_time_seconds", None)
    result.pop("_model", None)
    return json.dumps(result)


def _ngrok_analyze_vision(image_path: str) -> str:
    """
    POST to /analyze_vision — full image-to-JSON endpoint.

    The Kaggle notebook handles the two-step flow internally:
      1. Gemma 4 vision: image -> plain text scene description
      2. Finetuned model: scene description -> structured incident JSON

    No AI Studio API key required. Timeout is 180s (two model passes on T4).
    """
    image_data, mime_type = _encode_image(image_path)
    payload = {
        "image_base64": image_data,
        "mime_type": mime_type,
        "max_new_tokens": 400,
    }
    with httpx.Client(timeout=_NGROK_TIMEOUT) as client:
        resp = client.post(f"{GEMMA_ENDPOINT}/analyze_vision", json=payload)
        resp.raise_for_status()
    result = resp.json()
    result.pop("_inference_time_seconds", None)
    result.pop("_model", None)
    return json.dumps(result)


def _extract_report_block(raw: str) -> str:
    """
    Extract just the INCIDENT REPORT block from Gemma's output.

    Thinking models (gemma-4-26b-a4b-it) output their reasoning chain before the
    final answer. Using rfind() grabs the LAST occurrence of the header — which is
    always the actual report, after all the scratchpad.
    """
    marker = "INCIDENT REPORT"
    idx = raw.rfind(marker)
    if idx != -1:
        return raw[idx:].strip()
    return raw.strip()


# ── Local formatting / search (Phase 3 — no API call needed) ─────────────────

def _format_report_local(
    timestamp: str,
    source: str,
    scene_description: str,
    activity: str,
    severity: str,
    category: str,
    confidence: int,
    persons_count: int = 0,
    recommended_action: str = "",
    objects_of_interest: list = None,
) -> str:
    """
    Build a formatted INCIDENT REPORT string from structured incident fields.
    Used in Phase 3 — all data already comes from the finetuned model,
    so a second LLM call for formatting is unnecessary.
    """
    objects = objects_of_interest or []
    objects_str = ", ".join(str(o) for o in objects) if objects else "None detected"
    persons_str = f"{persons_count} person(s) visible" if persons_count > 0 else "None detected"
    action = recommended_action.strip() or "Standard monitoring protocols apply."

    return (
        f"INCIDENT REPORT\n"
        f"===============\n"
        f"Timestamp:          {timestamp}\n"
        f"Source:             {source}\n"
        f"Severity:           {severity}\n"
        f"Category:           {category}\n"
        f"Confidence Score:   {confidence}%\n"
        f"\nDescription:\n{scene_description}\n"
        f"\nActivity Detected:\n{activity}\n"
        f"\nPersons Involved:\n{persons_str}\n"
        f"\nObjects / Evidence:\n{objects_str}\n"
        f"\nRecommended Action:\n{action}\n"
        f"\nReport Generated: {timestamp}"
    )


def _search_local(query: str, incidents_json: str) -> str:
    """
    Keyword-based incident search — no LLM call required.
    Scores each incident by how many query words appear in its text fields.
    Returns the same JSON array schema as the AI Studio NL search.
    """
    query_words = set(query.lower().split())
    if not query_words:
        return "[]"
    try:
        incidents = json.loads(incidents_json)
    except (json.JSONDecodeError, TypeError):
        return "[]"

    matches = []
    for inc in incidents:
        text = " ".join(filter(None, [
            inc.get("scene_description", ""),
            inc.get("activity_detected", ""),
            inc.get("category", ""),
            inc.get("severity", ""),
        ])).lower()

        matched = sum(1 for w in query_words if w in text)
        score = min(100, int(matched / len(query_words) * 100))
        if score > 40:
            matches.append({
                "incident_id": inc.get("id"),
                "timestamp": inc.get("timestamp"),
                "relevance_score": score,
                "reason": f"Matched {matched} of {len(query_words)} query terms in incident record",
            })

    matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    return json.dumps(matches)


# ── Fallback JSON ─────────────────────────────────────────────────────────────

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

    Phase 1 (GEMMA_ENDPOINT=aistudio):
      Single AI Studio vision call — image + JSON prompt -> incident JSON string.

    Phase 3 (GEMMA_ENDPOINT=ngrok URL):
      Single call to /analyze_vision — notebook handles vision + finetuned model
      internally. No AI Studio API key required.

    Returns:
        Raw JSON string matching the FRAME_ANALYSIS_PROMPT schema.
        Never raises — returns a safe fallback on any error.
    """
    print(f"  [Gemma] Analyzing frame: {Path(image_path).name} | endpoint={GEMMA_ENDPOINT}")
    try:
        if GEMMA_ENDPOINT == "aistudio":
            print('using ai stusio')
            raw = _aistudio_vision(image_path, FRAME_ANALYSIS_PROMPT)
        else:
            print(f"  [Gemma] Calling /analyze_vision (vision + finetuned model on Kaggle GPU)...")
            raw = _ngrok_analyze_vision(image_path)

        print(f"  [Gemma] Response received ({len(raw)} chars)")
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
    persons_count: int = 0,
    recommended_action: str = "",
    objects_of_interest: list = None,
) -> str:
    """
    Generate a formatted incident report from structured incident data.

    Phase 1 (GEMMA_ENDPOINT=aistudio):
      Calls AI Studio to generate a narrative report.

    Phase 3 (GEMMA_ENDPOINT=ngrok URL):
      Formats the report locally — all fields already come from the finetuned
      model, so a second LLM call adds no value and costs ~35s of GPU time.

    Returns:
        Formatted INCIDENT REPORT string. Never raises.
    """
    print(f"  [Gemma] Generating report | severity={severity}")
    try:
        if GEMMA_ENDPOINT == "aistudio":
            prompt = REPORT_GENERATION_PROMPT.format(
                timestamp=timestamp,
                source=source,
                scene_description=scene_description,
                activity=activity,
                severity=severity,
                category=category,
                confidence=confidence,
            )
            raw = _aistudio_text(prompt)
            report = _extract_report_block(raw)
        else:
            report = _format_report_local(
                timestamp=timestamp,
                source=source,
                scene_description=scene_description,
                activity=activity,
                severity=severity,
                category=category,
                confidence=confidence,
                persons_count=persons_count,
                recommended_action=recommended_action,
                objects_of_interest=objects_of_interest,
            )

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
    Search incidents against a natural language query.

    Phase 1 (GEMMA_ENDPOINT=aistudio):
      Uses AI Studio to semantically rank incidents against the query.

    Phase 3 (GEMMA_ENDPOINT=ngrok URL):
      Local keyword-based search — fast, no GPU time, no API key needed.
      Returns the same JSON array schema as the AI Studio search.

    Returns:
        Raw JSON array string of matching incidents. Never raises — returns "[]" on error.
    """
    print(f"  [Gemma] Searching incidents | query='{query[:60]}'")
    try:
        if GEMMA_ENDPOINT == "aistudio":
            prompt = NL_QUERY_PROMPT.format(query=query, incidents_json=incidents_json)
            raw = _aistudio_text(prompt)
            print(f"  [Gemma] Search response received ({len(raw)} chars)")
            return raw
        else:
            result = _search_local(query, incidents_json)
            print(f"  [Gemma] Local search complete ({len(json.loads(result))} matches)")
            return result

    except Exception as e:
        print(f"  [Gemma] ERROR in search_incidents: {e}")
        return "[]"


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Smoke-test Gemma calls.

    Usage:
        python gemma_client.py                   # uses first frame in data/frames/
        python gemma_client.py path/to/frame.jpg
    """
    import sys

    print("=" * 60)
    print("HawkWatch — gemma_client.py smoke test")
    print(f"  GEMMA_ENDPOINT : {GEMMA_ENDPOINT}")
    print(f"  GEMMA_MODEL    : {GEMMA_MODEL}")
    if GEMMA_ENDPOINT == "aistudio":
        print(f"  GEMMA_API_KEY  : {'set' if GEMMA_API_KEY else 'NOT SET — check .env'}")
    else:
        print(f"  Mode           : Phase 3 — fully self-hosted on Kaggle GPU (no AI Studio)")
        print(f"  Endpoints      : {GEMMA_ENDPOINT}/analyze_vision")
    print("=" * 60)

    if len(sys.argv) > 1:
        test_frame = sys.argv[1]
    else:
        frames_dir = Path("data/frames")
        candidates = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
        if not candidates:
            print("\nNo frames in data/frames/. Run frame_extractor.py first.")
            print("Or pass a JPEG path: python gemma_client.py frame.jpg")
            sys.exit(1)
        test_frame = str(candidates[0])

    print(f"\nTest frame: {test_frame}")

    print("\n--- Test 1: analyze_frame() ---")
    raw_json = analyze_frame(test_frame)
    print("Raw output:")
    print(raw_json)

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        print("\nWarning: response is not valid JSON.")
        parsed = {
            "scene_description": raw_json[:200],
            "activity_detected": "See raw output",
            "severity": "CLEAR",
            "category": "Normal",
            "confidence": 0,
        }

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
        persons_count=int(parsed.get("persons_count", 0)),
        recommended_action=parsed.get("recommended_action", ""),
        objects_of_interest=parsed.get("objects_of_interest", []),
    )
    print(report)

    print("\n" + "=" * 60)
    print("Smoke test complete.")
