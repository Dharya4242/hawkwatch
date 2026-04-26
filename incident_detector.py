"""
incident_detector.py — Parses Gemma's raw JSON output into a clean Incident dataclass.

What this does:
- Takes the raw string returned by gemma_client.analyze_frame()
- Handles malformed JSON (Gemma sometimes wraps output in markdown, adds extra text, etc.)
- Validates and normalises severity / category to the allowed enum values
- Returns a typed Incident dataclass ready for report_generator.py and the DB

Never raises — bad input returns a safe fallback Incident so the pipeline keeps running.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


VALID_SEVERITIES = {"CRITICAL", "WARNING", "CLEAR"}
VALID_CATEGORIES = {"Crime", "Medical Emergency", "Suspicious Activity", "Disaster", "Normal"}


@dataclass
class Incident:
    """One detected security incident from a single surveillance frame."""
    id: str                          # uuid4 string
    timestamp: str                   # ISO format e.g. "2026-04-15T14:32:15"
    video_source: str                # video filename or stream URL
    frame_path: str                  # path to the JPEG frame on disk
    scene_description: str           # raw scene text from Gemma vision call
    activity_detected: str           # specific activity string
    persons_count: int               # number of people detected
    severity: str                    # CRITICAL | WARNING | CLEAR
    category: str                    # Crime | Medical Emergency | Suspicious Activity | Disaster | Normal
    confidence: int                  # 0–100
    recommended_action: str          # what security should do right now
    objects_of_interest: list        # notable objects / people
    report: str = ""                 # filled in later by report_generator.py
    alert_sent: bool = False         # set True after Twilio SMS is sent


# ── JSON extraction helpers ────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences Gemma sometimes adds."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _extract_json_block(text: str) -> Optional[str]:
    """Return the first balanced { ... } block found in text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_raw_json(raw: str) -> Optional[dict]:
    """
    Robustly parse Gemma's output into a dict.
    Tries three strategies in order; returns None if all fail.
    """
    # 1. Direct parse (fast path — finetuned model should always hit this)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences then parse
    cleaned = _strip_markdown(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Find the first { ... } block embedded in prose
    block = _extract_json_block(cleaned)
    if block:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

    return None


# ── Field normalisation ────────────────────────────────────────────────────────

def _normalise_severity(value: str) -> str:
    upper = str(value).upper().strip()
    if upper in VALID_SEVERITIES:
        return upper
    if any(k in upper for k in ("CRIT", "URGENT", "IMMEDIATE", "DANGER")):
        return "CRITICAL"
    if any(k in upper for k in ("WARN", "SUSPIC", "ALERT", "CAUTION")):
        return "WARNING"
    return "CLEAR"


def _normalise_category(value: str) -> str:
    lowered = str(value).lower().strip()
    if lowered in {c.lower() for c in VALID_CATEGORIES}:
        # Exact match ignoring case — find the canonical form
        for cat in VALID_CATEGORIES:
            if cat.lower() == lowered:
                return cat
    if any(k in lowered for k in ("crime", "criminal", "theft", "assault", "weapon", "fight", "robbery")):
        return "Crime"
    if any(k in lowered for k in ("medical", "injury", "injur", "unconscious", "health", "ambulance")):
        return "Medical Emergency"
    if any(k in lowered for k in ("suspicious", "loiter", "trespass", "vandal")):
        return "Suspicious Activity"
    if any(k in lowered for k in ("disaster", "fire", "flood", "collapse", "smoke", "explosion")):
        return "Disaster"
    return "Normal"


def _safe_int(value, default: int = 0, lo: int = 0, hi: int = 100) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_gemma_output(
    raw: str,
    frame_path: str,
    video_source: str,
    timestamp: Optional[str] = None,
) -> Incident:
    """
    Parse raw JSON string from gemma_client.analyze_frame() into a typed Incident.

    Args:
        raw:          Raw string from analyze_frame() — may be malformed JSON.
        frame_path:   Path to the JPEG frame that was analyzed.
        video_source: Video filename or stream URL (recorded in the incident).
        timestamp:    ISO timestamp; defaults to now if omitted.

    Returns:
        Always a valid Incident dataclass — never raises.
        Parse failures produce a CLEAR / Normal incident with confidence=0.
    """
    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    incident_id = str(uuid.uuid4())

    data = _parse_raw_json(raw)

    if data is None:
        print(f"  [Detector] WARNING: Could not parse Gemma JSON.")
        print(f"             Raw (first 200 chars): {raw[:200]!r}")
        return Incident(
            id=incident_id,
            timestamp=ts,
            video_source=video_source,
            frame_path=frame_path,
            scene_description=raw[:500] if raw else "No output from Gemma",
            activity_detected="Parse error — manual review required",
            persons_count=0,
            severity="CLEAR",
            category="Normal",
            confidence=0,
            recommended_action="Manual review required — automated analysis failed",
            objects_of_interest=[],
        )

    severity = _normalise_severity(data.get("severity", "CLEAR"))
    category = _normalise_category(data.get("category", "Normal"))

    incident = Incident(
        id=incident_id,
        timestamp=ts,
        video_source=video_source,
        frame_path=frame_path,
        scene_description=str(data.get("scene_description", "")),
        activity_detected=str(data.get("activity_detected", "")),
        persons_count=_safe_int(data.get("persons_count", 0), lo=0, hi=999),
        severity=severity,
        category=category,
        confidence=_safe_int(data.get("confidence", 0)),
        recommended_action=str(data.get("recommended_action", "Monitor situation")),
        objects_of_interest=list(data.get("objects_of_interest", [])),
    )

    print(
        f"  [Detector] Incident parsed | severity={incident.severity}"
        f" | category={incident.category} | confidence={incident.confidence}%"
        f" | persons={incident.persons_count}"
    )
    return incident


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Smoke-test the parser against several input shapes.

    Usage:
        python incident_detector.py
    """
    print("=" * 60)
    print("HawkWatch — incident_detector.py smoke test")
    print("=" * 60)

    test_cases = [
        (
            "Valid clean JSON",
            json.dumps({
                "scene_description": "Two individuals near a rear entrance at night.",
                "activity_detected": "Loitering near restricted access door",
                "persons_count": 2,
                "severity": "WARNING",
                "category": "Suspicious Activity",
                "confidence": 78,
                "recommended_action": "Dispatch security to rear entrance",
                "objects_of_interest": ["dark clothing", "backpack"],
            }),
        ),
        (
            "Markdown-fenced JSON",
            '```json\n{"scene_description":"Person running through corridor",'
            '"activity_detected":"Fast movement away from alarm",'
            '"persons_count":1,"severity":"CRITICAL","category":"Crime","confidence":91,'
            '"recommended_action":"Intercept immediately","objects_of_interest":["person in hoodie"]}\n```',
        ),
        (
            "JSON buried in prose",
            'Here is my analysis of the frame: {"scene_description":"Empty parking lot at night",'
            '"activity_detected":"No activity detected","persons_count":0,'
            '"severity":"CLEAR","category":"Normal","confidence":99,'
            '"recommended_action":"No action needed","objects_of_interest":[]} End of analysis.',
        ),
        (
            "Unknown severity and category values",
            json.dumps({
                "scene_description": "Person collapsed on the ground near entrance.",
                "activity_detected": "Possible medical event",
                "persons_count": 1,
                "severity": "URGENT",
                "category": "health emergency",
                "confidence": 85,
                "recommended_action": "Call ambulance immediately",
                "objects_of_interest": ["person lying down"],
            }),
        ),
        (
            "Completely malformed — free text",
            "The scene appears to be a normal office environment. No threats detected.",
        ),
    ]

    all_passed = True
    for name, raw_input in test_cases:
        print(f"\n--- {name} ---")
        incident = parse_gemma_output(
            raw=raw_input,
            frame_path="data/frames/test_frame.jpg",
            video_source="test_video.mp4",
        )
        assert incident.severity in VALID_SEVERITIES, f"Bad severity: {incident.severity}"
        assert incident.category in VALID_CATEGORIES, f"Bad category: {incident.category}"
        assert 0 <= incident.confidence <= 100, f"Bad confidence: {incident.confidence}"
        print(f"  severity    : {incident.severity}")
        print(f"  category    : {incident.category}")
        print(f"  confidence  : {incident.confidence}%")
        print(f"  description : {incident.scene_description[:80]}")
        print(f"  PASS")

    print("\n" + "=" * 60)
    print("All cases passed." if all_passed else "Some cases failed.")
