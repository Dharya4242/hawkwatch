"""
report_generator.py — Generates a formatted incident report from a parsed Incident.

What this does:
- Takes an Incident dataclass (from incident_detector.py)
- Calls gemma_client.generate_report() with the structured fields
- Writes the formatted report string back onto incident.report
- Returns the updated Incident

This is the last step before saving to DB and triggering alerts.
"""

from incident_detector import Incident
import gemma_client


def generate_report(incident: Incident) -> Incident:
    """
    Call Gemma to produce a formatted INCIDENT REPORT and attach it to the incident.

    Args:
        incident: Parsed Incident from incident_detector.parse_gemma_output().
                  incident.report should be "" at this point.

    Returns:
        The same Incident object with incident.report filled in.
    """
    print(f"  [Reporter] Generating report | id={incident.id[:8]}... | severity={incident.severity}")

    report = gemma_client.generate_report(
        timestamp=incident.timestamp,
        source=incident.video_source,
        scene_description=incident.scene_description,
        activity=incident.activity_detected,
        severity=incident.severity,
        category=incident.category,
        confidence=incident.confidence,
        persons_count=incident.persons_count,
        recommended_action=incident.recommended_action,
        objects_of_interest=incident.objects_of_interest,
    )

    incident.report = report
    print(f"  [Reporter] Report attached ({len(report)} chars)")
    return incident


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Smoke-test report generation using a hand-crafted Incident.
    Requires GEMMA_API_KEY in .env.

    Usage:
        python report_generator.py
    """
    import uuid
    from datetime import datetime

    print("=" * 60)
    print("SecureSight — report_generator.py smoke test")
    print("=" * 60)

    # Build a realistic test incident (as if incident_detector.py produced it)
    test_incident = Incident(
        id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        video_source="testing/videos/10.mp4",
        frame_path="data/frames/frame_0000003000ms.jpg",
        scene_description=(
            "Two individuals are standing near a rear entrance in low light. "
            "One is wearing dark clothing and carrying a backpack. "
            "They appear to be testing the door handle repeatedly."
        ),
        activity_detected="Loitering and attempting entry at restricted access door",
        persons_count=2,
        severity="WARNING",
        category="Suspicious Activity",
        confidence=78,
        recommended_action="Dispatch security to rear entrance immediately",
        objects_of_interest=["dark clothing", "backpack", "restricted door"],
    )

    print(f"\nTest incident:")
    print(f"  severity : {test_incident.severity}")
    print(f"  category : {test_incident.category}")
    print(f"  activity : {test_incident.activity_detected}")
    print(f"\nCalling generate_report()...")

    updated = generate_report(test_incident)

    print("\n" + "=" * 60)
    print("Generated Report:")
    print("=" * 60)
    print(updated.report)
    print("=" * 60)
    print(f"\nincident.report is set: {bool(updated.report)}")
    print("Smoke test complete.")
