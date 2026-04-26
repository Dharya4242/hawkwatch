"""
backend/routes/query.py — POST /query

Natural language search across the incident library.
Fetches all incidents from SQLite, sends them + the user's query to Gemma,
and returns ranked matches with frame paths attached.
"""

import json
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import gemma_client
from backend.database import IncidentDB, get_db
from backend.models import QueryMatch, QueryRequest, QueryResult

router = APIRouter()


def _parse_matches_json(raw: str) -> list[dict]:
    """
    Robustly extract the JSON array from Gemma's response.
    Handles markdown fences and JSON buried in prose.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()

    # Try direct parse first
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        pass

    # Find first [ ... ] array block
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    return []


@router.post("/query", response_model=QueryResult)
def query_incidents(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Search incidents using natural language.

    Gemma ranks all stored incidents against the query and returns
    the most relevant matches with relevance scores and reasons.

    Example queries:
      "find a person waving for help near debris"
      "suspicious activity near entrance at night"
      "medical emergency in the last hour"
    """
    if not request.query.strip():
        raise HTTPException(400, "query must not be empty.")

    all_incidents = db.query(IncidentDB).order_by(IncidentDB.timestamp.desc()).all()

    if not all_incidents:
        return QueryResult(query=request.query, matches=[])

    # Build a compact incident list for the prompt (keep tokens low)
    incidents_for_prompt = [
        {
            "id": inc.id,
            "timestamp": inc.timestamp,
            "scene_description": inc.scene_description[:300],
            "activity_detected": inc.activity_detected,
            "category": inc.category,
            "severity": inc.severity,
        }
        for inc in all_incidents
    ]

    raw = gemma_client.search_incidents(
        query=request.query,
        incidents_json=json.dumps(incidents_for_prompt, indent=2),
    )

    matches_data = _parse_matches_json(raw)

    # Build lookup tables for enrichment
    id_to_row = {inc.id: inc for inc in all_incidents}

    matches = []
    for m in matches_data:
        iid = m.get("incident_id", "")
        row = id_to_row.get(iid)
        if not row:
            continue
        matches.append(
            QueryMatch(
                incident_id=iid,
                timestamp=row.timestamp,
                relevance_score=int(m.get("relevance_score", 0)),
                reason=str(m.get("reason", "")),
                frame_path=row.frame_path,
            )
        )

    # Sort by relevance descending (Gemma should already do this, but enforce it)
    matches.sort(key=lambda x: x.relevance_score, reverse=True)

    return QueryResult(query=request.query, matches=matches)
