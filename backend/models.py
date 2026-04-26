"""
backend/models.py — Pydantic schemas for all API request and response bodies.

These are the shapes the FastAPI routes accept and return.
They are separate from both the pipeline Incident dataclass and the SQLAlchemy ORM model.
"""

import json
from dataclasses import asdict
from typing import List

from pydantic import BaseModel, field_validator


class IncidentRead(BaseModel):
    """Full incident record returned by the API."""

    id: str
    timestamp: str
    video_source: str
    frame_path: str
    scene_description: str
    activity_detected: str
    persons_count: int
    severity: str
    category: str
    confidence: int
    recommended_action: str
    objects_of_interest: List[str]
    report: str
    alert_sent: bool

    @field_validator("objects_of_interest", mode="before")
    @classmethod
    def parse_objects(cls, v):
        """Accept either a list or a JSON-encoded string (as stored in SQLite)."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return []
        return v or []

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str


class QueryMatch(BaseModel):
    """A single incident that matched a natural language query."""

    incident_id: str
    timestamp: str
    relevance_score: int
    reason: str
    frame_path: str


class QueryResult(BaseModel):
    """Response body for POST /query."""

    query: str
    matches: List[QueryMatch]


def incident_to_read(incident) -> IncidentRead:
    """Convert a pipeline Incident dataclass directly to an IncidentRead schema."""
    return IncidentRead(**asdict(incident))
