"""
backend/routes/incidents.py — GET /incidents, GET /incidents/{id}

Read endpoints for the incident library.
Supports filtering by severity and category via query parameters.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import IncidentDB, get_db
from backend.models import IncidentRead

router = APIRouter()


@router.get("/incidents", response_model=List[IncidentRead])
def list_incidents(
    severity: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    Return all incidents, newest first.

    Optional filters:
      severity — CRITICAL | WARNING | CLEAR
      category — Crime | Medical Emergency | Suspicious Activity | Disaster | Normal
      limit    — max records to return (default 100)
      offset   — pagination offset (default 0)
    """
    query = db.query(IncidentDB)

    if severity:
        query = query.filter(IncidentDB.severity == severity.upper())
    if category:
        query = query.filter(IncidentDB.category == category)

    rows = query.order_by(IncidentDB.timestamp.desc()).offset(offset).limit(limit).all()
    return [IncidentRead.model_validate(row) for row in rows]


@router.get("/incidents/{incident_id}", response_model=IncidentRead)
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    """Return a single incident by its UUID."""
    row = db.query(IncidentDB).filter(IncidentDB.id == incident_id).first()
    if not row:
        raise HTTPException(404, f"Incident '{incident_id}' not found.")
    return IncidentRead.model_validate(row)
