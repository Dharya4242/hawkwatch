"""
backend/database.py — SQLAlchemy engine, session, and ORM model for incidents.

Creates a SQLite database at the path set in DATABASE_URL (.env).
Call create_tables() once at app startup to initialise the schema.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/incidents.db")

# Ensure the data/ directory exists before SQLite tries to create the file
Path("data").mkdir(exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class IncidentDB(Base):
    """SQLAlchemy ORM model — mirrors the Incident dataclass from incident_detector.py."""

    __tablename__ = "incidents"

    id = Column(String, primary_key=True)
    timestamp = Column(String, index=True)
    video_source = Column(String)
    frame_path = Column(String)
    scene_description = Column(Text)
    activity_detected = Column(Text)
    persons_count = Column(Integer, default=0)
    severity = Column(String, index=True)
    category = Column(String, index=True)
    confidence = Column(Integer, default=0)
    recommended_action = Column(Text)
    objects_of_interest = Column(Text, default="[]")  # JSON-encoded list
    report = Column(Text, default="")
    alert_sent = Column(Boolean, default=False)


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables. Safe to call repeatedly (uses CREATE IF NOT EXISTS)."""
    Base.metadata.create_all(bind=engine)


def incident_to_db(incident) -> IncidentDB:
    """Convert a pipeline Incident dataclass to an IncidentDB ORM object."""
    return IncidentDB(
        id=incident.id,
        timestamp=incident.timestamp,
        video_source=incident.video_source,
        frame_path=incident.frame_path,
        scene_description=incident.scene_description,
        activity_detected=incident.activity_detected,
        persons_count=incident.persons_count,
        severity=incident.severity,
        category=incident.category,
        confidence=incident.confidence,
        recommended_action=incident.recommended_action,
        objects_of_interest=json.dumps(incident.objects_of_interest),
        report=incident.report,
        alert_sent=incident.alert_sent,
    )
