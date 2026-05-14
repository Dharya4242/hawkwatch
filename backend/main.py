"""
backend/main.py — FastAPI application entry point.

Mounts all route modules and initialises the database on startup.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    POST /upload              — upload MP4, run pipeline, get incidents
    POST /stream              — analyze stream URL, get incidents
    GET  /incidents           — list all incidents (filterable)
    GET  /incidents/{id}      — get one incident by UUID
    POST /query               — natural language search across incidents
    GET  /health              — liveness check
    GET  /docs                — Swagger UI (auto-generated)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.database import create_tables
from backend.routes import incidents, query, stream, upload

app = FastAPI(
    title="SecureSight API",
    description=(
        "Open-source AI video surveillance and disaster response platform. "
        "Powered by finetuned Gemma 4."
    ),
    version="1.0.0",
)

# Allow all origins — fine for hackathon / local demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted frames as static files so the frontend can display them
frames_dir = Path("data/frames")
frames_dir.mkdir(parents=True, exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(frames_dir)), name="frames")

# Register route modules
app.include_router(upload.router, tags=["Pipeline"])
app.include_router(stream.router, tags=["Pipeline"])
app.include_router(incidents.router, tags=["Incidents"])
app.include_router(query.router, tags=["Search"])


@app.on_event("startup")
def on_startup():
    """Create database tables on first run."""
    create_tables()
    print("[SecureSight] Database ready.")
    print("[SecureSight] API running — visit http://localhost:8000/docs")


@app.get("/health", tags=["Meta"])
def health():
    """Liveness check — returns OK if the server is up."""
    return {"status": "ok", "service": "SecureSight"}
