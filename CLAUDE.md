# HawkWatch Gemma Edition — Claude Code Context

## What This Project Is
An AI-powered video surveillance and disaster response platform built for the
Kaggle Gemma 4 Good Hackathon. It analyzes video footage using Google Gemma 4
(multimodal), detects incidents, generates structured reports, and supports
natural language querying of footage.

## Team
- 2 people
- No local GPU — using Kaggle free tier + Google AI Studio free tier
- Stack comfort: Python (primary), will learn what is needed

---

## Project Structure
```
hawkwatch/
├── CLAUDE.md                  ← you are here
├── README.md
├── requirements.txt
├── .env.example               ← API keys template (never commit .env)
│
├── pipeline/                  ← Core AI pipeline (build this first)
│   ├── frame_extractor.py     ← OpenCV: video → frames with motion detection
│   ├── gemma_client.py        ← Gemma 4 API calls (vision + text)
│   ├── incident_detector.py   ← Parse Gemma output → structured incident
│   └── report_generator.py   ← Generate formatted incident reports
│
├── backend/                   ← FastAPI server
│   ├── main.py                ← FastAPI app entry point
│   ├── routes/
│   │   ├── upload.py          ← POST /upload — accept MP4
│   │   ├── stream.py          ← POST /stream — accept stream URL
│   │   ├── query.py           ← POST /query — natural language search
│   │   └── incidents.py       ← GET /incidents — fetch incident library
│   ├── models.py              ← Pydantic models / DB schemas
│   └── database.py            ← SQLite setup with SQLAlchemy
│
├── frontend/                  ← Streamlit UI
│   ├── app.py                 ← Main Streamlit entry point
│   └── pages/
│       ├── live_feed.py       ← Live stream viewer + analysis
│       ├── upload.py          ← MP4 upload page
│       ├── query.py           ← Natural language query interface
│       └── library.py         ← Incident library browser
│
├── models/                    ← Finetuning related
│   ├── finetune.py            ← Unsloth finetuning script
│   ├── prepare_dataset.py     ← UCF-Crime → training pairs
│   └── prompts.py             ← All prompt templates in one place
│
├── data/
│   ├── sample_videos/         ← Put test MP4s here
│   ├── frames/                ← Extracted frames (temp, gitignored)
│   └── incidents.db           ← SQLite database (gitignored)
│
├── notebooks/                 ← Kaggle notebooks
│   ├── 01_test_gemma_api.ipynb
│   ├── 02_dataset_prep.ipynb
│   └── 03_finetune_unsloth.ipynb
│
└── scripts/
    └── test_pipeline.py       ← Quick end-to-end test script
```

---

## Build Order (follow this exactly)
```
Phase 1 — Pipeline (no server, no UI, just Python scripts)
  Step 1: pipeline/frame_extractor.py    ← start here, needs only OpenCV
  Step 2: pipeline/gemma_client.py       ← Gemma API wrapper
  Step 3: pipeline/incident_detector.py  ← parse Gemma JSON output
  Step 4: pipeline/report_generator.py  ← format final report
  Step 5: scripts/test_pipeline.py       ← glue all 4 together, test on sample video

Phase 2 — Backend
  Step 6:  backend/database.py           ← SQLite + SQLAlchemy setup
  Step 7:  backend/models.py             ← Pydantic schemas
  Step 8:  backend/routes/upload.py      ← MP4 upload endpoint
  Step 9:  backend/routes/incidents.py   ← fetch incidents
  Step 10: backend/routes/query.py       ← NL query endpoint
  Step 11: backend/main.py               ← wire everything together

Phase 3 — Frontend
  Step 12: frontend/app.py               ← Streamlit skeleton
  Step 13: frontend/pages/upload.py      ← upload page
  Step 14: frontend/pages/library.py     ← incident library
  Step 15: frontend/pages/query.py       ← query interface

Phase 4 — Finetuning (run on Kaggle, separate from main app)
  Step 16: models/prompts.py             ← centralise all prompts
  Step 17: models/prepare_dataset.py     ← UCF-Crime → training data
  Step 18: models/finetune.py            ← Unsloth training script
```

---

## Tech Stack
- **Python 3.10+**
- **OpenCV** — frame extraction, motion detection
- **FastAPI** — backend API server
- **SQLAlchemy + SQLite** — database (incidents, reports)
- **Streamlit** — frontend UI
- **Pydantic** — data validation
- **httpx** — async HTTP client for Gemma API calls
- **Twilio** — SMS alerts (add in Phase 2)
- **Unsloth** — finetuning (Kaggle only, not in main app)
- **python-dotenv** — environment variable management

---

## Environment Variables
```
# .env (never commit this file)
GEMMA_API_KEY=           # Google AI Studio API key
TWILIO_ACCOUNT_SID=      # Twilio (for SMS alerts)
TWILIO_AUTH_TOKEN=       # Twilio
TWILIO_PHONE_NUMBER=     # Twilio sender number
ALERT_PHONE_NUMBER=      # Number to send alerts to
DATABASE_URL=sqlite:///./data/incidents.db
FRAME_INTERVAL_SECONDS=3       # extract 1 frame every N seconds
MOTION_THRESHOLD=500           # pixel diff to trigger motion detection
SEVERITY_ALERT_THRESHOLD=CRITICAL  # only alert on CRITICAL or above
```

---

## Gemma 4 API — How We Call It

### Vision Call (frame analysis)
```python
# Send a frame image to Gemma 4, get back structured JSON
# Endpoint: Google AI Studio (free tier) or Kaggle Models API
# Model: gemma-4 multimodal (check latest model name on AI Studio)
# Input: base64 encoded image + text prompt
# Output: JSON with severity, category, description, confidence
```

### Text Call (report generation)
```python
# Send incident description to Gemma 4 text, get back formatted report
# Same API, no image, just text prompt
# Output: formatted INCIDENT REPORT string
```

### All Prompts Live In: models/prompts.py
Never hardcode prompts in other files. Always import from prompts.py.

---

## Key Data Structures

### Incident (what we save to DB)
```python
{
  "id": "uuid",
  "timestamp": "2026-04-15T14:32:15",
  "video_source": "mall_entrance.mp4",
  "frame_path": "data/frames/frame_14321500.jpg",
  "scene_description": "raw Gemma vision output",
  "severity": "CRITICAL | WARNING | CLEAR",
  "category": "Crime | Medical | Suspicious | Disaster | Normal",
  "confidence": 87,
  "report": "full formatted report text",
  "recommended_action": "Dispatch security immediately",
  "alert_sent": True
}
```

### Query Result
```python
{
  "query": "person waving for help",
  "matches": [
    {
      "incident_id": "uuid",
      "timestamp": "...",
      "relevance_score": 92,
      "reason": "Person detected waving arms near debris",
      "frame_path": "..."
    }
  ]
}
```

---

## Coding Rules for This Project
1. **Every file must have a module docstring** explaining what it does
2. **No hardcoded API keys** — always use os.getenv()
3. **All Gemma responses must be try/caught** — API can fail, pipeline must not crash
4. **Prompts only in models/prompts.py** — never inline
5. **Type hints on all functions**
6. **Each pipeline step must be independently testable** — no tight coupling
7. **Print progress to console** — this is a hackathon, visibility > clean logs

---

## Current Status
- [ ] Phase 1 — Pipeline (NOT STARTED)
- [ ] Phase 2 — Backend (NOT STARTED)
- [ ] Phase 3 — Frontend (NOT STARTED)
- [ ] Phase 4 — Finetuning (NOT STARTED)

## What To Build Next
**Start with: pipeline/frame_extractor.py**
- Input: path to any MP4 file
- Output: saves JPEG frames to data/frames/ folder
- Feature: motion detection (only save frames where something moved)
- No API keys needed, no GPU needed, testable immediately
