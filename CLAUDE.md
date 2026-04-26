# HawkWatch Gemma Edition — Claude Code Context
> Hackathon: Kaggle × Google DeepMind — Gemma 4 Good Hackathon
> Deadline: May 18, 2026 at 11:59 PM UTC
> Prize Pool: $200,000 | Special mention track for Unsloth
> Team: 2 people | No local GPU

---

## What This Project Is

An open-source AI-powered video surveillance and disaster response platform
that helps underserved communities and disaster response teams monitor footage
using natural language — without expensive proprietary systems.

Core idea: security personnel describe what they are looking for in plain
English ("find a person waving for help") and the system finds it across
live or recorded drone/CCTV footage, then auto-generates structured incident
reports and sends real-time alerts.

Built entirely on Gemma 4 (open source) — finetuned with Unsloth on
surveillance-specific data so it outputs structured security reports instead
of generic descriptions.

---

## Hackathon Requirements (source of truth)

From the official Kaggle competition page:

SUBMISSION MUST INCLUDE:
  1. Working demo (the actual app running)
  2. Public GitHub repository (clean, documented code)
  3. Technical write-up (how Gemma 4 is used)
  4. Short demo video (2 min max, shows real usage)

JUDGING CRITERIA:
  - Real-world impact (not theoretical, not fake UI)
  - Technical execution (working prototype)
  - Clear use case communication
  - Solutions that work in low-bandwidth / privacy-sensitive environments

SPECIAL MENTION TRACKS:
  - Global Resilience / Climate / Disaster Response  <- our project fits here
  - Unsloth technology                               <- our finetuning fits here
  - Ollama technology                                <- mention as future work

FOCUS AREAS: health, education, climate/global resilience

OUR ANGLE:
  "Open-source disaster response intelligence for underserved communities
   that cannot afford proprietary CCTV systems — powered by finetuned Gemma 4"

---

## How Gemma Is Used — The Correct Mental Model

THIS IS IMPORTANT. Read carefully before touching any Gemma-related code.

### The 3 Phases of Gemma Usage

```
PHASE 1 — Development & Testing (temporary)
  Use: Google AI Studio free API (base Gemma 4)
  Why: Test that pipeline works before finetuning exists
  Code: pipeline/gemma_client.py points to AI Studio
  This is throwaway — not in final product

PHASE 2 — Dataset Generation (one-time job, run on Kaggle)
  Use: Google AI Studio free API (base Gemma 4)
  Why: Auto-label UCF-Crime frames to create training data
  Input: surveillance video frame (image)
  Output: scene description + structured incident report
  Save these (description, report) pairs as your training dataset
  Code: models/prepare_dataset.py

PHASE 3 — Final Product (what judges see)
  Use: YOUR finetuned Gemma 4 model (hosted on HuggingFace)
  Why: Custom model outputs structured security reports directly
  Running: Kaggle notebook with GPU acts as inference server
  App talks to Kaggle notebook via ngrok tunnel
  Google AI Studio API is NOT used at all in final product
```

### Why We Finetune

Base Gemma 4 output (generic):
  "Two people are standing near a building. One appears to be holding something."

Finetuned Gemma 4 output (what we need):
  INCIDENT REPORT
  Severity:    WARNING
  Category:    Suspicious Activity
  Confidence:  78%
  Description: Two individuals loitering near rear entrance.
               One suspect holding unidentified object.
  Action:      Monitor closely. Dispatch if behavior escalates.

Finetuning teaches Gemma the exact output format and security
terminology we need. Without it every response needs heavy
post-processing. With it the pipeline is clean and reliable.

---

## Full System Architecture

```
[Video Input]
      |
      |-- MP4 file upload
      |-- Live stream URL (RTSP / YouTube)
            |
            v
[Frame Extractor]                     <- OpenCV, local machine, no GPU
  - Screenshot every 3 seconds
  - Skip if no motion detected
  - Save JPEG to data/frames/
            |
            v
[Finetuned Gemma 4]                   <- Running on Kaggle GPU via ngrok
  - Input: frame image + prompt
  - Output: structured JSON incident
            |
            v
[Incident Detector]                   <- Parse + validate Gemma JSON output
  - Severity: CRITICAL / WARNING / CLEAR
  - Category: Crime / Medical / Disaster / Suspicious / Normal
  - Confidence score
            |
            v
[Report Generator]                    <- Finetuned Gemma text call
  - Formats full incident report
  - Saves to SQLite DB with frame path
            |
      ------+------
      |           |
      v           v
[Alert System]  [Incident Library]    <- Twilio SMS + SQLite
  SMS to           All incidents
  security         searchable
            |
            v
[NL Query Engine]                     <- Finetuned Gemma text call
  User types query -> matches incidents
  Returns timestamps + frame previews
            |
            v
[Streamlit Frontend]
  - Live feed viewer
  - MP4 upload page
  - Query interface
  - Incident library
```

---

## Project Structure

```
hawkwatch/
|-- CLAUDE.md                    <- you are here (read first always)
|-- README.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
|
|-- pipeline/                    <- Core AI pipeline (build Phase 1 first)
|   |-- frame_extractor.py       DONE - OpenCV frame extraction + motion detection
|   |-- gemma_client.py          <- Gemma inference wrapper (Phase 1: AI Studio API,
|   |                               Phase 3: swap to finetuned model endpoint)
|   |-- incident_detector.py     <- Parse + validate Gemma JSON -> Incident object
|   |-- report_generator.py      <- Format final human-readable report
|
|-- backend/                     <- FastAPI server
|   |-- main.py                  <- App entry point, mounts all routes
|   |-- database.py              <- SQLAlchemy + SQLite setup
|   |-- models.py                <- Pydantic schemas (Incident, QueryResult etc.)
|   |-- routes/
|       |-- upload.py            <- POST /upload - accept MP4, run pipeline
|       |-- stream.py            <- POST /stream - accept URL, run pipeline
|       |-- query.py             <- POST /query - NL search across incidents
|       |-- incidents.py         <- GET /incidents - fetch incident library
|
|-- frontend/                    <- Streamlit UI
|   |-- app.py                   <- Entry point, navigation
|   |-- pages/
|       |-- live_feed.py         <- Live stream viewer + real-time analysis
|       |-- upload.py            <- MP4 upload + analysis page
|       |-- query.py             <- Natural language query interface
|       |-- library.py           <- Incident library with filters
|
|-- models/                      <- Finetuning (runs on Kaggle, not in main app)
|   |-- prompts.py               DONE - all prompt templates centralised here
|   |-- prepare_dataset.py       <- UCF-Crime frames -> (description, report) pairs
|   |-- finetune.py              <- Unsloth LoRA finetuning script
|
|-- notebooks/                   <- Kaggle notebooks (upload and run on Kaggle)
|   |-- 01_test_gemma_vision.ipynb    <- test base Gemma 4 on sample frames
|   |-- 02_prepare_dataset.ipynb      <- generate training data from UCF-Crime
|   |-- 03_finetune_unsloth.ipynb     <- run finetuning, push to HuggingFace
|   |-- 04_serve_model.ipynb          <- load finetuned model + expose via ngrok
|
|-- data/
|   |-- sample_videos/           <- put test MP4s here (gitignored)
|   |-- frames/                  <- extracted frames, temp (gitignored)
|   |-- incidents.db             <- SQLite DB (gitignored)
|
|-- scripts/
    |-- test_pipeline.py         <- end-to-end test: video -> report (no UI)
```

---

## Build Order (follow exactly, do not skip steps)

```
PHASE 1 — Pipeline (no server, no UI, pure Python, test immediately)
------------------------------------------------------------------
Step 1: pipeline/frame_extractor.py     DONE
        Test: python pipeline/frame_extractor.py data/sample_videos/test.mp4

Step 2: pipeline/gemma_client.py
        - Wraps Google AI Studio API (temporary, for testing only)
        - Two methods: analyze_frame(image_path) and generate_report(description)
        - Returns raw JSON string from Gemma
        - Must be swappable later (see GEMMA_ENDPOINT in .env)
        Test: python pipeline/gemma_client.py (sends one test frame)

Step 3: pipeline/incident_detector.py
        - Takes raw JSON string from gemma_client
        - Parses and validates into clean Incident dataclass
        - Handles malformed JSON gracefully (Gemma sometimes returns bad JSON)
        Test: python pipeline/incident_detector.py (parses sample JSON)

Step 4: pipeline/report_generator.py
        - Takes Incident object
        - Calls gemma_client.generate_report()
        - Returns formatted report string
        Test: python pipeline/report_generator.py

Step 5: scripts/test_pipeline.py
        - Full end-to-end: MP4 -> frames -> Gemma -> incident -> report
        - Prints everything to console, saves to data/test_report.txt
        Test: python scripts/test_pipeline.py data/sample_videos/test.mp4

PHASE 2 — Backend
------------------------------------------------------------------
Step 6:  backend/database.py       SQLite + SQLAlchemy, Incident table
Step 7:  backend/models.py         Pydantic: IncidentCreate, IncidentRead, QueryRequest
Step 8:  backend/routes/upload.py  POST /upload - receive MP4, run pipeline, save to DB
Step 9:  backend/routes/incidents.py GET /incidents - list, filter, get by id
Step 10: backend/routes/query.py   POST /query - NL search, call Gemma, return matches
Step 11: backend/main.py           Wire everything, test with curl

PHASE 3 — Frontend
------------------------------------------------------------------
Step 12: frontend/app.py           Streamlit skeleton, sidebar navigation
Step 13: frontend/pages/upload.py  Upload MP4, show progress, show report
Step 14: frontend/pages/library.py Table of all incidents, click to expand
Step 15: frontend/pages/query.py   Text input, show matching frames + reports

PHASE 4 — Finetuning (Kaggle notebooks, run in parallel with Phase 2-3)
------------------------------------------------------------------
Step 16: notebooks/01 - confirm base Gemma 4 vision works on sample frames
Step 17: models/prepare_dataset.py - UCF-Crime -> labeled training pairs
Step 18: notebooks/02 - run dataset prep on Kaggle GPU
Step 19: models/finetune.py - Unsloth LoRA training script
Step 20: notebooks/03 - run training on Kaggle, push model to HuggingFace
Step 21: notebooks/04 - load finetuned model, expose via ngrok
Step 22: pipeline/gemma_client.py - swap GEMMA_ENDPOINT to ngrok URL in .env
```

---

## The Swap — How Phase 1 Becomes Phase 3

This is the key design decision. gemma_client.py supports both phases
by reading GEMMA_ENDPOINT from .env:

```
Phase 1 (testing):    GEMMA_ENDPOINT=aistudio
Phase 3 (final demo): GEMMA_ENDPOINT=https://xxxx.ngrok.io

gemma_client.py checks this env var and calls the right place.
Everything else in the codebase stays exactly the same.
This is the ONLY file + .env that changes between phases.
```

---

## Finetuning Data Flow

```
UCF-Crime dataset (1900 video clips, 13 crime categories)
          |
          v  models/prepare_dataset.py
Extract 1 frame per 5 seconds using OpenCV
          |
          v  Google AI Studio API (base Gemma 4, free tier)
Prompt: "Describe this surveillance frame in detail."
          |
          v
Save: { "frame_path": "...", "scene_description": "..." }
          |
          v  second Gemma call
Prompt: "Given this scene, write a structured security incident report"
          |
          v
Save: { "scene_description": "...", "incident_report": "..." }
          |
          v
Training dataset: 2000-5000 (description, report) pairs saved as JSONL
          |
          v  models/finetune.py on Kaggle GPU (Unsloth + LoRA)
Finetune Gemma 4 — 3 epochs, 4-bit quantization, gradient checkpointing
          |
          v
Push to HuggingFace: "your-username/hawkwatch-gemma4"
          |
          v  notebooks/04_serve_model.ipynb on Kaggle GPU
Load model + expose inference endpoint via ngrok
          |
          v
Set GEMMA_ENDPOINT=https://xxxx.ngrok.io in .env
App now uses your finetuned model
```

---

## GPU and Compute Plan

```
Google AI Studio free tier   -> Phase 1 testing + dataset labeling (API calls only)
Kaggle Account 1 (30hr/wk)   -> Main dev, integration testing, demo day serving
Kaggle Account 2 (30hr/wk)   -> Finetuning runs (Phase 4)
Vast.ai / RunPod (paid)       -> Backup if Kaggle hours run out (~$1.50 for 4hrs)
```

What needs GPU vs what does not:
```
Frame extraction (OpenCV)   -> NO  -> local machine
FastAPI backend              -> NO  -> local machine
Streamlit frontend           -> NO  -> local machine
Dataset generation (API)     -> NO  -> API calls to AI Studio
Finetuning (Unsloth)         -> YES -> Kaggle Account 2
Serving finetuned model      -> YES -> Kaggle Account 1 (demo day)
```

---

## Environment Variables

```bash
# .env - never commit this file

# Phase 1 + dataset generation
GEMMA_API_KEY=your_google_ai_studio_key

# Swap this to your ngrok URL after finetuning is done
# Phase 1:  GEMMA_ENDPOINT=aistudio
# Phase 3:  GEMMA_ENDPOINT=https://xxxx.ngrok.io
GEMMA_ENDPOINT=aistudio

# Finetuned model ID (set after finetuning)
HUGGINGFACE_MODEL_ID=your-username/hawkwatch-gemma4

# Alerts
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=+1234567890
ALERT_PHONE_NUMBER=+1234567890

# App config
DATABASE_URL=sqlite:///./data/incidents.db
FRAME_INTERVAL_SECONDS=3
MOTION_THRESHOLD=500
SEVERITY_ALERT_THRESHOLD=CRITICAL
```

---

## Prompts — All in models/prompts.py (DONE)

Never hardcode prompts anywhere else. Always import from prompts.py.

FRAME_ANALYSIS_PROMPT     - vision call, asks Gemma to return JSON
REPORT_GENERATION_PROMPT  - text call, formats full incident report
NL_QUERY_PROMPT            - text call, ranked search across incidents
FINETUNING_PROMPT_TEMPLATE - used only during Unsloth training

---

## Key Data Structures

Incident (saved to SQLite):
```python
{
  "id": "uuid4 string",
  "timestamp": "2026-04-15T14:32:15",
  "video_source": "mall_entrance.mp4",
  "frame_path": "data/frames/frame_0143200000ms.jpg",
  "scene_description": "raw text from Gemma vision call",
  "severity": "CRITICAL | WARNING | CLEAR",
  "category": "Crime | Medical Emergency | Suspicious Activity | Disaster | Normal",
  "confidence": 87,
  "report": "full formatted INCIDENT REPORT string",
  "recommended_action": "Dispatch security to east entrance immediately",
  "alert_sent": True
}
```

QueryResult:
```python
{
  "query": "person waving for help near debris",
  "matches": [
    {
      "incident_id": "uuid4 string",
      "timestamp": "2026-04-15T14:32:15",
      "relevance_score": 92,
      "reason": "Person detected waving arms near collapsed structure",
      "frame_path": "data/frames/frame_0143200000ms.jpg"
    }
  ]
}
```

---

## Coding Rules

1. Every file has a module docstring explaining what it does
2. No hardcoded API keys — always os.getenv()
3. All Gemma calls wrapped in try/except — pipeline must never crash on bad JSON
4. Prompts only in models/prompts.py — never inline anywhere else
5. Type hints on all functions
6. Each pipeline step independently testable via if __name__ == "__main__"
7. Print progress to console — hackathon, visibility matters
8. gemma_client.py + .env are the ONLY things that change between Phase 1 and 3

---

## Current Build Status

- [x] Phase 1 Step 1 — pipeline/frame_extractor.py       DONE
- [x] models/prompts.py                                   DONE
- [x] Phase 1 Step 2 — pipeline/gemma_client.py          DONE
- [ ] Phase 1 Step 3 — pipeline/incident_detector.py     TODO
- [ ] Phase 1 Step 4 — pipeline/report_generator.py      TODO
- [ ] Phase 1 Step 5 — scripts/test_pipeline.py          TODO
- [ ] Phase 2 — Backend                                   NOT STARTED
- [ ] Phase 3 — Frontend                                  NOT STARTED
- [ ] Phase 4 — Finetuning                                NOT STARTED

---

## Submission Checklist

CODE:
  [ ] All pipeline steps working end-to-end
  [ ] FastAPI backend running
  [ ] Streamlit frontend working
  [ ] Finetuned model on HuggingFace Hub
  [ ] Clean public GitHub repo with README

DEMO VIDEO (2 min target):
  [ ] Show MP4 upload -> analysis -> incident report generated
  [ ] Show natural language query returning matching frames
  [ ] Show SMS alert being sent
  [ ] Show incident library with saved reports
  [ ] THE WOW MOMENT: type "find person waving for help" -> instant result

WRITE-UP NARRATIVE:
  [ ] Problem: underserved communities, expensive proprietary systems
  [ ] Solution: open-source Gemma 4, self-hostable
  [ ] Technical: finetuning with Unsloth (hits special mention track)
  [ ] Impact: disaster responders, public safety, reduced manual monitoring
  [ ] Future: edge deployment via Ollama for fully offline use

---

## Context Block — Paste This When Starting Any New Claude Session

"I am building HawkWatch Gemma Edition for the Kaggle Gemma 4 Good Hackathon
(deadline May 18 2026, $200K prize). Open-source AI video surveillance and
disaster response platform for underserved communities.

Core flow: OpenCV extracts frames from video -> finetuned Gemma 4 analyzes
frames -> structured incident reports generated -> SQLite saves incidents ->
Streamlit UI shows everything -> Twilio sends SMS alerts.

Gemma usage: Phase 1 uses Google AI Studio free API for testing. Phase 2 uses
same API to generate training data from UCF-Crime dataset. Phase 3 (final
product) uses our own Gemma 4 model finetuned with Unsloth, running on Kaggle
GPU and exposed via ngrok. Only gemma_client.py and .env change between phases.

Team: 2 people. No local GPU. Stack: Python, FastAPI, OpenCV, Streamlit, SQLite.
CLAUDE.md has full architecture, build order, data structures, and all decisions."