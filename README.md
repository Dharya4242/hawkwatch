# 🦅 SecureSight Gemma Edition

> Open-source AI-powered video surveillance and disaster response platform
> Built for the [Kaggle × Google DeepMind Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)

[![Demo Video](https://img.shields.io/badge/Demo-Watch%20Video-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=njbNn9m34H4)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/Dharya/secure_sight-gemma4-security)
[![Kaggle](https://img.shields.io/badge/Kaggle-Submission-blue?style=for-the-badge&logo=kaggle)](YOUR_KAGGLE_SUBMISSION_URL)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)](LICENSE)

---

## The Problem

Manual video surveillance is expensive, labor-intensive, and inaccessible to underserved communities. Existing systems cost thousands of dollars per month, require constant human monitoring, and miss critical incidents. Communities that need safety the most cannot afford the tools that provide it.

During disasters, response teams are overwhelmed with drone footage but have no way to quickly search it. Every minute of delay costs lives.

---

## The Solution

SecureSight Gemma Edition lets security personnel describe what they are looking for in plain English — and the system finds it.

```
"Find a person waving for help near collapsed debris"
        ↓
System searches all footage instantly
        ↓
Returns matching timestamps + frame evidence + incident report
        ↓
SMS alert sent to responders in real time
```

Powered by a **finetuned Gemma 4** model, open-source, and self-hostable by anyone with a laptop.

---

## Demo

[![SecureSight Demo](https://img.shields.io/badge/▶_Watch_Demo-2_minutes-red?style=for-the-badge)](https://www.youtube.com/watch?v=njbNn9m34H4)

**Key moments in the demo:**
- Upload any MP4 → automatic incident detection
- Natural language query → instant frame matching
- Real-time SMS alert on phone
- Incident library with full report history

---

## Features

**1. Automatic Incident Detection**
Upload any MP4 or connect a live stream. The system extracts frames, analyzes each one with Gemma 4, and detects crimes, suspicious activity, medical emergencies, and disaster scenarios automatically.

**2. Natural Language Querying**
Search your footage like a database. Type "find anyone running near the exit" or "show me all CRITICAL incidents from today" — the system understands and returns results instantly.

**3. Structured Incident Reports**
Every detected incident generates a structured report with severity level, category, confidence score, frame evidence, and recommended action for security personnel.

**4. Real-Time Alerts**
Configurable SMS alerts via Twilio. Get notified the moment a CRITICAL incident is detected — with incident details and timestamp.

**5. Incident Library**
Full searchable history of all detected incidents with frame screenshots, reports, and timeline view.

---

## Architecture

```
Video Input (MP4 / Live Stream)
          │
          ▼
Frame Extractor (OpenCV)
  • Screenshot every 3 seconds
  • Motion detection skips static frames
  • No GPU needed locally
          │
          ▼
Finetuned Gemma 4 E4B
  • Hosted on Kaggle GPU
  • Exposed via ngrok REST API
  • Input: frame image + prompt
  • Output: structured JSON incident
          │
          ▼
Incident Engine
  • Severity: CRITICAL / WARNING / CLEAR
  • Category: Crime / Medical / Disaster / Suspicious / Normal
  • Confidence score + recommended action
          │
     ─────┴─────
     │         │
     ▼         ▼
  Alerts    Incident
 (Twilio)   Library
             (SQLite)
          │
          ▼
NL Query Engine
  • Plain English search
  • Semantic matching
  • Returns ranked results
          │
          ▼
Streamlit Dashboard
  • Live feed viewer
  • Upload interface
  • Query interface
  • Incident library
```

---

## Finetuning with Unsloth

We finetuned Gemma 4 E4B using Unsloth LoRA to specialize it for security incident reporting.

**Why finetune?**

Base Gemma produces generic descriptions:
> "Two people are standing near a building. One appears to be holding something."

Finetuned Gemma produces structured security reports:
```json
{
  "severity": "WARNING",
  "category": "Suspicious Activity",
  "confidence": 82,
  "activity_detected": "Two individuals loitering near rear entrance, one holding unidentified object",
  "recommended_action": "Monitor closely. Dispatch if behavior escalates."
}
```

**Dataset**
- Source: UCF-Crime surveillance dataset (14 crime categories)
- Labeling: Gemini 2.5 Flash as teacher model (knowledge distillation)
- Size: 698 balanced training examples + 149 test examples
- Format: Alpaca instruction format for Unsloth

**Training**
- Framework: Unsloth LoRA (text-only, language layers)
- Model: unsloth/gemma-4-E4B-it
- Time: 28 minutes on T4 GPU
- Trainable params: 0.46% (LoRA r=16)
- Final loss: 0.27

**Results**

| Metric | Base Gemma 4 | Finetuned (Ours) | Improvement |
|--------|-------------|-----------------|-------------|
| Severity Accuracy | 58.4% | 65.8% | ↑ 7.4% |
| Category Accuracy | 57.0% | 66.4% | ↑ 9.4% |
| JSON Compliance | 100% | 100% | — |
| All Fields Present | 100% | 100% | — |

Evaluated on 149 held-out test samples from UCF-Crime dataset.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI Model | Gemma 4 E4B (finetuned) |
| Finetuning | Unsloth LoRA |
| Video Processing | OpenCV |
| Backend | Python + FastAPI |
| Frontend | Streamlit |
| Database | SQLite + SQLAlchemy |
| Alerts | Twilio SMS |
| Model Serving | Kaggle GPU + ngrok |

---

## Setup

### Prerequisites
- Python 3.10+
- Google AI Studio API key (free at aistudio.google.com)
- Twilio account (free tier works)
- ngrok account (free at ngrok.com)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Dharya4242/secure_sight-gemma4.git
cd secure_sight-gemma4

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys
```

### Environment Variables

```bash
# .env
GEMMA_API_KEY=your_google_ai_studio_key

# Development: use AI Studio base model
GEMMA_ENDPOINT=aistudio

# Production: use finetuned model via ngrok
# GEMMA_ENDPOINT=https://xxxx.ngrok-free.app

TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+1234567890
ALERT_PHONE_NUMBER=+1234567890

DATABASE_URL=sqlite:///./data/incidents.db
FRAME_INTERVAL_SECONDS=3
MOTION_THRESHOLD=500
SEVERITY_ALERT_THRESHOLD=CRITICAL
```

### Run The App

```bash
# Terminal 1 — Backend
uvicorn backend.main:app --reload

# Terminal 2 — Frontend
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

### Run With Finetuned Model (Optional)

To use our finetuned Gemma 4 instead of the base API:

1. Open `notebooks/04_serve_model.py` in a Kaggle notebook with GPU
2. Set your HuggingFace model ID and ngrok token
3. Run all cells — you get a public URL
4. Set `GEMMA_ENDPOINT=https://your-url.ngrok-free.app` in `.env`

---

## Kaggle Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| 01_test_gemma_vision | Confirm Gemma 4 loads + vision works | ✅ |
| 02_prepare_dataset | UCF-Crime → labeled training data | ✅ |
| 03_finetune_unsloth | Unsloth LoRA finetuning | ✅ |
| 04_serve_model | Load model + expose via ngrok | ✅ |

---

## Finetuned Model

🤗 **[Dharya/secure_sight-gemma4-security](https://huggingface.co/Dharya/secure_sight-gemma4-security)**

```python
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastModel.from_pretrained(
    "Dharya/secure_sight-gemma4-security",
    load_in_4bit = True,
)
tokenizer = get_chat_template(tokenizer, "gemma-4")
```

---

## Project Structure

```
SecureSight/
├── pipeline/
│   ├── frame_extractor.py      # OpenCV frame extraction + motion detection
│   ├── gemma_client.py         # Gemma 4 API wrapper (swappable endpoint)
│   ├── incident_detector.py    # Parse Gemma output → Incident object
│   └── report_generator.py     # Format structured incident reports
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── database.py             # SQLite + SQLAlchemy
│   ├── models.py               # Pydantic schemas
│   └── routes/                 # upload, incidents, query, stream
├── frontend/
│   ├── app.py                  # Streamlit entry point
│   └── pages/                  # upload, library, query, live_feed
├── models/
│   ├── prompts.py              # All Gemma prompts centralised
│   ├── prepare_dataset.py      # UCF-Crime dataset preparation
│   └── finetune.py             # Unsloth finetuning script
├── notebooks/                  # Kaggle notebooks (01-04)
├── data/
│   └── sample_videos/          # Put test MP4s here
├── requirements.txt
└── .env.example
```

---

## AI for Good Impact

> This project qualifies for the **Global Resilience / Disaster Response** special mention track.

- **Cost**: Replaces $500+/month proprietary systems with a free open-source alternative
- **Access**: Any community with a laptop can deploy this — no expensive hardware
- **Efficiency**: Reduces manual monitoring labor by ~80% through automated incident detection
- **Disaster response**: Search drone footage in plain English during emergencies — every minute saved matters
- **Privacy**: Self-hostable — footage never leaves your own infrastructure
- **Future**: Edge deployment via Ollama for fully offline, no-internet environments

---

## Team

Built for the Kaggle × Google DeepMind Gemma 4 Good Hackathon (April–May 2026)

- **[Dharya Jasuja](https://github.com/Dharya4242)** — AI pipeline, finetuning, backend, frontend

---

## License

Apache 2.0 — free to use, modify, and deploy commercially.

---

## Acknowledgements

- [Unsloth](https://unsloth.ai) — LoRA finetuning framework
- [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/real-world) — training data
- [SecureSight (TreeHacks 2025)](https://github.com/Grace-Shao/Treehacks2025) — inspiration
- Google DeepMind — Gemma 4 open weights
- Kaggle — free GPU compute