
# HawkWatch Gemma Edition — CLAUDE.md v3
> Kaggle x Google DeepMind — Gemma 4 Good Hackathon
> Deadline: May 18, 2026 | Prize: $200,000
> Team: 2 people | No local GPU
> Special mention tracks: Unsloth + Global Resilience/Disaster Response

---

## WHAT WE ARE BUILDING

Two cooperating AI systems in one platform:

SYSTEM 1 — SURVEILLANCE INTELLIGENCE
  Video/audio surveillance platform for underserved communities
  and disaster response teams. Analyzes footage using finetuned
  Gemma 4, stores incidents in Elasticsearch, supports natural
  language querying across all saved footage.

SYSTEM 2 — FIELD LEGAL ASSISTANT (the novel differentiator)
  A Gemma 4 model finetuned on criminal law + police procedure
  that helps detectives and officers make fast, legally sound
  decisions at a crime scene. Officer describes what they see
  verbally or in text and gets: applicable laws, required
  procedures, evidence collection checklist, rights to recite.

Both systems are open-source, self-hostable, and designed for
environments where expensive proprietary software is not an option.

---

## HACKATHON POSITIONING

Our angle: "AI for public safety in underserved communities"

Why we win:
  - Two finetuned models (rare in hackathon submissions)
  - Unsloth special mention track targeted directly
  - Real multimodal pipeline: video + audio + text
  - Novel idea (legal assistant for field officers) — no one else doing this
  - Elasticsearch for scalable NL retrieval (not just SQLite)
  - Gemma 4 audio support used (E2B/E4B models support audio natively)
  - Working demo with clear real-world impact story

Submission checklist:
  [ ] Working demo (both systems)
  [ ] Public GitHub repo
  [ ] Technical write-up (how Gemma 4 used in each system)
  [ ] 2 min demo video
  [ ] Finetuned models on HuggingFace (one per system)

---

## GEMMA 4 MODEL FAMILY — KEY FACTS

From HuggingFace (April 2026):

Four sizes:
  gemma-4-E2B     — 2B params, edge/mobile, supports AUDIO + vision + text
  gemma-4-E4B     — 4B params, edge/mobile, supports AUDIO + vision + text
  gemma-4-26B-A4B — 26B MoE (only 4B active), runs on consumer GPU
  gemma-4-31B     — 31B dense, best quality, ideal for finetuning

Context windows:
  E2B / E4B       — 128K tokens
  26B / 31B       — 256K tokens

CRITICAL FOR OUR PROJECT:
  Audio support exists ONLY on E2B and E4B (the small edge models)
  For our surveillance pipeline we use E4B for audio+vision analysis
  For our legal assistant we finetune 26B-A4B (fits Kaggle GPU, best quality)

Apache 2.0 license — fully open, commercial use allowed

Unsloth collection: huggingface.co/collections/unsloth/gemma-4
  Available: unsloth/gemma-4-E4B, unsloth/gemma-4-26B-A4B-it-GGUF

---

## FULL SYSTEM ARCHITECTURE

```
===== SYSTEM 1: SURVEILLANCE INTELLIGENCE =====

[Video Input]
      |
      |-- MP4 upload
      |-- Live stream (RTSP / YouTube)
      |
      v
[Multimodal Extractor]                  <- runs locally, no GPU
      |
      |-- FRAMES: OpenCV extracts 1 frame per 3s
      |   Motion detection skips static frames
      |   Saved as JPEG, low token budget (140 tokens)
      |   for speed (Gemma 4 supports configurable visual tokens)
      |
      |-- AUDIO: FFmpeg extracts audio track
      |   Chunked into 10s segments
      |   Transcribed via Gemma 4 E4B audio capability
      |   OR Whisper (fallback, runs on CPU)
      |
      v
[Segment Builder]
      Pairs each frame with its audio transcript (same timestamp window)
      Output: { frame_jpg, audio_transcript, timestamp }

      v
[Gemma 4 E4B — Multimodal Analysis]    <- Kaggle GPU / finetuned model
      Input: frame image + audio transcript + prompt
      Output: structured JSON incident
      {
        scene_description, activity_detected,
        audio_cues (shouting, glass breaking, gunshot),
        severity, category, confidence,
        recommended_action, persons_count,
        objects_of_interest
      }

      v
[Elasticsearch]                         <- the retrieval database
      What gets stored (see Section: Elasticsearch Schema)
      - NOT raw video files
      - NOT raw frame images (stored on disk, only path in ES)
      - YES: rich text metadata + embeddings for NL search

      v
[Natural Language Query Engine]
      User: "find anyone waving for help near debris"
            |
            v  Step 1: Elasticsearch BM25 keyword search
            |  Fast pre-filter across all incident metadata
            v  Step 2: Gemma 4 re-ranking
            |  Send top 10 ES results to Gemma for exact match
            v
      Returns: ranked incidents with frame paths + timestamps

      v
[Alert System]                          <- Twilio SMS
      Only on CRITICAL severity
      SMS: incident type + timestamp + confidence

      v
[Streamlit Frontend]
      Live feed | Upload | Query | Library


===== SYSTEM 2: FIELD LEGAL ASSISTANT =====

[Officer Input]
      |
      |-- Text: "I found a weapon near the suspect"
      |-- Voice: speak into phone -> Whisper -> text
      |
      v
[Finetuned Gemma 4 26B-A4B]            <- Legal assistant model
      Input: scene description + jurisdiction (optional)
      Output:
        LEGAL ASSESSMENT
        Applicable laws: [list with citations]
        Required procedure: [step by step]
        Rights to recite: [Miranda / local equivalent]
        Evidence checklist: [what to collect, how]
        Caution flags: [what NOT to do legally]
        Confidence: high/medium/low

      v
[Streamlit Frontend — Field Tab]
      Simple mobile-friendly interface
      Officer types or speaks, gets instant legal guidance
      Works offline if model loaded locally (Ollama future work)
```

---

## ELASTICSEARCH — EXACTLY WHAT TO STORE

This is the critical design decision for efficient NL retrieval.

DO NOT store in Elasticsearch:
  - Raw video files (too large, use filesystem)
  - Raw frame images (too large, store path only)
  - Base64 encoded anything (defeats the purpose)

DO store in Elasticsearch:

```json
{
  "incident_id": "uuid4",
  "timestamp": "2026-04-15T14:32:15",
  "video_source": "entrance_cam_01.mp4",
  "frame_path": "data/frames/frame_0143200000ms.jpg",
  "clip_start_seconds": 143.2,
  "clip_end_seconds": 153.2,

  // === TEXT FIELDS (BM25 searchable) ===
  "scene_description": "Two men near rear exit, one holding pipe",
  "activity_detected": "Assault in progress",
  "audio_cues": "Shouting detected, glass breaking sound at 14:32:18",
  "audio_transcript": "raw whisper/gemma transcript of audio segment",
  "category": "Crime",
  "severity": "CRITICAL",
  "recommended_action": "Dispatch immediately to rear exit",
  "objects_of_interest": ["pipe", "red jacket", "blue car"],
  "persons_count": 2,

  // === NUMERIC FIELDS (range queries) ===
  "confidence": 87,
  "motion_score": 0.92,

  // === KEYWORD FIELDS (exact filter) ===
  "alert_sent": true,
  "reviewed": false,
  "location_zone": "rear_exit",

  // === DENSE VECTOR (semantic search) ===
  "embedding": [0.023, -0.412, ...]
  // 768-dim vector of scene_description + activity_detected
  // generated using sentence-transformers/all-MiniLM-L6-v2
  // enables "find similar incidents" even without exact keywords
}
```

### Query Flow Explained

```
User types: "person waving for help near collapsed structure"
      |
      v Step 1 — Elasticsearch hybrid search
        BM25 on: scene_description, activity_detected, audio_cues
        KNN on:  embedding field (semantic similarity)
        Filter:  severity != CLEAR (skip boring frames)
        Returns: top 10 candidates fast (<50ms)
      |
      v Step 2 — Gemma 4 re-ranking
        Send top 10 incident descriptions to Gemma
        Prompt: "rank these by relevance to: {query}"
        Returns: ordered list with relevance_score + reason
      |
      v Step 3 — Frontend display
        Show ranked frames + reports + timestamps
        Click any result to see full incident report
```

Why this two-step approach:
  Elasticsearch alone = fast but misses semantic nuance
  Gemma alone = accurate but too slow for large libraries
  Combined = fast + accurate (best of both)

---

## FRAME + AUDIO EXTRACTION — EFFICIENCY DETAILS

### Frame Extraction Strategy

```python
# Configurable visual token budget (Gemma 4 feature)
# Lower = faster inference, higher = more detail
VISUAL_TOKEN_BUDGET = 140   # for surveillance (speed priority)
                            # options: 70, 140, 280, 560, 1120

# Motion detection threshold
MOTION_THRESHOLD = 500      # pixels changed to trigger save

# Interval
FRAME_INTERVAL_SECONDS = 3  # 1 frame per 3 seconds

# Storage
# - Save JPEG at 75% quality (good balance size vs detail)
# - Filename: frame_{timestamp_ms:010d}ms.jpg
# - Keep frames for 7 days then auto-delete (configurable)
# - Only keep frames where motion detected
```

### Audio Extraction Strategy

```python
# FFmpeg extracts audio from video
# Chunk into 10-second segments aligned with frame timestamps
# Two options for transcription:

OPTION A — Gemma 4 E4B audio (preferred, uses same model)
  Input: raw audio chunk (WAV 16kHz mono)
  Output: transcript + audio event labels
  (shouting, glass breaking, gunshot, crying, alarm)

OPTION B — Whisper (fallback, CPU, slower)
  whisper.transcribe(audio_chunk)
  No audio event detection, text only

# Store: transcript text + detected audio events + timestamp
# Do NOT store raw audio files
# Audio is ephemeral — transcribe then discard the WAV chunk
```

### What Gets Saved To Disk (data/ folder)

```
data/
|-- frames/              <- JPEG frames (motion-detected only)
|   |-- cam01/
|       |-- frame_0000000000ms.jpg
|       |-- frame_0003000000ms.jpg
|-- incidents.db         <- SQLite (lightweight incident index)
|                           mirrors Elasticsearch for local fallback
```

---

## LEGAL ASSISTANT — DETAILED FINETUNING PIPELINE

### The Novel Idea

No existing model is finetuned specifically for:
  "Officer at a crime scene describes what they see ->
   model gives applicable law + exact procedure + evidence checklist"

This is different from general legal QA because:
  - Field-focused (not courtroom, not academic)
  - Procedural (what to DO, not just what the law says)
  - Multimodal input (officer might describe visual scene)
  - Time-sensitive (decision support in minutes, not hours)
  - Jurisdiction-aware (law varies by state/country)

### Model Choice for Legal Assistant

Use: gemma-4-26B-A4B-it (26B MoE, 4B active)
Why:
  - Fits on Kaggle T4 GPU with 4-bit quantization via Unsloth
  - 256K context window (can handle long case descriptions)
  - Strong reasoning capability for legal logic
  - MoE architecture = fast inference despite large param count

### Training Data — What to Finetune On

Combine these datasets (all on HuggingFace):

TIER 1 — Legal foundation (what the law says):
  pile-of-law/pile-of-law          — 40B tokens of legal text
                                     court opinions, statutes, regulations
  nguha/legalbench                 — legal reasoning benchmark dataset
                                     use for both training + evaluation
  free-law/Caselaw_Access_Project  — US case law, 6.7M decisions
  HFforLegal/case-law              — structured case law
  isaacus/LegalQAEval              — legal QA pairs for eval

TIER 2 — Police procedure (what to DO):
  Synthesize this — no clean dataset exists, so generate it:
    Source texts: Miranda rights, arrest procedures, evidence
    handling protocols, use-of-force guidelines, chain of custody
    requirements, search and seizure law (4th Amendment)
    Use base Gemma 4 to generate (scenario -> procedure) pairs
    Target: 2000 high-quality synthetic procedure examples

TIER 3 — Crime scene scenarios (the novel part):
  Synthesize from UCF-Crime descriptions:
    Input:  "Officer arrives at scene: two men fighting near ATM,
             one has a weapon on the ground, crowd gathering"
    Output: LEGAL FIELD ASSESSMENT
            Applicable: 18 U.S.C. 111 (assault), State Penal 245
            Immediate steps: 1. Secure the scene...
            Miranda required: Yes, if arresting
            Evidence: photograph weapon before touching...
            Caution: do not move weapon without forensics present

### Finetuning Data Format

```python
# Instruction format for legal assistant finetuning
training_example = {
    "system": """You are a field legal assistant for law enforcement.
                 Given a scene description, provide: applicable laws,
                 required procedures, evidence checklist, and caution flags.
                 Be concise. Officers need fast, accurate guidance.""",

    "user": """Scene: {officer_description}
               Jurisdiction: {state_or_country} (if provided)
               Situation type: {crime_category}""",

    "assistant": """FIELD LEGAL ASSESSMENT
========================
Applicable Laws:
  - {law_1_name}: {brief_description} [{citation}]
  - {law_2_name}: {brief_description} [{citation}]

Required Procedure:
  1. {step_1}
  2. {step_2}
  3. {step_3}

Rights to Recite:
  {miranda_or_equivalent}

Evidence Checklist:
  [ ] {evidence_item_1}
  [ ] {evidence_item_2}

Caution Flags:
  ! {legal_risk_1}
  ! {legal_risk_2}

Confidence: {HIGH | MEDIUM | LOW}
Note: Always verify with supervising officer / legal counsel."""
}
```

### Legal Assistant — Unsloth Finetuning Config

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-26B-A4B-it",
    max_seq_length=4096,     # legal texts can be long
    load_in_4bit=True,       # essential for Kaggle T4 GPU
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,                    # higher rank for complex legal reasoning
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Training args
# batch_size=1, gradient_accumulation=8, epochs=2
# learning_rate=1e-4, warmup_ratio=0.1
# Push to: "your-username/hawkwatch-legal-gemma4"
```

### Legal Assistant — Benchmarks to Evaluate

After finetuning, evaluate on these (all available on HuggingFace):

| Benchmark | What it tests | Target score |
|---|---|---|
| nguha/legalbench | Legal reasoning tasks (61 subtasks) | Beat base Gemma 4 by >5% |
| isaacus/LegalQAEval | Legal QA accuracy | Beat base Gemma 4 by >5% |
| Custom eval set | Scene->procedure accuracy | Human eval by you (20 examples) |
| Response latency | Time to first token | <3 seconds on T4 GPU |

Custom eval set — create 20 hand-written examples:
  - 5 assault scenarios
  - 5 robbery scenarios
  - 5 disaster/emergency scenarios
  - 5 ambiguous/edge cases
  Score: did the model give the correct law + correct procedure?
  This is your novel benchmark — include it in the write-up

---

## SURVEILLANCE MODEL — FINETUNING PIPELINE

### Model Choice for Surveillance

Use: gemma-4-E4B-it (4B, edge model)
Why:
  - Supports audio natively (ONLY E2B and E4B do)
  - Small enough for fast inference on Kaggle free GPU
  - Still strong enough for surveillance scene understanding
  - Edge-deployable (future: run on Raspberry Pi at camera site)

### Training Data

Same approach as before but now includes audio:

```
UCF-Crime video clips
      |
      v FFmpeg
Extract frame + audio chunk (10s window)
      |
      v Base Gemma 4 E4B (AI Studio API)
Send frame + audio + prompt:
  "Describe this surveillance scene.
   What do you see? What sounds are described?
   What is happening? Generate a structured security report."
      |
      v
Save: (frame_path, audio_transcript, audio_events, structured_report)
      |
      v Unsloth finetuning on Kaggle
Target: 3000 training examples with audio context included
```

---

## PROJECT STRUCTURE

```
hawkwatch/
|-- CLAUDE.md                         <- you are here
|-- README.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- docker-compose.yml                <- Elasticsearch + app together
|
|-- pipeline/                         <- Surveillance pipeline
|   |-- frame_extractor.py            DONE - OpenCV + motion detection
|   |-- audio_extractor.py            <- FFmpeg audio chunking + transcription
|   |-- segment_builder.py            <- pair frame + audio by timestamp
|   |-- gemma_client.py               <- Gemma inference (swappable endpoint)
|   |-- incident_detector.py          <- parse Gemma JSON -> Incident
|   |-- report_generator.py           <- format final report
|   |-- embedder.py                   <- generate text embeddings for ES
|
|-- storage/
|   |-- elasticsearch_client.py       <- ES connection + index setup
|   |-- incident_store.py             <- save/search incidents in ES
|   |-- sqlite_fallback.py            <- local SQLite (ES not available)
|
|-- legal/                            <- Legal assistant system
|   |-- legal_assistant.py            <- query finetuned legal model
|   |-- scene_parser.py               <- parse officer description -> structured input
|
|-- backend/                          <- FastAPI
|   |-- main.py
|   |-- routes/
|       |-- upload.py                 <- POST /upload
|       |-- stream.py                 <- POST /stream
|       |-- query.py                  <- POST /query (NL search)
|       |-- incidents.py              <- GET /incidents
|       |-- legal.py                  <- POST /legal (field assistant)
|
|-- frontend/                         <- Streamlit
|   |-- app.py
|   |-- pages/
|       |-- upload.py
|       |-- live_feed.py
|       |-- query.py
|       |-- library.py
|       |-- legal_assistant.py        <- field legal assistant tab
|
|-- models/
|   |-- prompts.py                    DONE - all prompts centralised
|   |-- surveillance_finetune.py      <- Unsloth for E4B surveillance model
|   |-- legal_finetune.py             <- Unsloth for 26B legal model
|   |-- prepare_surveillance_data.py  <- UCF-Crime -> training pairs
|   |-- prepare_legal_data.py         <- law datasets -> training pairs
|   |-- evaluate_legal.py             <- run legal benchmarks
|
|-- notebooks/                        <- Run on Kaggle
|   |-- 01_test_gemma_e4b_audio.ipynb
|   |-- 02_prepare_surveillance_data.ipynb
|   |-- 03_finetune_surveillance.ipynb
|   |-- 04_prepare_legal_data.ipynb
|   |-- 05_finetune_legal.ipynb
|   |-- 06_evaluate_legal_benchmarks.ipynb
|   |-- 07_serve_models.ipynb         <- load both models + expose ngrok
|
|-- data/
|   |-- frames/                       <- extracted frames (gitignored)
|   |-- sample_videos/                <- test videos (gitignored)
|
|-- scripts/
    |-- test_surveillance_pipeline.py
    |-- test_legal_assistant.py
    |-- setup_elasticsearch.py        <- create ES index with correct mapping
```

---

## BUILD ORDER

```
PHASE 1 — Surveillance Pipeline Core (no server, no UI)
----------------------------------------------------------
Step 1:  pipeline/frame_extractor.py          DONE
Step 2:  pipeline/audio_extractor.py
         - FFmpeg extracts audio from MP4
         - Chunks into 10s WAV segments
         - Transcribes via Whisper (CPU) or Gemma E4B audio
         - Returns: {timestamp, transcript, audio_events}

Step 3:  pipeline/segment_builder.py
         - Pairs frame + audio segment by overlapping timestamp
         - Returns: {frame_path, transcript, audio_events, timestamp}

Step 4:  pipeline/gemma_client.py
         - analyze_segment(frame_path, audio_transcript) -> JSON
         - generate_report(incident_dict) -> formatted string
         - GEMMA_ENDPOINT env var controls AI Studio vs finetuned model

Step 5:  pipeline/incident_detector.py
         - Parse + validate Gemma JSON output
         - Return typed Incident dataclass

Step 6:  pipeline/embedder.py
         - Load sentence-transformers/all-MiniLM-L6-v2 (CPU, fast)
         - embed_text(scene_description + activity) -> 768-dim vector
         - Used for semantic search in Elasticsearch

Step 7:  storage/elasticsearch_client.py
         - Connect to ES (local Docker)
         - Create index with mapping (see ES Schema section)

Step 8:  storage/incident_store.py
         - save_incident(incident) -> ES index
         - search_incidents(query, filters) -> top 10 candidates
         - rerank_with_gemma(query, candidates) -> final ranked list

Step 9:  scripts/test_surveillance_pipeline.py
         - End-to-end: MP4 -> frames+audio -> Gemma -> ES -> query

PHASE 2 — Legal Assistant
----------------------------------------------------------
Step 10: legal/scene_parser.py
         - Parse free text officer description into structured fields
         - Extract: crime_type, persons_involved, objects, location

Step 11: legal/legal_assistant.py
         - Query finetuned legal Gemma model (same gemma_client pattern)
         - Return structured FIELD LEGAL ASSESSMENT

Step 12: scripts/test_legal_assistant.py
         - Test with 5 sample scenarios

PHASE 3 — Backend + Frontend
----------------------------------------------------------
Step 13: backend/ (FastAPI, all routes)
Step 14: frontend/ (Streamlit, all pages including legal tab)

PHASE 4 — Finetuning (Kaggle, parallel to Phase 2-3)
----------------------------------------------------------
Step 15: models/prepare_surveillance_data.py  <- UCF-Crime + audio labels
Step 16: notebooks/02 + 03                    <- finetune E4B, push to HF
Step 17: models/prepare_legal_data.py         <- law datasets + synthetic
Step 18: models/evaluate_legal.py             <- benchmark script
Step 19: notebooks/04 + 05 + 06              <- finetune 26B, eval, push to HF
Step 20: notebooks/07                         <- serve both models via ngrok
Step 21: Swap GEMMA_ENDPOINT in .env
```

---

## ELASTICSEARCH SETUP

Run locally via Docker for development:

```yaml
# docker-compose.yml
version: '3'
services:
  elasticsearch:
    image: elasticsearch:8.13.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "9200:9200"
```

```bash
docker-compose up -d
python scripts/setup_elasticsearch.py   # creates index with mapping
```

Index mapping (setup_elasticsearch.py must create this):
```json
{
  "mappings": {
    "properties": {
      "incident_id":        { "type": "keyword" },
      "timestamp":          { "type": "date" },
      "video_source":       { "type": "keyword" },
      "frame_path":         { "type": "keyword" },
      "clip_start_seconds": { "type": "float" },
      "clip_end_seconds":   { "type": "float" },
      "scene_description":  { "type": "text", "analyzer": "english" },
      "activity_detected":  { "type": "text", "analyzer": "english" },
      "audio_cues":         { "type": "text", "analyzer": "english" },
      "audio_transcript":   { "type": "text", "analyzer": "english" },
      "category":           { "type": "keyword" },
      "severity":           { "type": "keyword" },
      "recommended_action": { "type": "text" },
      "objects_of_interest":{ "type": "keyword" },
      "persons_count":      { "type": "integer" },
      "confidence":         { "type": "integer" },
      "alert_sent":         { "type": "boolean" },
      "reviewed":           { "type": "boolean" },
      "embedding": {
        "type": "dense_vector",
        "dims": 384,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

---

## FRONTEND — ALL PAGES

### Page 1: Upload / Live Feed
- Upload MP4 or paste stream URL
- Progress bar: extracting frames... analyzing audio... running Gemma...
- Show incidents as they are detected in real time
- Each incident card: frame thumbnail + severity badge + report preview

### Page 2: Incident Library
- Table of all saved incidents (from Elasticsearch)
- Filters: severity, category, date range, video source
- Click row -> full incident report + frame image
- Export selected incidents as PDF

### Page 3: Natural Language Query
- Large text input: "find anyone waving for help near debris"
- Results: ranked list of matching frames with timestamps
- Each result shows: frame image, relevance score, reason, full report
- Click to jump to that moment in the video (if MP4 available)

### Page 4: Field Legal Assistant (the differentiator)
- Clean mobile-friendly layout (officers use phones in the field)
- Text area: "Describe what you see at the scene"
- Optional: jurisdiction dropdown (US states, India states, etc.)
- Optional: crime category selector (helps model focus)
- OUTPUT: structured FIELD LEGAL ASSESSMENT
  - Applicable laws with citations
  - Step-by-step procedure
  - Rights to recite
  - Evidence checklist (checkboxes, officer can check off)
  - Caution flags in red
- Save assessment to incident record (links surveillance + legal)
- Print/share button for the assessment

### Page 5: Dashboard (overview)
- Stats: total incidents today, CRITICAL count, query count
- Recent incidents feed
- System status (Elasticsearch, Gemma endpoint, alert system)

---

## ENVIRONMENT VARIABLES

```bash
# .env — never commit

# Gemma endpoints
GEMMA_API_KEY=your_google_ai_studio_key
SURVEILLANCE_GEMMA_ENDPOINT=aistudio      # swap to ngrok after finetuning
LEGAL_GEMMA_ENDPOINT=aistudio             # swap to ngrok after finetuning

# After finetuning — your HuggingFace models
SURVEILLANCE_MODEL_ID=your-username/hawkwatch-surveillance-gemma4-e4b
LEGAL_MODEL_ID=your-username/hawkwatch-legal-gemma4-26b

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=hawkwatch_incidents

# Alerts
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=+1234567890
ALERT_PHONE_NUMBER=+1234567890

# Pipeline config
FRAME_INTERVAL_SECONDS=3
MOTION_THRESHOLD=500
VISUAL_TOKEN_BUDGET=140          # Gemma 4 feature: lower = faster inference
AUDIO_CHUNK_SECONDS=10
SEVERITY_ALERT_THRESHOLD=CRITICAL

# Storage
FRAME_RETENTION_DAYS=7
DATABASE_URL=sqlite:///./data/incidents.db   # SQLite fallback
```

---

## GPU AND COMPUTE PLAN

```
Google AI Studio free tier    -> Testing + dataset generation (API calls)
Kaggle Account 1 (30hr/wk)   -> Surveillance E4B finetuning + demo serving
Kaggle Account 2 (30hr/wk)   -> Legal 26B-A4B finetuning
Vast.ai / RunPod (paid)       -> Backup if hours run out
```

Estimated GPU time:
  Surveillance E4B finetune   -> ~4-6 hours on T4
  Legal 26B-A4B finetune      -> ~8-10 hours on T4 (use A100 if available)
  Dataset generation          -> 0 GPU (API calls)
  Demo day serving            -> ~2 hours total

```
No GPU needed (local machine):
  Frame extraction, audio extraction, FastAPI, Streamlit,
  Elasticsearch (Docker), SQLite, embedder (MiniLM is CPU-fast)
```

---

## WHAT MODELS EXIST ON HUGGINGFACE ALREADY

As of April 2026, existing Gemma 4 finetuned models found:

  Jackrong/Gemopus-4-26B-A4B-it-GGUF
    -> SFT on general instruction following
    -> NOT domain-specific, not surveillance, not legal

  unsloth/gemma-4-26B-A4B-it-GGUF
    -> Unsloth quantized base (not finetuned on domain data)

GAP WE ARE FILLING:
  No finetuned Gemma 4 model exists for:
  - Surveillance incident report generation
  - Field legal assistance for law enforcement
  Both are genuinely novel. This is our hackathon edge.

---

## CODING RULES

1. Every file has a module docstring
2. No hardcoded API keys — always os.getenv()
3. All Gemma calls wrapped in try/except — never crash on bad JSON
4. All prompts in models/prompts.py only
5. Type hints on all functions
6. Every module independently testable via __main__
7. Print progress to console
8. gemma_client.py reads SURVEILLANCE_GEMMA_ENDPOINT and
   LEGAL_GEMMA_ENDPOINT separately — two swappable endpoints
9. Elasticsearch operations must have SQLite fallback
10. Frontend pages must work even if ES is down (graceful degradation)

---

## CURRENT BUILD STATUS

- [x] pipeline/frame_extractor.py         DONE
- [x] models/prompts.py                   DONE
- [ ] pipeline/audio_extractor.py         TODO (build next)
- [ ] pipeline/segment_builder.py         TODO
- [ ] pipeline/gemma_client.py            TODO
- [ ] pipeline/incident_detector.py       TODO
- [ ] pipeline/embedder.py                TODO
- [ ] storage/elasticsearch_client.py     TODO
- [ ] storage/incident_store.py           TODO
- [ ] legal/legal_assistant.py            TODO
- [ ] backend/ (all routes)               NOT STARTED
- [ ] frontend/ (all pages)               NOT STARTED
- [ ] Surveillance finetuning             NOT STARTED
- [ ] Legal finetuning                    NOT STARTED

---

## CONTEXT BLOCK — PASTE TO START ANY NEW CLAUDE SESSION

"I am building HawkWatch Gemma Edition for the Kaggle Gemma 4 Good Hackathon
(deadline May 18 2026, $200K prize). Two AI systems in one platform:

SYSTEM 1 — Surveillance Intelligence: OpenCV extracts frames + FFmpeg
extracts audio -> paired segments sent to finetuned Gemma 4 E4B (supports
audio natively) -> structured incidents stored in Elasticsearch with
embeddings for hybrid BM25 + semantic NL search -> Twilio alerts.

SYSTEM 2 — Field Legal Assistant: Gemma 4 26B-A4B finetuned on pile-of-law
+ legalbench + synthetic crime scene procedure data -> officers describe
scene in text/voice -> model returns applicable laws, procedures, evidence
checklist, caution flags.

Both models finetuned with Unsloth (hackathon special mention track).
No local GPU. Stack: Python, FastAPI, Streamlit, OpenCV, FFmpeg, Whisper,
Elasticsearch (Docker), SQLite fallback, Twilio, sentence-transformers.
CLAUDE.md has full architecture, build order, ES schema, legal finetuning
pipeline, benchmark plan, and all design decisions documented."
```
