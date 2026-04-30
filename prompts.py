"""
prompts.py — All Gemma 4 prompt templates in one place.
Never hardcode prompts in other files. Always import from here.
"""


FRAME_ANALYSIS_PROMPT = """You are an intelligent security monitoring system analyzing surveillance footage.

Look at this image carefully and respond with ONLY a valid JSON object — no extra text, no markdown, no explanation.

{{
  "scene_description": "detailed description of everything visible in the frame",
  "activity_detected": "specific activity or event occurring — be precise",
  "persons_count": 0,
  "severity": "CRITICAL or WARNING or CLEAR",
  "category": "Crime or Medical Emergency or Suspicious Activity or Disaster or Normal",
  "confidence": 0,
  "recommended_action": "specific action for security personnel to take right now",
  "objects_of_interest": ["list", "of", "notable", "objects", "or", "people"]
}}

Severity guide:
- CRITICAL: immediate threat to life or active crime in progress
- WARNING: suspicious behavior, potential threat, needs monitoring
- CLEAR: normal activity, nothing concerning

Be precise. Real security personnel will act on this information."""


REPORT_GENERATION_PROMPT = """You are a security incident report writer. Output ONLY the formatted report below — no reasoning, no notes, no preamble.

Incident Data:
- Timestamp: {timestamp}
- Location/Source: {source}
- Scene Description: {scene_description}
- Activity: {activity}
- Severity: {severity}
- Category: {category}
- Confidence: {confidence}%

INCIDENT REPORT
===============
Timestamp:          {timestamp}
Source:             {source}
Severity:           {severity}
Category:           {category}
Confidence Score:   {confidence}%

Description:
[2-3 sentences describing what happened clearly and professionally]

Persons Involved:
[describe individuals — clothing, actions, count]

Objects / Evidence:
[list notable objects, weapons, vehicles etc. or "None detected"]

Recommended Action:
[one clear actionable instruction for security personnel]

Report Generated: [auto]"""


NL_QUERY_PROMPT = """You are searching through a database of security incident records.

User is looking for: "{query}"

Here are the incident records to search through:
{incidents_json}

Each record has: id, timestamp, scene_description, activity_detected, category, severity.

Return ONLY a valid JSON array of matches, ordered by relevance (most relevant first):
[
  {{
    "incident_id": "the id from the record",
    "timestamp": "the timestamp",
    "relevance_score": 0-100,
    "reason": "one sentence explaining why this matches the query"
  }}
]

Only include incidents with relevance_score above 40.
If nothing matches, return an empty array: []
No extra text, just the JSON array."""


SCENE_DESCRIPTION_PROMPT = """You are a surveillance camera assistant.

Describe this camera frame in precise detail. Cover:
- Who is present (count, clothing, actions)
- What is happening
- Location context (indoor/outdoor, lighting, setting)
- Any objects of interest

Write 2-4 clear sentences. Plain text only — no JSON, no lists."""


FINETUNING_PROMPT_TEMPLATE = """Below is a security footage description. Generate a structured incident report.

### Description:
{scene_description}

### Incident Report:
{incident_report}"""
