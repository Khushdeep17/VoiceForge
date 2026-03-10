# VoiceForge 🎙️

VoiceForge is an emotion-aware Text-to-Speech inference system that detects emotion from text using a transformer model and generates expressive speech with dynamically conditioned voice parameters, YAML-driven personas, MLflow observability, and AWS S3 artifact storage.

Built as a modular production service — not a script.

---

## ⚙️ System Architecture

```
Text Input (API / CLI)
        ↓
Emotion Detection (DistilRoBERTa + VADER fallback)
        ↓
Voice Parameter Conditioning (Emotion → Rate / Volume)
        ↓  ↕ Persona Override (YAML profiles)
TTS Synthesis (pyttsx3 → .wav)
        ↓
S3 Artifact Upload (boto3 → presigned URL)
        ↓
MLflow Inference Logging (confidence, params, latency)
```

Modular package structure — emotion, TTS, storage, and metrics layers fully separated.

---

## 🚀 Key Features

- **Hybrid emotion pipeline** — DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`) classifying 7 emotions with full per-class confidence scores; VADER as silent fallback
- **Emotion-conditioned TTS** — rate and volume dynamically scaled by emotion label and confidence intensity via pyttsx3
- **5 YAML-driven voice personas** — narrator, therapist, broadcaster, assistant, storyteller; override emotion params without touching code
- **MLflow inference tracking** — every request logs emotion, confidence, voice params, per-class scores, emotion latency, and P95 end-to-end latency
- **AWS S3 artifact storage** — audio and visualizations uploaded via boto3 with presigned URL delivery; graceful local fallback if S3 unavailable
- **`/metrics` aggregation endpoint** — live emotion distribution, avg/P95 latency, total request count from MLflow store

---

## 🧠 Tech Stack

| Layer | Technologies |
|-------|-------------|
| API | FastAPI, Pydantic, Jinja2 |
| Emotion | Transformers (DistilRoBERTa), VADER |
| TTS | pyttsx3, SSML |
| Observability | MLflow |
| Storage | AWS S3, boto3 |
| Config | PyYAML |
| CLI | Python argparse |

---

## 🗂️ Project Structure

```
VoiceForge/
├── voiceforge/
│   ├── emotion/
│   │   ├── detector.py       # DistilRoBERTa + VADER hybrid pipeline
│   │   └── schemas.py        # EmotionResult dataclass
│   ├── tts/
│   │   ├── engine.py         # pyttsx3 synthesis + emotion conditioning
│   │   ├── personas.py       # YAML-driven persona loader
│   │   └── ssml.py           # SSML prosody builder
│   ├── metrics/
│   │   └── tracker.py        # MLflow logging + aggregate stats
│   └── storage/
│       └── s3.py             # S3 upload with local fallback
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── speak.py          # /speak, /speak-ui
│       └── health.py         # /health, /personas, /metrics
├── cli/main.py
└── configs/personas.yaml
```

---

## 📡 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/speak` | Emotion detection → TTS synthesis → S3 upload |
| GET | `/metrics` | Emotion distribution, avg/P95 latency, total requests |
| GET | `/personas` | List all YAML-configured voice personas |
| GET | `/health` | Service health check |

**Request:**
```json
{ "text": "I just got the job!", "mode": "hybrid", "persona": "broadcaster" }
```

**Response:**
```json
{
  "emotion": "joy",
  "confidence": 0.912,
  "voice_parameters": { "rate": 185, "volume": 0.95 },
  "voice_style": "energetic",
  "audio_file": "/outputs/abc.wav",
  "s3_audio_url": "https://s3.amazonaws.com/..."
}
```

---

## ☁️ AWS S3 Setup

1. Create IAM user → attach `AmazonS3FullAccess` → generate access keys
2. Create S3 bucket (Block Public Access ON — presigned URLs handle access)
3. Configure:

```bash
aws configure          # key, secret, region
```

---

## 🏃 Quick Start

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload    # http://127.0.0.1:8000
mlflow ui                        # http://127.0.0.1:5000
python -m cli.main
```

---

## 🧩 What This Project Demonstrates

- Transformer-based ML inference pipeline with production fallback strategy
- Emotion-conditioned generative audio output with configurable persona system
- MLflow observability instrumentation on a live inference service
- AWS S3 artifact management with presigned URL delivery and graceful degradation
- Modular FastAPI service design with separated concerns across emotion, TTS, storage, and metrics layers