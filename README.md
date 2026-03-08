# VoiceForge 🎙️

Emotion-aware Text-to-Speech system that detects emotion from text and generates expressive speech with dynamically adjusted voice parameters and named voice personas.

---

## Project Structure

```
VoiceForge/
├── voiceforge/              # core package
│   ├── emotion/
│   │   ├── detector.py      # unified emotion detection (transformer + VADER fallback)
│   │   └── schemas.py       # EmotionResult dataclass
│   └── tts/
│       ├── engine.py        # pyttsx3 TTS synthesis
│       ├── personas.py      # named voice persona profiles
│       └── ssml.py          # SSML prosody builder
├── api/
│   ├── main.py              # FastAPI app
│   ├── schemas.py           # Pydantic request/response models
│   └── routes/
│       ├── speak.py         # /speak, /, /speak-ui
│       └── health.py        # /health, /personas
├── cli/
│   └── main.py              # interactive CLI
├── configs/
│   └── personas.yaml        # voice persona definitions
├── outputs/                 # generated audio and visualizations
└── templates/               # Jinja2 HTML templates
```

---

## Setup

```bash
git clone https://github.com/Khushdeep17/VoiceForge.git
cd VoiceForge
python -m venv venv
# Windows: venv\Scripts\activate  |  Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

---

## Run

**CLI**
```bash
python -m cli.main
```

**Web server**
```bash
uvicorn api.main:app --reload
```
Open `http://127.0.0.1:8000`

**API docs**
```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/speak` | JSON API — emotion detection + TTS |
| GET | `/health` | Service health check |
| GET | `/personas` | List all available voice personas |
| GET | `/` | Web UI |
| POST | `/speak-ui` | Web UI form handler |

**Example request with persona:**
```json
POST /speak
{
  "text": "I just got the job!",
  "mode": "hybrid",
  "persona": "broadcaster"
}
```

---

## Voice Personas

Named profiles defined in `configs/personas.yaml`. Add your own by editing the file.

| Persona | Style | Rate | Volume |
|---------|-------|------|--------|
| narrator | authoritative | 160 | 0.85 |
| therapist | gentle | 140 | 0.65 |
| broadcaster | energetic | 185 | 0.95 |
| assistant | neutral | 170 | 0.80 |
| storyteller | expressive | 155 | 0.75 |

---

## Emotion → Voice Mapping

| Emotion | Effect |
|---------|--------|
| Joy | Faster, louder — energetic |
| Sadness | Slower, quieter — soft |
| Anger | Faster, firm |
| Fear | Slower, medium — cautious |
| Neutral | Balanced |

---

## Tech Stack

Python · FastAPI · Transformers · pyttsx3 · VADER · Matplotlib · PyYAML
