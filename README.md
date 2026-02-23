# Empathy Engine 🎙️

Emotion-Aware Text-to-Speech system that detects emotion from text and generates expressive speech with dynamically adjusted voice parameters.

The system analyzes input text, classifies emotion, and modulates speech rate, volume, and pitch to produce emotionally appropriate audio output.

---

# Features

* Emotion detection from text input
* Emotion-based speech modulation (rate, volume, pitch)
* Generates playable audio file (.wav)
* CLI interface for quick testing
* FastAPI web interface with browser playback
* Fully local and runnable

---

# Setup Instructions

## Step 1 — Clone the repository

```bash
git clone https://github.com/Khushdeep17/empathy-engine.git
cd empathy-engine
```

---

## Step 2 — (Optional) Create virtual environment

Recommended but not required.

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / Mac**

```bash
python -m venv venv
source venv/bin/activate
```

---

## Step 3 — Install dependencies

This step is REQUIRED.

```bash
pip install -r requirements.txt
```

---

# Run the Project

You can run the project using either CLI or Web Interface.

---

## Option 1 — Run CLI

```bash
python cli.py
```

Enter text when prompted.

Audio file will be generated in the outputs folder.

---

## Option 2 — Run FastAPI Web Server

Start the server:

```bash
uvicorn app:app --reload
```

Open browser and go to:

```
http://127.0.0.1:8000
```

Enter text and audio will be generated and played in browser.

---

# Emotion to Voice Mapping Logic

The system detects emotion and adjusts speech parameters dynamically:

| Emotion | Rate   | Volume | Effect          |
| ------- | ------ | ------ | --------------- |
| Joy     | Faster | Higher | Energetic voice |
| Sadness | Slower | Lower  | Calm voice      |
| Anger   | Faster | Higher | Intense voice   |
| Fear    | Slower | Medium | Soft voice      |
| Neutral | Normal | Normal | Balanced voice  |

This mapping creates emotionally expressive speech output instead of robotic voice.

---

# Output

Generated audio files are saved in:

```
outputs/
```

Format: `.wav`

---

# Tech Stack

* Python
* FastAPI
* Transformers
* PyTorch
* VADER Sentiment
* Local Neural TTS

---

