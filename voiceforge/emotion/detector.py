import os
import uuid

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — fixes threading crash in FastAPI
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

from voiceforge.emotion.schemas import EmotionResult


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load once at module import — not per request
_analyzer = SentimentIntensityAnalyzer()
_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)


def _visualize(results: list, output_dir: str = OUTPUT_DIR) -> str:
    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]

    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, scores)
    ax.set_ylim(0, 1)
    ax.set_title("Emotion Probabilities")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Confidence")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)

    return filepath


def detect_emotion(text: str, mode: str = "hybrid") -> EmotionResult:
    """Detect emotion from text.

    Args:
        text: Input string to classify.
        mode: "hybrid" (transformer + VADER fallback) or "basic" (VADER only).

    Returns:
        EmotionResult with emotion label, confidence, full scores, and viz path.
    """
    vader = _analyzer.polarity_scores(text)
    compound = vader["compound"]

    if mode == "basic":
        if compound >= 0.5:
            return EmotionResult("joy", compound, None, None)
        elif compound <= -0.5:
            return EmotionResult("anger", abs(compound), None, None)
        else:
            return EmotionResult("neutral", abs(compound), None, None)

    # hybrid: transformer with VADER fallback
    try:
        results = _classifier(text)
        if isinstance(results[0], list):
            results = results[0]

        scores = {r["label"]: r["score"] for r in results}
        emotion = max(scores, key=scores.get)
        confidence = scores[emotion]
        viz_path = _visualize(results)

        return EmotionResult(emotion, confidence, scores, viz_path)

    except Exception:
        if compound >= 0.5:
            return EmotionResult("joy", compound, None, None)
        elif compound <= -0.5:
            return EmotionResult("anger", abs(compound), None, None)
        else:
            return EmotionResult("neutral", abs(compound), None, None)
