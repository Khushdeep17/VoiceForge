from transformers import pipeline
import matplotlib.pyplot as plt
import uuid
import os


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# load model once at startup
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


def visualize_emotions(results):

    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]

    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores)
    plt.ylim(0, 1)
    plt.title("Emotion Probabilities")
    plt.xlabel("Emotion")
    plt.ylabel("Confidence")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


def detect_emotion_transformer(text: str):

    results = emotion_classifier(text)

    if isinstance(results[0], list):
        results = results[0]

    scores = {r["label"]: r["score"] for r in results}

    emotion = max(scores, key=scores.get)
    confidence = scores[emotion]

    viz_path = visualize_emotions(results)

    return emotion, confidence, scores, viz_path