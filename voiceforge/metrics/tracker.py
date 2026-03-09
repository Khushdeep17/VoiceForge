
"""
voiceforge/metrics/tracker.py

Central MLflow tracking module.

Used by both API and CLI to log inference runs and retrieve
aggregated observability metrics.

Tracks:
- emotion classification
- voice parameters
- latency metrics
- persona usage
"""
import os
import mlflow
import statistics
from pathlib import Path
from typing import Optional, Dict
from contextlib import contextmanager


# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------

EXPERIMENT_NAME = "VoiceForge-Inference"

MLFLOW_DIR = Path("./mlruns")
MLFLOW_DIR.mkdir(exist_ok=True)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR.resolve()}")
mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# Internal helper
# --------------------------------------------------

@contextmanager
def _run(run_name: Optional[str] = None):
    """Context manager wrapper for MLflow runs."""
    with mlflow.start_run(run_name=run_name) as run:
        yield run


# --------------------------------------------------
# Logging function
# --------------------------------------------------

def log_inference(
    *,
    text: str,
    emotion: str,
    confidence: float,
    scores: Optional[Dict[str, float]],
    rate: int,
    volume: float,
    style: str,
    persona: Optional[str],
    mode: str,
    emotion_latency_ms: float,
    total_latency_ms: float,
) -> str:
    """
    Log a single VoiceForge inference event to MLflow.

    Returns:
        run_id (str)
    """

    with _run(run_name=f"{emotion}-{style}") as run:

        # PARAMETERS (categorical metadata)
        mlflow.log_params({
            "mode": mode,
            "persona": persona or "none",
            "voice_style": style,
            "emotion": emotion,
        })

        # METRICS (numerical measurements)
        mlflow.log_metrics({
            "confidence": round(confidence, 4),
            "rate": rate,
            "volume": round(volume, 4),
            "emotion_latency_ms": round(emotion_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "text_length": len(text),
        })

        # Optional transformer probability scores
        if scores:
            for label, score in scores.items():
                mlflow.log_metric(f"score_{label}", round(score, 4))

        return run.info.run_id


# --------------------------------------------------
# Aggregated metrics for /metrics endpoint
# --------------------------------------------------

def get_aggregate_stats() -> dict:
    """Compute aggregated observability metrics across all runs."""

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if not experiment:
        return {"total_requests": 0}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000
    )

    if not runs:
        return {"total_requests": 0}

    total = len(runs)

    latencies = []
    confidence_values = []

    emotion_counts = {}
    persona_counts = {}

    for run in runs:

        params = run.data.params
        metrics = run.data.metrics

        # Emotion distribution
        emotion = params.get("emotion", "unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Persona usage
        persona = params.get("persona", "none")
        persona_counts[persona] = persona_counts.get(persona, 0) + 1

        # Latency
        if "total_latency_ms" in metrics:
            latencies.append(metrics["total_latency_ms"])

        # Confidence
        if "confidence" in metrics:
            confidence_values.append(metrics["confidence"])

    # ---------- latency stats ----------

    if latencies:
        latencies.sort()
        avg_latency = round(statistics.mean(latencies), 2)
        p50_latency = round(statistics.median(latencies), 2)
        p95_latency = round(latencies[int(len(latencies) * 0.95)], 2)
        max_latency = round(max(latencies), 2)
    else:
        avg_latency = p50_latency = p95_latency = max_latency = 0

    # ---------- confidence stats ----------

    avg_confidence = round(statistics.mean(confidence_values), 4) if confidence_values else 0

    # ---------- distributions ----------

    emotion_distribution = {
        emotion: {
            "count": count,
            "pct": round(count / total * 100, 1)
        }
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1])
    }

    persona_distribution = {
        persona: {
            "count": count,
            "pct": round(count / total * 100, 1)
        }
        for persona, count in sorted(persona_counts.items(), key=lambda x: -x[1])
    }

    return {
        "total_requests": total,
        "avg_confidence": avg_confidence,

        "latency": {
            "avg_ms": avg_latency,
            "p50_ms": p50_latency,
            "p95_ms": p95_latency,
            "max_ms": max_latency,
        },

        "emotion_distribution": emotion_distribution,
        "persona_distribution": persona_distribution,
    }