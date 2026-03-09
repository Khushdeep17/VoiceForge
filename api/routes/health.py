from fastapi import APIRouter
from voiceforge.tts.personas import load_personas
from voiceforge.metrics.tracker import get_aggregate_stats

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "service": "VoiceForge", "version": "2.0.0"}


@router.get("/personas")
def list_personas():
    """Return all available voice personas."""
    personas = load_personas()
    return {
        name: {"style": p.style, "description": p.description, "rate": p.rate, "volume": p.volume}
        for name, p in personas.items()
    }


@router.get("/metrics")
def metrics():
    """Aggregate inference stats from MLflow: emotion distribution, latency, totals."""
    return get_aggregate_stats()
