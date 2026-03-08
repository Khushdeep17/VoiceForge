from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    scores: Optional[Dict[str, float]]
    viz_path: Optional[str]
