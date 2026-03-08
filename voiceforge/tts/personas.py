import os
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class Persona:
    name: str
    rate: int
    volume: float
    style: str
    description: str


_DEFAULTS: dict[str, Persona] = {
    "narrator": Persona(
        name="narrator", rate=160, volume=0.85,
        style="authoritative",
        description="Deep, measured storytelling voice",
    ),
    "therapist": Persona(
        name="therapist", rate=140, volume=0.65,
        style="gentle",
        description="Slow, soft, reassuring tone",
    ),
    "broadcaster": Persona(
        name="broadcaster", rate=185, volume=0.95,
        style="energetic",
        description="Fast, punchy, confident delivery",
    ),
    "assistant": Persona(
        name="assistant", rate=170, volume=0.80,
        style="neutral",
        description="Balanced, clear, professional voice",
    ),
    "storyteller": Persona(
        name="storyteller", rate=155, volume=0.75,
        style="expressive",
        description="Warm, engaging, varied pacing",
    ),
}


def load_personas(config_path: str = "configs/personas.yaml") -> dict[str, Persona]:
    """Load personas from YAML, falling back to built-in defaults."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return {k: Persona(**v) for k, v in data.items()}
    return _DEFAULTS


def get_persona(name: str, config_path: str = "configs/personas.yaml") -> Optional[Persona]:
    return load_personas(config_path).get(name.lower())
