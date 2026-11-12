"""Task / environment registry for RL."""

from __future__ import annotations

from ..env import Env
from .math import SynthMathEnv

ENV_REGISTRY: dict[str, type[Env]] = {
    "synth_math": SynthMathEnv,
}


def build_env(name: str, options: dict | None = None) -> Env:
    key = str(name).strip()
    if key not in ENV_REGISTRY:
        known = ", ".join(sorted(ENV_REGISTRY))
        raise ValueError(f"Unknown env {name!r}. Registered: {known}")
    return ENV_REGISTRY[key](options)
