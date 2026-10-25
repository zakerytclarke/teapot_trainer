"""Composable trainer for small causal language models."""
from .config import ModelSpec, Stage, TrainConfig, preset
from .data import (
    HuggingFaceSFT,
    HuggingFaceText,
    JsonlMessages,
    ListMessages,
    ListText,
    MemmapTokens,
    chat_turn,
    rag_messages,
    tool_call_text,
    tool_messages,
)
from .datasets import fineweb_edu, gsm8k, smoltalk, ultrafineweb
from .evals import GenerateEval, TaskEval, TokenPerplexity, parse_tool_call, score_completion
from .models import build_model
from .trainer import TeapotTrainer

__version__ = "0.2.0"
__all__ = [
    "GenerateEval",
    "HuggingFaceSFT",
    "HuggingFaceText",
    "JsonlMessages",
    "ListMessages",
    "ListText",
    "MemmapTokens",
    "ModelSpec",
    "Stage",
    "TaskEval",
    "TeapotTrainer",
    "TokenPerplexity",
    "TrainConfig",
    "build_model",
    "chat_turn",
    "fineweb_edu",
    "gsm8k",
    "parse_tool_call",
    "preset",
    "rag_messages",
    "score_completion",
    "smoltalk",
    "tool_call_text",
    "tool_messages",
    "ultrafineweb",
]
