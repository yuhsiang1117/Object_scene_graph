"""`llm` group: the OpenAI-compatible endpoint shared by the text scorer and
the vision verifier. They differ only by model name."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMConfig:
    base_url: str = "${oc.env:OLLAMA_HOST,http://localhost:11434}/v1"
    api_key: str = "ollama"
    text_model: str = "qwen2.5vl:3b"
    vlm_model: str = "qwen2.5vl:3b"
    timeout_s: float = 120.0
    max_image_px: int = 512
    # Some OpenAI-compatible providers (e.g. NVIDIA NIM vision models) return
    # malformed output when sent response_format=json_object; setting this false
    # omits that param and parses the JSON out of the plain-text reply instead.
    send_response_format: bool = True


