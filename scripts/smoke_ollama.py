"""M0 smoke test: text + vision round-trip through the ollama service."""
from __future__ import annotations

import os
import sys

import numpy as np

from osg.llm.client import ChatClient

MODEL = os.environ.get("OSG_SMOKE_MODEL", "qwen2.5vl:3b")


def main() -> int:
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1"
    client = ChatClient(base_url, MODEL, timeout_s=120.0)

    resp = client.chat(
        "You answer with JSON only.",
        'Reply with JSON: {"ping": "pong"}',
    )
    print(f"text round-trip: {resp}")
    assert resp.get("ping") == "pong", resp

    # Vision: a red square on white — ask for the dominant color
    img = np.full((128, 128, 3), 255, dtype=np.uint8)
    img[32:96, 32:96] = (220, 30, 30)
    resp = client.chat(
        "You answer with JSON only.",
        'What is the dominant color of the square in the image? JSON: {"color": "<name>"}',
        images=[img],
    )
    print(f"vision round-trip: {resp}")
    color = str(resp.get("color", "")).lower()
    ok = "red" in color
    print("SMOKE OK" if ok else f"SMOKE WARNING: expected red, got {color!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
