"""Offline VLM prompt experiments against saved verification images.
Finds a phrasing the small VLM answers correctly before baking it in.

Usage: python scripts/prompt_lab.py <image.jpg> <target> [...more pairs]
"""
from __future__ import annotations

import os
import sys

import imageio.v2 as imageio
import numpy as np

from osg.llm.client import ChatClient

MODEL = os.environ.get("OSG_VLM", "qwen2.5vl:3b")

PROMPTS = {
    "current-verify": (
        "You verify object detections for a robot. Reject only clearly mislabeled "
        "objects; partial views or unusual angles of the right category count as "
        "correct. Answer with JSON only.",
        'The robot is looking for a {t}. The attached image(s) show the\n'
        'candidate object (possibly from a distance or partially occluded).\n'
        'Is there a {t} in the image(s)?\n'
        'Respond as JSON: {{"is_target": true/false, "confidence": <0-1>}}',
    ),
    "classify": (
        "You identify objects in images. Answer with JSON only.",
        'What is the main piece of furniture or object in this image? '
        'Respond as JSON: {{"object": "<one or two words>"}}',
    ),
    "simple-yesno": (
        "Answer with JSON only.",
        'Is there a {t} (or part of one) visible in this image? '
        'Respond as JSON: {{"answer": "yes" or "no"}}',
    ),
    "describe-then-decide": (
        "You help a robot double-check its object detector. Answer with JSON only.",
        'First describe what you see, then decide: does the image show a {t}, '
        'even partially or occluded? '
        'Respond as JSON: {{"description": "<short>", "is_target": true/false}}',
    ),
}


def main() -> None:
    args = sys.argv[1:]
    pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    client = ChatClient(os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1",
                        MODEL, timeout_s=180.0, max_image_px=512)
    for img_path, target in pairs:
        img = np.asarray(imageio.imread(img_path))[..., :3]
        print(f"\n=== {img_path} (target: {target}, {img.shape[1]}x{img.shape[0]}) ===")
        for name, (system, user) in PROMPTS.items():
            try:
                resp = client.chat(system, user.format(t=target), images=[img])
            except Exception as e:
                resp = f"ERROR {e}"
            print(f"  {name:22s} -> {resp}")


if __name__ == "__main__":
    main()
