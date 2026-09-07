"""OpenAI-compatible chat client. Points at Ollama's /v1 endpoint by default;
swapping to a cloud provider is a base_url/api_key config change.
"""
from __future__ import annotations

import base64
import json
import re
from typing import List, Optional

import numpy as np


def _encode_image(img: np.ndarray, max_px: int = 512, quality: int = 80) -> str:
    import cv2

    h, w = img.shape[:2]
    scale = max_px / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in a model response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"no JSON object in response: {text[:200]!r}")


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        timeout_s: float = 60.0,
        max_image_px: int = 512,
        send_response_format: bool = True,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.model = model
        self.max_image_px = max_image_px
        # When False, do not send response_format=json_object (some providers,
        # e.g. NIM vision models, mangle their output in that mode); JSON is
        # still parsed from the text reply below.
        self.send_response_format = send_response_format

    def chat(
        self,
        system: str,
        user: str,
        images: Optional[List[np.ndarray]] = None,
        json_response: bool = True,
        temperature: float = 0.0,
    ) -> dict:
        """Returns the parsed JSON response; retries once on parse failure.

        temperature 0, not 0.1. Measured (docs/AB_RESULTS, "The A/B noise
        floor"): a baseline replicated against itself is only 48/50
        bit-identical, and every episode that diverged had made model calls
        while every episode that made none was identical in every comparison.
        Sampling noise in these calls is the mechanism, and it puts a floor of
        net 0..+2 under every A/B in this repo. ASCENT runs its planner at
        temperature 0 with do_sample=False (model_api/qwen25_out.py:62-66);
        there is no reason to pay for sampling in a forced-choice task.
        """
        content: list = [{"type": "text", "text": user}]
        for img in images or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(img, self.max_image_px)}"},
                }
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        kwargs = {}
        if json_response and self.send_response_format:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(2):
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, **kwargs
            )
            text = resp.choices[0].message.content or ""
            if not json_response:
                return {"text": text}
            try:
                return extract_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {"role": "user", "content": "Respond with ONLY the JSON object, nothing else."}
                )
        raise ValueError(f"LLM did not return parseable JSON: {last_err}")
