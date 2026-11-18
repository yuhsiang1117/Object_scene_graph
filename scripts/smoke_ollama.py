#!/usr/bin/env python3
"""M0 smoke: is the configured LLM endpoint up, and does it hold the models the
config asks for?

A run refuses to start when a model server it needs is down, and that refusal is
deliberate (see CLAUDE.md) -- but it arrives after Habitat has loaded, which is
slow and buries the reason. This does the same round-trip in a second, against
whatever `llm` group a run would actually compose.

    python scripts/smoke_ollama.py                 # the default llm group
    python scripts/smoke_ollama.py llm=qwen_local  # any Hydra override

Exits non-zero if the endpoint is unreachable, if a configured model is not
present, or if the round-trip does not come back as JSON.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO = Path(__file__).resolve().parent.parent


def compose_llm(overrides: list[str]):
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        return compose(config_name="config", overrides=overrides).llm


def tags(base_url: str) -> list[str]:
    """Model names the server holds. `base_url` ends in /v1; /api/tags does not."""
    import json
    import urllib.error
    import urllib.request

    root = base_url.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=10) as fh:
            return sorted(m["name"] for m in json.load(fh).get("models", []))
    except urllib.error.URLError as exc:
        raise SystemExit(f"FAIL: {root} is unreachable -- {exc.reason}")


def main(argv: list[str]) -> int:
    cfg = compose_llm(argv)
    print(f"base_url    {cfg.base_url}")
    print(f"text_model  {cfg.text_model}")
    print(f"vlm_model   {cfg.vlm_model}")

    present = tags(cfg.base_url)
    print(f"served      {', '.join(present) or '(none)'}")

    wanted = {str(cfg.text_model), str(cfg.vlm_model)}
    missing = sorted(name for name in wanted if name not in present)
    if missing:
        # Worth failing on rather than warning: the run would start, reach its
        # first LLM call, and only then report a model the host never had.
        print(f"FAIL: not served: {', '.join(missing)}")
        print(f"      pull them with: ollama pull {missing[0]}")
        return 1

    from osg.llm.client import ChatClient

    client = ChatClient(
        base_url=str(cfg.base_url),
        model=str(cfg.text_model),
        api_key=str(cfg.api_key),
        timeout_s=float(cfg.timeout_s),
        send_response_format=bool(cfg.send_response_format),
    )
    reply = client.chat(
        system="You answer with JSON only.",
        user='Reply with exactly {"ok": true}.',
    )
    print(f"round-trip  {reply}")
    if not isinstance(reply, dict):
        print("FAIL: reply did not parse as a JSON object")
        return 1
    print("OK: the LLM endpoint answers and parses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
