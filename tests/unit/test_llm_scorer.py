from __future__ import annotations

from osg.exploration.llm_scorer import LLMTextScorer


def test_reset_clears_room_label_cache():
    """P1i follow-up: room.id restarts from 1 each episode (fresh
    RoomSegmenter per NavAgent), but the scorer instance persists across the
    whole eval run -- without reset(), a new scene's room 1 would silently
    inherit whatever a previous, unrelated scene's room 1 was labeled."""
    scorer = LLMTextScorer(client=None)
    scorer._room_label_cache[1] = "bedroom"
    scorer._room_label_cache[2] = "kitchen"

    scorer.reset()

    assert scorer._room_label_cache == {}
