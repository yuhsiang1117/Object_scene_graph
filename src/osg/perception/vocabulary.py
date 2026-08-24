"""Which query strings the detector is asked about.

An open-vocabulary head does not detect objects, it scores NAMES -- and it runs
class-competitive NMS over its own vocabulary, so which names are in the list
changes what gets found. Two whole experiment conditions were about nothing
else, and both of their results are counter-intuitive enough to be worth
stating where the rule lives:

  the name must be the one the DETECTOR answers to, not the one a person would
  use. "pitcher" scores 0.00 on the YCB asset at every resolution and against
  eight synonyms; "blue plastic pitcher" reaches 0.71. Over 96 episodes the
  agent had the tomato soup can at least half visible in 431 keyframes and the
  detector named it twelve times -- an in-situ recall of 0.03 against 0.90 for
  the bowl.

  but it must be the SPECIFIC name, not the highest-scoring one. Condition J
  took the top scorers from a re-probe -- "cylindrical can" 0.75 over "tin can"
  0.55, "red dish" 0.79 over "red plate" 0.76 -- and they cost more than they
  were worth. On their own targets they were a large win (soup can recall
  0.03 -> 0.21, SR 0.083 -> 0.500) but the rest of the vocabulary gave back
  eight episodes for their seven, because "cylindrical can" describes a shape
  the pitcher and the bleach bottle also have: the cracker box went SR
  0.667 -> 0.333. Condition K kept the specific names and took in_anchor to its
  best value of the whole campaign, 0.625.

The measurement tables behind the shipped vocabulary and the per-class gates are
in docs/ARCHITECTURE.md; `configs/experiment/ycb_authored_nav.yaml` holds the
list itself.
"""
from __future__ import annotations

from typing import List


def target_vocabulary(target: str, vocabulary) -> List[str]:
    """Target first, then the generic list with anything that COLLIDES removed.

    Measured on the YCB benchmark: with the target "cracker box" the vocabulary
    also offered the generic "box", and YOLOE labelled every sighting "box" --
    263 mapped tracks, 3 of them the target, none of them proposable, because
    candidates() matches on the target category. The specific class was in the
    vocabulary and still never won.

    So drop a generic entry that is a whole-word part of the target ("box" for
    "cracker box"), and drop an exact duplicate of the target. Anything that is
    not a sub-phrase of the target is left alone -- this narrows the vocabulary
    only where it was actively competing with the goal.
    """
    target = str(target).replace("_", " ").strip()
    words = target.lower().split()
    out = [target]
    for entry in vocabulary:
        text = str(entry).replace("_", " ").strip()
        low = text.lower()
        if low == target.lower():
            continue
        parts = low.split()
        n = len(parts)
        if n < len(words) and any(words[i:i + n] == parts for i in range(len(words) - n + 1)):
            continue
        out.append(text)
    return out
