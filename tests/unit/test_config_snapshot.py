"""Every calibrated constant, pinned.

Almost every default in `OSGConfig` was chosen by an experiment and carries the
measurement in a comment beside it: `search_proximity_len_m` is 1.0 because a
114-relocation offline ranking said so, `min_presence` is 0.45 because that is
where one detector-strength absence reading lands and one VLM reading does not,
`stair_min_rise_m` is 1.0 because every false positive on the single-floor gate
rose 0.33-0.75 m.

A refactor that moves those files must not move those numbers. This flattens the
whole config to `{dotted.key: value}` and compares it against a committed
snapshot, so a constant can only change when someone changes the snapshot too --
which makes it show up in the diff as what it is, rather than as a line lost in a
file move.

To change a default on purpose:  python tests/unit/test_config_snapshot.py
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from osg.core.config import OSGConfig

SNAPSHOT = Path(__file__).parent / "golden" / "config_snapshot.json"


def flatten(obj, prefix: str = "") -> dict:
    """`{dotted.key: value}` over a nested dataclass tree.

    Tuples become lists so the JSON round-trip is exact -- `container_top_h_m`
    is a tuple in the dataclass and a list once it has been through a file.
    """
    out = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        key = f"{prefix}{field.name}"
        if dataclasses.is_dataclass(value):
            out.update(flatten(value, key + "."))
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def test_no_calibrated_default_has_moved():
    current = flatten(OSGConfig())
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    added = sorted(set(current) - set(expected))
    removed = sorted(set(expected) - set(current))
    changed = {
        k: (expected[k], current[k])
        for k in sorted(set(current) & set(expected))
        if current[k] != expected[k]
    }
    assert not changed, f"defaults changed (was, is): {changed}"
    assert not removed, f"config fields removed: {removed}"
    assert not added, f"config fields added without updating the snapshot: {added}"


def test_every_field_is_reachable_by_its_dotted_name():
    """The snapshot is only a lock if its keys are the names Hydra overrides
    use -- `exploration.search_posterior=true` on the command line has to reach
    the same field this file pins."""
    cfg = OSGConfig()
    for dotted in flatten(cfg):
        node = cfg
        for part in dotted.split("."):
            node = getattr(node, part)


if __name__ == "__main__":  # regenerate deliberately, never automatically
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(
        json.dumps(flatten(OSGConfig()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {SNAPSHOT}")
