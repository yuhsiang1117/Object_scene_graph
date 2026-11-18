"""Subset ASCENT's statistical priors into a few KB this repo can load.

ASCENT ships two priors that its LLM planner puts straight into the prompt:

  knowledge_graph.json                 2 MB networkx node-link graph, 1634
                                       objects x 10 room types, edge weight =
                                       co-occurrence probability
  hm3d_floor_object_possibility.xlsx   per-category distribution over storeys,
                                       conditioned on how many storeys the
                                       building has

Neither is usable here as-is: loading them the way ASCENT does needs networkx
and pandas, and this project has neither. Both are also far larger than the
part that matters -- we only ever ask about the six HM3D ObjectNav goal
categories.

So this converts once, host-side, into plain JSON keyed by those six
categories, small enough to commit. Rerun only if the goal set changes.

    python scripts/make_priors.py --ascent ../ascent
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

# HM3D ObjectNav goal categories, and the synonyms ASCENT's priors are keyed by
# (its knowledge graph says "sofa", the dataset says "couch", and so on).
GOALS = {
    "chair": ["chair"],
    "bed": ["bed"],
    "plant": ["potted plant", "plant"],
    "toilet": ["toilet"],
    "tv_monitor": ["tv", "tv_monitor", "television", "monitor"],
    "sofa": ["sofa", "couch"],
}
# ASCENT's REFERENCE_ROOMS (constants.py).
ROOMS = [
    "bathroom", "bedroom", "dining_room", "garage", "hall",
    "kitchen", "laundry_room", "living_room", "office", "rec_room",
]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def room_priors(kg_path: Path) -> dict:
    """{goal: {room: probability}} from the node-link graph, without networkx."""
    graph = json.loads(kg_path.read_text())
    rooms = set(ROOMS)
    edges: dict = {}
    for link in graph["links"]:
        src, dst, w = link["source"], link["target"], float(link.get("weight", 0.0))
        if dst in rooms:
            edges.setdefault(src, {})[dst] = w
        elif src in rooms:
            edges.setdefault(dst, {})[src] = w

    out = {}
    for goal, aliases in GOALS.items():
        found = next((edges[a] for a in aliases if a in edges), None)
        if found:
            out[goal] = {r: round(found.get(r, 0.0), 4) for r in ROOMS}
    return out


def _cells(xlsx: Path) -> list:
    """Rows of the first sheet as strings, resolving the shared-string table."""
    with zipfile.ZipFile(xlsx) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t"))
                      for si in root.findall("m:si", NS)]
        root = ElementTree.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.iter(f"{{{NS['m']}}}row"):
            cells = {}
            for c in row.iter(f"{{{NS['m']}}}c"):
                v = c.find("m:v", NS)
                if v is None or v.text is None:
                    continue
                text = shared[int(v.text)] if c.get("t") == "s" else v.text
                col = re.match(r"[A-Z]+", c.get("r") or "A").group(0)
                cells[col] = text
            rows.append(cells)
        return rows


def floor_priors(xlsx_path: Path) -> dict:
    """{goal: {"<total_floors>": {"<floor>": pct}}}.

    ASCENT's columns are train_floor{N}_{y}: given a building with N storeys,
    the share of this category's instances found on storey y (1 = ground).
    """
    rows = _cells(xlsx_path)
    if not rows:
        return {}
    header = rows[0]
    cols = {col: name for col, name in header.items()}
    cat_col = next((c for c, n in cols.items() if n.strip().lower() == "category"), None)
    if cat_col is None:
        return {}

    alias_to_goal = {a: g for g, aliases in GOALS.items() for a in aliases}
    out: dict = {}
    for row in rows[1:]:
        goal = alias_to_goal.get((row.get(cat_col) or "").strip().lower())
        if goal is None:
            continue
        per_total: dict = {}
        for col, name in cols.items():
            m = re.fullmatch(r"train_floor(\d+)_(\d+)", (name or "").strip())
            if not m or col not in row:
                continue
            total, floor = m.groups()
            try:
                per_total.setdefault(total, {})[floor] = round(float(row[col]), 3)
            except ValueError:
                continue
        if per_total:
            out[goal] = per_total
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ascent", default="../ascent", help="path to the ASCENT checkout")
    ap.add_argument("--out", default="data/priors")
    args = ap.parse_args()

    src = Path(args.ascent) / "statistic_priors"
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    rooms = room_priors(src / "knowledge_graph.json")
    floors = floor_priors(src / "hm3d_floor_object_possibility.xlsx")
    (dest / "hm3d_room_prior.json").write_text(json.dumps(rooms, indent=2, sort_keys=True))
    (dest / "hm3d_floor_prior.json").write_text(json.dumps(floors, indent=2, sort_keys=True))
    (dest / "README.md").write_text(
        "# Statistical priors\n\n"
        "Generated by `scripts/make_priors.py` from ASCENT\n"
        "(https://github.com/zeying-gong/ascent, arXiv:2505.23019), subset to the\n"
        "six HM3D ObjectNav goal categories. See that repo for provenance and\n"
        "licence of the underlying `statistic_priors/` data.\n\n"
        "- `hm3d_room_prior.json` — P(room type | goal category)\n"
        "- `hm3d_floor_prior.json` — P(storey | goal category, building storeys)\n"
    )
    print(f"rooms: {sorted(rooms)}")
    print(f"floors: {sorted(floors)}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
