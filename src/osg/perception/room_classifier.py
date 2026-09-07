"""Room type from a single RGB frame, via Places365 scene classification.

Ported from ASCENT, which classifies every keyframe with a Places365 ResNet-50
and maps the result onto ten reference room types
(`ascent/map_controller.py:816-830`, `ascent/utils.py:209-229`).

This exists because of a measured hole rather than a hunch. `RoomNode.label` had
exactly one writer -- `LLMTextScorer.score` -- and every arm in docs/AB_RESULTS
runs `exploration.scorer=nearest`, a NullScorer. So room labels were always
None, and the S10 ranker experiment described every frontier to the model as
"a unknown room containing objects: ...". The room-to-goal priors were in the
prompt with nothing to match them against.

Places365 is 97 MB and runs in ~3 ms; the alternative (asking an LLM to name a
room from its object list) costs a network round trip and was what the disabled
scorer did.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_WEIGHTS = "data/place365/resnet50_places365.pth.tar"
DEFAULT_CATEGORIES = "data/place365/categories_places365.txt"
DEFAULT_MAP = "data/priors/place365_room_map.json"

# ImageNet statistics, matching the transform ASCENT applies
# (map_controller.py:114-119).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_categories(path: str) -> List[str]:
    """`/a/airfield 0` -> `airfield`; `/b/garage/indoor 1` -> `garage/indoor`.

    Places365 prefixes each line with a letter directory and suffixes the class
    index. The mapping table is keyed on the middle part, so both must go.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        name = line.rsplit(" ", 1)[0]
        out.append(name[3:] if name.startswith("/") else name)
    return out


def map_to_room(top_classes: Sequence[str], direct_mapping: dict,
                reference_rooms: Sequence[str]) -> str:
    """Port of `extract_room_categories` (ascent/utils.py:209-229).

    Walks the top-k in rank order and returns the first that maps into the
    reference set. Falls back to the raw top-1 -- deliberately, and it matters:
    ASCENT would rather tell the LLM "attic" than "unknown", because an
    unmapped-but-real room name is still information.
    """
    for name in top_classes:
        mapped = direct_mapping.get(name)
        if mapped is not None and mapped in reference_rooms:
            return mapped
    return top_classes[0] if top_classes else "unknown room"


class Place365RoomClassifier:
    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        categories: str = DEFAULT_CATEGORIES,
        room_map: str = DEFAULT_MAP,
        device: str = "cuda",
        topk: int = 5,
    ) -> None:
        import torch
        from torchvision import models

        self.topk = topk
        self.device = device
        self.categories = load_categories(categories)
        table = json.loads(Path(room_map).read_text())
        self.direct_mapping = table["direct_mapping"]
        self.reference_rooms = set(table["reference_rooms"])

        model = models.resnet50(num_classes=365)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(state, strict=False)
        self.model = model.to(device).eval()
        self._torch = torch

    def _preprocess(self, rgb: np.ndarray):
        """Resize 256 -> centre crop 224 -> normalise, as ASCENT's transform does.

        cv2 rather than PIL, to stay inside the dependencies already installed.
        """
        import cv2

        h, w = rgb.shape[:2]
        scale = 256.0 / min(h, w)
        img = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))))
        y = (img.shape[0] - 224) // 2
        x = (img.shape[1] - 224) // 2
        img = img[y:y + 224, x:x + 224].astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        t = self._torch.from_numpy(img).permute(2, 0, 1)[None]
        return t.to(self.device)

    def classify(self, rgb: np.ndarray) -> str:
        with self._torch.no_grad():
            logits = self.model(self._preprocess(rgb))
            idx = logits.softmax(1)[0].argsort(descending=True)[: self.topk]
        top = [self.categories[int(i)] for i in idx]
        return map_to_room(top, self.direct_mapping, self.reference_rooms)


def build_room_classifier(cfg) -> Optional["Place365RoomClassifier"]:
    if str(getattr(cfg.scene_graph, "room_classifier", "none")) != "place365":
        return None
    try:
        return Place365RoomClassifier(device=cfg.detector.device)
    except (OSError, FileNotFoundError):
        # Weights are downloaded separately; exploration must not become
        # unrunnable because they are absent.
        return None
