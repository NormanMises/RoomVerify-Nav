from __future__ import annotations

from typing import Dict, Iterable

from vlfm.utils.room_schema import (
    get_room_reasoner_keywords,
    normalize_scene_object_label,
)

RoomDistribution = Dict[str, float]


class RoomReasoner:
    """
    Extremely lightweight room-type reasoner.

    It takes a list of observed object class names (strings) and produces a
    probability distribution over coarse room categories such as
    \"kitchen\", \"bedroom\", etc.

    This is intentionally heuristic and cheap; if you want to plug in an
    external LLM, you can subclass this class or wrap `infer_room_type`.
    """

    def __init__(self) -> None:
        self._room_keywords = get_room_reasoner_keywords()
        self._shared_keywords = {"sink", "dining table"}

    def infer_room_type(self, observed_objects: Iterable[str]) -> RoomDistribution:
        scores: Dict[str, float] = {room: 0.0 for room in self._room_keywords}

        for name in set(observed_objects):
            lower_name = normalize_scene_object_label(name)
            for room, keywords in self._room_keywords.items():
                for kw in keywords:
                    if kw in lower_name:
                        # Shared keywords (e.g., sink) contribute less
                        weight = 0.5 if kw in self._shared_keywords else 1.0
                        scores[room] += weight
                        break

        total = sum(scores.values())
        if total <= 0.0:
            return {}

        return {room: s / total for room, s in scores.items() if s > 0.0}
