from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

# Centralized room-type schema used by:
# - current-room reasoning (RoomReasoner / LLMRoomReasoner / RoomStateTracker)
# - target guidance provider (TargetGuidanceProvider)

# Keep the order as provided by the user.
EXTENDED_ROOM_TYPES: Tuple[str, ...] = (
    "unknown",
    "bedroom",
    "bathroom",
    "living_room",
    "lounge",
    "kitchen",
    "storeroom",
    "hallway",
    "gym",
    "dining_room",
)

_SCENE_ROOM_CUES: Dict[str, Tuple[str, ...]] = {
    "bedroom": ("bed", "wardrobe", "dresser", "nightstand", "teddy bear"),
    "bathroom": ("toilet", "bathtub", "shower", "sink", "hair drier"),
    "living_room": ("couch", "tv"),
    "lounge": ("desk", "laptop", "computer"),
    "kitchen": ("refrigerator", "oven", "microwave", "toaster", "sink"),
    "dining_room": ("dining table", "chair"),
    "storeroom": ("shelf", "box", "carton", "cabinet"),
    "hallway": ("stairs",),
    "gym": ("treadmill", "dumbbell"),
}

# phrase -> canonical object label
_SCENE_OBJECT_ALIASES: Dict[str, str] = {
    "sofa": "couch",
    "television": "tv",
    "fridge": "refrigerator",
    "staircase": "stairs",
    "dining_table": "dining table",
    "hair dryer": "hair drier",
    # Common plural variants from open-vocab detectors.
    "boxes": "box",
    "cabinets": "cabinet",
    "drawers": "dresser",
    "dumbbells": "dumbbell",
}

# Sparse GDINO supplement classes (coverage-first, cost-aware).
_SCENE_GDINO_SPARSE_LABELS: Tuple[str, ...] = (
    "bathtub",
    "shower",
    "wardrobe",
    "desk",
    "computer",
    "shelf",
    "box",
    "carton",
    "cabinet",
    "stairs",
    "staircase",
    "treadmill",
    "dumbbell",
    "dresser",
    "nightstand",
)


def get_current_room_types(*, include_unknown: bool = True) -> Tuple[str, ...]:
    # Fixed: extended 10-class taxonomy (already includes unknown).
    if include_unknown:
        return EXTENDED_ROOM_TYPES
    return tuple(r for r in EXTENDED_ROOM_TYPES if r != "unknown")


def get_target_room_types() -> Tuple[str, ...]:
    # Target prior should not include "unknown" as a room type. "No strong prior"
    # is represented by returning {} (or "everywhere"/"anywhere" from the LLM).
    return get_current_room_types(include_unknown=False)


def get_scene_room_cues() -> Dict[str, Set[str]]:
    return {room: set(cues) for room, cues in _SCENE_ROOM_CUES.items()}


def get_room_reasoner_keywords() -> Dict[str, Set[str]]:
    """
    Keywords used by heuristic room reasoner.

    This extends canonical cues with a few compatibility aliases so historical
    detector phrases still map to the intended room.
    """
    kws = get_scene_room_cues()
    kws["living_room"].update({"sofa", "television"})
    kws["kitchen"].update({"fridge"})
    kws["dining_room"].update({"dining_table"})
    kws["storeroom"].update({"storage"})
    kws["hallway"].update({"staircase", "hallway"})
    return kws


def get_scene_detector_labels() -> Tuple[str, ...]:
    labels: Set[str] = set()
    for cues in _SCENE_ROOM_CUES.values():
        labels.update(cues)
    return tuple(sorted(labels))


def get_scene_gdino_caption_labels(exclude_canonical: Iterable[str] = ()) -> Tuple[str, ...]:
    excluded = {normalize_scene_object_label(label) for label in exclude_canonical}
    if not excluded:
        return _SCENE_GDINO_SPARSE_LABELS
    return tuple(label for label in _SCENE_GDINO_SPARSE_LABELS if normalize_scene_object_label(label) not in excluded)


def normalize_scene_object_label(label: object) -> str:
    s = str(label).strip().lower().replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return _SCENE_OBJECT_ALIASES.get(s, s)


def normalize_room_type_key(key: object) -> str:
    """Normalize room-type keys from LLM outputs."""
    s = str(key).strip().lower().replace(" ", "_")

    # Common punctuation variants.
    s = s.replace("-", "_")

    # Canonicalize common synonyms/variants.
    base_aliases: Dict[str, str] = {
        "livingroom": "living_room",
        "bath_room": "bathroom",
        "diningroom": "dining_room",
        "store_room": "storeroom",
        "storage_room": "storeroom",
        "storage": "storeroom",
        "hall_way": "hallway",
        "corridor": "hallway",
        "exercise_room": "gym",
        "workout_room": "gym",
    }

    # Map work-like terms to lounge (no workspace class).
    schema_aliases = {
        "office": "lounge",
        "study": "lounge",
        "workspace": "lounge",
    }

    if s in schema_aliases:
        return schema_aliases[s]
    return base_aliases.get(s, s)


def room_list_str(rooms: Iterable[str]) -> str:
    items = [str(r) for r in rooms]
    return ", ".join(items)
