import os
from collections import deque
from typing import Dict, List, Tuple
from roomverify_nav.mapping.llm_room_reasoner import LLMRoomReasoner
from roomverify_nav.utils.timing import timed


class RoomStateTracker:
    """Track room state from recent object observations."""

    def __init__(
        self,
        window_size: int = 10,
        change_threshold: float = 0.5,
        min_confidence_for_switch: float = 0.4,
        max_per_class_per_frame: int = 1,
    ):
        self.observation_window = deque(maxlen=window_size)
        self.current_room_type: str = "unknown"
        self.current_room_confidence: float = 0.0
        self.last_room_dist: Dict[str, float] = {}
        self.last_room_confidence: float = 0.0
        self.change_threshold = change_threshold
        self.min_confidence_for_switch = min_confidence_for_switch
        self.max_per_class_per_frame = int(
            os.environ.get("PROMPTING_ROOM_TRACKER_MAX_PER_CLASS", str(max_per_class_per_frame))
        )

        self.last_room_signature: Dict[str, float] = {}

        self._room_reasoner = LLMRoomReasoner()

    def reset(self) -> None:
        self.observation_window.clear()
        self.current_room_type = "unknown"
        self.current_room_confidence = 0.0
        self.last_room_dist = {}
        self.last_room_confidence = 0.0
        self.last_room_signature = {}
        self._room_reasoner.reset_episode()

    def get_room_dist(self, top_k: int = 3) -> Dict[str, float]:
        """Return the most recent room probabilities, limited to top_k entries."""
        if not self.last_room_dist:
            return {}
        items = sorted(self.last_room_dist.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return dict(items)

    def update(self, detected_objects: List[str]) -> Tuple[str, float, bool]:
        """Update room state and report a confirmed room change."""
        self.observation_window.append(detected_objects)

        aggregated_objects = self._aggregate_observations()

        room_dist = self._infer_room_from_objects(aggregated_objects)
        self.last_room_dist = room_dist
        new_room_type = max(room_dist, key=room_dist.get) if room_dist else "unknown"
        new_confidence = room_dist.get(new_room_type, 0.0)
        self.last_room_confidence = new_confidence

        type_changed = new_room_type != self.current_room_type
        object_dist_changed = self._detect_object_distribution_change(aggregated_objects)

        is_room_changed = False

        if type_changed:
            if new_confidence >= self.min_confidence_for_switch and object_dist_changed:
                self.current_room_type = new_room_type
                self.current_room_confidence = new_confidence
                self.last_room_signature = self._compute_signature(aggregated_objects)
                is_room_changed = True
        else:
            if new_confidence > self.current_room_confidence:
                self.current_room_confidence = new_confidence
                self.last_room_signature = self._compute_signature(aggregated_objects)

        return self.current_room_type, self.current_room_confidence, is_room_changed

    def _aggregate_observations(self) -> Dict[str, float]:
        """Aggregate recent observations with recency weights and per-class limits."""
        all_objects = []
        weights = []

        for i, objects in enumerate(self.observation_window):
            weight = (i + 1) / len(self.observation_window)
            if self.max_per_class_per_frame > 0:
                counts: Dict[str, int] = {}
                capped_objects: List[str] = []
                for obj in objects:
                    c = counts.get(obj, 0)
                    if c >= self.max_per_class_per_frame:
                        continue
                    counts[obj] = c + 1
                    capped_objects.append(obj)
            else:
                capped_objects = list(objects)

            all_objects.extend(capped_objects)
            weights.extend([weight] * len(capped_objects))

        weighted_counts = {}
        for obj, w in zip(all_objects, weights):
            weighted_counts[obj] = weighted_counts.get(obj, 0) + w

        total = sum(weighted_counts.values())
        if total > 0:
            return {k: v / total for k, v in weighted_counts.items()}
        return {}

    def _infer_room_from_objects(self, object_freq: Dict[str, float]) -> Dict[str, float]:
        """Infer a room distribution from object frequencies."""
        return self._room_reasoner.infer_room_type_from_freq(object_freq)

    def _compute_signature(self, object_freq: Dict[str, float]) -> Dict[str, float]:
        """Select frequent objects for room-change detection."""
        threshold = 0.1
        return {k: v for k, v in object_freq.items() if v > threshold}

    def _detect_object_distribution_change(self, current_objects: Dict[str, float]) -> bool:
        """Compare object distributions using weighted Jaccard similarity."""
        if not self.last_room_signature:
            return True

        current_sig = self._compute_signature(current_objects)

        all_keys = set(current_sig.keys()) | set(self.last_room_signature.keys())
        intersection = sum(min(current_sig.get(k, 0), self.last_room_signature.get(k, 0)) for k in all_keys)
        union = sum(max(current_sig.get(k, 0), self.last_room_signature.get(k, 0)) for k in all_keys)

        if union == 0:
            return False

        similarity = intersection / union

        return similarity < (1 - self.change_threshold)

    def get_stable_objects(self, top_k: int = 5) -> List[str]:
        """Return the most frequent objects in recent observations."""
        aggregated = self._aggregate_observations()
        sorted_objects = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        return [obj for obj, _freq in sorted_objects[:top_k]]
