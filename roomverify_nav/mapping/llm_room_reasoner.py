from __future__ import annotations

from collections import OrderedDict, Counter
from typing import Dict, Iterable, Optional
import os
import numpy as np

from roomverify_nav.mapping.room_reasoner import RoomReasoner, RoomDistribution
from roomverify_nav.utils.timing import TimingContext

from roomverify_nav.utils.query_llm import OpenAIClient
from roomverify_nav.utils.room_schema import (
    get_current_room_types,
    normalize_room_type_key,
    room_list_str,
)


class LLMRoomReasoner(RoomReasoner):
    """Infer room distributions using cached observations, heuristics, and an LLM."""

    def __init__(self, max_cache_size: Optional[int] = None):
        super().__init__()
        self._client = None
        self._cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._llm_call_count = 0

        self._debug = os.environ.get("DEBUG_ROOM_REASONER") == "1"

        self._last_llm_call_step = -999
        self._current_step = 0
        self._last_llm_result: Optional[RoomDistribution] = None

        self.cache_size = int(
            os.environ.get("PROMPTING_ROOM_REASONER_CACHE_SIZE", "128") if max_cache_size is None else max_cache_size
        )
        self.min_freq_threshold = float(os.environ.get("PROMPTING_ROOM_REASONER_MIN_FREQ", "0.15"))
        self.confidence_threshold = float(os.environ.get("PROMPTING_ROOM_REASONER_CONFIDENCE_THRESH", "0.6"))
        self.api_timeout = float(os.environ.get("PROMPTING_ROOM_REASONER_TIMEOUT", "5.0"))
        self.min_call_interval = int(os.environ.get("PROMPTING_ROOM_REASONER_MIN_INTERVAL", "10"))
        self.reasoning_effort = os.environ.get("PROMPTING_ROOM_REASONER_REASONING_EFFORT", "").strip()

        self._allowed_room_types = get_current_room_types(include_unknown=True)
        self._allowed_room_set = set(self._allowed_room_types)

        # System prompt for LLM (role and task definition)
        allowed_str = room_list_str(self._allowed_room_types)
        # Do NOT use f-strings here because the prompt includes JSON braces like
        # {"unknown": 1.0} which would be interpreted as a format expression.
        _prompt = """
You are a room type inference expert. Infer a probability distribution over a fixed set of room types from observed objects and their frequencies.

Hard constraints (must follow exactly):
- Allowed room types (keys): __ALLOWED_ROOMS__
- Output: a single flat JSON object only (no markdown, no extra text)
- Keys must be exactly from the allowed set (do NOT output synonyms; use the allowed keys only)
- Values must be decimal numbers and must sum to exactly 1.0
- If evidence is weak/ambiguous, output {"unknown": 1.0}

Input format:
- A single line like: chair(0.6), table(0.3), lamp(0.1)

Examples:
Input: chair(0.6), table(0.3), lamp(0.1)
Output: {"living_room": 0.6, "dining_room": 0.2, "unknown": 0.2}

Input: toilet(0.7), sink(0.2), shower(0.1)
Output: {"bathroom": 1.0}

Input: tv(0.4), bed(0.3), desk(0.3)
Output: {"living_room": 0.4, "bedroom": 0.3, "unknown": 0.3}

Input: objectA(0.5), objectB(0.5)
Output: {"unknown": 1.0}
"""
        self.system_prompt = _prompt.replace("__ALLOWED_ROOMS__", allowed_str)

        # User prompt template (specific task)
        self.user_prompt_template = "{}"

    def reset_episode(self) -> None:
        self._current_step = 0
        self._last_llm_call_step = -999
        self._last_llm_result = None

    def _get_client(self) -> Optional[OpenAIClient]:
        """Initialize and reuse the API client."""
        if self._client is not None:
            return self._client

        self._client = OpenAIClient.from_env()
        return self._client

    def _compute_frequencies(self, observed_objects: Iterable[str]) -> Dict[str, float]:
        """Normalize observed object counts into frequencies."""
        counts = Counter(observed_objects)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {obj: count / total for obj, count in counts.items()}

    def _make_cache_key(self, object_freq: Dict[str, float]) -> frozenset:
        """Select object classes above the frequency threshold."""
        filtered_objects = {obj for obj, freq in object_freq.items() if freq >= self.min_freq_threshold}
        return frozenset(filtered_objects)

    def _update_cache(self, key: frozenset, value: RoomDistribution) -> None:
        """Update the LRU cache and evict its oldest entry if needed."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value

        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    @property
    def cache_stats(self) -> Dict[str, float]:
        """Return cache usage and LLM call statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "capacity": self.cache_size,
            "llm_calls": self._llm_call_count,
        }

    def _should_call_llm(self, object_freq: Dict[str, float], heuristic_result: RoomDistribution) -> bool:
        """Check the call interval, heuristic confidence, and object diversity."""
        steps_since_last_call = self._current_step - self._last_llm_call_step
        if steps_since_last_call < self.min_call_interval:
            if self._debug:
                print(f"[LLMRoomReasoner] Skipping LLM (interval={steps_since_last_call}/{self.min_call_interval})")
            return False

        max_confidence = max(heuristic_result.values()) if heuristic_result else 0.0
        if max_confidence >= self.confidence_threshold:
            return False

        if len(object_freq) < 3:
            return False

        entropy = -sum(p * np.log(p + 1e-10) for p in object_freq.values())
        if entropy <= 0.5:
            return False

        return True

    def _build_user_message(self, object_freq: Dict[str, float]) -> str:
        """Format objects in descending frequency order."""
        objects_str = ", ".join(
            f"{obj}({freq:.2f})"
            for obj, freq in sorted(object_freq.items(), key=lambda x: (-x[1], x[0]))
        )
        return self.user_prompt_template.format(objects_str)

    def _query_llm(self, object_freq: Dict[str, float]) -> RoomDistribution:
        """Query the LLM and normalize its room distribution."""
        user_message = self._build_user_message(object_freq)

        client = self._get_client()
        if client is None:
            raise RuntimeError("OpenAI client not available")

        with TimingContext("LLMRoomReasoner._query_llm()", "LLM call"):
            data = client.call_openai_api(
                system_prompt=self.system_prompt,
                user_prompt=user_message,
                reasoning_effort=self.reasoning_effort or None,
            )

        try:
            if not isinstance(data, dict):
                return {}

            lowered_keys = {normalize_room_type_key(k) for k in data.keys()}
            lowered_vals = {
                normalize_room_type_key(v)
                for v in data.values()
                if isinstance(v, str)
            }
            if (
                "everywhere" in lowered_keys
                or "everywhere" in lowered_vals
                or "anywhere" in lowered_keys
                or "anywhere" in lowered_vals
            ):
                return {}

            numeric_items: Dict[str, float] = {}
            for k, v in data.items():
                if not isinstance(v, (int, float)):
                    continue
                kk = normalize_room_type_key(k)
                if kk in {"everywhere", "anywhere"}:
                    continue
                if kk not in self._allowed_room_set:
                    continue
                numeric_items[kk] = numeric_items.get(kk, 0.0) + float(v)

            total = sum(numeric_items.values())
            if total <= 0.0:
                return {}

            result = {k: v / total for k, v in numeric_items.items()}
        except Exception:
            return {}

        self._llm_call_count += 1
        return result

    def infer_room_type_from_freq(self, object_freq: Dict[str, float]) -> RoomDistribution:
        """Infer room probabilities from object frequencies."""
        self._current_step += 1

        cache_key = self._make_cache_key(object_freq)

        debug_mode = os.environ.get("DEBUG_ROOM_REASONER") == "1"

        if cache_key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(cache_key)
            result = self._cache[cache_key]
            if debug_mode:
                print(f"[LLMRoomReasoner] Cache HIT: {result}")
            return result

        self._cache_misses += 1

        expanded_objects = []
        for obj, freq in object_freq.items():
            count = int(freq * 100)
            expanded_objects.extend([obj] * count)

        heuristic_result = super().infer_room_type(expanded_objects)

        if self._get_client() is None:
            self._update_cache(cache_key, heuristic_result)
            return heuristic_result

        steps_since_last = self._current_step - self._last_llm_call_step
        within_interval = steps_since_last < self.min_call_interval

        if not self._should_call_llm(object_freq, heuristic_result):
            if within_interval and self._last_llm_result:
                if debug_mode:
                    print(
                        f"[LLMRoomReasoner] Reusing last LLM result (interval={steps_since_last}/{self.min_call_interval})"
                    )
                return self._last_llm_result

            if debug_mode:
                max_conf = max(heuristic_result.values()) if heuristic_result else 0.0
                print(f"[LLMRoomReasoner] Using heuristic (conf={max_conf:.2f})")

            self._update_cache(cache_key, heuristic_result)
            return heuristic_result

        if debug_mode:
            objects_str = ", ".join(f"{obj}({freq:.2f})" for obj, freq in object_freq.items())
            print(f"[LLMRoomReasoner] Calling LLM for: {objects_str}")

        try:
            llm_result = self._query_llm(object_freq)

            self._last_llm_call_step = self._current_step
            self._last_llm_result = llm_result

            if debug_mode:
                print(f"[LLMRoomReasoner] LLM result: {llm_result}")

            self._update_cache(cache_key, llm_result)

            return llm_result

        except Exception as e:
            if debug_mode:
                print(f"[LLMRoomReasoner] LLM query failed: {e}, falling back to heuristic")
            self._update_cache(cache_key, heuristic_result)
            return heuristic_result

    def infer_room_type(self, observed_objects: Iterable[str]) -> RoomDistribution:
        """Infer room probabilities from observed objects."""
        object_freq = self._compute_frequencies(observed_objects)

        return self.infer_room_type_from_freq(object_freq)
