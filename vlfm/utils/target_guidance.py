from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from icecream import ic

from vlfm.utils.timing import TimingContext
from vlfm.utils.query_llm import OpenAIClient
from vlfm.utils.room_schema import (
    get_target_room_types,
    normalize_room_type_key,
    room_list_str,
)

RoomDistribution = Dict[str, float]
RoomPriorWithImpossible = Tuple[RoomDistribution, RoomDistribution]


@dataclass
class TargetGuidanceProvider:
    """Estimate target-level room priors and impossible rooms."""

    _cache: Dict[str, RoomPriorWithImpossible] = field(default_factory=dict)
    _client: Optional[OpenAIClient] = field(default=None)
    _debug: bool = field(default_factory=lambda: os.environ.get("DEBUG_TARGET_GUIDANCE", "0") == "1")

    # ------------------------------------------------------------------
    # SiliconFlow client (requests)
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[OpenAIClient]:
        if self._client is not None:
            return self._client

        self._client = OpenAIClient.from_env()
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_room_prior_with_impossible(self, target_name: str) -> RoomPriorWithImpossible:
        target_name = target_name.strip().lower()
        if not target_name:
            return {}, {}

        if target_name in self._cache:
            if self._debug:
                ic("TargetGuidanceProvider.room_prior.cache_hit", target_name, self._cache[target_name])
            return self._cache[target_name]

        client = self._get_client()
        if client is not None:
            result = self._query_openai_backend(client, target_name)
        else:
            result = (self._heuristic_prior(target_name), {})
            if self._debug:
                ic(
                    "TargetGuidanceProvider.room_prior.no_client_fallback",
                    target_name,
                    result,
                )

        self._cache[target_name] = result
        if self._debug:
            ic("TargetGuidanceProvider.room_prior.result", target_name, result)
        return result

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    def _query_openai_backend(
        self, client: OpenAIClient, target_name: str
    ) -> RoomPriorWithImpossible:
        allowed_rooms = get_target_room_types()
        allowed_str = room_list_str(allowed_rooms)
        # Do NOT use f-strings here because the prompt includes JSON braces like
        # {"room_prior": {...}} which would be interpreted as a format expression.
        _prompt = """
You are an expert in indoor semantic priors for Embodied Navigation.

### Task
Given one target object category, estimate:
1. **room_prior**: a probability distribution over rooms where this target is likely to physically appear.
2. **room_impossible**: a soft incompatibility map over rooms where this target is unlikely to physically appear.

### Input
The user prompt contains exactly one target object category name, such as:
- bed
- toilet
- refrigerator
- dining table

### Allowed Room Keys
You may ONLY use the following room keys:
__ALLOWED_ROOMS__

### Output Requirements
- Output a single JSON object only.
- Do not output markdown fences, explanations, comments, or extra fields.
- The top-level JSON may contain only these keys:
  - "room_prior"
  - "room_impossible"
- "room_prior" must be a JSON object whose values are decimal numbers summing to exactly 1.0.
- "room_impossible" must be a JSON object whose values are decimal numbers in [0, 1].
- All room keys must use the allowed snake_case names exactly.

### Important Decision Rule
- Use {} only when the target is genuinely room-agnostic or too ambiguous to support a meaningful room prior.
- Do NOT use {} for common indoor furniture or appliances with clear room semantics.
- For example, targets like bed, toilet, bathtub, refrigerator, oven, dining table, couch, tv, desk, wardrobe should produce a non-empty prior.

### Examples
Input: bed
Output:
{
  "room_prior": {
    "bedroom": 1.0
  },
  "room_impossible": {
    "bathroom": 0.9,
    "kitchen": 0.8,
    "gym": 0.7
  }
}

Input: toilet
Output:
{
  "room_prior": {
    "bathroom": 1.0
  },
  "room_impossible": {
    "bedroom": 0.8,
    "kitchen": 0.9,
    "dining_room": 0.9
  }
}

Input: bottle
Output:
{}
"""
        system_prompt = _prompt.replace("__ALLOWED_ROOMS__", allowed_str)
        user_prompt = (
            "### Target Object\n"
            f"{target_name}\n\n"
            "### Required Output\n"
            "Return JSON only with room_prior and room_impossible for this target object."
        )

        try:
            empty: RoomPriorWithImpossible = ({}, {})
            with TimingContext("TargetGuidanceProvider._openai_room_prior_call()", "OpenAI room-prior call"):
                data = client.call_openai_api(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    reasoning_effort=os.environ.get("TARGET_GUIDANCE_REASONING_EFFORT", "").strip(),
                )
            if self._debug:
                ic("TargetGuidanceProvider.room_prior.raw_response", target_name, data)

            if not isinstance(data, dict):
                if self._debug:
                    ic("TargetGuidanceProvider.room_prior.invalid_response_type", target_name, type(data))
                return empty

            allowed = set(allowed_rooms)

            def _normalize_prior(raw: object) -> RoomDistribution:
                if not isinstance(raw, dict):
                    return {}
                items: RoomDistribution = {}
                for k, v in raw.items():
                    if not isinstance(v, (int, float)):
                        continue
                    kk = normalize_room_type_key(k)
                    if kk not in allowed:
                        continue
                    vv = float(v)
                    if vv <= 0.0:
                        continue
                    items[kk] = items.get(kk, 0.0) + vv
                total = sum(items.values())
                if total <= 0.0:
                    return {}
                return {k: v / total for k, v in items.items()}

            def _normalize_impossible(raw: object) -> RoomDistribution:
                if not isinstance(raw, dict):
                    return {}
                items: RoomDistribution = {}
                for k, v in raw.items():
                    if not isinstance(v, (int, float)):
                        continue
                    kk = normalize_room_type_key(k)
                    if kk not in allowed:
                        continue
                    vv = float(v)
                    vv = min(1.0, max(0.0, vv))
                    items[kk] = max(items.get(kk, 0.0), vv)
                return items

            if not data or ("room_prior" not in data and "room_impossible" not in data):
                if self._debug:
                    ic("TargetGuidanceProvider.room_prior.empty_payload", target_name, data)
                return empty

            prior = _normalize_prior(data.get("room_prior", {}))
            impossible = _normalize_impossible(data.get("room_impossible", {}))
            if self._debug:
                ic(
                    "TargetGuidanceProvider.room_prior.normalized",
                    target_name,
                    {"room_prior": prior, "room_impossible": impossible},
                )
            return prior, impossible

        except Exception as exc:
            fallback = (self._heuristic_prior(target_name), {})
            if self._debug:
                ic(
                    "TargetGuidanceProvider.room_prior.exception_fallback",
                    target_name,
                    repr(exc),
                    fallback,
                )
            return fallback

    # ------------------------------------------------------------------
    # Heuristic backend (clean, room-indicative only)
    # ------------------------------------------------------------------

    def _heuristic_prior(self, target_name: str) -> RoomDistribution:
        name = target_name.lower()
        scores: RoomDistribution = {}

        def add(room: str, score: float) -> None:
            scores[room] = scores.get(room, 0.0) + score

        # ---------------- kitchen ----------------
        if any(k in name for k in [
            "refrigerator", "fridge", "oven", "microwave", "toaster", "sink"
        ]):
            add("kitchen", 1.0)

        # ---------------- dining room ----------------
        if any(k in name for k in [
            "dining_table", "dining table"
        ]):
            add("dining_room", 1.0)

        # ---------------- bathroom ----------------
        if any(k in name for k in [
            "toilet", "bathtub", "shower"
        ]):
            add("bathroom", 1.0)

        # ---------------- bedroom ----------------
        if any(k in name for k in [
            "bed", "wardrobe"
        ]):
            add("bedroom", 1.0)

        # ---------------- living room ----------------
        if any(k in name for k in [
            "couch", "sofa", "tv", "television"
        ]):
            add("living_room", 1.0)

        # ---------------- lounge ----------------
        if any(k in name for k in [
            "desk", "laptop", "computer", "office", "study"
        ]):
            add("lounge", 1.0)

        # ---------------- storeroom ----------------
        if any(k in name for k in [
            "box", "carton", "storage", "shelf", "broom"
        ]):
            add("storeroom", 1.0)

        # ---------------- hallway ----------------
        if any(k in name for k in [
            "hallway", "corridor", "stair", "staircase"
        ]):
            add("hallway", 1.0)

        # ---------------- gym ----------------
        if any(k in name for k in [
            "treadmill", "dumbbell", "barbell"
        ]):
            add("gym", 1.0)

        total = sum(scores.values())
        if total <= 0.0:
            return {}

        return {room: s / total for room, s in scores.items()}
