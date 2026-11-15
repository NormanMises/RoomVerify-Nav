from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from vlfm.mapping.value_map import ValueMap
from vlfm.utils.room_schema import get_current_room_types


@dataclass(frozen=True)
class RoomQueryResult:
    dist: Dict[str, float]
    obs_conf: float


class RoomMemoryMap(ValueMap):
    """
    A ValueMap-shaped memory that stores a room-type distribution per cell.

    We reuse ValueMap's FOV confidence mask + localization, and fuse room
    distributions using the same confidence-driven fusion rule.
    """

    DEFAULT_ROOM_TYPES: Tuple[str, ...] = get_current_room_types(include_unknown=True)

    _radius_masks: Dict[int, np.ndarray] = {}

    def __init__(
        self,
        room_types: Sequence[str] = DEFAULT_ROOM_TYPES,
        size: int = 1000,
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> None:
        # Room memory benefits from averaging multiple steps (more robust than max).
        super().__init__(
            value_channels=len(room_types),
            size=size,
            use_max_confidence=False,
            fusion_type="default",
            obstacle_map=obstacle_map,
        )
        self.room_types = tuple(str(r) for r in room_types)

    def update_room_dist(
        self,
        room_dist: Dict[str, float],
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
        step_weight: float,
    ) -> None:
        """
        Write the per-step room distribution to the observed FOV region.

        Args:
            room_dist: Raw room distribution (may be partial/un-normalized).
            step_weight: Soft write strength in [0,1] (e.g., margin confidence).
        """
        step_weight = float(np.clip(float(step_weight), 0.0, 1.0))
        if step_weight <= 0.0:
            return

        if not room_dist:
            return

        values = np.array([float(room_dist.get(r, 0.0)) for r in self.room_types], dtype=np.float32)
        if float(values.sum()) <= 0.0:
            return

        # Localize FOV confidence mask, then scale by step_weight.
        curr_map = self._localize_new_data(depth, tf_camera_to_episodic, min_depth, max_depth, fov)
        curr_map = curr_map.astype(np.float32) * step_weight

        # Fuse into the distribution memory.
        self._fuse_new_data(curr_map, values)

    def query_room_dist(self, point_xy: np.ndarray, radius_m: float) -> Optional[RoomQueryResult]:
        """
        Query the stored room distribution around a metric (x,y) point.

        Returns:
            RoomQueryResult or None if the area has no observation confidence.
        """
        radius_px = int(max(0.0, float(radius_m)) * float(self.pixels_per_meter))

        pt = np.asarray(point_xy, dtype=np.float32).reshape(-1)
        if pt.shape[0] < 2:
            return None
        center = self._xy_to_value_map_px(pt[:2])
        if center is None:
            return None
        row, col = center

        conf_map = self._map
        dist_map = self._value_map
        h, w = conf_map.shape

        r0 = max(0, row - radius_px)
        r1 = min(h, row + radius_px + 1)
        c0 = max(0, col - radius_px)
        c1 = min(w, col + radius_px + 1)

        conf_patch = conf_map[r0:r1, c0:c1].astype(np.float32, copy=False)
        dist_patch = dist_map[r0:r1, c0:c1, :].astype(np.float32, copy=False)

        mask = self._get_radius_mask(radius_px)
        # Crop mask if patch is clipped by map boundary.
        mr0 = r0 - (row - radius_px)
        mc0 = c0 - (col - radius_px)
        mask = mask[mr0 : mr0 + (r1 - r0), mc0 : mc0 + (c1 - c0)]

        weights = conf_patch * mask.astype(np.float32)
        total_w = float(np.sum(weights))
        if total_w <= 0.0:
            return None

        vec = np.sum(dist_patch * weights[:, :, None], axis=(0, 1)) / total_w
        vec = np.clip(vec, 0.0, None)
        vec_sum = float(np.sum(vec))
        if vec_sum <= 0.0:
            return None
        vec = vec / vec_sum

        obs_conf = float(np.max(weights))

        dist = {self.room_types[i]: float(vec[i]) for i in range(len(self.room_types)) if float(vec[i]) > 0.0}
        return RoomQueryResult(dist=dist, obs_conf=obs_conf)

    def _xy_to_value_map_px(self, point_xy: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Match ValueMap.sort_waypoints' xy->pixel conversion (baseline_89c3 behavior).
        """
        x, y = float(point_xy[0]), float(point_xy[1])
        px = int(-x * self.pixels_per_meter) + int(self._episode_pixel_origin[0])
        py = int(-y * self.pixels_per_meter) + int(self._episode_pixel_origin[1])
        row = int(self._value_map.shape[0] - px)
        col = int(py)
        if row < 0 or row >= self._value_map.shape[0] or col < 0 or col >= self._value_map.shape[1]:
            return None
        return row, col

    @classmethod
    def _get_radius_mask(cls, radius_px: int) -> np.ndarray:
        radius_px = int(max(0, radius_px))
        mask = cls._radius_masks.get(radius_px)
        if mask is not None:
            return mask
        if radius_px <= 0:
            mask = np.ones((1, 1), dtype=bool)
        else:
            y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
            mask = (x * x + y * y) <= (radius_px * radius_px)
        cls._radius_masks[radius_px] = mask
        return mask
