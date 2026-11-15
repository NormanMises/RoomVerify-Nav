"""Export episode data in native map coordinates for paper figures."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class FigureDump:
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.episode: Any = None
        self.reset()

    def reset(self) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.frontiers: Optional[Dict[str, np.ndarray]] = None
        self.last_rgb: Optional[np.ndarray] = None

    def _output_directory(self) -> Path:
        scene_name = Path(self.episode.scene_id).name.split(".")[0]
        output = self.directory / f"{self.episode.episode_id}_{scene_name}"
        output.mkdir(parents=True, exist_ok=True)
        return output

    def record_frontiers(
        self,
        frontiers: np.ndarray,
        values: List[float],
        scales: Optional[List[float]] = None,
        scores: Optional[List[float]] = None,
        order: Optional[List[int]] = None,
    ) -> None:
        count = len(frontiers)
        self.frontiers = {
            "frontiers_base": np.asarray(frontiers).reshape(-1, 2).copy(),
            "base_values": np.asarray(values).copy(),
            "scales": np.ones(count) if scales is None else np.asarray(scales).copy(),
            "final_scores": np.asarray(values if scores is None else scores).copy(),
            "rerank_order": np.arange(count) if order is None else np.asarray(order, dtype=np.int64),
        }

    def record_step(self, policy: Any, mode: str, observations: Any = None) -> None:
        if self.frontiers is None:
            self.record_frontiers(np.empty((0, 2)), [])
        frontier_data = self.frontiers
        assert frontier_data is not None
        order = frontier_data["rerank_order"]
        changed = bool(len(order) and order[0] != 0)
        if observations is None:
            cache = policy._observations_cache
            robot_xy = cache["robot_xy"]
            heading = cache["robot_heading"]
            rgb = cache["value_map_rgbd"][0][0]
        else:
            # Habitat can stop at the map boundary before its observation cache is built.
            robot_xy = observations["gps"][0].cpu().numpy().copy()
            robot_xy[1] *= -1
            heading = observations["compass"][0].item()
            rgb = observations["rgb"][0].cpu().numpy()
        self.last_rgb = rgb.astype(np.uint8)
        self.steps.append(
            dict(
                robot_xy=robot_xy.copy(),
                robot_heading=float(heading),
                mode=mode,
                rerank_changed=changed,
                **frontier_data,
            )
        )
        if changed:
            self._write_snapshot(policy, len(self.steps) - 1, frontier_data)
        self.frontiers = None

    def _write_snapshot(self, policy: Any, step: int, frontier_data: Dict[str, Any]) -> None:
        value_map = policy._value_map
        room_map = policy._room_memory_map
        room_argmax = room_map._value_map.argmax(axis=-1).astype(np.uint8)
        room_maxprob = room_map._value_map.max(axis=-1).astype(np.float16)
        room_conf = room_map._map.astype(np.float16)
        explored = policy._obstacle_map.explored_area.astype(bool)
        value = policy._vis_reduce_fn(value_map._value_map).copy()
        value[~explored] = 0
        np.savez_compressed(
            self._output_directory() / f"snapshot_{step:04d}.npz",
            room_argmax=room_argmax,
            room_maxprob=room_maxprob,
            room_conf=room_conf,
            explored=explored,
            navigable=policy._obstacle_map._navigable_map.astype(bool),
            value=value.astype(np.float16),
            rgb=self.last_rgb,
            **{
                key: frontier_data[key]
                for key in ("frontiers_base", "base_values", "scales", "final_scores", "rerank_order")
            },
        )

    def finish(self, policy: Any, success: float) -> None:
        episode = self.episode
        output = self._output_directory()
        value_map = policy._value_map
        room_map = policy._room_memory_map
        prior, impossible = policy._target_guidance_provider.get_room_prior_with_impossible(policy._target_object)
        meta = dict(
            episode_id=str(episode.episode_id),
            scene_id=episode.scene_id,
            target_object=policy._target_object,
            start_position=list(episode.start_position),
            start_rotation=list(episode.start_rotation),
            success=float(success),
            pixels_per_meter=value_map.pixels_per_meter,
            size=value_map.size,
            episode_pixel_origin=value_map._episode_pixel_origin.tolist(),
            coordinate_system="episodic x forward from start heading, y left; Habitat GPS y negated",
            pixel_coordinates="col = size - (round(y * pixels_per_meter) + origin[0]); "
            "row = round(x * pixels_per_meter) + origin[1]; arrays indexed [row, col]",
            room_types=list(room_map.room_types),
            room_prior=prior,
            room_impossible=impossible,
        )
        with (output / "meta.json").open("w") as stream:
            json.dump(meta, stream, indent=2)
        arrays: Dict[str, np.ndarray] = {}
        for key in ("robot_xy", "robot_heading", "mode", "rerank_changed"):
            arrays[key] = np.asarray([step[key] for step in self.steps])
        for key in ("frontiers_base", "base_values", "scales", "final_scores", "rerank_order"):
            rows = np.empty(len(self.steps), dtype=object)
            for index, step in enumerate(self.steps):
                rows[index] = step[key]
            arrays[key] = rows
        np.savez_compressed(output / "steps.npz", **arrays)
        if self.steps and not self.steps[-1]["rerank_changed"]:
            self._write_snapshot(policy, len(self.steps) - 1, self.steps[-1])
