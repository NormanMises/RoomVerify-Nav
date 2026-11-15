# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from roomverify_nav.mapping.frontier_map import FrontierMap
from roomverify_nav.mapping.room_memory_map import RoomMemoryMap
from roomverify_nav.mapping.value_map import ValueMap
from roomverify_nav.mapping.room_tracker import RoomStateTracker
from roomverify_nav.policy.base_objectnav_policy import BaseObjectNavPolicy
from roomverify_nav.policy.utils.acyclic_enforcer import make_acyclic_enforcer
from roomverify_nav.utils.figure_dump import FigureDump
from roomverify_nav.utils.geometry_utils import closest_point_within_threshold
from roomverify_nav.utils.room_schema import (
    get_scene_detector_labels,
    get_scene_gdino_caption_labels,
    normalize_scene_object_label,
)
from roomverify_nav.utils.target_guidance import TargetGuidanceProvider
from roomverify_nav.utils.timing import TimingContext
from roomverify_nav.vlm.coco_classes import COCO_CLASSES
from roomverify_nav.vlm.blip2itm import BLIP2ITMClient
from roomverify_nav.vlm.detections import ObjectDetections

from icecream import ic

SCENE_OBJECT_LABELS: Set[str] = set(get_scene_detector_labels())


try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        figure_dump_dir = os.environ.get("FIGURE_DUMP_DIR", "").strip()
        frontier_rerank_enabled = os.environ.get("PROMPTING_FRONTIER_RERANK", "0") == "1"
        if figure_dump_dir and not frontier_rerank_enabled:
            raise ValueError("开启 FIGURE_DUMP_DIR 导出论文插图数据时，必须同时设置 PROMPTING_FRONTIER_RERANK=1。")
        super().__init__(*args, **kwargs)
        self._figure_dump = FigureDump(figure_dump_dir) if figure_dump_dir else None
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        # Context-aware prompting components
        self.room_tracker = RoomStateTracker(window_size=12, change_threshold=0.6)  # 保留最近10步
        self._target_guidance_provider = TargetGuidanceProvider()
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = make_acyclic_enforcer()

        # 场景物体检测的置信度阈值
        self._scene_detection_threshold: float = float(os.environ.get("SCENE_DETECTION_THRESHOLD", "0.5"))
        # 场景物体检测 bbox 面积下限（相对整图面积的比例）
        # 目的：抑制大量“小框噪声”（弱证据/远处误检）污染 room_dist
        self._scene_detection_min_bbox_area_ratio: float = float(
            os.environ.get("SCENE_DETECTION_MIN_BBOX_AREA_RATIO", "0.002")
        )
        self._scene_detection_gdino_enabled: bool = os.environ.get("SCENE_DETECTION_GDINO_ENABLED", "0") == "1"
        self._scene_detection_gdino_min_yolo_classes: int = int(
            os.environ.get("SCENE_DETECTION_GDINO_MIN_YOLO_CLASSES", "2")
        )
        self._scene_detection_gdino_interval: int = max(
            1, int(os.environ.get("SCENE_DETECTION_GDINO_INTERVAL", "5"))
        )
        self._scene_detection_gdino_threshold: float = float(
            os.environ.get("SCENE_DETECTION_GDINO_THRESHOLD", "0.4")
        )
        self._scene_detection_last_gdino_step: int = -10**9
        self._scene_detection_gdino_labels: Tuple[str, ...] = get_scene_gdino_caption_labels()

        # ------------------------------------------------------------------
        # Frontier rerank (use room context at decision level, not BLIP2 prompt)
        # ------------------------------------------------------------------
        # PROMPTING_FRONTIER_RERANK=1: enable room-prior-based frontier reranking.
        # We keep it gated behind an env var for clean A/B.
        self._frontier_rerank_enabled: bool = frontier_rerank_enabled
        # PROMPTING_FRONTIER_RERANK_ALPHA: mix weight in [0,1] for room bonus.
        self._frontier_rerank_alpha: float = float(os.environ.get("PROMPTING_FRONTIER_RERANK_ALPHA", "0.3"))
        # PROMPTING_FRONTIER_RERANK_ROOM_RADIUS: query radius (meters) around a frontier.
        self._frontier_rerank_room_radius: float = float(os.environ.get("PROMPTING_FRONTIER_RERANK_ROOM_RADIUS", "0.5"))
        # FOV-diffused room memory (no robot-position keying).
        self._room_memory_map: Optional[RoomMemoryMap] = (
            RoomMemoryMap(
                size=self._value_map.size,
                obstacle_map=self._obstacle_map if sync_explored_areas else None,
            )
            if self._frontier_rerank_enabled
            else None
        )

    def _reset(self) -> None:
        super()._reset()
        if self._figure_dump is not None:
            self._figure_dump.reset()
        self.room_tracker.reset()
        self._value_map.reset()
        self._acyclic_enforcer = make_acyclic_enforcer()
        self._scene_detection_last_gdino_step = -10**9
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)
        if self._room_memory_map is not None:
            self._room_memory_map.reset()

    def _update_room_memory_map(self) -> None:
        if not self._frontier_rerank_enabled or self._room_memory_map is None:
            return

        dist = getattr(self.room_tracker, "last_room_dist", None)
        if not dist:
            return

        room_conf = float(getattr(self.room_tracker, "last_room_confidence", 0.0))
        step_weight = float(np.clip(room_conf, 0.0, 1.0))
        if step_weight <= 0.0:
            return

        for _rgb, depth, tf, min_depth, max_depth, fov in self._observations_cache["value_map_rgbd"]:
            self._room_memory_map.update_room_dist(
                dist,
                depth,
                tf,
                float(min_depth),
                float(max_depth),
                float(fov),
                step_weight=step_weight,
            )

    def _detect_scene_objects(self, rgb: np.ndarray) -> List[str]:
        """
        使用 YOLOv7 检测场景中的室内物体（用于房间推理）。

        与目标物体检测不同，这里检测所有可见的室内物体，
        而不仅仅是导航目标。

        Args:
            rgb: RGB 图像

        Returns:
            检测到的室内物体类别列表（可能包含重复）
        """
        try:
            detections = self._predict_yolov7_cached(rgb)
            detections.filter_by_conf(self._scene_detection_threshold)
            yolo_objects = self._normalize_scene_objects(detections)

            gdino_objects: List[str] = []
            if self._should_run_scene_gdino(yolo_objects):
                gdino_caption = self._build_scene_gdino_caption(yolo_objects)
                if gdino_caption:
                    gdino_detections = self._predict_grounding_dino_cached(rgb, caption=gdino_caption)
                    gdino_detections.filter_by_conf(self._scene_detection_gdino_threshold)
                    gdino_objects = self._normalize_scene_objects(gdino_detections)
                    self._scene_detection_last_gdino_step = self._num_steps

            merged_objects = list(yolo_objects)
            covered = set(yolo_objects)
            for obj in gdino_objects:
                if obj in covered:
                    continue
                merged_objects.append(obj)
                covered.add(obj)

            if os.environ.get("DEBUG_ROOM_REASONER") == "1":
                if yolo_objects:
                    print(f"[Scene-YOLO] {yolo_objects}")
                if gdino_objects:
                    print(f"[Scene-GDINO] {gdino_objects}")
                if merged_objects:
                    print(f"[Scene-Merged] {merged_objects}")

            return merged_objects

        except Exception as e:
            # 检测失败时返回空列表，不影响导航
            if os.environ.get("DEBUG_ROOM_REASONER") == "1":
                print(f"[Scene Detection] Error: {e}")
            return []

    def _normalize_scene_objects(self, detections: ObjectDetections) -> List[str]:
        labels: List[str] = []
        for phrase in detections.phrases:
            norm = normalize_scene_object_label(phrase)
            if norm in SCENE_OBJECT_LABELS:
                labels.append(norm)
        return labels

    def _build_scene_gdino_caption(self, yolo_objects: List[str]) -> str:
        labels: List[str] = list(get_scene_gdino_caption_labels(exclude_canonical=yolo_objects))
        for target_label in self._get_scene_target_gdino_labels():
            if target_label not in labels:
                labels.append(target_label)
        if not labels:
            return ""
        return f"{' . '.join(labels)} ."

    def _get_scene_target_gdino_labels(self) -> List[str]:
        labels: List[str] = []
        for raw_label in self._target_object.split(PROMPT_SEPARATOR):
            label = raw_label.strip().lower().replace("_", " ")
            if not label or label in COCO_CLASSES or label in labels:
                continue
            labels.append(label)
        return labels

    def _should_run_scene_gdino(self, yolo_objects: List[str]) -> bool:
        if not self._scene_detection_gdino_enabled:
            return False
        if not self._scene_detection_gdino_labels:
            return False
        unique_yolo = len(set(yolo_objects))
        if unique_yolo >= self._scene_detection_gdino_min_yolo_classes:
            return False
        steps_since_last = self._num_steps - self._scene_detection_last_gdino_step
        return steps_since_last >= self._scene_detection_gdino_interval

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        self._add_step_debug(f"Best value: {best_value*100:.2f}%")
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        top_two_values = tuple(sorted_values[:2])
        best_frontier_idx = None
        frontier_print = os.environ.get("DEBUG_FRONTIER_PRINT", "0") == "1"
        cyclic_suppressed = 0
        cyclic_checked = 0
        all_cyclic_fallback = False

        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)
                if closest_index != -1:
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    if frontier_print:
                        print("Sticking to last point.")
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic_checked += 1
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    cyclic_suppressed += 1
                    if frontier_print:
                        print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            all_cyclic_fallback = True
            if frontier_print:
                print("All frontiers are cyclic. Just choosing the closest one.")
            best_frontier_idx = max(
                range(len(sorted_pts)),
                key=lambda i: float(np.linalg.norm(sorted_pts[i] - robot_xy)),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier

        if os.environ.get("DEBUG_FRONTIER", "0") == "1":
            self._add_step_debug(
                f"Frontier: (cyc={cyclic_suppressed}/{cyclic_checked},fallback={int(all_cyclic_fallback)})"
            )

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        target_for_itm = self._target_object.replace("|", "/")
        itm_prompts = [
            p.replace("target_object", target_for_itm) for p in self._text_prompt.split(PROMPT_SEPARATOR)
        ]
        # Store the exact prompts sent to BLIP2 for per-step visualization.
        self._step_blip2_prompt = " | ".join(itm_prompts)
        # Keep the list around for any offline debugging (not used for video overlay).
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p,
                )
                for p in itm_prompts
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()  # 前沿点管理系统

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._scan_value_radius = float(os.environ.get("DYNAMIC_RELOCATION_VALUE_RADIUS", "0.5"))
        self._scan_value_threshold_mode = os.environ.get(
            "DYNAMIC_RELOCATION_VALUE_THRESHOLD_MODE", "mean_std"
        ).strip().lower()
        self._scan_value_std_k = float(os.environ.get("DYNAMIC_RELOCATION_VALUE_STD_K", "0.5"))
        self._scan_value_quantile = float(os.environ.get("DYNAMIC_RELOCATION_VALUE_QUANTILE", "0.75"))

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)

        frontier_rerank = bool(getattr(self, "_frontier_rerank_enabled", False))
        if frontier_rerank:
            with TimingContext("ITMPolicyV2._frontier_rerank_context()"):
                rgb = self._observations_cache["object_map_rgbd"][0][0]
                current_scene_objects = self._detect_scene_objects(rgb)

                room_type, confidence, is_changed = self.room_tracker.update(current_scene_objects)
                if is_changed:
                    print(f"[Room Change] Entered {room_type} (confidence: {confidence:.2f})")

                if os.environ.get("DEBUG_ROOM_INFO", "0") == "1":
                    stable_objects = self.room_tracker.get_stable_objects(top_k=5)
                    room_info = f"Room: {room_type}({confidence:.0%})"
                    if is_changed:
                        room_info += "[CHANGED]"
                    if stable_objects:
                        room_info += f" Observed: {', '.join(stable_objects)}"
                    self._add_step_debug(room_info)

                self._update_room_memory_map()

        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _get_scan_value_reduce_fn(self) -> Optional[Any]:
        if getattr(self._value_map, "_value_channels", 1) <= 1:
            return None
        return getattr(self, "_reduce_values", None)

    def _scan_frontier_value_allows(self, observations: Dict, frontier_xy: np.ndarray) -> bool:
        del observations
        reduce_fn = self._get_scan_value_reduce_fn()
        if getattr(self._value_map, "_value_channels", 1) > 1 and reduce_fn is None:
            self._add_step_debug("ScanGate: missing_reduce_fn")
            return False

        value_radius = float(self._scan_value_radius)
        curr_value = float(self._value_map.get_value_at_point(frontier_xy, value_radius, reduce_fn=reduce_fn))
        if curr_value < 0.0:
            return False

        mean_value = float(self._value_map.get_mean_value(reduce_fn=reduce_fn))
        std_value = float(self._value_map.get_std_value(reduce_fn=reduce_fn))
        mode = self._scan_value_threshold_mode

        if mode == "quantile":
            quantile_value = float(
                self._value_map.get_quantile_value(self._scan_value_quantile, reduce_fn=reduce_fn)
            )
            threshold = max(mean_value, quantile_value)
        else:
            if mode != "mean_std" and mode != "mean":
                mode = "mean_std"
            if mode == "mean":
                threshold = mean_value
            else:
                threshold = max(mean_value, mean_value + self._scan_value_std_k * std_value)
            quantile_value = None

        if curr_value <= threshold:
            extra = f",q={quantile_value:.2f}" if quantile_value is not None else ""
            self._add_step_debug(
                "ScanGate: below_thr("
                f"mode={mode},curr={curr_value:.2f},thr={threshold:.2f},"
                f"mean={mean_value:.2f},std={std_value:.2f}{extra})"
            )
            return False

        extra = f",q={quantile_value:.2f}" if quantile_value is not None else ""
        self._add_step_debug(
            "ScanGate: pass("
            f"mode={mode},curr={curr_value:.2f},thr={threshold:.2f},"
            f"mean={mean_value:.2f},std={std_value:.2f}{extra})"
        )
        return True

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)

        if self._figure_dump is not None:
            self._figure_dump.record_frontiers(sorted_frontiers, sorted_values)

        # Optional frontier rerank using room priors + room memory (decision-level context).
        if not getattr(self, "_frontier_rerank_enabled", False):
            return sorted_frontiers, sorted_values
        if getattr(self, "_room_memory_map", None) is None:
            return sorted_frontiers, sorted_values

        # Frontier rerank depends on target-level room semantics.
        room_prior_provider = getattr(self, "_target_guidance_provider", None)
        if room_prior_provider is None:
            return sorted_frontiers, sorted_values
        room_prior, room_impossible = room_prior_provider.get_room_prior_with_impossible(self._target_object)
        if not room_prior and not room_impossible:
            return sorted_frontiers, sorted_values

        alpha = float(getattr(self, "_frontier_rerank_alpha", 0.0))
        if alpha <= 0.0:
            return sorted_frontiers, sorted_values
        alpha = float(np.clip(alpha, 0.0, 1.0))
        radius_m = float(getattr(self, "_frontier_rerank_room_radius", 0.5))

        base_values = [float(v) for v in sorted_values]
        return_final_values = os.environ.get("PROMPTING_FRONTIER_RERANK_RETURN_FINAL", "0") == "1"

        topk = int(min(32, len(sorted_frontiers)))
        compats: List[float] = [0.0] * topk
        incompats: List[float] = [0.0] * topk
        obs_confs: List[float] = [0.0] * topk
        raws: List[float] = [0.0] * topk
        scales: List[float] = [1.0] * topk
        final_scores: List[float] = [0.0] * topk
        room_dist_summaries: List[str] = ["none"] * topk

        prior_norm = {str(k): float(v) for k, v in room_prior.items()}
        impossible_norm = {str(k): float(v) for k, v in room_impossible.items()}
        if not prior_norm and not impossible_norm:
            return sorted_frontiers, sorted_values
        prior_sorted = sorted(prior_norm.items(), key=lambda kv: kv[1], reverse=True)
        # ic(topk, prior_sorted, impossible_norm)
        for j in range(topk):
            # print("%" * 50)
            pt = sorted_frontiers[j]
            base = base_values[j]

            query = self._room_memory_map.query_room_dist(pt, radius_m=radius_m)
            if query is None:
                scale = 1.0
            else:
                dist_norm = query.dist
                dist_sorted = sorted(
                    ((str(k), float(v)) for k, v in dist_norm.items()),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                obs_conf = float(query.obs_conf)
                compat = sum(float(prior_norm.get(r, 0.0)) * float(p) for r, p in dist_norm.items())
                incompat = sum(float(impossible_norm.get(r, 0.0)) * float(p) for r, p in dist_norm.items())
                raw = float((compat - incompat) * obs_conf)
                raw = float(np.clip(raw, -1.0, 1.0))
                if raw >= 0.0:
                    delta = 0.4 * raw
                else:
                    delta = 0.2 * raw
                scale = float(np.clip(1.0 + alpha * delta, 0.8, 1.4))
                compats[j] = float(compat)  # 纯 debug 用，保留房间兼容度
                incompats[j] = float(incompat)  # 纯 debug 用，保留房间不兼容度
                obs_confs[j] = float(obs_conf)  # 纯 debug 用，保留观测置信
                raws[j] = float(raw)  # 纯 debug 用，保留 raw room bonus
                room_dist_summaries[j] = ",".join(f"{room}:{prob:.2f}" for room, prob in dist_sorted[:3]) or "empty"
                # ic(dist_sorted, obs_conf, compat, incompat, scale)

            scales[j] = float(scale)  # 纯 debug 用，用来在 overlay 里查看缩放因子

            final_scores[j] = float(base * scale)
            # ic(base, final_scores[j])

        # Sort only the prefix by final score (descending), keep the remainder stable.
        final_values = list(base_values)
        if return_final_values:
            for j in range(topk):
                final_values[j] = float(final_scores[j])
        # ic(base_values, final_values)

        top_inds = list(np.argsort([-s for s in final_scores] ,kind="stable"))
        if self._figure_dump is not None:
            self._figure_dump.record_frontiers(sorted_frontiers, base_values, scales, final_scores, top_inds)
        inds = top_inds + list(range(topk, len(sorted_frontiers)))
        sorted_frontiers = np.array([sorted_frontiers[i] for i in inds])
        sorted_values = [final_values[i] for i in inds] if return_final_values else [base_values[i] for i in inds]

        if os.environ.get("DEBUG_FRONTIER", "0") == "1" and len(top_inds) > 0:
            best_i = int(top_inds[0])
            self._add_step_debug(
                f"Rerank: (base={base_values[best_i]:.2f},scale={scales[best_i]:.2f},final={final_scores[best_i]:.2f})"
            )
            self._add_step_debug(
                "RerankCtx: "
                f"rooms={room_dist_summaries[best_i]} "
                f"obs={obs_confs[best_i]:.2f} "
                f"compat={compats[best_i]:.2f} "
                f"incompat={incompats[best_i]:.2f} "
                f"raw={raws[best_i]:.2f}"
            )
        if os.environ.get("DEBUG_FRONTIER_VALUE", "0") == "1" and len(top_inds) > 0:
            every = max(1, int(os.environ.get("DEBUG_FRONTIER_VALUE_EVERY", "1")))
            if int(getattr(self, "_num_steps", 0)) % every == 0:
                best_i = int(top_inds[0])
                print(
                    "[Frontier-Rerank] "
                    f"target={self._target_object} "
                    f"prior={prior_sorted[:3]} "
                    f"impossible={sorted(impossible_norm.items(), key=lambda kv: kv[1], reverse=True)[:3]} "
                    f"rooms={room_dist_summaries[best_i]} "
                    f"obs={obs_confs[best_i]:.2f} "
                    f"compat={compats[best_i]:.2f} "
                    f"incompat={incompats[best_i]:.2f} "
                    f"raw={raws[best_i]:.2f} "
                    f"scale={scales[best_i]:.2f} "
                    f"base={base_values[best_i]:.2f} "
                    f"final={final_scores[best_i]:.2f} "
                    f"returned={(final_values[best_i] if return_final_values else base_values[best_i]):.2f}"
                )

        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return target_values
