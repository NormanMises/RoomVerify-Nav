# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, fields
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from roomverify_nav.mapping.object_point_cloud_map import ObjectPointCloudMap
from roomverify_nav.mapping.obstacle_map import ObstacleMap
from roomverify_nav.obs_transformers.utils import image_resize
from roomverify_nav.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from roomverify_nav.utils.geometry_utils import get_fov, get_point_cloud, rho_theta, transform_points
from roomverify_nav.utils.query_llm import OpenAIClient
from roomverify_nav.vlm.blip2 import BLIP2Client
from roomverify_nav.vlm.coco_classes import COCO_CLASSES
from roomverify_nav.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from roomverify_nav.vlm.sam import MobileSAMClient
from roomverify_nav.vlm.yolov7 import YOLOv7Client

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from roomverify_nav.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True
    _safe_name_re = re.compile(r"[^0-9A-Za-z._-]+")

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_candidate_threshold: float = 0.8,
        non_coco_candidate_threshold: float = 0.4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_candidate_threshold = float(coco_candidate_threshold)
        self._non_coco_candidate_threshold = float(non_coco_candidate_threshold)
        self._current_detector_type = "unknown"
        self._applied_detection_threshold = self._coco_candidate_threshold
        # Per-step cache for YOLOv7 predictions (shared by scene + target detection).
        self._yolov7_cache: Dict[Tuple[int, Tuple[int, ...], str], ObjectDetections] = {}
        # Per-step cache for GroundingDINO predictions (caption-sensitive).
        self._gdino_cache: Dict[Tuple[int, Tuple[int, ...], str, str], ObjectDetections] = {}
        self._gdino_cache_labels: Dict[Tuple[int, Tuple[int, ...], str, str], Tuple[str, ...]] = {}

        # Per-step overlays (avoid writing to global env vars like DEBUG_INFO).
        self._step_debug_msgs: List[str] = []
        self._step_blip2_prompt: Optional[str] = None
        # Some policies call `_pre_step()` manually before delegating to `super().act()`;
        # guard against double-resetting per-step overlays within the same step.
        self._last_pre_step_num_steps: int = -1

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._locked_goal_xy: Optional[np.ndarray] = None
        self._locked_instance_center_xy: Optional[np.ndarray] = None
        self._locked_instance_center_xyz: Optional[np.ndarray] = None
        self._locked_instance_candidate_key: Optional[Tuple[Any, ...]] = None
        self._locked_instance_radius: float = 0.0
        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        self._dynamic_relocation_enabled = os.environ.get("DYNAMIC_RELOCATION_ENABLED", "0") == "1"
        self._scan_turn_steps = max(1, int(os.environ.get("DYNAMIC_RELOCATION_SCAN_STEPS", "12")))
        self._scan_cooldown_steps = max(
            0, int(os.environ.get("DYNAMIC_RELOCATION_SCAN_COOLDOWN_STEPS", str(self._scan_turn_steps)))
        )
        scan_reach_dist = os.environ.get("DYNAMIC_RELOCATION_FRONTIER_REACH_DIST")
        self._scan_frontier_reach_dist_override = None if scan_reach_dist is None else float(scan_reach_dist)
        self._scan_frontier_q = float(os.environ.get("DYNAMIC_RELOCATION_FRONTIER_Q", "0.5"))
        self._scan_max_scans_per_episode = max(
            0, int(os.environ.get("DYNAMIC_RELOCATION_MAX_SCANS_PER_EPISODE", "3"))
        )
        self._scan_region_q = float(os.environ.get("DYNAMIC_RELOCATION_REGION_Q", "1.0"))
        self._scan_region_revisit_steps = max(
            0, int(os.environ.get("DYNAMIC_RELOCATION_REGION_REVISIT_STEPS", "60"))
        )
        self._scan_active = False
        self._scan_steps_remaining = 0
        self._scan_total_steps = 0
        self._scan_frontier = np.zeros(2, dtype=np.float32)
        self._scan_last_trigger_step = -10**9
        self._scan_num_triggers = 0
        self._scan_last_frontier_key: Optional[Tuple[float, float]] = None
        self._scan_region_last_trigger_steps: Dict[Tuple[float, float], int] = {}
        self._scan_candidate_frontier = np.zeros(2, dtype=np.float32)
        self._scan_candidate_frontier_key: Optional[Tuple[float, float]] = None
        self._scan_pending_frontier = np.zeros(2, dtype=np.float32)
        self._scan_pending_frontier_key: Optional[Tuple[float, float]] = None
        self._reset_pointnav_on_next_call = False
        self._scan_episode_total_steps = 0
        self._scan_episode_trigger_steps: List[int] = []
        self._scan_episode_frontier_keys: List[Tuple[float, float]] = []
        self._scan_episode_gate_reject_counts: Dict[str, int] = {}
        self._scan_last_gate_reject_signature: Optional[str] = None
##################################
        self._stop_verify_enabled = os.environ.get("STOP_VERIFY_ENABLED", "0") == "1"
        self._stop_verify_near_goal_dist = float(
            os.environ.get("STOP_VERIFY_NEAR_GOAL_DIST", str(pointnav_stop_radius))
        )
        self._first_detection_verify_enabled = os.environ.get("FIRST_DETECTION_VERIFY_ENABLED", "1") == "1"
        self._first_detection_verify_categories = self._parse_first_detection_verify_categories(
            os.environ.get("FIRST_DETECTION_VERIFY_CATEGORIES", "")
        )
        self._target_verify_openai_model = os.environ.get("OPENAI_VISION_MODEL", "").strip()
        self._target_verify_reasoning_effort = os.environ.get("TARGET_VERIFY_REASONING_EFFORT", "").strip()
        self._stop_verify_debug_save_images = os.environ.get("STOP_VERIFY_DEBUG_SAVE_IMAGES", "0") == "1"
        self._stop_verify_debug_dir = os.environ.get("STOP_VERIFY_DEBUG_DIR", "").strip()
        self._stop_verify_history_size = max(0, int(os.environ.get("STOP_VERIFY_HISTORY_SIZE", "5")))
        self._stop_verify_goal_match_radius = float(os.environ.get("STOP_VERIFY_GOAL_MATCH_RADIUS", "1.0"))
        self._stop_verify_use_history = os.environ.get("STOP_VERIFY_USE_HISTORY", "1") == "1"
        self._stop_verify_no_match_fail_open = os.environ.get("STOP_VERIFY_NO_MATCH_FAIL_OPEN", "0") == "1"
        self._rejected_region_enabled = os.environ.get("REJECTED_REGION_ENABLED", "0") == "1"
        self._rejected_region_radius = float(os.environ.get("REJECTED_REGION_RADIUS", "0.75"))
        self._rejected_region_decay = float(os.environ.get("REJECTED_REGION_DECAY", "0.5"))
        self._stop_verify_client: Optional[OpenAIClient] = None
        self._stop_verify_history: Deque[Dict[str, Any]] = deque(maxlen=max(1, self._stop_verify_history_size))
        self._first_detection_verify_category_skip_logged = False
        self._first_detection_verify_calls = 0
        self._first_detection_verify_pass = 0
        self._first_detection_verify_reject = 0
        self._second_detection_verify_calls = 0
        self._second_detection_verify_pass = 0
        self._second_detection_verify_reject = 0
        self._stop_verify_calls = 0
        self._stop_verify_pass = 0
        self._stop_verify_reject = 0
        self._stop_verify_current_match = 0
        self._stop_verify_history_match = 0
        self._stop_verify_no_goal_matched_detection = 0
        self._stop_verify_bypass_no_client = 0
        self._stop_verify_bypass_no_image = 0
        self._stop_verify_bypass_api_error = 0
        self._stop_verify_goal_match_dist: Optional[float] = None
        self._rejected_region_writes = 0
        self._reject_memory_hits = 0
        self._target_cloud_clears = 0
        self._stop_verify_last_reject_center: Optional[Tuple[float, float]] = None
        self._accepted_target_regions: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._accepted_target_instances: List[Dict[str, Any]] = []
        self._pending_target_candidates: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._rejected_target_regions: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._rejected_target_instances: List[Dict[str, Any]] = []
        self._candidate_accept_radius = float(os.environ.get("TARGET_VERIFY_ACCEPT_RADIUS", "1.0"))
        self._candidate_region_q = float(os.environ.get("TARGET_VERIFY_REGION_Q", "0.5"))
        self._candidate_view_center_q = float(os.environ.get("TARGET_VERIFY_VIEW_CENTER_Q", "0.1"))
        self._candidate_retry_ttl = max(1, int(os.environ.get("TARGET_VERIFY_REJECT_TTL", "80")))
        self._candidate_blacklist_radius = float(os.environ.get("TARGET_VERIFY_BLACKLIST_RADIUS", "0.6"))
        self._candidate_retry_center_ratio = float(os.environ.get("TARGET_VERIFY_CENTER_RATIO", "0.25"))
        self._candidate_retry_min_area_gain = float(os.environ.get("TARGET_VERIFY_MIN_AREA_GAIN", "0.01"))
        self._candidate_retry_min_score_gain = float(os.environ.get("TARGET_VERIFY_MIN_SCORE_GAIN", "0.05"))
        self._candidate_retry_min_depth = float(os.environ.get("TARGET_VERIFY_MIN_DEPTH", "1.0"))
        self._candidate_retry_max_depth = float(os.environ.get("TARGET_VERIFY_MAX_DEPTH", "4.75"))
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )

    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._locked_goal_xy = None
        self._locked_instance_center_xy = None
        self._locked_instance_center_xyz = None
        self._locked_instance_candidate_key = None
        self._locked_instance_radius = 0.0
        self._num_steps = 0
        self._last_pre_step_num_steps = -1
        self._done_initializing = False
        self._called_stop = False
        self._scan_active = False
        self._scan_steps_remaining = 0
        self._scan_total_steps = 0
        self._scan_frontier = np.zeros(2, dtype=np.float32)
        self._scan_last_trigger_step = -10**9
        self._scan_num_triggers = 0
        self._scan_last_frontier_key = None
        self._scan_region_last_trigger_steps = {}
        self._scan_candidate_frontier = np.zeros(2, dtype=np.float32)
        self._scan_candidate_frontier_key = None
        self._scan_pending_frontier = np.zeros(2, dtype=np.float32)
        self._scan_pending_frontier_key = None
        self._reset_pointnav_on_next_call = False
        self._scan_episode_total_steps = 0
        self._scan_episode_trigger_steps = []
        self._scan_episode_frontier_keys = []
        self._scan_episode_gate_reject_counts = {}
        self._scan_last_gate_reject_signature = None
        self._current_detector_type = "unknown"
        self._applied_detection_threshold = self._coco_candidate_threshold
        self._first_detection_verify_category_skip_logged = False
        self._first_detection_verify_calls = 0
        self._first_detection_verify_pass = 0
        self._first_detection_verify_reject = 0
        self._second_detection_verify_calls = 0
        self._second_detection_verify_pass = 0
        self._second_detection_verify_reject = 0
        self._stop_verify_calls = 0
        self._stop_verify_pass = 0
        self._stop_verify_reject = 0
        self._stop_verify_current_match = 0
        self._stop_verify_history_match = 0
        self._stop_verify_no_goal_matched_detection = 0
        self._stop_verify_bypass_no_client = 0
        self._stop_verify_bypass_no_image = 0
        self._stop_verify_bypass_api_error = 0
        self._stop_verify_goal_match_dist = None
        self._stop_verify_history = deque(maxlen=max(1, self._stop_verify_history_size))
        self._rejected_region_writes = 0
        self._reject_memory_hits = 0
        self._target_cloud_clears = 0
        self._stop_verify_last_reject_center = None
        self._accepted_target_regions = {}
        self._accepted_target_instances = []
        self._pending_target_candidates = {}
        self._rejected_target_regions = {}
        self._rejected_target_instances = []
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_navigation_goal(robot_xy)

        # 模式切换优先级：
        # 1) initialize：episode 开始时先执行子类定义的初始化动作。
        # 2) navigate：一旦拿到目标点，立即抢占 scan / explore。
        # 3) scan：已经开始的 360 扫描会持续到结束，再回落到 explore。
        # 4) explore：默认兜底模式；过程中可以在合适的 frontier 上临时插入 scan。
        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
        elif goal is not None:
            # 一旦有合法 goal，就立刻终止当前 scan，并清空待触发的 scan 候选，
            # 避免已经缓存的 frontier 扫描在切回导航后又被误触发。
            if self._scan_active:
                self._stop_360_scan()
            self._clear_pending_360_scan(clear_candidate=True)
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True, detections=detections[0] if detections else None)
        elif self._scan_active:
            # scan 一旦启动，会跨 step 持续执行，直到 _perform_360_scan()
            # 返回 None 表示扫描完成，此时才退回普通 explore。
            mode = "scan"
            pointnav_action = self._perform_360_scan()
            if pointnav_action is None:
                mode = "explore"
                pointnav_action = self._explore(observations)
        else:  # Haven't found target object yet
            mode = "explore"
            if self._should_perform_360_scan_in_explore(observations):
                # explore 阶段如果判断当前 frontier 值得做一次原地 360 扫描，
                # 会在当前 step 直接启动 scan 并执行第一步转向；如果 scan 立刻
                # no-op，则本 step 退回普通 explore 动作。
                self._start_360_scan()
                mode = "scan"
                pointnav_action = self._perform_360_scan()
                if pointnav_action is None:
                    mode = "explore"
                    pointnav_action = self._explore(observations)
            else:
                pointnav_action = self._explore(observations)

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        self._add_step_debug(f"Step {self._num_steps}, {mode}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        if getattr(self, "_figure_dump", None) is not None:
            self._figure_dump.record_step(self, mode)
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _reset_step_overlays(self) -> None:
        """Reset all per-step overlay text; called at the beginning of each step."""
        self._step_debug_msgs = []
        self._step_blip2_prompt = None
        # This cache is intended to be per-step (shared between scene + target detection).
        # If not cleared, it grows unbounded and can retain large RGB arrays via
        # ObjectDetections.image_source, leading to CPU OOM (often surfaced as OOM-killer
        # killing "pt_main_thread").
        self._yolov7_cache = {}
        self._gdino_cache = {}
        self._gdino_cache_labels = {}

        # Preserve existing user workflow: allow injecting a static debug line via env var,
        # but do NOT mutate it from code.
        env_debug = os.environ.get("DEBUG_INFO", "").strip()
        if env_debug:
            self._step_debug_msgs.append(env_debug)

    def _add_step_debug(self, msg: str) -> None:
        m = str(msg).strip()
        if not m:
            return
        self._step_debug_msgs.append(m)

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        # Reset per-step caches (must run before any detection in this step).
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        if self._last_pre_step_num_steps != int(self._num_steps):
            self._reset_step_overlays()
            self._last_pre_step_num_steps = int(self._num_steps)
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}
        self._cleanup_rejected_target_regions()

    def _predict_yolov7_cached(self, img: np.ndarray) -> ObjectDetections:
        key = (int(img.ctypes.data), tuple(img.shape), str(img.dtype))
        cached = self._yolov7_cache.get(key)
        if cached is None:
            cached = self._coco_object_detector.predict(img)
            # Avoid retaining the full RGB frame in the cache (it can be very large).
            cached.image_source = None
            cached._annotated_frame = None
            self._yolov7_cache[key] = cached

        # Return a copy so callers can filter in-place without mutating the cache.
        return ObjectDetections(
            boxes=cached.boxes.clone(),
            logits=cached.logits.clone(),
            phrases=list(cached.phrases),
            image_source=img,
            fmt="xyxy",
        )

    def _predict_grounding_dino_cached(self, img: np.ndarray, caption: str) -> ObjectDetections:
        key = (int(img.ctypes.data), tuple(img.shape), str(img.dtype), str(caption))
        cached = self._gdino_cache.get(key)
        if cached is None:
            request_labels = self._parse_gdino_caption_labels(caption)
            superset_key = self._find_cached_gdino_superset(key, request_labels)
            if superset_key is not None:
                cached = self._clone_detections(self._gdino_cache[superset_key], img)
                cached.filter_by_class(list(request_labels))
                return cached

            cached = self._object_detector.predict(img, caption=caption)
            cached.image_source = None
            cached._annotated_frame = None
            self._gdino_cache[key] = cached
            self._gdino_cache_labels[key] = request_labels

        return self._clone_detections(cached, img)

    def _parse_gdino_caption_labels(self, caption: str) -> Tuple[str, ...]:
        labels: List[str] = []
        for label in str(caption).split("."):
            norm = label.strip()
            if norm and norm not in labels:
                labels.append(norm)
        return tuple(labels)

    def _find_cached_gdino_superset(
        self,
        key: Tuple[int, Tuple[int, ...], str, str],
        request_labels: Tuple[str, ...],
    ) -> Optional[Tuple[int, Tuple[int, ...], str, str]]:
        if not request_labels:
            return None

        image_key = key[:3]
        request_set = set(request_labels)
        best_key: Optional[Tuple[int, Tuple[int, ...], str, str]] = None
        best_size: Optional[int] = None
        for candidate_key, candidate_labels in self._gdino_cache_labels.items():
            if candidate_key[:3] != image_key:
                continue
            candidate_set = set(candidate_labels)
            if not request_set.issubset(candidate_set):
                continue
            candidate_size = len(candidate_set)
            if best_size is None or candidate_size < best_size:
                best_key = candidate_key
                best_size = candidate_size
        return best_key

    def _clone_detections(self, detections: ObjectDetections, img: np.ndarray) -> ObjectDetections:
        return ObjectDetections(
            boxes=detections.boxes.clone(),
            logits=detections.logits.clone(),
            phrases=list(detections.phrases),
            image_source=img,
            fmt="xyxy",
        )

    def _prepare_grounding_dino_caption(self, img: np.ndarray, caption: str) -> str:
        return str(caption)

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_scan_turn_action(self) -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        if self._object_map.has_object(self._target_object):
            target_cloud = self._object_map.get_target_cloud(self._target_object)
            target_cloud = self._filter_target_cloud_for_navigation(target_cloud)
            if len(target_cloud) == 0:
                return None
            return self._object_map.get_best_object_from_cloud(target_cloud, position)
        else:
            return None

    def _filter_target_cloud_for_navigation(self, target_cloud: np.ndarray) -> np.ndarray:
        if len(target_cloud) == 0 or not self._rejected_target_instances:
            return target_cloud

        filtered_cloud = target_cloud
        target_name = self._target_object.split("|")[0].strip().lower()
        for item in self._rejected_target_instances:
            item_target = str(item.get("target_object", "")).strip().lower()
            if item_target and item_target != target_name:
                continue
            center_xyz = np.asarray(item.get("center_xyz", np.zeros(3)), dtype=np.float32).reshape(-1)
            if center_xyz.shape[0] != 3:
                continue
            radius = float(item.get("radius", self._candidate_blacklist_radius))
            keep = np.linalg.norm(filtered_cloud[:, :3] - center_xyz[None, :], axis=1) > radius
            filtered_cloud = filtered_cloud[keep]
            if len(filtered_cloud) == 0:
                break
        return filtered_cloud

    def _get_accepted_instance_reference_xy(self, item: Dict[str, Any]) -> Optional[np.ndarray]:
        center_xy = np.asarray(item.get("center_xy", np.zeros(2)), dtype=np.float32).reshape(-1)
        if center_xy.shape[0] == 2:
            return center_xy
        center_xyz = np.asarray(item.get("center_xyz", np.zeros(3)), dtype=np.float32).reshape(-1)
        if center_xyz.shape[0] == 3:
            return center_xyz[:2].astype(np.float32)
        return None

    def _select_locked_instance_for_goal(self, goal_xy: np.ndarray) -> Optional[Dict[str, Any]]:
        goal_xy_arr = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
        if goal_xy_arr.shape[0] != 2:
            return None

        target_name = self._target_object.split("|")[0].strip().lower()
        best_item: Optional[Dict[str, Any]] = None
        best_dist: Optional[float] = None
        for item in self._accepted_target_instances:
            item_target = str(item.get("target_object", "")).strip().lower()
            if item_target and item_target != target_name:
                continue
            ref_xy = self._get_accepted_instance_reference_xy(item)
            if ref_xy is None:
                continue
            dist = float(np.linalg.norm(goal_xy_arr - ref_xy))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_item = item
        return best_item

    def _bind_locked_instance(self, item: Optional[Dict[str, Any]]) -> None:
        if item is None:
            self._locked_instance_center_xy = None
            self._locked_instance_center_xyz = None
            self._locked_instance_candidate_key = None
            self._locked_instance_radius = float(self._candidate_accept_radius)
            return

        center_xy = np.asarray(item.get("center_xy", np.zeros(2)), dtype=np.float32).reshape(-1)
        self._locked_instance_center_xy = center_xy if center_xy.shape[0] == 2 else None
        center_xyz = np.asarray(item.get("center_xyz", np.zeros(3)), dtype=np.float32).reshape(-1)
        self._locked_instance_center_xyz = center_xyz if center_xyz.shape[0] == 3 else None
        candidate_key = item.get("candidate_key")
        self._locked_instance_candidate_key = candidate_key if isinstance(candidate_key, tuple) else None
        self._locked_instance_radius = float(item.get("radius", self._candidate_accept_radius))

    def _filter_target_cloud_for_locked_instance(self, target_cloud: np.ndarray) -> np.ndarray:
        if len(target_cloud) == 0:
            return target_cloud

        radius = float(self._locked_instance_radius or self._candidate_accept_radius)
        locked_center_xyz = self._locked_instance_center_xyz
        if locked_center_xyz is not None:
            keep = np.linalg.norm(target_cloud[:, :3] - locked_center_xyz[None, :], axis=1) <= radius
            subset = target_cloud[keep]
            if len(subset) > 0:
                return subset

        locked_center_xy = self._locked_instance_center_xy
        if locked_center_xy is not None:
            keep = np.linalg.norm(target_cloud[:, :2] - locked_center_xy[None, :], axis=1) <= radius
            subset = target_cloud[keep]
            if len(subset) > 0:
                return subset

        locked_candidate_key = self._locked_instance_candidate_key
        if locked_candidate_key is not None:
            keys = [self._quantize_candidate_xy(point[:2]) == locked_candidate_key for point in target_cloud]
            subset = target_cloud[np.asarray(keys, dtype=bool)]
            if len(subset) > 0:
                return subset

        return np.empty((0, target_cloud.shape[1]), dtype=target_cloud.dtype)

    def _is_candidate_for_locked_instance(self, metrics: Dict[str, Any]) -> bool:
        if self._locked_goal_xy is None:
            return False

        radius = float(self._locked_instance_radius or self._candidate_accept_radius)

        candidate_xyz = metrics.get("candidate_xyz")
        if candidate_xyz is not None and self._locked_instance_center_xyz is not None:
            candidate_xyz_arr = np.asarray(candidate_xyz, dtype=np.float32).reshape(-1)
            if candidate_xyz_arr.shape[0] == 3:
                if float(np.linalg.norm(candidate_xyz_arr - self._locked_instance_center_xyz)) <= radius:
                    return True

        candidate_xy = metrics.get("candidate_xy")
        if candidate_xy is not None and self._locked_instance_center_xy is not None:
            candidate_xy_arr = np.asarray(candidate_xy, dtype=np.float32).reshape(-1)
            if candidate_xy_arr.shape[0] == 2:
                if float(np.linalg.norm(candidate_xy_arr - self._locked_instance_center_xy)) <= radius:
                    return True

        candidate_key = metrics.get("candidate_key")
        if candidate_key is not None and self._locked_instance_candidate_key is not None:
            if candidate_key == self._locked_instance_candidate_key:
                return True

        return False

    def _update_locked_instance_from_cloud(self, cloud_subset: np.ndarray) -> None:
        if len(cloud_subset) == 0:
            return
        self._locked_instance_center_xy = np.median(cloud_subset[:, :2], axis=0).astype(np.float32)
        if cloud_subset.shape[1] >= 3:
            self._locked_instance_center_xyz = np.median(cloud_subset[:, :3], axis=0).astype(np.float32)
        if self._locked_instance_center_xy is not None:
            self._locked_instance_candidate_key = self._quantize_candidate_xy(self._locked_instance_center_xy)

    def _acquire_navigation_goal_lock(self, goal: np.ndarray) -> np.ndarray:
        locked_goal = np.asarray(goal[:2], dtype=np.float32).copy()
        self._locked_goal_xy = locked_goal
        locked_instance = self._select_locked_instance_for_goal(locked_goal)
        if locked_instance is None:
            locked_instance = {
                "center_xy": locked_goal.astype(float).tolist(),
                "center_xyz": None,
                "candidate_key": self._quantize_candidate_xy(locked_goal),
                "radius": float(self._candidate_accept_radius),
                "target_object": self._target_object.split("|")[0].strip().lower(),
            }
        self._bind_locked_instance(locked_instance)
        self._add_step_debug(f"GoalLock: acquire({locked_goal[0]:.2f},{locked_goal[1]:.2f})")
        return locked_goal.copy()

    def _release_navigation_goal_lock(self, reason: str) -> None:
        if self._locked_goal_xy is None:
            return
        locked_goal = self._locked_goal_xy.copy()
        self._locked_goal_xy = None
        self._bind_locked_instance(None)
        self._add_step_debug(
            f"GoalLock: release({reason},{locked_goal[0]:.2f},{locked_goal[1]:.2f})"
        )

    def _get_navigation_goal(self, position: np.ndarray) -> Union[None, np.ndarray]:
        # Navigation owns the lock. Once we have committed to a target instance, new
        # detections should not immediately rewrite the pointnav goal mid-approach.
        if self._locked_goal_xy is not None:
            if not self._object_map.has_object(self._target_object):
                self._release_navigation_goal_lock("target_cloud_missing")
                return None
            target_cloud = self._object_map.get_target_cloud(self._target_object)
            target_cloud = self._filter_target_cloud_for_navigation(target_cloud)
            locked_cloud = self._filter_target_cloud_for_locked_instance(target_cloud)
            if len(locked_cloud) == 0:
                self._release_navigation_goal_lock("locked_instance_missing")
                return None

            refreshed_goal = self._object_map.get_closest_point_from_cloud(locked_cloud, position)[:2].astype(np.float32)
            prev_goal = self._locked_goal_xy.copy()
            self._locked_goal_xy = refreshed_goal
            self._update_locked_instance_from_cloud(locked_cloud)
            if float(np.linalg.norm(refreshed_goal - prev_goal)) > 0.05:
                self._add_step_debug(
                    "GoalLock: refresh("
                    f"({prev_goal[0]:.2f},{prev_goal[1]:.2f})->({refreshed_goal[0]:.2f},{refreshed_goal[1]:.2f}))"
                )
            else:
                self._add_step_debug(f"GoalLock: keep({self._locked_goal_xy[0]:.2f},{self._locked_goal_xy[1]:.2f})")
            return self._locked_goal_xy.copy()

        detected_goal = self._get_target_object_location(position)
        if detected_goal is None:
            return None

        # During initialization the policy is not actually navigating yet, so delay
        # lock acquisition until the first real navigate step.
        if not self._done_initializing:
            return detected_goal

        return self._acquire_navigation_goal_lock(detected_goal)

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "locked_nav_goal": None if self._locked_goal_xy is None else self._locked_goal_xy.copy(),
            "stop_called": self._called_stop,
            "episode_log_extra": self._get_episode_log_extra(),
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # `annotated_frame` can be None if the detections were created without an
            # image_source (e.g., future callers / custom detectors). Be defensive.
            base_rgb = detections.annotated_frame
            if base_rgb is None:
                base_rgb = self._observations_cache["object_map_rgbd"][0][0]
            else:
                base_rgb = base_rgb.copy()
            annotated_rgb = cv2.drawContours(base_rgb, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        # 添加 BLIP2 提示词到视频标注（仅本 step 内实际发送的 prompt）
        if self._step_blip2_prompt:
            policy_info["render_below_images"].append("blip2_prompt")
            policy_info["blip2_prompt"] = self._step_blip2_prompt

        if self._step_debug_msgs:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "\n".join(self._step_debug_msgs)

        return policy_info

    def _reset_scan_gate_reject_dedup(self) -> None:
        self._scan_last_gate_reject_signature = None

    def _record_scan_gate_reject(self, reason: str, key: Optional[str] = None) -> None:
        signature = f"{reason}:{key}" if key is not None else reason
        if self._scan_last_gate_reject_signature == signature:
            return
        self._scan_episode_gate_reject_counts[reason] = self._scan_episode_gate_reject_counts.get(reason, 0) + 1
        self._scan_last_gate_reject_signature = signature

    def _get_episode_log_extra(self) -> Dict[str, Any]:
        return {
            "scan_num_triggers": int(self._scan_num_triggers),
            "scan_total_steps": int(self._scan_episode_total_steps),
            "scan_trigger_steps": [int(step) for step in self._scan_episode_trigger_steps],
            "scan_frontier_keys": [[float(x), float(y)] for (x, y) in self._scan_episode_frontier_keys],
            "scan_gate_reject_counts": dict(self._scan_episode_gate_reject_counts),
            "applied_detector_type": self._current_detector_type,
            "applied_detection_threshold": float(self._applied_detection_threshold),
            "first_detection_verify_calls": int(self._first_detection_verify_calls),
            "first_detection_verify_pass": int(self._first_detection_verify_pass),
            "first_detection_verify_reject": int(self._first_detection_verify_reject),
            "second_detection_verify_calls": int(self._second_detection_verify_calls),
            "second_detection_verify_pass": int(self._second_detection_verify_pass),
            "second_detection_verify_reject": int(self._second_detection_verify_reject),
            "stop_verify_calls": int(self._stop_verify_calls),
            "stop_verify_pass": int(self._stop_verify_pass),
            "stop_verify_reject": int(self._stop_verify_reject),
            "stop_verify_current_match": int(self._stop_verify_current_match),
            "stop_verify_history_match": int(self._stop_verify_history_match),
            "stop_verify_no_goal_matched_detection": int(self._stop_verify_no_goal_matched_detection),
            "stop_verify_bypass_no_client": int(self._stop_verify_bypass_no_client),
            "stop_verify_bypass_no_image": int(self._stop_verify_bypass_no_image),
            "stop_verify_bypass_api_error": int(self._stop_verify_bypass_api_error),
            "stop_verify_goal_match_dist": (
                None if self._stop_verify_goal_match_dist is None else float(self._stop_verify_goal_match_dist)
            ),
            "rejected_region_writes": int(self._rejected_region_writes),
            "reject_memory_hits": int(self._reject_memory_hits),
            "rejected_instance_memory_count": int(len(self._rejected_target_instances)),
            "target_cloud_clears": int(self._target_cloud_clears),
            "goal_lock_active": bool(self._locked_goal_xy is not None),
            "goal_lock_xy": (
                [float(self._locked_goal_xy[0]), float(self._locked_goal_xy[1])]
                if self._locked_goal_xy is not None
                else None
            ),
            "stop_verify_last_reject_center": (
                [float(self._stop_verify_last_reject_center[0]), float(self._stop_verify_last_reject_center[1])]
                if self._stop_verify_last_reject_center is not None
                else None
            ),
        }

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes_raw = self._target_object.split("|")
        # GroundingDINO/YOLO 类名通常是自然语言（空格），避免使用下划线造成严格匹配失败
        target_classes = [c.replace("_", " ") for c in target_classes_raw]
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        # GroundingDINO caption 需要用 " . " 分隔并以 " ." 结尾；为空会导致全部被 filter 掉
        gdino_caption = self._non_coco_caption.strip()
        if not gdino_caption:
            gdino_caption = " . ".join(target_classes) + " ."
        gdino_caption = self._prepare_grounding_dino_caption(img, gdino_caption)

        active_detector = "yolo" if has_coco else "gdino"
        self._current_detector_type = active_detector
        self._applied_detection_threshold = (
            self._coco_candidate_threshold if has_coco else self._non_coco_candidate_threshold
        )
        detections = (
            self._predict_yolov7_cached(img)
            if active_detector == "yolo"
            else self._predict_grounding_dino_cached(img, caption=gdino_caption)
        )
        detections.filter_by_class(target_classes)
        detections.filter_by_conf(float(self._applied_detection_threshold))

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            self._current_detector_type = "gdino"
            self._applied_detection_threshold = self._non_coco_candidate_threshold
            detections = self._predict_grounding_dino_cached(img, caption=gdino_caption)
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(float(self._applied_detection_threshold))

        self._add_step_debug(
            f"DetectorThreshold: detector={self._current_detector_type} "
            f"threshold={self._applied_detection_threshold:.2f}"
        )
        return detections

    def _should_perform_360_scan_in_explore(self, observations: Dict) -> bool:
        # 总开关未启用，或者 scan 已经在执行中时，explore 分支不能再启动新的 scan。
        # 这里直接返回，避免 explore/scan 两套控制流交错。
        if not self._dynamic_relocation_enabled or self._scan_active:
            return False

        # 先把“当前正在追的 frontier”同步成 scan candidate；如果机器人已经真正走到
        # candidate 附近，再把它升级为 pending scan。这样 scan 的触发时机是“到达后”，
        # 而不是“选中 frontier 的当下”。
        self._sync_scan_frontier_candidate()
        self._arm_pending_360_scan_if_reached()

        # 还没有形成待执行的 pending scan，说明当前没有满足“到达 frontier 后再扫”
        # 的前提，explore 继续正常走。
        if self._scan_pending_frontier_key is None:
            return False

        # 单个 episode 的 scan 预算控制。即使后面还有高价值 frontier，也不允许无限
        # 触发原地扫描，避免把探索 budget 过多消耗在旋转上。
        if self._scan_max_scans_per_episode > 0 and self._scan_num_triggers >= self._scan_max_scans_per_episode:
            self._record_scan_gate_reject("budget_exhausted")
            self._add_step_debug(
                f"ScanGate: budget_exhausted({self._scan_num_triggers}/{self._scan_max_scans_per_episode})"
            )
            self._clear_pending_360_scan()
            return False

        # 同一个 frontier 在 cooldown 窗口内不重复触发。这里不清 pending，是为了保留
        # 这次候选 frontier 的身份，等 cooldown 过后再自然评估。
        if (
            self._scan_pending_frontier_key == self._scan_last_frontier_key
            and int(self._num_steps) - int(self._scan_last_trigger_step) < self._scan_cooldown_steps
        ):
            self._record_scan_gate_reject("frontier_cooldown", key=str(self._scan_pending_frontier_key))
            return False

        frontier_xy = self._scan_pending_frontier.copy()
        region_key = self._get_scan_region_key(frontier_xy)
        last_region_step = self._scan_region_last_trigger_steps.get(region_key)
        # 区域级去重：frontier 本身可能随着 explored 区域扩大而轻微漂移，但本质上还是
        # 同一片局部未知区域。这里按量化后的 region_key 去重，避免相邻 frontier 反复扫。
        if (
            last_region_step is not None
            and self._scan_region_revisit_steps > 0
            and int(self._num_steps) - int(last_region_step) < self._scan_region_revisit_steps
        ):
            self._record_scan_gate_reject("recent_region", key=str(region_key))
            self._add_step_debug(
                "ScanGate: recent_region("
                f"region={region_key},delta={int(self._num_steps) - int(last_region_step)},"
                f"ttl={self._scan_region_revisit_steps})"
            )
            self._clear_pending_360_scan()
            return False

        # 最后再做“值不值得扫”的 gate。子类可以基于 frontier value / 全图统计量
        # 否决这次 scan；一旦否决，就清掉 pending，避免旧 frontier 卡在队列里反复尝试。
        if not self._scan_frontier_value_allows(observations, frontier_xy):
            self._record_scan_gate_reject("value_gate", key=str(self._scan_pending_frontier_key))
            self._add_step_debug(f"ScanGate: drop_pending(frontier={self._scan_pending_frontier_key})")
            self._clear_pending_360_scan()
            return False

        # 走到这里，说明：已到达候选 frontier、没超预算、没撞 cooldown、不是近期重复区域，
        # 且 frontier value 也通过了子类 gate。本 step 允许进入 scan。
        return True

    def _scan_frontier_value_allows(self, observations: Dict, frontier_xy: np.ndarray) -> bool:
        """Optional hook for scan gating.

        Subclasses may override this to veto starting a 360 scan at a frontier.
        BaseObjectNavPolicy keeps the default permissive behavior so non-ITM
        policies do not need to implement value-based gating.
        """
        del observations, frontier_xy
        return True

    def _get_scan_frontier_reach_dist(self) -> float:
        if self._scan_frontier_reach_dist_override is not None:
            return self._scan_frontier_reach_dist_override
        return 0.6

    def _quantize_frontier(self, xy: np.ndarray, q: float) -> Tuple[float, float]:
        q = max(float(q), 1e-6)
        quantized = np.round(np.asarray(xy, dtype=np.float32) / q) * q
        return (float(quantized[0]), float(quantized[1]))

    def _get_scan_region_key(self, xy: np.ndarray) -> Tuple[float, float]:
        return self._quantize_frontier(xy, self._scan_region_q)

    def _clear_pending_360_scan(self, clear_candidate: bool = False) -> None:
        self._scan_pending_frontier = np.zeros(2, dtype=np.float32)
        self._scan_pending_frontier_key = None
        if clear_candidate:
            self._scan_candidate_frontier = np.zeros(2, dtype=np.float32)
            self._scan_candidate_frontier_key = None
        self._reset_scan_gate_reject_dedup()

    def _get_current_scan_frontier(self) -> Optional[np.ndarray]:
        last_frontier = getattr(self, "_last_frontier", None)
        if last_frontier is None:
            return None
        frontier_xy = np.asarray(last_frontier, dtype=np.float32).reshape(-1)
        if frontier_xy.shape[0] != 2 or np.array_equal(frontier_xy, np.zeros(2, dtype=np.float32)):
            return None
        return frontier_xy

    def _sync_scan_frontier_candidate(self) -> None:
        if self._scan_pending_frontier_key is not None:
            return

        frontier_xy = self._get_current_scan_frontier()
        if frontier_xy is None:
            self._scan_candidate_frontier = np.zeros(2, dtype=np.float32)
            self._scan_candidate_frontier_key = None
            return

        frontier_key = self._quantize_frontier(frontier_xy, self._scan_frontier_q)
        if frontier_key == self._scan_candidate_frontier_key:
            return

        self._scan_candidate_frontier = frontier_xy.copy()
        self._scan_candidate_frontier_key = frontier_key
        self._reset_scan_gate_reject_dedup()

    def _arm_pending_360_scan_if_reached(self) -> None:
        # 这里不是“看到一个合适 frontier 就立刻扫描”，而是分两步：
        # 1) _sync_scan_frontier_candidate() 先记录当前想观察的 frontier；
        # 2) 只有机器人真的走到该 frontier 附近后，才把它升级为 pending scan。
        # 这样可以避免机器人还在远处时就原地转圈扫描，浪费 step。
        if self._scan_candidate_frontier_key is None or self._scan_pending_frontier_key is not None:
            return

        if (
            self._scan_candidate_frontier_key == self._scan_last_frontier_key
            and int(self._num_steps) - int(self._scan_last_trigger_step) < self._scan_cooldown_steps
        ):
            # 同一个 frontier 在 cooldown 窗口内不重复 arm，避免刚扫描完又立刻触发。
            self._record_scan_gate_reject("frontier_cooldown", key=str(self._scan_candidate_frontier_key))
            return

        robot_xy = np.asarray(self._observations_cache["robot_xy"], dtype=np.float32).reshape(-1)
        reach_dist = self._get_scan_frontier_reach_dist()
        if float(np.linalg.norm(robot_xy - self._scan_candidate_frontier)) > reach_dist:
            # 还没真正走到 frontier 附近时，只保留 candidate，不启动 pending scan。
            return

        # 到达 frontier 后，冻结这次待执行的 scan 目标。
        # 后续即使 best frontier 继续变化，本次 pending scan 也不会被漂移的候选点打断。
        self._scan_pending_frontier = self._scan_candidate_frontier.copy()
        self._scan_pending_frontier_key = self._scan_candidate_frontier_key
        self._scan_candidate_frontier = np.zeros(2, dtype=np.float32)
        self._scan_candidate_frontier_key = None
        self._add_step_debug(f"ScanArm: frontier={self._scan_pending_frontier_key}")

    def _start_360_scan(self) -> None:
        frontier_xy = self._scan_pending_frontier.copy()
        frontier_key = self._scan_pending_frontier_key
        if frontier_key is None:
            fallback_frontier = self._get_current_scan_frontier()
            if fallback_frontier is None:
                return
            frontier_xy = fallback_frontier.copy()
            frontier_key = self._quantize_frontier(frontier_xy, self._scan_frontier_q)

        self._scan_active = True
        self._scan_steps_remaining = int(self._scan_turn_steps)
        self._scan_total_steps = int(self._scan_turn_steps)
        self._scan_frontier = frontier_xy.copy()
        self._scan_last_frontier_key = frontier_key
        self._scan_num_triggers += 1
        self._scan_episode_trigger_steps.append(int(self._num_steps))
        self._scan_episode_frontier_keys.append((float(frontier_key[0]), float(frontier_key[1])))
        self._scan_region_last_trigger_steps[self._get_scan_region_key(frontier_xy)] = int(self._num_steps)
        self._clear_pending_360_scan(clear_candidate=True)
        self._add_step_debug(
            "ScanStart: "
            f"frontier={self._scan_last_frontier_key},steps={self._scan_total_steps},"
            f"budget={self._scan_num_triggers}/{self._scan_max_scans_per_episode}"
        )

    def _stop_360_scan(self, reason: str = "") -> None:
        if self._scan_active:
            self._scan_last_trigger_step = int(self._num_steps)
            self._reset_pointnav_on_next_call = True
            reason_suffix = f",reason={reason}" if reason else ""
            self._add_step_debug(
                f"ScanStop: frontier={self._scan_last_frontier_key},step={self._scan_last_trigger_step}{reason_suffix}"
            )
        self._scan_active = False
        self._scan_steps_remaining = 0
        self._scan_total_steps = 0
        self._scan_frontier = np.zeros(2, dtype=np.float32)

    def _perform_360_scan(self) -> Optional[Tensor]:
        if not self._scan_active:
            return None

        if self._scan_steps_remaining <= 0:
            self._stop_360_scan(reason="complete")
            return None

        current_step = self._scan_total_steps - self._scan_steps_remaining + 1
        self._add_step_debug(f"Scan: {current_step}/{self._scan_total_steps}")
        self._scan_episode_total_steps += 1
        self._scan_steps_remaining -= 1
        return self._get_scan_turn_action()

    def _pointnav(self, goal: np.ndarray, stop: bool = False, detections: Optional[ObjectDetections] = None) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        reset_pointnav = bool(self._reset_pointnav_on_next_call)
        self._reset_pointnav_on_next_call = False
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                reset_pointnav = True
            self._last_goal = goal
        if reset_pointnav:
            self._pointnav_policy.reset()
            masks = torch.zeros_like(masks)
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if stop:  # 在 navigate 模式时输出距离信息
            print(f"[Navigate] Distance to target: {rho:.2f}m (stop_radius: {self._pointnav_stop_radius}m)")
        if rho < self._pointnav_stop_radius and stop:
            if self._should_run_stop_verification(rho):
                stop_verify_snapshot = self._select_stop_verify_snapshot(goal, rho)
                if stop_verify_snapshot is None:
                    reason = "no_goal_matched_detection"
                    self._stop_verify_no_goal_matched_detection += 1
                    self._add_step_debug(f"stop_verify: {reason}")
                    if not self._stop_verify_no_match_fail_open:
                        self._record_target_verification_outcome("stop_verify", False, reason)
                        self._handle_stop_verification_reject(goal, reason)
                        return self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
                else:
                    passed, reason = self._run_stop_verification(stop_verify_snapshot)
                    if not passed:
                        self._handle_stop_verification_reject(goal, reason)
                        return self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            print(f"[Navigate] Reached target! Stopping.")
            self._called_stop = True
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _get_stop_verify_client(self) -> Optional[OpenAIClient]:
        if self._stop_verify_client is not None:
            return self._stop_verify_client
        self._stop_verify_client = OpenAIClient.from_env()
        return self._stop_verify_client

    def _should_run_stop_verification(self, rho: float) -> bool:
        if not self._stop_verify_enabled:
            return False
        if rho > self._stop_verify_near_goal_dist:
            return False
        return True

    def _get_candidate_xy_from_metrics(self, metrics: Dict[str, Any]) -> Optional[np.ndarray]:
        candidate_xy = metrics.get("candidate_xy")
        if candidate_xy is not None:
            candidate_xy_arr = np.asarray(candidate_xy, dtype=np.float32).reshape(-1)
            if candidate_xy_arr.shape[0] >= 2:
                return candidate_xy_arr[:2].astype(np.float32)

        candidate_xyz = metrics.get("candidate_xyz")
        if candidate_xyz is not None:
            candidate_xyz_arr = np.asarray(candidate_xyz, dtype=np.float32).reshape(-1)
            if candidate_xyz_arr.shape[0] >= 2:
                return candidate_xyz_arr[:2].astype(np.float32)

        return None

    def _to_debug_list(self, value: Any) -> Optional[List[float]]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return [float(v) for v in arr.tolist()]

    def _clone_single_detection_for_history(self, detections: ObjectDetections, idx: int) -> ObjectDetections:
        item = self._single_detection(detections, idx)
        if item.image_source is not None:
            item.image_source = item.image_source.copy()
            item._annotated_frame = None
        return item

    def _build_stop_verify_snapshot(
        self,
        detections: ObjectDetections,
        idx: int,
        metrics: Dict[str, Any],
        goal_xy: np.ndarray,
        match_dist: float,
    ) -> Dict[str, Any]:
        return {
            "detections": self._clone_single_detection_for_history(detections, idx),
            "step": int(self._num_steps),
            "source_step": int(self._num_steps),
            "candidate_xy": self._to_debug_list(metrics.get("candidate_xy")),
            "candidate_xyz": self._to_debug_list(metrics.get("candidate_xyz")),
            "goal_xy": self._to_debug_list(goal_xy),
            "box_xyxy": [float(v) for v in metrics.get("box_xyxy", [])],
            "center_x": float(metrics.get("center_x", 0.0)),
            "center_y": float(metrics.get("center_y", 0.0)),
            "depth_median": (
                None if metrics.get("depth_median") is None else float(metrics.get("depth_median"))
            ),
            "score": float(metrics.get("score", 0.0)),
            "phrase": str(metrics.get("phrase", "")),
            "match_dist": float(match_dist),
        }

    def _record_stop_verify_history(
        self,
        detections: ObjectDetections,
        metrics_by_idx: List[Dict[str, Any]],
        goal_xy: Optional[np.ndarray],
        allowed_indices: Optional[set[int]] = None,
    ) -> None:
        if not self._stop_verify_enabled or self._stop_verify_history_size <= 0:
            return
        if goal_xy is None or detections.num_detections <= 0:
            return

        goal_xy_arr = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
        if goal_xy_arr.shape[0] < 2:
            return
        goal_xy_arr = goal_xy_arr[:2]

        best_idx: Optional[int] = None
        best_dist: Optional[float] = None
        for idx, metrics in enumerate(metrics_by_idx):
            if allowed_indices is not None and idx not in allowed_indices:
                continue
            candidate_xy = self._get_candidate_xy_from_metrics(metrics)
            if candidate_xy is None:
                continue
            dist = float(np.linalg.norm(candidate_xy - goal_xy_arr))
            if dist > float(self._stop_verify_goal_match_radius):
                continue
            if best_dist is None or dist < best_dist:
                best_idx = int(idx)
                best_dist = dist

        if best_idx is None or best_dist is None:
            return

        snapshot = self._build_stop_verify_snapshot(
            detections,
            best_idx,
            metrics_by_idx[best_idx],
            goal_xy_arr,
            best_dist,
        )
        self._stop_verify_history.append(snapshot)

    def _select_stop_verify_snapshot(self, goal: np.ndarray, rho: float) -> Optional[Dict[str, Any]]:
        if not self._stop_verify_history:
            return None

        goal_xy = np.asarray(goal[:2], dtype=np.float32).reshape(-1)
        if goal_xy.shape[0] < 2:
            return None
        goal_xy = goal_xy[:2]

        current_step = int(self._num_steps)
        best_current: Optional[Tuple[float, int, Dict[str, Any]]] = None
        best_history: Optional[Tuple[float, int, Dict[str, Any]]] = None
        for snapshot in self._stop_verify_history:
            candidate_xy = snapshot.get("candidate_xy")
            if candidate_xy is None:
                continue
            candidate_xy_arr = np.asarray(candidate_xy, dtype=np.float32).reshape(-1)
            if candidate_xy_arr.shape[0] < 2:
                continue
            dist = float(np.linalg.norm(candidate_xy_arr[:2] - goal_xy))
            if dist > float(self._stop_verify_goal_match_radius):
                continue
            source_step = int(snapshot.get("source_step", snapshot.get("step", -1)))
            item = (dist, -source_step, snapshot)
            if source_step == current_step:
                if best_current is None or (dist, -source_step) < (best_current[0], best_current[1]):
                    best_current = item
            elif self._stop_verify_use_history:
                if best_history is None or (dist, -source_step) < (best_history[0], best_history[1]):
                    best_history = item

        selected = best_current if best_current is not None else best_history
        if selected is None:
            return None

        match_dist, neg_source_step, snapshot = selected
        source_step = -neg_source_step
        source = "current" if source_step == current_step else "history"
        age_steps = max(0, current_step - source_step)
        selected_snapshot = dict(snapshot)
        selected_snapshot.update(
            {
                "source": source,
                "source_step": int(source_step),
                "history_age_steps": int(age_steps),
                "goal_xy": self._to_debug_list(goal_xy),
                "rho": float(rho),
                "match_dist": float(match_dist),
            }
        )
        self._stop_verify_goal_match_dist = float(match_dist)
        if source == "current":
            self._stop_verify_current_match += 1
        else:
            self._stop_verify_history_match += 1
        self._add_step_debug(f"StopVerify: source={source} age={age_steps} dist={match_dist:.2f}")
        return selected_snapshot

    @staticmethod
    def _parse_first_detection_verify_categories(raw: str) -> FrozenSet[str]:
        return frozenset(category.strip().lower() for category in raw.split(",") if category.strip())

    def _should_run_first_detection_verification(
        self,
        detections: ObjectDetections,
        had_target_cloud: bool,
    ) -> bool:
        del had_target_cloud
        if not self._first_detection_verify_enabled:
            return False
        if detections.num_detections <= 0:
            return False
        if self._first_detection_verify_categories:
            target_name = self._target_object.split("|")[0].strip().lower()
            if target_name not in self._first_detection_verify_categories:
                if not self._first_detection_verify_category_skip_logged:
                    self._add_step_debug(f"FirstDetectVerify: skip(category={target_name} not_selected)")
                    self._first_detection_verify_category_skip_logged = True
                return False
        return True

    def _score_candidate_for_verification(self, metrics: Dict[str, Any], is_pending: bool) -> float:
        score = float(metrics.get("score", 0.0))
        center_dist = float(metrics.get("center_dist", 1.0))
        area_ratio = float(metrics.get("area_ratio", 0.0))
        depth_median = metrics.get("depth_median")

        center_bonus = max(0.0, 1.0 - center_dist / 0.75)
        area_bonus = min(1.0, area_ratio / 0.1)

        depth_bonus = 0.0
        if depth_median is not None:
            min_depth = float(self._candidate_retry_min_depth)
            max_depth = float(self._candidate_retry_max_depth)
            if max_depth > min_depth and min_depth <= float(depth_median) <= max_depth:
                depth_mid = 0.5 * (min_depth + max_depth)
                half_span = max(0.5 * (max_depth - min_depth), 1e-6)
                depth_bonus = max(0.0, 1.0 - abs(float(depth_median) - depth_mid) / half_span)

        pending_bonus = 0.15 if is_pending else 0.0
        return score + 0.2 * center_bonus + 0.1 * area_bonus + 0.1 * depth_bonus + pending_bonus

    def _get_first_detection_proposal_indices(
        self,
        detections: ObjectDetections,
        metrics_by_idx: List[Dict[str, Any]],
    ) -> List[int]:
        if detections.num_detections <= 0 or not metrics_by_idx:
            return []

        ranked_by_key: Dict[Any, Tuple[float, int]] = {}
        for idx, metrics in enumerate(metrics_by_idx):
            score = float(metrics.get("score", 0.0))
            if score < float(self._applied_detection_threshold):
                continue
            if self._is_candidate_rejected(metrics):
                continue
            if self._is_candidate_accepted(metrics):
                continue
            candidate_key = metrics.get("candidate_key")
            pending = self._pending_target_candidates.get(candidate_key) if candidate_key is not None else None
            if pending is not None:
                continue
            verify_rank = self._score_candidate_for_verification(metrics, is_pending=pending is not None)
            dedup_key = candidate_key if candidate_key is not None else ("idx", int(idx))
            prev = ranked_by_key.get(dedup_key)
            if prev is None or verify_rank > prev[0]:
                ranked_by_key[dedup_key] = (verify_rank, int(idx))

        if not ranked_by_key:
            return []

        ranked = list(ranked_by_key.values())
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, idx in ranked]

    def _get_stop_verify_debug_dir(self) -> Optional[str]:
        if not self._stop_verify_debug_save_images:
            return None
        if self._stop_verify_debug_dir:
            return self._stop_verify_debug_dir
        log_dir = os.environ.get("ZSOS_LOG_DIR", "").strip()
        if not log_dir:
            return None
        return os.path.join(log_dir, "stop_verify_images")

    def _save_stop_verify_debug_image(
        self,
        image: np.ndarray,
        *,
        stage: str,
        detections: ObjectDetections,
        user_prompt: str,
        system_prompt: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        debug_dir = self._get_stop_verify_debug_dir()
        if not debug_dir:
            return None

        try:
            os.makedirs(debug_dir, exist_ok=True)
            target_name = self._target_object.split("|")[0].replace(" ", "_")
            episode_id = os.environ.get("VLFM_CURRENT_EPISODE_ID", "").strip() or "unknown_ep"
            scene_id_raw = os.environ.get("VLFM_CURRENT_SCENE_ID", "").strip()
            scene_name = self._sanitize_debug_name_component(os.path.basename(scene_id_raw).split(".")[0] or "unknown_scene")
            stamp = time.time_ns()
            stem = (
                f"episode={episode_id}-scene={scene_name}-step={int(self._num_steps):04d}-"
                f"stage={self._sanitize_debug_name_component(stage)}-target={target_name}-{stamp}"
            )
            image_path = os.path.join(debug_dir, f"{stem}.jpg")
            meta_path = os.path.join(debug_dir, f"{stem}.json")
            run_video_dir = os.environ.get("VLFM_CURRENT_VIDEO_DIR", "").strip()
            video_match_glob = (
                os.path.join(run_video_dir, f"episode={episode_id}-*.mp4")
                if run_video_dir
                else None
            )
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            with open(meta_path, "w") as f:
                payload: Dict[str, Any] = {
                    "episode_id": episode_id,
                    "scene_id": scene_id_raw or None,
                    "scene_name": scene_name,
                    "step": int(self._num_steps),
                    "stage": stage,
                    "target_object": target_name,
                    "model": self._target_verify_openai_model,
                    "detector_type": self._current_detector_type,
                    "applied_detection_threshold": float(self._applied_detection_threshold),
                    "detections": self._serialize_detection_debug(detections),
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "image_path": image_path,
                    "video_match_glob": video_match_glob,
                }
                if extra_meta:
                    payload.update(extra_meta)
                json.dump(payload, f, indent=2)
            self._add_step_debug(f"StopVerify: saved_image({os.path.basename(image_path)})")
            return meta_path
        except Exception as exc:
            self._add_step_debug(f"StopVerify: save_image_failed({type(exc).__name__})")
            return None

    def _update_stop_verify_debug_meta(self, meta_path: Optional[str], extra: Dict[str, Any]) -> None:
        if not meta_path:
            return
        try:
            data: Dict[str, Any] = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
            data.update(extra)
            with open(meta_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            self._add_step_debug(f"StopVerify: update_meta_failed({type(exc).__name__})")

    def _sanitize_debug_name_component(self, value: str) -> str:
        sanitized = self._safe_name_re.sub("_", value.strip())
        sanitized = sanitized.strip("._-")
        return sanitized or "unknown"

    def _serialize_detection_debug(self, detections: ObjectDetections) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        img_h = int(detections.image_source.shape[0]) if detections.image_source is not None else 0
        img_w = int(detections.image_source.shape[1]) if detections.image_source is not None else 0
        for idx, (box, logit, phrase) in enumerate(zip(detections.boxes, detections.logits, detections.phrases)):
            box_arr = box.detach().cpu().numpy().astype(float)
            if box_arr.max() <= 1.0 + 1e-6 and img_h > 0 and img_w > 0:
                box_xyxy = (
                    box_arr * np.array([img_w, img_h, img_w, img_h], dtype=float)
                ).tolist()
            else:
                box_xyxy = box_arr.tolist()
            score = float(logit.detach().cpu().item())
            items.append(
                {
                    "index": int(idx),
                    "phrase": str(phrase),
                    "logit": score,
                    "score_percent": round(score * 100.0, 4),
                    "box_xyxy": [float(v) for v in box_xyxy],
                }
            )
        return items

    def _subset_detections(self, detections: ObjectDetections, indices: List[int]) -> ObjectDetections:
        tensor_idx = torch.tensor(indices, dtype=torch.long, device=detections.boxes.device)
        return ObjectDetections(
            boxes=detections.boxes.index_select(0, tensor_idx).clone(),
            logits=detections.logits.index_select(0, tensor_idx).clone(),
            phrases=[detections.phrases[int(i)] for i in indices],
            image_source=detections.image_source,
            fmt="xyxy",
        )

    def _single_detection(self, detections: ObjectDetections, idx: int) -> ObjectDetections:
        return ObjectDetections(
            boxes=detections.boxes[idx : idx + 1].clone(),
            logits=detections.logits[idx : idx + 1].clone(),
            phrases=[detections.phrases[idx]],
            image_source=detections.image_source,
            fmt="xyxy",
        )

    def _build_som_verify_image(
        self,
        detections: ObjectDetections,
        candidate_indices: List[int],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        assert detections.image_source is not None
        image = detections.image_source.copy()
        img_h, img_w = image.shape[:2]
        label_meta: List[Dict[str, Any]] = []
        colors = [
            (255, 64, 64),
            (64, 255, 64),
            (64, 160, 255),
            (255, 210, 64),
        ]

        for display_idx, det_idx in enumerate(candidate_indices, start=1):
            box = detections.boxes[det_idx].detach().cpu().numpy().astype(float)
            if box.max() <= 1.0 + 1e-6:
                box *= np.array([img_w, img_h, img_w, img_h], dtype=float)
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            color = colors[(display_idx - 1) % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=4)

            tag = f"[{display_idx}]"
            tag_font = cv2.FONT_HERSHEY_SIMPLEX
            tag_scale = 0.9
            tag_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(tag, tag_font, tag_scale, tag_thickness)
            text_x = max(0, min(x1, img_w - text_w - 8))
            text_y = max(text_h + 8, y1)
            cv2.rectangle(
                image,
                (text_x - 4, text_y - text_h - 6),
                (text_x + text_w + 4, text_y + 4),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                image,
                tag,
                (text_x, text_y),
                tag_font,
                tag_scale,
                color,
                tag_thickness,
                cv2.LINE_AA,
            )
            label_meta.append(
                {
                    "label": int(display_idx),
                    "detection_index": int(det_idx),
                    "phrase": str(detections.phrases[det_idx]),
                    "score": float(detections.logits[det_idx].detach().cpu().item()),
                    "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        return image, label_meta

    def _quantize_candidate_xy(self, xy: np.ndarray) -> Tuple[float, float]:
        q = max(float(self._candidate_region_q), 1e-6)
        quantized = np.round(np.asarray(xy, dtype=np.float32) / q) * q
        return (float(quantized[0]), float(quantized[1]))

    def _quantize_scalar(self, value: float, q: float) -> float:
        q = max(float(q), 1e-6)
        return float(np.round(float(value) / q) * q)

    def _build_fallback_candidate_key(self, center_x: float, center_y: float) -> Tuple[Any, ...]:
        robot_xy = np.asarray(self._observations_cache.get("robot_xy", np.zeros(2)), dtype=np.float32).reshape(-1)
        q = float(self._candidate_region_q)
        center_q = float(self._candidate_view_center_q)
        return (
            "view2d",
            self._quantize_scalar(float(robot_xy[0]), q),
            self._quantize_scalar(float(robot_xy[1]), q),
            self._quantize_scalar(center_x, center_q),
            self._quantize_scalar(center_y, center_q),
        )

    def _extract_candidate_metrics(
        self,
        detections: ObjectDetections,
        idx: int,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        object_mask: np.ndarray,
    ) -> Dict[str, Any]:
        height, width = depth.shape[:2]
        box = detections.boxes[idx].detach().cpu().numpy().astype(float)
        if box.max() <= 1.0 + 1e-6:
            box *= np.array([width, height, width, height], dtype=float)
        x1, y1, x2, y2 = box.tolist()
        center_x = float((x1 + x2) / 2.0 / max(width, 1))
        center_y = float((y1 + y2) / 2.0 / max(height, 1))
        area_ratio = float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(float(height * width), 1.0))
        center_dist = float(np.linalg.norm(np.array([center_x - 0.5, center_y - 0.5], dtype=np.float32)))

        # object_map_rgbd follows the same contract as ObjectPointCloudMap / ValueMap:
        # depth is normalized to [0, 1] and converted to metric depth locally.
        depth_normalized = np.asarray(depth, dtype=np.float32)
        valid_mask = (
            (object_mask > 0)
            & np.isfinite(depth_normalized)
            & (depth_normalized > 0.0)
            & (depth_normalized <= 1.0)
        )
        depth_median: Optional[float] = None
        candidate_xy: Optional[np.ndarray] = None
        candidate_xyz: Optional[np.ndarray] = None
        candidate_key: Optional[Tuple[Any, ...]] = None
        if np.any(valid_mask):
            depth_metric = depth_normalized * float(max_depth - min_depth) + float(min_depth)
            depth_median = float(np.median(depth_metric[valid_mask]))
            local_cloud = get_point_cloud(depth_metric, valid_mask.astype(np.uint8), fx, fy)
            if len(local_cloud) > 0:
                global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
                candidate_xyz = np.median(global_cloud[:, :3], axis=0).astype(np.float32)
                candidate_xy = np.median(global_cloud[:, :2], axis=0).astype(np.float32)
                candidate_key = self._quantize_candidate_xy(candidate_xy)
        if candidate_key is None:
            candidate_key = self._build_fallback_candidate_key(center_x, center_y)

        return {
            "score": float(detections.logits[idx].detach().cpu().item()),
            "phrase": str(detections.phrases[idx]),
            "center_x": center_x,
            "center_y": center_y,
            "center_dist": center_dist,
            "area_ratio": area_ratio,
            "depth_median": depth_median,
            "candidate_xy": candidate_xy,
            "candidate_xyz": candidate_xyz,
            "candidate_key": candidate_key,
            "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        }

    def _is_candidate_centered(self, metrics: Dict[str, Any]) -> bool:
        ratio = float(self._candidate_retry_center_ratio)
        return (
            abs(float(metrics["center_x"]) - 0.5) <= ratio
            and abs(float(metrics["center_y"]) - 0.5) <= ratio
        )

    def _is_candidate_depth_valid(self, metrics: Dict[str, Any]) -> bool:
        depth_median = metrics.get("depth_median")
        if depth_median is None:
            return False
        return float(self._candidate_retry_min_depth) <= float(depth_median) <= float(self._candidate_retry_max_depth)

    def _is_candidate_evidence_improved(self, previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
        prev_center = float(previous.get("center_dist", 1.0))
        curr_center = float(current.get("center_dist", 1.0))
        prev_area = float(previous.get("area_ratio", 0.0))
        curr_area = float(current.get("area_ratio", 0.0))
        prev_score = float(previous.get("score", 0.0))
        curr_score = float(current.get("score", 0.0))
        return (
            curr_center < prev_center
            or curr_area >= prev_area + float(self._candidate_retry_min_area_gain)
            or curr_score >= prev_score + float(self._candidate_retry_min_score_gain)
        )

    def _is_candidate_in_rejected_instance_memory(self, metrics: Dict[str, Any]) -> bool:
        candidate_xyz = metrics.get("candidate_xyz")
        if candidate_xyz is None:
            return False
        candidate_xyz = np.asarray(candidate_xyz, dtype=np.float32).reshape(-1)
        if candidate_xyz.shape[0] != 3:
            return False

        target_name = self._target_object.split("|")[0].strip().lower()
        for item in self._rejected_target_instances:
            item_target = str(item.get("target_object", "")).strip().lower()
            if item_target and item_target != target_name:
                continue
            center_xyz = np.asarray(item.get("center_xyz", np.zeros(3)), dtype=np.float32).reshape(-1)
            if center_xyz.shape[0] != 3:
                continue
            radius = float(item.get("radius", self._candidate_blacklist_radius))
            if float(np.linalg.norm(candidate_xyz - center_xyz)) <= radius:
                return True
        return False

    def _is_candidate_in_accepted_instance_memory(self, metrics: Dict[str, Any]) -> bool:
        candidate_xyz = metrics.get("candidate_xyz")
        candidate_xy = metrics.get("candidate_xy")
        candidate_xyz_arr: Optional[np.ndarray] = None
        candidate_xy_arr: Optional[np.ndarray] = None
        if candidate_xyz is not None:
            candidate_xyz_arr = np.asarray(candidate_xyz, dtype=np.float32).reshape(-1)
            if candidate_xyz_arr.shape[0] != 3:
                candidate_xyz_arr = None
        if candidate_xy is not None:
            candidate_xy_arr = np.asarray(candidate_xy, dtype=np.float32).reshape(-1)
            if candidate_xy_arr.shape[0] != 2:
                candidate_xy_arr = None
        if candidate_xyz_arr is None and candidate_xy_arr is None:
            return False

        target_name = self._target_object.split("|")[0].strip().lower()
        for item in self._accepted_target_instances:
            item_target = str(item.get("target_object", "")).strip().lower()
            if item_target and item_target != target_name:
                continue
            radius = float(item.get("radius", self._candidate_accept_radius))

            center_xyz = np.asarray(item.get("center_xyz", np.zeros(3)), dtype=np.float32).reshape(-1)
            if candidate_xyz_arr is not None and center_xyz.shape[0] == 3:
                if float(np.linalg.norm(candidate_xyz_arr - center_xyz)) <= radius:
                    return True

            center_xy = np.asarray(item.get("center_xy", np.zeros(2)), dtype=np.float32).reshape(-1)
            if candidate_xy_arr is not None and center_xy.shape[0] == 2:
                if float(np.linalg.norm(candidate_xy_arr - center_xy)) <= radius:
                    return True
        return False

    def _is_candidate_accepted(self, metrics: Dict[str, Any]) -> bool:
        candidate_key = metrics.get("candidate_key")
        if self._is_candidate_in_accepted_instance_memory(metrics):
            return True
        return candidate_key is not None and candidate_key in self._accepted_target_regions

    def _is_candidate_rejected(self, metrics: Dict[str, Any]) -> bool:
        candidate_key = metrics.get("candidate_key")
        if self._is_candidate_in_rejected_instance_memory(metrics):
            return True
        return candidate_key is not None and candidate_key in self._rejected_target_regions

    def _cleanup_rejected_target_regions(self) -> None:
        now = int(self._num_steps)
        if self._rejected_target_regions:
            expired = [k for k, state in self._rejected_target_regions.items() if int(state.get("until_step", -1)) <= now]
            for k in expired:
                self._rejected_target_regions.pop(k, None)
        if self._rejected_target_instances:
            self._rejected_target_instances = [
                item for item in self._rejected_target_instances if int(item.get("until_step", -1)) > now
            ]

    def _write_rejected_candidate_memory(self, metrics: Dict[str, Any], reason: str) -> None:
        candidate_key = metrics.get("candidate_key")
        if candidate_key is None:
            return
        prev = self._rejected_target_regions.get(candidate_key, {})
        self._rejected_target_regions[candidate_key] = {
            "until_step": int(self._num_steps) + int(self._candidate_retry_ttl),
            "reason": reason,
            "reject_count": int(prev.get("reject_count", 0)) + 1,
        }
        candidate_xyz = metrics.get("candidate_xyz")
        if candidate_xyz is not None:
            candidate_xyz_arr = np.asarray(candidate_xyz, dtype=np.float32).reshape(-1)
            if candidate_xyz_arr.shape[0] == 3:
                self._rejected_target_instances.append(
                    {
                        "center_xyz": candidate_xyz_arr.astype(float).tolist(),
                        "radius": float(self._candidate_blacklist_radius),
                        "until_step": int(self._num_steps) + int(self._candidate_retry_ttl),
                        "reason": reason,
                        "target_object": self._target_object.split("|")[0].strip().lower(),
                    }
                )
        if self._rejected_region_enabled and hasattr(self, "_value_map") and metrics.get("candidate_xy") is not None:
            value_map = getattr(self, "_value_map", None)
            if value_map is not None and hasattr(value_map, "apply_negative_bias"):
                wrote = value_map.apply_negative_bias(
                    np.asarray(metrics["candidate_xy"], dtype=np.float32),
                    radius_m=self._rejected_region_radius,
                    value_decay=self._rejected_region_decay,
                    bias_increment=1.0,
                )
                if wrote:
                    self._rejected_region_writes += 1

    def _write_accepted_candidate_memory(self, metrics: Dict[str, Any], reason: str) -> None:
        if self._is_candidate_accepted(metrics):
            return
        candidate_key = metrics.get("candidate_key")
        if candidate_key is not None:
            self._accepted_target_regions[candidate_key] = {
                "reason": reason,
                "accepted_step": int(self._num_steps),
            }
        candidate_xy = metrics.get("candidate_xy")
        candidate_xy_arr: Optional[np.ndarray] = None
        if candidate_xy is not None:
            candidate_xy_arr = np.asarray(candidate_xy, dtype=np.float32).reshape(-1)
            if candidate_xy_arr.shape[0] != 2:
                candidate_xy_arr = None
        candidate_xyz = metrics.get("candidate_xyz")
        candidate_xyz_arr: Optional[np.ndarray] = None
        if candidate_xyz is not None:
            candidate_xyz_arr = np.asarray(candidate_xyz, dtype=np.float32).reshape(-1)
            if candidate_xyz_arr.shape[0] != 3:
                candidate_xyz_arr = None
        if candidate_xyz_arr is None and candidate_xy_arr is None:
            return

        self._accepted_target_instances.append(
            {
                "center_xyz": candidate_xyz_arr.astype(float).tolist() if candidate_xyz_arr is not None else None,
                "center_xy": candidate_xy_arr.astype(float).tolist() if candidate_xy_arr is not None else None,
                "candidate_key": candidate_key,
                "radius": float(self._candidate_accept_radius),
                "accepted_step": int(self._num_steps),
                "reason": reason,
                "target_object": self._target_object.split("|")[0].strip().lower(),
            }
        )

    def _forget_accepted_candidate_memory(self, metrics: Dict[str, Any]) -> None:
        candidate_key = metrics.get("candidate_key")
        if candidate_key is not None:
            self._accepted_target_regions.pop(candidate_key, None)

        candidate_xyz = metrics.get("candidate_xyz")
        candidate_xy = metrics.get("candidate_xy")
        candidate_xyz_arr: Optional[np.ndarray] = None
        candidate_xy_arr: Optional[np.ndarray] = None
        if candidate_xyz is not None:
            candidate_xyz_arr = np.asarray(candidate_xyz, dtype=np.float32).reshape(-1)
            if candidate_xyz_arr.shape[0] != 3:
                candidate_xyz_arr = None
        if candidate_xy is not None:
            candidate_xy_arr = np.asarray(candidate_xy, dtype=np.float32).reshape(-1)
            if candidate_xy_arr.shape[0] != 2:
                candidate_xy_arr = None
        if candidate_xyz_arr is None and candidate_xy_arr is None:
            return

        filtered_items: List[Dict[str, Any]] = []
        for item in self._accepted_target_instances:
            radius = float(item.get("radius", self._candidate_accept_radius))
            center_xyz = np.asarray(item.get("center_xyz", np.zeros(3)), dtype=np.float32).reshape(-1)
            if candidate_xyz_arr is not None and center_xyz.shape[0] == 3:
                if float(np.linalg.norm(candidate_xyz_arr - center_xyz)) <= radius:
                    continue
            center_xy = np.asarray(item.get("center_xy", np.zeros(2)), dtype=np.float32).reshape(-1)
            if candidate_xy_arr is not None and center_xy.shape[0] == 2:
                if float(np.linalg.norm(candidate_xy_arr - center_xy)) <= radius:
                    continue
            filtered_items.append(item)
        self._accepted_target_instances = filtered_items

    def _build_goal_rejection_metrics(self, goal: np.ndarray) -> Dict[str, Any]:
        candidate_xy = np.asarray(goal[:2], dtype=np.float32).reshape(-1)
        metrics: Dict[str, Any] = {
            "candidate_xy": candidate_xy,
            "candidate_key": self._quantize_candidate_xy(candidate_xy),
            "candidate_xyz": None,
        }
        if not self._object_map.has_object(self._target_object):
            return metrics

        target_cloud = self._object_map.get_target_cloud(self._target_object)
        if len(target_cloud) == 0:
            return metrics

        nearest_idx = int(np.argmin(np.linalg.norm(target_cloud[:, :2] - candidate_xy[None, :], axis=1)))
        candidate_xyz = target_cloud[nearest_idx, :3].astype(np.float32)
        metrics["candidate_xyz"] = candidate_xyz
        metrics["candidate_xy"] = candidate_xyz[:2].astype(np.float32)
        metrics["candidate_key"] = self._quantize_candidate_xy(metrics["candidate_xy"])
        return metrics

    def _record_target_verification_outcome(self, stage: str, passed: bool, reason: str) -> None:
        if stage.startswith("first_detect"):
            self._first_detection_verify_calls += 1
        elif stage.startswith("second_detect"):
            self._second_detection_verify_calls += 1
        else:
            self._stop_verify_calls += 1

        if passed:
            if stage.startswith("first_detect"):
                self._first_detection_verify_pass += 1
            elif stage.startswith("second_detect"):
                self._second_detection_verify_pass += 1
            else:
                self._stop_verify_pass += 1
            self._add_step_debug(f"{stage}: pass({reason})")
        else:
            if stage.startswith("first_detect"):
                self._first_detection_verify_reject += 1
            elif stage.startswith("second_detect"):
                self._second_detection_verify_reject += 1
            else:
                self._stop_verify_reject += 1
            self._add_step_debug(f"{stage}: reject({reason})")

    def _run_target_verification_group(
        self,
        detections: ObjectDetections,
        candidate_indices: List[int],
        stage: str,
    ) -> Tuple[List[int], str]:
        if not candidate_indices:
            return [], "no_candidates"
        if len(candidate_indices) == 1:
            passed, reason = self._run_target_verification(
                self._single_detection(detections, candidate_indices[0]),
                stage=stage,
            )
            return ([candidate_indices[0]] if passed else []), reason

        client = self._get_stop_verify_client()
        if client is None:
            self._add_step_debug(f"{stage}: no_client")
            return list(candidate_indices), "no_client"
        if detections.image_source is None:
            self._add_step_debug(f"{stage}: no_image")
            return list(candidate_indices), "no_image"

        verify_image, label_meta = self._build_som_verify_image(detections, candidate_indices)
        target_name = self._target_object.split("|")[0]
        labels_str = ", ".join(f"[{item['label']}]" for item in label_meta)
        detector_text = (
            "YOLOv7"
            if self._current_detector_type == "yolo"
            else "Grounding-DINO"
            if self._current_detector_type == "gdino"
            else self._current_detector_type
        )
        candidate_details = "; ".join(
            f"[{item['label']}]: detector label='{item['phrase']}', score={float(item['score']):.3f}"
            for item in label_meta
        )
        system_prompt_base = (
            "You are an expert visual perception assistant for an embodied navigating robot. "
            "Your task is to verify which numbered bounding boxes contain the requested target. "
            "CRITICAL RULES:\n"
            "1. PHYSICALITY: A valid target must be a real, physical, 3D object that the robot can navigate to. "
            "Reject any 2D pictures, posters, reflections in mirrors, TV screens, or miniature toys.\n"
            '2. FORMAT: Return ONLY raw JSON with exactly two keys: "selected_indices" (array of integers) and '
            '"reason" (short string). Do NOT wrap the JSON in markdown such as ```json.'
        )
        if stage.startswith("second_detect"):
            system_prompt = (
                system_prompt_base
                + "\nThis is a re-verification of previously rejected candidates. "
                "Be extremely strict. Only keep boxes that unmistakably contain the correct target."
            )
            user_prompt = (
                f"The image shows candidate boxes labeled {labels_str}. "
                f"The intended target category is '{target_name}'. "
                f"The current detector is {detector_text}. "
                f"Candidate details: {candidate_details}. "
                f"Select every numbered box that clearly contains a real, physical '{target_name}'. "
                "If none are true targets, return an empty array. Output raw JSON only."
            )
        else:
            system_prompt = system_prompt_base
            user_prompt = (
                f"The image shows candidate boxes labeled {labels_str}. "
                f"The intended target category is '{target_name}'. "
                f"The current detector is {detector_text}. "
                f"Candidate details: {candidate_details}. "
                f"Select every numbered box that contains a real, physical '{target_name}' that the robot can navigate to. "
                "If none are true targets, return an empty array. Output raw JSON only."
            )

        subset = self._subset_detections(detections, candidate_indices)
        meta_path = self._save_stop_verify_debug_image(
            verify_image,
            stage=stage,
            detections=subset,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            extra_meta={"som_candidates": label_meta},
        )
        try:
            data = client.call_openai_vision_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image=verify_image,
                json_mode=True,
                model=self._target_verify_openai_model or None,
                reasoning_effort=self._target_verify_reasoning_effort or None,
            )
        except Exception as exc:
            self._update_stop_verify_debug_meta(
                meta_path,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            self._add_step_debug(f"{stage}: api_error({type(exc).__name__})")
            return list(candidate_indices), f"api_error:{type(exc).__name__}"

        raw_selected = data.get("selected_indices", [])
        if not raw_selected and "selected_index" in data:
            raw_selected = [data.get("selected_index")]
        selected_labels: List[int] = []
        if isinstance(raw_selected, list):
            for value in raw_selected:
                if isinstance(value, (int, float)):
                    selected_labels.append(int(value))
                elif isinstance(value, str):
                    stripped = value.strip()
                    if stripped.isdigit():
                        selected_labels.append(int(stripped))
        label_to_idx = {int(item["label"]): int(item["detection_index"]) for item in label_meta}
        selected_indices_global = [label_to_idx[label] for label in selected_labels if label in label_to_idx]
        selected_indices_global = sorted(set(selected_indices_global))
        reason = str(data.get("reason", "")).strip() or (
            "verified" if selected_indices_global else "rejected"
        )
        self._update_stop_verify_debug_meta(
            meta_path,
            {
                "response": data,
                "selected_detection_indices": selected_indices_global,
                "reason": reason,
            },
        )
        self._record_target_verification_outcome(stage, bool(selected_indices_global), reason)
        return selected_indices_global, reason

    def _record_stop_verify_bypass(self, reason: str) -> None:
        if reason == "no_client":
            self._stop_verify_bypass_no_client += 1
        elif reason == "no_image":
            self._stop_verify_bypass_no_image += 1
        elif reason.startswith("api_error"):
            self._stop_verify_bypass_api_error += 1

    def _run_target_verification(
        self,
        detections: ObjectDetections,
        stage: str,
        verification_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        client = self._get_stop_verify_client()
        if client is None:
            self._add_step_debug(f"{stage}: no_client")
            if stage == "stop_verify":
                self._record_stop_verify_bypass("no_client")
                self._record_target_verification_outcome(stage, True, "no_client")
            return True, "no_client"

        verify_image = detections.annotated_frame
        if verify_image is None:
            verify_image = detections.image_source
        if verify_image is None:
            self._add_step_debug(f"{stage}: no_image")
            if stage == "stop_verify":
                self._record_stop_verify_bypass("no_image")
                self._record_target_verification_outcome(stage, True, "no_image")
            return True, "no_image"

        target_name = self._target_object.split("|")[0]
        detector_text = (
            "YOLOv7"
            if self._current_detector_type == "yolo"
            else "Grounding-DINO"
            if self._current_detector_type == "gdino"
            else self._current_detector_type
        )
        detected_label = str(detections.phrases[0]).strip() if detections.num_detections > 0 else ""
        detected_label_prompt = (
            f"The detector label for the highlighted bounding box is '{detected_label}'. "
            if detected_label
            else ""
        )
        detection_confidence = 0.0
        if detections.num_detections > 0:
            detection_confidence = float(detections.logits[0].detach().cpu().item())
        confidence_text = f"{detection_confidence:.3f}"
        system_prompt_base = (
            "You are an expert visual perception assistant for an embodied navigating robot. "
            "Your task is to verify whether the object inside the highlighted bounding box is the requested target. "
            "CRITICAL RULES:\n"
            "1. PHYSICALITY: The object must be a real, physical, 3D object that the robot can navigate to. "
            "Reject any 2D pictures, posters, reflections in mirrors, TV screens, or miniature toys.\n"
            '2. FORMAT: Return ONLY raw JSON with exactly two keys: "is_target" (boolean) and "reason" (short string). '
            "Do NOT wrap the JSON in markdown such as ```json."
        )
        if stage.startswith("first_detect"):
            system_prompt = system_prompt_base
            user_prompt = (
                "Please verify the object inside the highlighted bounding box. "
                f"The intended target category is '{target_name}'. "
                f"The current detector is {detector_text}. "
                f"{detected_label_prompt}"
                f"The detector confidence score for this box is {confidence_text}. "
                f"Based on the rules above, is the object inside the highlighted box a real, physical instance of '{target_name}'? "
                "Output raw JSON only."
            )
        elif stage.startswith("second_detect"):
            system_prompt = (
                system_prompt_base
                + "\nThis is a re-verification of a previously rejected candidate. "
                "Be extremely strict. Only confirm if the object in the highlighted box is unmistakably the correct target."
            )
            user_prompt = (
                "Please re-verify the object inside the highlighted bounding box in this review frame. "
                f"The intended target category is '{target_name}'. "
                f"The current detector is {detector_text}. "
                f"{detected_label_prompt}"
                f"The detector confidence score for this box is {confidence_text}. "
                f"Is the object inside the highlighted box genuinely a real, physical '{target_name}'? "
                "Output raw JSON only."
            )
        else:
            system_prompt = system_prompt_base
            context_text = ""
            if verification_context:
                source = str(verification_context.get("source", "unknown"))
                match_dist = verification_context.get("match_dist")
                rho = verification_context.get("rho")
                age_steps = verification_context.get("history_age_steps")
                context_parts = [f"This candidate was selected from {source} visual evidence"]
                if age_steps is not None:
                    context_parts.append(f"history age is {int(age_steps)} navigation steps")
                if match_dist is not None:
                    context_parts.append(f"its 2D map distance to the current navigation goal is {float(match_dist):.2f} meters")
                if rho is not None:
                    context_parts.append(f"the robot is {float(rho):.2f} meters from the navigation goal")
                context_text = ". ".join(context_parts) + ". "
            user_prompt = (
                "Please verify the object inside the highlighted bounding box for a robot stop decision. "
                f"The intended target category is '{target_name}'. "
                f"The current detector is {detector_text}. "
                f"{detected_label_prompt}"
                f"The detector confidence score for this box is {confidence_text}. "
                f"{context_text}"
                f"Is the object inside the highlighted box truly a real, physical '{target_name}'? "
                "Output raw JSON only."
            )
        extra_meta = None
        if stage == "stop_verify" and verification_context:
            extra_meta = {
                "stop_verify_source": verification_context.get("source"),
                "stop_verify_goal_match_dist": verification_context.get("match_dist"),
                "stop_verify_history_age_steps": verification_context.get("history_age_steps"),
                "rho": verification_context.get("rho"),
                "candidate_xy": verification_context.get("candidate_xy"),
                "candidate_xyz": verification_context.get("candidate_xyz"),
                "goal_xy": verification_context.get("goal_xy"),
                "box_xyxy": verification_context.get("box_xyxy"),
                "box_center": [
                    float(verification_context.get("center_x", 0.0)),
                    float(verification_context.get("center_y", 0.0)),
                ],
                "depth_median": verification_context.get("depth_median"),
                "detector_score": verification_context.get("score"),
                "source_step": verification_context.get("source_step"),
            }
        meta_path = self._save_stop_verify_debug_image(
            verify_image,
            stage=stage,
            detections=detections,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            extra_meta=extra_meta,
        )

        try:
            data = client.call_openai_vision_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image=verify_image,
                json_mode=True,
                model=self._target_verify_openai_model or None,
                reasoning_effort=self._target_verify_reasoning_effort or None,
            )
        except Exception as exc:
            self._update_stop_verify_debug_meta(
                meta_path,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            self._add_step_debug(f"{stage}: api_error({type(exc).__name__})")
            if stage == "stop_verify":
                reason = f"api_error:{type(exc).__name__}"
                self._record_stop_verify_bypass(reason)
                self._record_target_verification_outcome(stage, True, reason)
            return True, f"api_error:{type(exc).__name__}"

        is_target = bool(data.get("is_target", False))
        reason = str(data.get("reason", "")).strip() or ("verified" if is_target else "rejected")
        self._update_stop_verify_debug_meta(
            meta_path,
            {
                "response": data,
                "is_target": is_target,
                "reason": reason,
            },
        )
        self._record_target_verification_outcome(stage, is_target, reason)
        return is_target, reason

    def _run_stop_verification(self, snapshot: Dict[str, Any]) -> Tuple[bool, str]:
        detections = snapshot.get("detections")
        if not isinstance(detections, ObjectDetections):
            self._stop_verify_no_goal_matched_detection += 1
            self._record_target_verification_outcome("stop_verify", False, "invalid_snapshot")
            return False, "invalid_snapshot"
        return self._run_target_verification(
            detections,
            stage="stop_verify",
            verification_context=snapshot,
        )

    def _handle_stop_verification_reject(self, goal: np.ndarray, reason: str) -> None:
        reject_metrics = self._build_goal_rejection_metrics(goal)
        self._forget_accepted_candidate_memory(reject_metrics)
        self._write_rejected_candidate_memory(reject_metrics, f"stop_verify:{reason}")
        reject_xy = reject_metrics.get("candidate_xy")
        if reject_xy is not None:
            reject_xy_arr = np.asarray(reject_xy, dtype=np.float32).reshape(-1)
            if reject_xy_arr.shape[0] >= 2:
                self._stop_verify_last_reject_center = (float(reject_xy_arr[0]), float(reject_xy_arr[1]))
                self._add_step_debug(
                    f"StopVerify: reject_memory(center=({reject_xy_arr[0]:.2f},{reject_xy_arr[1]:.2f}))"
                )

        self._release_navigation_goal_lock("stop_verify_reject")
        self._last_goal = np.zeros(2)
        self._reset_pointnav_on_next_call = True
        self._called_stop = False

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        had_target_cloud = self._object_map.has_object(self._target_object)
        should_run_first_detection_verify = self._should_run_first_detection_verification(
            detections,
            had_target_cloud,
        )
        verify_candidate_indices: Optional[set[int]] = None
        verify_candidate_order: List[int] = []
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)

        object_masks: List[np.ndarray] = []
        metrics_by_idx: List[Dict[str, Any]] = []
        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())
            metrics = self._extract_candidate_metrics(
                detections,
                idx,
                depth,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
                object_mask,
            )
            object_masks.append(object_mask)
            metrics_by_idx.append(metrics)

        locked_navigation_active = self._locked_goal_xy is not None and self._done_initializing
        if locked_navigation_active:
            allowed_update_indices = set()
            refresh_indices: List[int] = []
            ignored_indices: List[int] = []
            for idx, metrics in enumerate(metrics_by_idx):
                if self._is_candidate_for_locked_instance(metrics):
                    allowed_update_indices.add(idx)
                    refresh_indices.append(idx)
                else:
                    ignored_indices.append(idx)

            if refresh_indices:
                self._add_step_debug(f"GoalLock: refresh_candidates(idxs={refresh_indices})")
            if ignored_indices:
                self._add_step_debug(f"GoalLock: ignore_new_targets(idxs={ignored_indices})")
            self._record_stop_verify_history(
                detections,
                metrics_by_idx,
                self._locked_goal_xy,
                allowed_update_indices,
            )

            for idx, (object_mask, metrics) in enumerate(zip(object_masks, metrics_by_idx)):
                if idx not in allowed_update_indices:
                    continue

                if self._use_vqa:
                    contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                    question = f"Question: {self._vqa_prompt}"
                    if not detections.phrases[idx].endswith("ing"):
                        question += "a "
                    question += detections.phrases[idx] + "? Answer:"
                    answer = self._vqa.ask(annotated_rgb, question)
                    if not answer.lower().startswith("yes"):
                        continue

                self._object_masks[object_mask > 0] = 1
                self._object_map.update_map(
                    self._target_object,
                    depth,
                    object_mask,
                    tf_camera_to_episodic,
                    min_depth,
                    max_depth,
                    fx,
                    fy,
                )

            cone_fov = get_fov(fx, depth.shape[1])
            self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)
            return detections

        if should_run_first_detection_verify:
            verify_candidate_indices = set()
            verify_candidate_order = self._get_first_detection_proposal_indices(detections, metrics_by_idx)
            verify_candidate_indices = set(verify_candidate_order)
            if verify_candidate_indices:
                ranked_debug = [
                    (
                        idx,
                        round(
                            self._score_candidate_for_verification(
                                metrics_by_idx[idx],
                                is_pending=metrics_by_idx[idx].get("candidate_key") in self._pending_target_candidates,
                            ),
                            3,
                        ),
                    )
                    for idx in sorted(int(i) for i in verify_candidate_indices)
                ]
                self._add_step_debug(f"FirstDetectVerify: proposals={ranked_debug}")

        allowed_update_indices = (
            set() if verify_candidate_indices is not None else set(range(len(detections.logits)))
        )
        if verify_candidate_indices is not None:
            first_verify_group: List[int] = []
            second_verify_group: List[int] = []
            for idx, metrics in enumerate(metrics_by_idx):
                candidate_key = metrics.get("candidate_key")
                pending = self._pending_target_candidates.get(candidate_key) if candidate_key is not None else None
                score = float(metrics["score"])

                if self._is_candidate_rejected(metrics):
                    self._reject_memory_hits += 1
                    self._add_step_debug(f"FirstDetect: reject_memory_hit(idx={idx})")
                    continue

                if pending is not None:
                    if not self._is_candidate_centered(metrics):
                        self._add_step_debug(f"SecondDetect: not_centered(idx={idx})")
                        continue
                    if not self._is_candidate_depth_valid(metrics):
                        self._add_step_debug(f"SecondDetect: bad_depth(idx={idx})")
                        continue
                    if not self._is_candidate_evidence_improved(pending, metrics):
                        self._add_step_debug(f"SecondDetect: no_improve(idx={idx})")
                        continue
                    second_verify_group.append(idx)
                    continue

                if score < float(self._applied_detection_threshold):
                    self._add_step_debug(f"FirstDetect: low_conf(idx={idx},score={score:.2f})")
                    continue

                if self._is_candidate_accepted(metrics):
                    allowed_update_indices.add(idx)
                    self._add_step_debug(f"FirstDetect: accepted_memory_hit(idx={idx},score={score:.2f})")
                    continue

                if idx in verify_candidate_indices:
                    first_verify_group.append(idx)
                    self._add_step_debug(f"FirstDetect: proposal(idx={idx},score={score:.2f})")
                else:
                    self._add_step_debug(f"FirstDetect: dedup_or_defer(idx={idx},score={score:.2f})")

            if verify_candidate_order:
                verify_rank = {idx: rank for rank, idx in enumerate(verify_candidate_order)}
                first_verify_group.sort(key=lambda idx: verify_rank.get(idx, 10**9))
            if first_verify_group:
                self._add_step_debug(f"FirstDetectVerify: group={first_verify_group}")

            if first_verify_group:
                selected_indices, reason = self._run_target_verification_group(
                    detections,
                    first_verify_group,
                    stage="first_detect_som" if len(first_verify_group) > 1 else f"first_detect_{first_verify_group[0]}",
                )
                selected_set = set(int(i) for i in selected_indices)
                # Dedup by candidate_key to avoid redundant writes for duplicate detections
                pending_keys_written = set()
                for idx in first_verify_group:
                    if idx in selected_set:
                        self._write_accepted_candidate_memory(metrics_by_idx[idx], reason)
                        allowed_update_indices.add(idx)
                        continue
                    candidate_key = metrics_by_idx[idx].get("candidate_key")
                    if candidate_key is not None and candidate_key not in pending_keys_written:
                        pending_state = dict(metrics_by_idx[idx])
                        pending_state["last_reason"] = reason
                        pending_state["first_reject_step"] = int(self._num_steps)
                        self._pending_target_candidates[candidate_key] = pending_state
                        pending_keys_written.add(candidate_key)

            if second_verify_group:
                selected_indices, reason = self._run_target_verification_group(
                    detections,
                    second_verify_group,
                    stage="second_detect_som" if len(second_verify_group) > 1 else f"second_detect_{second_verify_group[0]}",
                )
                selected_set = set(int(i) for i in selected_indices)
                # Dedup by candidate_key to avoid redundant ops for duplicate detections
                pending_keys_cleared = set()
                rejected_keys_written = set()
                for idx in second_verify_group:
                    candidate_key = metrics_by_idx[idx].get("candidate_key")
                    if candidate_key is not None and candidate_key not in pending_keys_cleared:
                        self._pending_target_candidates.pop(candidate_key, None)
                        pending_keys_cleared.add(candidate_key)
                    if idx in selected_set:
                        self._write_accepted_candidate_memory(metrics_by_idx[idx], reason)
                        allowed_update_indices.add(idx)
                    else:
                        # Only write rejection for this key once
                        if candidate_key is None or candidate_key not in rejected_keys_written:
                            self._write_rejected_candidate_memory(metrics_by_idx[idx], reason)
                            if candidate_key is not None:
                                rejected_keys_written.add(candidate_key)

        for idx, (object_mask, metrics) in enumerate(zip(object_masks, metrics_by_idx)):
            if idx not in allowed_update_indices:
                continue

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.
            if self._use_vqa:
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "
                question += detections.phrases[idx] + "? Answer:"
                answer = self._vqa.ask(annotated_rgb, question)
                if not answer.lower().startswith("yes"):
                    continue

            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map(
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_candidate_threshold: float = 0.8
    non_coco_candidate_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
