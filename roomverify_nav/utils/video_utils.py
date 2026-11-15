# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union

import numpy as np
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import images_to_video


_SAFE_COMPONENT_RE = re.compile(r"[^0-9A-Za-z._-]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def _sanitize_name_component(value: str, max_len: int = 80) -> str:
    value = value.strip()
    value = _SAFE_COMPONENT_RE.sub("_", value)
    value = _MULTI_UNDERSCORE_RE.sub("_", value)
    value = value.strip("._-")
    if not value:
        return "unknown"
    if len(value) > max_len:
        value = value[:max_len].rstrip("._-")
    return value


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
    verbose: bool = True,
    keys_to_include_in_name: Optional[List[str]] = None,
    name_metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Generate a video on disk and/or tensorboard, with optional string metadata in the filename.

    This is a thin wrapper around `habitat_baselines.utils.common.generate_video` that additionally
    supports including string fields (e.g. `scene_id`, `goal_category`) in the disk filename.
    """

    if len(images) < 1:
        return

    metric_strs: List[str] = []

    if name_metadata:
        for k, v in name_metadata.items():
            if v is None:
                continue
            v_str = str(v).strip()
            if not v_str:
                continue
            metric_strs.append(f"{k}={_sanitize_name_component(v_str)}")

    if keys_to_include_in_name is not None and len(keys_to_include_in_name) > 0:
        use_metrics_k = [
            k for k in metrics if any(to_include_k in k for to_include_k in keys_to_include_in_name)
        ]
    else:
        use_metrics_k = list(metrics.keys())

    for k in use_metrics_k:
        metric_strs.append(f"{k}={metrics[k]:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(metric_strs)

    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, fps=fps, verbose=verbose)

    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(f"episode{episode_id}", checkpoint_idx, images, fps=fps)
