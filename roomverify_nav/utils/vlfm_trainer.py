# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf

from roomverify_nav.utils.video_utils import generate_video


def _drop_none_from_info(info: Any) -> Any:
    """Recursively remove None entries before Habitat flattens scalar metrics."""
    if isinstance(info, dict):
        return {
            k: _drop_none_from_info(v)
            for k, v in info.items()
            if v is not None
        }
    if isinstance(info, list):
        return [_drop_none_from_info(v) for v in info if v is not None]
    return info


def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    """Extract scalar metrics from info dict and return in a specific order for video filenames.

    The order of keys in the returned dict determines the order of metrics in video filenames.
    """
    info_filtered = _drop_none_from_info(info)
    info_filtered = {k: v for k, v in info_filtered.items() if not isinstance(v, list)}
    metrics = extract_scalars_from_info_habitat(info_filtered)

    # Define the desired order of metrics in video filenames
    # Metrics not in this list will be appended at the end in their original order
    preferred_order = [
        "success",
        "spl",
        "soft_spl",
        "distance_to_goal",
        # "distance_to_goal_reward",
        "target_detected",
        "stop_called",
        # "start_yaw",
        # "yaw",
        "traveled_stairs",
    ]

    # Reorder metrics according to preferred_order
    ordered_metrics = {}
    for key in preferred_order:
        if key in metrics:
            ordered_metrics[key] = metrics[key]

    # Add remaining metrics not in preferred_order
    for key, value in metrics.items():
        if key not in ordered_metrics:
            ordered_metrics[key] = value

    return ordered_metrics


@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    @staticmethod
    def _parse_eval_episode_ids() -> List[str]:
        raw = os.environ.get("VLFM_EVAL_EPISODE_IDS", "").strip()
        if not raw:
            return []
        parts = [p.strip() for p in raw.replace(" ", "").split(",") if p.strip()]
        # Keep as strings to match Habitat episode_id type.
        return parts

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        # habitat_baselines 的 construct_envs 会原地 shuffle 显式指定的 content_scenes（分片评测），需要可写 config。
        with read_write(config):
            self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)
        failure_cause_counts: Dict[str, int] = defaultdict(int)
        failure_cause_by_goal: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        goal_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)]

        # ------------------------------------------------------------------
        # Run output directory + per-episode log directory (always created)
        #
        # Even when video export is disabled, we still create a run folder under
        # `habitat_baselines.video_dir` and save logs under `episode_logs/`.
        # ------------------------------------------------------------------
        run_started_at_dt = datetime.now()
        timestamp = run_started_at_dt.strftime("%Y%m%d_%H%M%S")
        video_dir_with_timestamp = os.path.join(self.config.habitat_baselines.video_dir, timestamp)
        episode_logs_dir = os.path.join(video_dir_with_timestamp, "episode_logs")
        try:
            os.makedirs(episode_logs_dir, exist_ok=True)
        except Exception:
            pass

        # Default per-episode logs (including failure_cause) to this run's episode_logs.
        # Keep user-provided ZSOS_LOG_DIR if set, but always save run_summary.json to episode_logs_dir.
        if "ZSOS_LOG_DIR" not in os.environ:
            os.environ["ZSOS_LOG_DIR"] = episode_logs_dir
        try:
            os.makedirs(os.environ["ZSOS_LOG_DIR"], exist_ok=True)
        except Exception:
            pass

        # Update config video_dir so generated videos (if enabled) land in this run folder.
        with read_write(self.config):
            self.config.habitat_baselines.video_dir = video_dir_with_timestamp

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            print(f"[Video] Saving videos to: {video_dir_with_timestamp}")
        else:
            print(f"[Video] Video export disabled; created run dir: {video_dir_with_timestamp}")
        print(f"[Log] Saving per-episode logs to: {os.environ['ZSOS_LOG_DIR']}")
        os.environ["VLFM_CURRENT_VIDEO_DIR"] = video_dir_with_timestamp

        desired_episode_ids = self._parse_eval_episode_ids()
        if desired_episode_ids and self.config.habitat_baselines.num_environments != 1:
            raise RuntimeError("VLFM_EVAL_EPISODE_IDS currently requires num_environments=1")

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        if desired_episode_ids:
            number_of_eval_episodes = len(desired_episode_ids)
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()

        from roomverify_nav.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis()
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
            current_episodes_info = self.envs.current_episodes()
            if self.envs.num_envs == 1 and current_episodes_info:
                os.environ["VLFM_CURRENT_EPISODE_ID"] = str(current_episodes_info[0].episode_id)
                os.environ["VLFM_CURRENT_SCENE_ID"] = str(current_episodes_info[0].scene_id)

            # Optionally skip episodes until we reach a specific episode_id.
            # Useful for reproducing a particular "episode=XX" seen in video filenames.
            if desired_episode_ids:
                cur_id = str(current_episodes_info[0].episode_id)
                if cur_id not in desired_episode_ids or (current_episodes_info[0].scene_id, cur_id) in stats_episodes:
                    observations = self.envs.reset()
                    batch = batch_obs(observations, device=self.device)
                    batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore
                    current_episode_reward.zero_()
                    test_recurrent_hidden_states.zero_()
                    prev_actions.zero_()
                    not_done_masks.zero_()
                    continue

            with inference_mode():
                if (figure_dump := getattr(self._agent.actor_critic, "_figure_dump", None)) is not None:
                    if not not_done_masks[0].item():
                        figure_dump.episode = self.envs.call_at(0, "current_episode", {"all_info": True})
                action_data = self._agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    action_id = action_data.actions.cpu()[0].item()
                    filepath = os.path.join(
                        os.environ["VLFM_RECORD_ACTIONS_DIR"],
                        "actions.txt",
                    )
                    # If the file doesn't exist, create it
                    if not os.path.exists(filepath):
                        open(filepath, "w").close()
                    with open(filepath, "a") as f:
                        f.write(f"{action_id}\n")

                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if should_insert.item():
                            test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            episode_log_extras: List[Dict[str, Any]] = []
            for i in range(len(policy_infos)):
                policy_info = dict(policy_infos[i])
                episode_log_extra = policy_info.pop("episode_log_extra", {})
                if not isinstance(episode_log_extra, dict):
                    episode_log_extra = {}
                episode_log_extras.append(episode_log_extra)
                infos[i].update(policy_info)
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

                # episode ended
                if not not_done_masks[i].item():
                    scene = os.path.basename(current_episodes_info[i].scene_id).split(".")[0]
                    goal_category: Optional[str]
                    try:
                        goal_category = action_data.policy_info[i].get("target_object", None)
                    except Exception:
                        goal_category = None

                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    if getattr(self._agent.actor_critic, "_figure_dump", None) is not None:
                        self._agent.actor_critic._figure_dump.finish(
                            self._agent.actor_critic, episode_stats.get("success", 0.0)
                        )
                    current_episode_reward[i] = 0

                    goal_key = (goal_category or "unknown").strip() or "unknown"
                    goal_stats[goal_key]["count"] += 1.0
                    try:
                        goal_stats[goal_key]["success_sum"] += float(episode_stats.get("success", 0.0))
                    except Exception:
                        pass
                    for k_metric in ["spl", "soft_spl", "distance_to_goal", "target_detected", "stop_called"]:
                        try:
                            if k_metric in episode_stats:
                                goal_stats[goal_key][f"{k_metric}_sum"] += float(episode_stats[k_metric])
                        except Exception:
                            pass

                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")

                    from roomverify_nav.utils.episode_stats_logger import (
                        log_episode_stats,
                    )

                    try:
                        episode_log_extra = episode_log_extras[i] if i < len(episode_log_extras) else {}
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                            extra={
                                "goal_category": goal_category,
                                **episode_log_extra,
                            },
                        )
                    except Exception:
                        failure_cause = "Unknown"

                    failure_cause_counts[failure_cause] += 1
                    if goal_category:
                        failure_cause_by_goal[goal_category][failure_cause] += 1

                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                            name_metadata={
                                "goal_category": goal_category,
                                "failure_cause": failure_cause,
                            },
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        def _sorted_counts(d: Dict[str, int]) -> Dict[str, int]:
            return dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))

        def _safe_json_dump(path: str, payload: Dict[str, Any]) -> None:
            try:
                import json

                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2)
            except Exception:
                return

        primary_log_dir = os.environ.get("ZSOS_LOG_DIR", "").strip()
        # Always write summaries into this run's episode_logs dir (requirement).
        # Also mirror to ZSOS_LOG_DIR if it's set to a different directory.
        summary_dirs = [episode_logs_dir]
        if primary_log_dir and primary_log_dir != episode_logs_dir:
            summary_dirs.append(primary_log_dir)

        failure_summary_payload = {
            "failure_cause_counts": _sorted_counts(failure_cause_counts),
            "failure_cause_by_goal": {
                goal: _sorted_counts(c)
                for goal, c in sorted(failure_cause_by_goal.items(), key=lambda kv: kv[0])
            },
        }
        for d in summary_dirs:
            _safe_json_dump(os.path.join(d, "failure_summary.json"), failure_summary_payload)

        run_finished_at_dt = datetime.now()
        elapsed_seconds = float((run_finished_at_dt - run_started_at_dt).total_seconds())

        goal_summary: Dict[str, Any] = {}
        for goal, s in sorted(goal_stats.items(), key=lambda kv: kv[0]):
            count = float(s.get("count", 0.0))
            if count <= 0:
                continue
            success_sum = float(s.get("success_sum", 0.0))
            entry: Dict[str, Any] = {
                "count": int(count),
                "success_rate": success_sum / count,
            }
            for k_metric in ["spl", "soft_spl", "distance_to_goal", "target_detected", "stop_called"]:
                denom = count
                total_key = f"{k_metric}_sum"
                if total_key in s and denom > 0:
                    entry[f"avg_{k_metric}"] = float(s[total_key]) / denom
            goal_summary[goal] = entry

        run_meta: Optional[Dict[str, Any]] = None
        try:
            from roomverify_nav.utils.log_saver import _collect_run_meta  # type: ignore

            run_meta = _collect_run_meta()
        except Exception:
            run_meta = None

        # Include a resolved config snapshot for reproducibility.
        # This can be large, but is safer than relying on implicit defaults.
        config_snapshot: Optional[Dict[str, Any]] = None
        try:
            config_snapshot = OmegaConf.to_container(self.config, resolve=True)  # type: ignore
        except Exception:
            config_snapshot = None

        env_keys = [
            # Frontier rerank (room-prior decision-level context)
            "PROMPTING_FRONTIER_RERANK",
            "PROMPTING_FRONTIER_RERANK_ALPHA",
            "POINTNAV_STOP_RADIUS",
            "SCENE_DETECTION_THRESHOLD",
            "SCENE_DETECTION_MIN_BBOX_AREA_RATIO",
            "SCENE_DETECTION_GDINO_ENABLED",
            "SCENE_DETECTION_GDINO_MIN_YOLO_CLASSES",
            "SCENE_DETECTION_GDINO_INTERVAL",
            "SCENE_DETECTION_GDINO_THRESHOLD",
            # Stop-only verification + rejected-region memory
            "OPENAI_API_BASE",
            "OPENAI_MODEL",
            "OPENAI_VISION_MODEL",
            "OPENAI_TEMPERATURE",
            "OPENAI_ENABLE_THINKING",
            "TARGET_GUIDANCE_REASONING_EFFORT",
            "TARGET_VERIFY_REASONING_EFFORT",
            "PROMPTING_ROOM_REASONER_REASONING_EFFORT",
            "FIRST_DETECTION_VERIFY_ENABLED",
            "FIRST_DETECTION_VERIFY_CATEGORIES",
            "VLFM_DATASET",
            "TEST_CONTENT_SCENES",
            "TARGET_VERIFY_ACCEPT_RADIUS",
            "STOP_VERIFY_ENABLED",
            "STOP_VERIFY_NEAR_GOAL_DIST",
            "STOP_VERIFY_DEBUG_SAVE_IMAGES",
            "STOP_VERIFY_DEBUG_DIR",
            "DEBUG_TARGET_GUIDANCE",
            "REJECTED_REGION_ENABLED",
            "REJECTED_REGION_RADIUS",
            "REJECTED_REGION_DECAY",
            "NEGATIVE_BIAS_ENABLED",
            "NEGATIVE_BIAS_LAMBDA",
            "TARGET_VERIFY_REGION_Q",
            "TARGET_VERIFY_REJECT_TTL",
            "TARGET_VERIFY_BLACKLIST_RADIUS",
            "TARGET_VERIFY_CENTER_RATIO",
            "TARGET_VERIFY_MIN_AREA_GAIN",
            "TARGET_VERIFY_MIN_SCORE_GAIN",
            "TARGET_VERIFY_MIN_DEPTH",
            "TARGET_VERIFY_MAX_DEPTH",
        ]
        env_snapshot = {k: os.environ.get(k) for k in env_keys if k in os.environ}

        num_evals_completed = int(len(stats_episodes))
        num_unique_episodes_completed = int(len(ep_eval_count))
        num_total_safe = max(1, int(num_total))
        run_summary_payload: Dict[str, Any] = {
            "run_started_at": run_started_at_dt.isoformat(),
            "run_finished_at": run_finished_at_dt.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "checkpoint_index": int(checkpoint_index),
            "checkpoint_path": checkpoint_path,
            "eval": {
                "split": str(self.config.habitat_baselines.eval.split),
                "requested_num_episodes": int(number_of_eval_episodes),
                "evals_per_ep": int(evals_per_ep),
                "num_evals_completed": num_evals_completed,
                "num_unique_episodes_completed": num_unique_episodes_completed,
            },
            "paths": {
                "run_dir": video_dir_with_timestamp,
                "episode_logs_dir": episode_logs_dir,
                "zsos_log_dir": primary_log_dir or None,
            },
            "overall": {
                "num_total": int(num_total),
                "num_successes": int(num_successes),
                "success_rate": float(num_successes) / float(num_total_safe),
                "avg_spl": float(aggregated_stats.get("spl", 0.0)),
                "avg_soft_spl": float(aggregated_stats.get("soft_spl", 0.0)),
                "avg_distance_to_goal": float(aggregated_stats.get("distance_to_goal", 0.0)),
                "avg_target_detected": float(aggregated_stats.get("target_detected", 0.0)),
                "avg_stop_called": float(aggregated_stats.get("stop_called", 0.0)),
            },
            "aggregated_stats": {k: float(v) for k, v in aggregated_stats.items()},
            "failure_summary": failure_summary_payload,
            "by_goal_category": goal_summary,
            "env": env_snapshot,
            "run_meta": run_meta,
            "config": config_snapshot,
        }
        for d in summary_dirs:
            _safe_json_dump(os.path.join(d, "run_summary.json"), run_summary_payload)

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
