import os
from collections import deque
from typing import Dict, List, Tuple
from roomverify_nav.mapping.llm_room_reasoner import LLMRoomReasoner
from roomverify_nav.utils.timing import timed


class RoomStateTracker:
    """跟踪机器人当前所在房间的状态"""

    def __init__(
        self,
        window_size: int = 10,  # 保留最近10步的观察
        change_threshold: float = 0.5,  # 物体分布变化阈值（Jaccard 距离）
        min_confidence_for_switch: float = 0.4,  # 房间切换所需的最小置信度
        max_per_class_per_frame: int = 1,  # 每帧同类计数上限（抑制多框噪声）
    ):
        self.observation_window = deque(maxlen=window_size)
        self.current_room_type: str = "unknown"
        self.current_room_confidence: float = 0.0
        # 最近一次推断的完整分布（用于保留不确定性）
        self.last_room_dist: Dict[str, float] = {}
        # 最近一次观测推断的 top1 置信度（观测层语义，不等同于 confirmed state）
        self.last_room_confidence: float = 0.0
        self.change_threshold = change_threshold
        self.min_confidence_for_switch = min_confidence_for_switch
        self.max_per_class_per_frame = int(
            os.environ.get("PROMPTING_ROOM_TRACKER_MAX_PER_CLASS", str(max_per_class_per_frame))
        )

        # 用于房间切换检测
        self.last_room_signature: Dict[str, float] = {}

        # 复用同一个 LLMRoomReasoner 实例，以保持调用间隔状态
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
        """
        返回最近一次推断的房间分布（截断 top_k），用于 prompt/策略层做不确定性门控。
        """
        if not self.last_room_dist:
            return {}
        items = sorted(self.last_room_dist.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return dict(items)

    def update(self, detected_objects: List[str]) -> Tuple[str, float, bool]:
        """
        更新房间状态

        Args:
            detected_objects: 当前帧检测到的物体列表

        Returns:
            (room_type, confidence, is_room_changed)
        """
        # 1. 添加当前观察到滑动窗口
        self.observation_window.append(detected_objects)

        # 2. 聚合窗口内的所有观察
        aggregated_objects = self._aggregate_observations()

        # 3. 推断当前房间类型
        room_dist = self._infer_room_from_objects(aggregated_objects)
        self.last_room_dist = room_dist
        new_room_type = max(room_dist, key=room_dist.get) if room_dist else "unknown"
        new_confidence = room_dist.get(new_room_type, 0.0)
        self.last_room_confidence = new_confidence

        # 4. 检测两个独立信号
        type_changed = new_room_type != self.current_room_type
        object_dist_changed = self._detect_object_distribution_change(aggregated_objects)

        # 5. 智能更新逻辑：分离"状态更新"和"房间切换信号"
        is_room_changed = False

        if type_changed:
            # 情况A：房间类型变化
            # 需要同时满足：置信度足够高 + 物体分布有显著变化
            if new_confidence >= self.min_confidence_for_switch and object_dist_changed:
                # 确认房间切换
                self.current_room_type = new_room_type
                self.current_room_confidence = new_confidence
                self.last_room_signature = self._compute_signature(aggregated_objects)
                is_room_changed = True
            # 否则：可能是推断噪声，保持当前状态不变
        else:
            # 情况B：房间类型未变
            # 只更新置信度（如果更高），不触发房间切换
            if new_confidence > self.current_room_confidence:
                self.current_room_confidence = new_confidence
                # 更新签名以反映最新物体分布
                self.last_room_signature = self._compute_signature(aggregated_objects)
            # is_room_changed 保持 False

        return self.current_room_type, self.current_room_confidence, is_room_changed

    def _aggregate_observations(self) -> Dict[str, float]:
        """
        聚合滑动窗口内的观察，返回加权物体频率

        返回格式: {"chair": 0.6, "table": 0.3, ...}
        越近的观察权重越高（时间衰减）
        """
        all_objects = []
        weights = []

        # 时间衰减权重：最近的观察权重更高
        for i, objects in enumerate(self.observation_window):
            weight = (i + 1) / len(self.observation_window)  # 线性衰减
            # 每帧对同类做封顶，避免 detector 多框/重复框把频率抬爆
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

        # 计算加权频率
        weighted_counts = {}
        for obj, w in zip(all_objects, weights):
            weighted_counts[obj] = weighted_counts.get(obj, 0) + w

        # 归一化
        total = sum(weighted_counts.values())
        if total > 0:
            return {k: v / total for k, v in weighted_counts.items()}
        return {}

    def _infer_room_from_objects(self, object_freq: Dict[str, float]) -> Dict[str, float]:
        """
        从物体频率推断房间类型分布

        直接使用频率字典，避免不必要的列表转换
        """
        # 复用实例以保持调用间隔状态
        return self._room_reasoner.infer_room_type_from_freq(object_freq)

    def _compute_signature(self, object_freq: Dict[str, float]) -> Dict[str, float]:
        """
        计算房间的"签名"（特征向量），用于检测房间切换
        只保留高频物体
        """
        # 过滤低频物体（噪声）
        threshold = 0.1
        return {k: v for k, v in object_freq.items() if v > threshold}

    def _detect_object_distribution_change(self, current_objects: Dict[str, float]) -> bool:
        """
        检测物体分布是否发生显著变化

        方法：计算当前物体分布与上次签名的 Jaccard 距离
        注意：这只检测"物体环境变化"，不直接判断"房间切换"
        """
        if not self.last_room_signature:
            return True  # 首次推断，认为有变化

        current_sig = self._compute_signature(current_objects)

        # 计算 Jaccard 相似度
        all_keys = set(current_sig.keys()) | set(self.last_room_signature.keys())
        intersection = sum(min(current_sig.get(k, 0), self.last_room_signature.get(k, 0)) for k in all_keys)
        union = sum(max(current_sig.get(k, 0), self.last_room_signature.get(k, 0)) for k in all_keys)

        if union == 0:
            return False

        similarity = intersection / union

        # 如果相似度低于阈值，认为物体分布发生了显著变化
        return similarity < (1 - self.change_threshold)

    def get_stable_objects(self, top_k: int = 5) -> List[str]:
        """
        返回滑动窗口内最稳定（高频）的物体
        用于生成 LLM 提示
        """
        aggregated = self._aggregate_observations()
        sorted_objects = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        return [obj for obj, _freq in sorted_objects[:top_k]]
