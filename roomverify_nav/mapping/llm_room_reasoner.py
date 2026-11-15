from __future__ import annotations

from collections import OrderedDict, Counter
from typing import Dict, Iterable, Optional
import os
import numpy as np

# 导入父类
from roomverify_nav.mapping.room_reasoner import RoomReasoner, RoomDistribution
from roomverify_nav.utils.timing import TimingContext

from roomverify_nav.utils.query_llm import OpenAIClient
from roomverify_nav.utils.room_schema import (
    get_current_room_types,
    normalize_room_type_key,
    room_list_str,
)


class LLMRoomReasoner(RoomReasoner):
    """
    LLM 增强的房间类型推理器。

    采用三层智能决策架构:
    1. 精确缓存查询 (O(1))
    2. 启发式预检 (O(n×m×k))
    3. LLM 推理 (1-3秒)

    通过智能缓存和启发式预检,在保证实时性能的同时提升泛化能力。
    """

    def __init__(self, max_cache_size: Optional[int] = None):
        super().__init__()
        self._client = None
        self._cache = OrderedDict()  # LRU 缓存
        self._cache_hits = 0
        self._cache_misses = 0
        self._llm_call_count = 0  # LLM 调用次数统计

        # 调试模式
        self._debug = os.environ.get("DEBUG_ROOM_REASONER") == "1"

        # 调用频率限制
        self._last_llm_call_step = -999  # 上次 LLM 调用的步数
        self._current_step = 0  # 当前步数
        self._last_llm_result: Optional[RoomDistribution] = None  # 上次 LLM 结果

        # 配置参数 (从环境变量读取,或使用默认值)
        self.cache_size = int(
            os.environ.get("PROMPTING_ROOM_REASONER_CACHE_SIZE", "128") if max_cache_size is None else max_cache_size
        )
        self.min_freq_threshold = float(os.environ.get("PROMPTING_ROOM_REASONER_MIN_FREQ", "0.15"))
        self.confidence_threshold = float(os.environ.get("PROMPTING_ROOM_REASONER_CONFIDENCE_THRESH", "0.6"))
        self.api_timeout = float(os.environ.get("PROMPTING_ROOM_REASONER_TIMEOUT", "5.0"))
        # LLM 调用最小间隔步数（默认 10 步调用一次）
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
        # episode 边界：只清运行态，不清全局缓存（缓存对跨 episode 复用通常是利大于弊）
        self._current_step = 0
        self._last_llm_call_step = -999
        self._last_llm_result = None

    def _get_client(self) -> Optional[OpenAIClient]:
        """
        延迟初始化和缓存 OpenAI 客户端。

        复制自 TargetGuidanceProvider,保持一致性。
        """
        if self._client is not None:
            return self._client

        self._client = OpenAIClient.from_env()
        return self._client

    def _compute_frequencies(self, observed_objects: Iterable[str]) -> Dict[str, float]:
        """
        计算观察物体的频率分布。

        Args:
            observed_objects: 观察到的物体列表（可能包含重复）

        Returns:
            物体→频率的字典，总频率为1.0

        Example:
            Input: ["chair", "chair", "table"]
            Output: {"chair": 0.667, "table": 0.333}
        """
        counts = Counter(observed_objects)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {obj: count / total for obj, count in counts.items()}

    def _make_cache_key(self, object_freq: Dict[str, float]) -> frozenset:
        """
        从物体频率生成缓存 key。

        使用频率阈值过滤，只保留相对高频的物体。
        这可以减少缓存 key 的数量，同时保留主要信号。

        Args:
            object_freq: 物体频率字典

        Returns:
            frozenset 对象，可作为字典 key

        Example:
            Input: {"chair": 0.6, "table": 0.3, "lamp": 0.05, "pen": 0.05}
            Output: frozenset({"chair", "table"})  # lamp 和 pen 频率低于阈值，被忽略
        """
        filtered_objects = {obj for obj, freq in object_freq.items() if freq >= self.min_freq_threshold}
        return frozenset(filtered_objects)

    def _update_cache(self, key: frozenset, value: RoomDistribution) -> None:
        """
        更新 LRU 缓存。

        如果 key 已存在，移动到末尾（表示最近使用）。
        如果缓存超过大小限制，移除最老的项。

        Args:
            key: 缓存 key (frozenset)
            value: 房间分布结果
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value

        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # 移除最老的项

    @property
    def cache_stats(self) -> Dict[str, float]:
        """
        获取缓存统计信息。

        Returns:
            包含命中率、缓存大小等信息的字典
        """
        total = self._cache_hits + self._cache_misses
        return {
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "capacity": self.cache_size,
            "llm_calls": self._llm_call_count,
        }

    def _should_call_llm(self, object_freq: Dict[str, float], heuristic_result: RoomDistribution) -> bool:
        """
        智能决定是否调用 LLM。

        触发 LLM 调用的条件:
        0. 距离上次 LLM 调用超过最小间隔步数
        1. 启发式置信度低于阈值（结果不够确定）
        2. 物体类型足够多样（需要更细致的语义理解）
        3. 至少包含 3 种不同物体（防止偶然噪声）

        Args:
            object_freq: 物体频率分布
            heuristic_result: 启发式推理结果

        Returns:
            True 表示应该调用 LLM，False 表示启发式足够好
        """
        # 条件0: 调用频率限制 - 避免连续步骤重复调用 LLM
        steps_since_last_call = self._current_step - self._last_llm_call_step
        if steps_since_last_call < self.min_call_interval:
            if self._debug:
                print(f"[LLMRoomReasoner] Skipping LLM (interval={steps_since_last_call}/{self.min_call_interval})")
            return False

        # 条件1: 启发式置信度低于阈值
        max_confidence = max(heuristic_result.values()) if heuristic_result else 0.0
        if max_confidence >= self.confidence_threshold:
            # 启发式已经很确定，不需要 LLM
            return False

        # 条件2: 至少 3 种不同物体
        if len(object_freq) < 3:
            # 物体太少，无法提供有效信息
            return False

        # 条件3: 物体分布多样（熵 > 0.5）
        # 熵高表示物体类型分散，需要更强的语义理解
        entropy = -sum(p * np.log(p + 1e-10) for p in object_freq.values())
        if entropy <= 0.5:
            # 物体太集中或太分散，启发式可能足够
            return False

        return True

    def _build_user_message(self, object_freq: Dict[str, float]) -> str:
        """
        构造用户消息（物体列表）。

        将物体频率格式化为字符串。

        Args:
            object_freq: 物体频率字典

        Returns:
            用户消息字符串
        """
        # 按频率降序排序（高频物体优先），同频时按名称排序以保证确定性
        objects_str = ", ".join(
            f"{obj}({freq:.2f})"
            for obj, freq in sorted(object_freq.items(), key=lambda x: (-x[1], x[0]))
        )
        return self.user_prompt_template.format(objects_str)

    def _query_llm(self, object_freq: Dict[str, float]) -> RoomDistribution:
        """
        调用 LLM 查询房间类型。

        包含：构造消息 → 调用 API → 解析响应

        Args:
            object_freq: 物体频率分布，如 {"chair": 0.6, "table": 0.3}

        Returns:
            房间类型概率分布，如 {"living_room": 0.8, "unknown": 0.2}
        """
        # 构造用户消息
        user_message = self._build_user_message(object_freq)

        # 调用 API
        client = self._get_client()
        if client is None:
            raise RuntimeError("OpenAI client not available")

        with TimingContext("LLMRoomReasoner._query_llm()", "LLM call"):
            data = client.call_openai_api(
                system_prompt=self.system_prompt,
                user_prompt=user_message,
                reasoning_effort=self.reasoning_effort or None,
            )

        # 解析 JSON 响应
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

        # 更新统计
        self._llm_call_count += 1
        return result

    def infer_room_type_from_freq(self, object_freq: Dict[str, float]) -> RoomDistribution:
        """
        从物体频率字典直接推断房间类型（避免重复转换）。

        决策流程：
        1. 检查缓存 → 命中则直接返回
        2. 运行启发式 → 置信度高则返回并缓存
        3. 智能决策 → 判断是否需要 LLM（包含调用间隔检查）
        4. 调用 LLM → 结果缓存并返回
        5. 异常处理 → 回退到启发式

        Args:
            object_freq: 物体频率字典，格式为 {"chair": 0.6, "table": 0.3, ...}

        Returns:
            房间类型概率分布
        """
        # 0. 更新步数计数
        self._current_step += 1

        # 1. 生成缓存 key
        cache_key = self._make_cache_key(object_freq)

        # 2. 检查缓存
        debug_mode = os.environ.get("DEBUG_ROOM_REASONER") == "1"

        if cache_key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(cache_key)
            result = self._cache[cache_key]
            if debug_mode:
                print(f"[LLMRoomReasoner] Cache HIT: {result}")
            return result

        self._cache_misses += 1

        # 3. 启发式预检（需要将频率转换为列表）
        expanded_objects = []
        for obj, freq in object_freq.items():
            count = int(freq * 100)  # 扩大到整数
            expanded_objects.extend([obj] * count)

        heuristic_result = super().infer_room_type(expanded_objects)

        # 如果没有 OpenAI client（未配置 key），永远不要尝试 LLM（避免热循环里抛异常/刷日志）。
        if self._get_client() is None:
            self._update_cache(cache_key, heuristic_result)
            return heuristic_result

        # 4. 智能决定是否调用 LLM（包含调用间隔检查）
        # 注意：_should_call_llm 内部已检查调用间隔
        steps_since_last = self._current_step - self._last_llm_call_step
        within_interval = steps_since_last < self.min_call_interval

        # 如果不需要调用 LLM
        if not self._should_call_llm(object_freq, heuristic_result):
            # 如果在间隔内且有上次 LLM 结果，优先返回上次结果
            if within_interval and self._last_llm_result:
                if debug_mode:
                    print(
                        f"[LLMRoomReasoner] Reusing last LLM result (interval={steps_since_last}/{self.min_call_interval})"
                    )
                return self._last_llm_result

            # 否则返回启发式结果
            if debug_mode:
                max_conf = max(heuristic_result.values()) if heuristic_result else 0.0
                print(f"[LLMRoomReasoner] Using heuristic (conf={max_conf:.2f})")

            self._update_cache(cache_key, heuristic_result)
            return heuristic_result

        # 5. 需要调用 LLM
        if debug_mode:
            objects_str = ", ".join(f"{obj}({freq:.2f})" for obj, freq in object_freq.items())
            print(f"[LLMRoomReasoner] Calling LLM for: {objects_str}")

        try:
            # 调用 LLM
            llm_result = self._query_llm(object_freq)

            # 更新调用记录
            self._last_llm_call_step = self._current_step
            self._last_llm_result = llm_result

            if debug_mode:
                print(f"[LLMRoomReasoner] LLM result: {llm_result}")

            # 缓存结果
            self._update_cache(cache_key, llm_result)

            return llm_result

        except Exception as e:
            # 6. 异常回退
            if debug_mode:
                print(f"[LLMRoomReasoner] LLM query failed: {e}, falling back to heuristic")
            self._update_cache(cache_key, heuristic_result)
            return heuristic_result

    def infer_room_type(self, observed_objects: Iterable[str]) -> RoomDistribution:
        """
        主推理方法：三层智能决策。

        决策流程：
        1. 检查缓存 → 命中则直接返回
        2. 运行启发式 → 置信度高则返回并缓存
        3. 智能决策 → 判断是否需要 LLM
        4. 调用 LLM → 结果缓存并返回
        5. 异常处理 → 回退到启发式

        Args:
            observed_objects: 观察到的物体列表（可重复）

        Returns:
            房间类型概率分布
        """
        # 1. 计算物体频率
        object_freq = self._compute_frequencies(observed_objects)

        # 2. 调用频率版本的方法
        return self.infer_room_type_from_freq(object_freq)
