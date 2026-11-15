# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Optional execution timing controlled by DEBUG_TIMING."""

import os
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from icecream import ic

F = TypeVar("F", bound=Callable[..., Any])

_TIMING_ENABLED: Optional[bool] = None


def is_timing_enabled() -> bool:
    """Read and cache whether timing output is enabled."""
    global _TIMING_ENABLED
    if _TIMING_ENABLED is None:
        _TIMING_ENABLED = os.environ.get("DEBUG_TIMING", "0") == "1"
    return _TIMING_ENABLED


_timing_stats: Dict[str, Dict[str, float]] = {}


def get_timing_stats() -> Dict[str, Dict[str, float]]:
    """Return accumulated timing statistics."""
    return _timing_stats.copy()


def reset_timing_stats() -> None:
    """Clear accumulated timing statistics."""
    global _timing_stats
    _timing_stats = {}


def print_timing_summary() -> None:
    """Print accumulated timing statistics."""
    if not _timing_stats:
        ic("No timing data collected")
        return

    lines = [
        "\n" + "=" * 60,
        "[TIMING SUMMARY]",
        "-" * 60,
        f"{'Function':<40} {'Calls':>6} {'Total(ms)':>10} {'Avg(ms)':>10}",
        "-" * 60,
    ]

    for name, stats in sorted(_timing_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        avg = stats["total"] / stats["count"] if stats["count"] > 0 else 0
        lines.append(f"{name:<40} {stats['count']:>6} {stats['total']:>10.1f} {avg:>10.2f}")

    lines.append("=" * 60 + "\n")
    ic("\n".join(lines))


def _record_timing(name: str, elapsed_ms: float, extra_info: str = "") -> None:
    """Record and print one timing measurement."""
    if name not in _timing_stats:
        _timing_stats[name] = {"count": 0, "total": 0.0, "min": float("inf"), "max": 0.0}

    stats = _timing_stats[name]
    stats["count"] += 1
    stats["total"] += elapsed_ms
    stats["min"] = min(stats["min"], elapsed_ms)
    stats["max"] = max(stats["max"], elapsed_ms)

    ms = f"{elapsed_ms:.2f}ms"
    if extra_info:
        ic(name, ms, extra_info)
    else:
        ic(name, ms)


def timed(func: F) -> F:
    """Measure function execution time when timing is enabled."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_timing_enabled():
            return func(*args, **kwargs)

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if args and hasattr(args[0], "__class__"):
                class_name = args[0].__class__.__name__
                name = f"{class_name}.{func.__name__}()"
            else:
                name = f"{func.__name__}()"
            _record_timing(name, elapsed_ms)

    return wrapper  # type: ignore


class TimingContext:
    """Measure a code block when timing is enabled."""

    def __init__(self, name: str, extra_info: str = ""):
        self.name = name
        self.extra_info = extra_info
        self.start_time = 0.0

    def __enter__(self) -> "TimingContext":
        if is_timing_enabled():
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if is_timing_enabled() and self.start_time > 0:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            _record_timing(self.name, elapsed_ms, self.extra_info)
