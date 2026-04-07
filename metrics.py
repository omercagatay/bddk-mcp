"""Simple in-process metrics tracking for BDDK MCP Server."""

import time
from collections import defaultdict


class Metrics:
    """Thread-safe metrics collector for request counts, latency, and errors."""

    def __init__(self) -> None:
        self._request_counts: dict[str, int] = defaultdict(int)
        self._error_counts: dict[str, int] = defaultdict(int)
        self._total_latency: dict[str, float] = defaultdict(float)
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._start_time: float = time.time()

    def record_request(self, tool_name: str, duration_ms: float) -> None:
        """Record a successful tool invocation."""
        self._request_counts[tool_name] += 1
        self._total_latency[tool_name] += duration_ms

    def record_error(self, tool_name: str) -> None:
        """Record a failed tool invocation."""
        self._error_counts[tool_name] += 1

    def record_cache_hit(self) -> None:
        self._cache_hits += 1

    def record_cache_miss(self) -> None:
        self._cache_misses += 1

    def summary(self) -> dict:
        """Return a summary of all metrics."""
        uptime = time.time() - self._start_time
        total_requests = sum(self._request_counts.values())
        total_errors = sum(self._error_counts.values())

        tools: list[dict] = []
        for name in sorted(set(self._request_counts) | set(self._error_counts)):
            count = self._request_counts.get(name, 0)
            errors = self._error_counts.get(name, 0)
            avg_ms = (self._total_latency.get(name, 0) / count) if count else 0
            tools.append(
                {
                    "tool": name,
                    "requests": count,
                    "errors": errors,
                    "avg_latency_ms": round(avg_ms, 1),
                }
            )

        cache_total = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / cache_total * 100) if cache_total else 0

        return {
            "uptime_seconds": round(uptime),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "cache_hit_rate": round(cache_hit_rate, 1),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "tools": tools,
        }


# Global metrics instance
metrics = Metrics()
