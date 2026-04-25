"""In-memory token-bucket rate limiter (per caller_id).

* Refill rate: ``rate_limit_per_min / 60`` tokens per second.
* Capacity: ``rate_limit_burst`` (max bursty tokens).
* Each request consumes one token. When empty, request is rejected with 429.

Process-local only — single-process deploys. For multi-replica setups, swap
this for a Redis-backed limiter (out of scope for now).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status

from ..config import get_settings
from .auth import require_api_key


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    def __init__(self, per_min: int, burst: int) -> None:
        self.per_min = max(1, per_min)
        self.burst = max(1, burst)
        self.refill_per_sec = self.per_min / 60.0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _capacity(self) -> float:
        # Allow short bursts up to `burst`, otherwise steady-state is per_min.
        return float(max(self.burst, self.per_min))

    def acquire(self, key: str) -> tuple[bool, float]:
        """Try to consume one token. Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        cap = self._capacity()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=cap, last_refill=now)
                self._buckets[key] = bucket

            # Refill based on elapsed time
            elapsed = now - bucket.last_refill
            bucket.tokens = min(cap, bucket.tokens + elapsed * self.refill_per_sec)
            bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            # Compute how long until 1 token is available
            deficit = 1.0 - bucket.tokens
            retry_after = deficit / self.refill_per_sec if self.refill_per_sec > 0 else 60.0
            return False, math.ceil(retry_after)


_limiter: TokenBucketLimiter | None = None


def _get_limiter() -> TokenBucketLimiter:
    global _limiter
    if _limiter is None:
        s = get_settings()
        _limiter = TokenBucketLimiter(s.rate_limit_per_min, s.rate_limit_burst)
    return _limiter


async def rate_limit(caller_id: str = Depends(require_api_key)) -> str:
    """FastAPI dependency: enforces rate limit, returns the caller_id through."""
    allowed, retry_after = _get_limiter().acquire(caller_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(int(retry_after))},
        )
    return caller_id
