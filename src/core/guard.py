import datetime
import hashlib
import time
from collections import OrderedDict, defaultdict

from nonebot import logger


class Guard:
    """速率限制 + 日预算 + 重复消息缓存 + API 调用计数"""

    def __init__(
        self,
        rate_per_minute: int = 5,
        daily_limit: int = 500,
        cache_ttl: int = 30,
        cache_size: int = 200,
    ):
        self._rpm = rate_per_minute
        self._daily_max = daily_limit
        self._cache_ttl = cache_ttl
        self._cache_size = cache_size

        self._user_hits: dict[str, list[float]] = defaultdict(list)
        self._daily_count = 0
        self._api_calls_today = 0  # 实际 LLM API 调用次数
        self._today = datetime.date.today()
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()

    # ── 速率 ──

    def record_api_call(self):
        """记录一次实际的 LLM API 调用。供 llm.py 调用。"""
        self._ensure_date()
        self._api_calls_today += 1

    def _ensure_date(self):
        """日预算重置（按自然日）。"""
        today = datetime.date.today()
        if today != self._today:
            self._daily_count = 0
            self._api_calls_today = 0
            self._today = today

    def check_rate(self, user_id: str, is_owner: bool = False) -> bool:
        """检查速率限制。超限返回 False（静默丢弃，不回复）。"""
        now = time.time()
        self._ensure_date()

        if self._daily_count >= self._daily_max:
            logger.warning(f"日预算耗尽 ({self._daily_count}/{self._daily_max})")
            return False

        # owner 放宽 3 倍速率，其他人正常
        rpm = self._rpm * 3 if is_owner else self._rpm

        hits = [t for t in self._user_hits[user_id] if now - t < 60]
        self._user_hits[user_id] = hits

        if len(hits) >= rpm:
            return False

        return True

    # ── 缓存 ──

    def get_cached(self, user_id: str, message: str) -> str | None:
        key = self._make_key(user_id, message)
        if key not in self._cache:
            return None
        resp, ts = self._cache[key]
        if time.time() - ts > self._cache_ttl:
            del self._cache[key]
            return None
        return resp

    def _cleanup_hits(self):
        """清理所有用户已过期的 hit 记录，防止长期运行内存增长。"""
        now = time.time()
        stale = [uid for uid, hits in self._user_hits.items()
                 if not any(now - t < 60 for t in hits)]
        for uid in stale:
            del self._user_hits[uid]
        if stale:
            logger.debug(f"速率 | 清理 {len(stale)} 个闲置用户 hit 记录")

    def record(self, user_id: str, message: str, response: str):
        self._user_hits[user_id].append(time.time())
        self._daily_count += 1
        if self._daily_count % 100 == 0:
            self._cleanup_hits()

        key = self._make_key(user_id, message)
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = (response, time.time())
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @property
    def daily_usage(self) -> tuple[int, int]:
        return self._daily_count, self._daily_max

    @property
    def api_calls_today(self) -> int:
        """今天实际的 LLM API 调用次数。"""
        self._ensure_date()
        return self._api_calls_today

    @staticmethod
    def _make_key(user_id: str, message: str) -> str:
        return hashlib.md5(f"{user_id}:{message}".encode()).hexdigest()
