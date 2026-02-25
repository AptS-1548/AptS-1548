import hashlib
import time
from collections import OrderedDict, defaultdict

from nonebot import logger


class Guard:
    """速率限制 + 日预算 + 重复消息缓存"""

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
        self._day_start = time.time()
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()

    # ── 速率 ──

    def check_rate(self, user_id: str, is_owner: bool = False) -> tuple[bool, str]:
        now = time.time()

        # 日预算重置
        if now - self._day_start > 86400:
            self._daily_count = 0
            self._day_start = now

        if self._daily_count >= self._daily_max:
            logger.warning(f"日预算耗尽 ({self._daily_count}/{self._daily_max})")
            return False, "今天说太多了，歇了。"

        # owner 放宽 3 倍速率，其他人正常
        rpm = self._rpm * 3 if is_owner else self._rpm

        hits = [t for t in self._user_hits[user_id] if now - t < 60]
        self._user_hits[user_id] = hits

        if len(hits) >= rpm:
            if is_owner:
                return False, "你慢点，我处理不过来。"
            return False, "你说得太快了，歇会。"

        return True, ""

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

    def record(self, user_id: str, message: str, response: str):
        self._user_hits[user_id].append(time.time())
        self._daily_count += 1

        key = self._make_key(user_id, message)
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = (response, time.time())
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @property
    def daily_usage(self) -> tuple[int, int]:
        return self._daily_count, self._daily_max

    @staticmethod
    def _make_key(user_id: str, message: str) -> str:
        return hashlib.md5(f"{user_id}:{message}".encode()).hexdigest()
