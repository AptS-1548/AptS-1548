from pydantic import BaseModel


class Config(BaseModel):
    owner_id: str | int = ""
    bot_name: str | int = "48"

    # 允许发言的群（空列表 = 不在任何群发言，必须显式配置）
    allowed_groups: list[str] = []

    # 速率限制（每用户每分钟）
    rate_per_minute: int = 5

    # 每日 API 调用上限
    daily_limit: int = 500

    # 重复消息缓存 TTL（秒），相同用户发相同内容直接返回缓存
    cache_ttl: int = 30

    # Qdrant 地址
    qdrant_url: str = "http://localhost:6333"

    # 记忆检索数量
    memory_search_limit: int = 4        # 相关记忆条数
    diary_limit: int = 3                # 日记（我的近况）条数
    impression_recent_limit: int = 3    # 近期印象条数
    impression_key_limit: int = 2       # 关键印象条数

    # 日程生成用的日记数量（比对话时多拉一些，让日程更贴合近况）
    schedule_diary_limit: int = 10      # 近期日记条数
    schedule_diary_key_limit: int = 10  # 重要日记条数

    # 主动行为
    proactive_check_sec: int = 7200     # 检查间隔（秒），默认 2 小时
    proactive_cooldown_sec: int = 21600 # 主动消息冷却（秒），默认 6 小时
    proactive_max_daily: int = 2        # 每天最多主动几次

    # 日记自动去重
    diary_dedup_enabled: bool = False   # 是否启用每日自动去重
    diary_dedup_threshold: float = 0.85 # 语义相似度阈值
    diary_dedup_window: int = 15        # 时间窗口（分钟）
