from pydantic import BaseModel


class Config(BaseModel):
    owner_id: str | int = ""
    bot_name: str = "48"

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
