"""
长期记忆存储：将对话向量化存入 Qdrant，供后续检索注入上下文。
"""

import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field

from nonebot import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

COLLECTION = "apts1548"
VECTOR_DIM = 512  # bge-small-zh-v1.5 输出维度


@dataclass
class MemoryEntry:
    user_id: str
    chat_type: str      # "private" | "group"
    chat_id: str        # 私聊=user_id，群聊=group_id
    user_name: str
    message: str        # 用户说的
    response: str       # 48 回的
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5


class Memory:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self._client = QdrantClient(url=qdrant_url)
        self._model: SentenceTransformer | None = None
        self._ensure_collection()
        self._load_model()  # 启动时预加载，避免首次对话延迟

    def _ensure_collection(self):
        names = [c.name for c in self._client.get_collections().collections]
        if COLLECTION not in names:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(f"记忆 | 创建集合 {COLLECTION}")

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("记忆 | 加载 Embedding 模型 bge-small-zh-v1.5 ...")
            self._model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            logger.info("记忆 | 模型加载完成")
        return self._model

    def _embed(self, text: str) -> list[float]:
        return self._load_model().encode(text, normalize_embeddings=True).tolist()

    async def store(self, entry: MemoryEntry) -> None:
        """将一次对话存入向量数据库"""
        doc = f"{entry.message} {entry.response}"
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, self._embed, doc)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "user_id": entry.user_id,
                "chat_type": entry.chat_type,
                "chat_id": entry.chat_id,
                "user_name": entry.user_name,
                "message": entry.message,
                "response": entry.response,
                "timestamp": entry.timestamp,
                "importance": entry.importance,
            },
        )

        await loop.run_in_executor(
            None,
            lambda: self._client.upsert(collection_name=COLLECTION, points=[point]),
        )
        logger.debug(f"记忆 | 存储 chat_id={entry.chat_id} user={entry.user_name}")

    async def search(
        self,
        query: str,
        chat_id: str | None = None,
        user_id: str | None = None,
        limit: int = 5,
        time_weight: float = 0.3,   # 0=纯相似度 1=纯时间
    ) -> list[MemoryEntry]:
        """检索相关历史记忆，余弦相似度 + 时间衰减加权重排。
        chat_id: 群聊按群隔离；user_id: 私聊按人检索（跨群+私聊全捞）。
        """
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, self._embed, query)

        query_filter = None
        if user_id:
            query_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )
        elif chat_id:
            query_filter = Filter(
                must=[FieldCondition(key="chat_id", match=MatchValue(value=chat_id))]
            )

        # 多取一些再重排
        raw_limit = max(limit * 2, 10)
        results = await loop.run_in_executor(
            None,
            lambda: self._client.query_points(
                collection_name=COLLECTION,
                query=vector,
                query_filter=query_filter,
                limit=raw_limit,
                with_payload=True,
            ).points,
        )

        SCORE_THRESHOLD = 0.5  # 余弦相似度低于此值直接丢弃

        now = time.time()
        scored = []
        for r in results:
            if r.score < SCORE_THRESHOLD:
                continue
            p = r.payload
            days_ago = (now - p.get("timestamp", 0.0)) / 86400
            recency = math.exp(-0.1 * days_ago)   # 半衰期约 7 天
            combined = r.score * (1 - time_weight) + recency * time_weight
            scored.append((combined, p))

        scored.sort(key=lambda x: x[0], reverse=True)

        entries = []
        for _, p in scored[:limit]:
            entries.append(MemoryEntry(
                user_id=p["user_id"],
                chat_type=p["chat_type"],
                chat_id=p["chat_id"],
                user_name=p.get("user_name", ""),
                message=p["message"],
                response=p["response"],
                timestamp=p.get("timestamp", 0.0),
                importance=p.get("importance", 0.5),
            ))
        return entries

    async def recent(
        self,
        chat_id: str | None = None,
        user_id: str | None = None,
        limit: int = 25,
    ) -> list[MemoryEntry]:
        """取最近 N 条对话记录（按时间正序），用于重启后重建上下文。"""
        must = []
        if chat_id:
            must.append(FieldCondition(key="chat_id", match=MatchValue(value=chat_id)))
        elif user_id:
            must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
            must.append(FieldCondition(key="chat_type", match=MatchValue(value="private")))

        query_filter = Filter(must=must) if must else None
        loop = asyncio.get_running_loop()

        results, _ = await loop.run_in_executor(
            None,
            lambda: self._client.scroll(
                collection_name=COLLECTION,
                scroll_filter=query_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            ),
        )

        entries = []
        for r in results:
            p = r.payload
            entries.append(MemoryEntry(
                user_id=p["user_id"],
                chat_type=p["chat_type"],
                chat_id=p["chat_id"],
                user_name=p.get("user_name", ""),
                message=p["message"],
                response=p["response"],
                timestamp=p.get("timestamp", 0.0),
                importance=p.get("importance", 0.5),
            ))

        entries.sort(key=lambda x: x.timestamp, reverse=True)
        return list(reversed(entries[:limit]))


def format_memories(memories: list[MemoryEntry]) -> str:
    """将记忆列表格式化为可注入 system prompt 的文本"""
    if not memories:
        return ""

    now = time.time()
    lines = []
    for m in memories:
        days = (now - m.timestamp) / 86400
        if days < 1:
            t = "今天"
        elif days < 2:
            t = "昨天"
        elif days < 7:
            t = f"{int(days)}天前"
        else:
            t = f"{int(days / 7)}周前"
        msg = m.message[:60].replace("\n", " ")
        resp = m.response[:60].replace("\n", " ")
        where = f"群{m.chat_id}" if m.chat_type == "group" else "私聊"
        lines.append(f"- [{t}][{where}] {m.user_name}({m.user_id})说: {msg} → 我: {resp}")

    return "## 相关记忆\n" + "\n".join(lines)
