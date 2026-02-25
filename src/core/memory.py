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
    Direction,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    PointStruct,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

COLLECTION = "apts1548"
VECTOR_DIM = 512  # bge-small-zh-v1.5 输出维度
EMBED_CACHE_SIZE = 512  # 嵌入 LRU 缓存条数


@dataclass
class MemoryEntry:
    user_id: str
    chat_type: str      # "private" | "group"
    chat_id: str        # 私聊=user_id，群聊=group_id
    user_name: str
    message: str        # 用户说的（impression 类型时为印象文本）
    response: str       # 48 回的（impression 类型时为空）
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    record_type: str = "dialog"  # "dialog" | "impression" | "diary"


class Memory:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self._client = QdrantClient(url=qdrant_url)
        self._model: SentenceTransformer | None = None
        self._embed_cache: dict[str, list[float]] = {}  # Fix 4: 嵌入缓存
        self._ensure_collection()
        self._load_model()  # 启动时预加载，避免首次对话延迟

    # ── 集合初始化 ──

    def _ensure_collection(self):
        names = [c.name for c in self._client.get_collections().collections]
        if COLLECTION not in names:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(f"记忆 | 创建集合 {COLLECTION}")
        self._ensure_payload_indices()
        self._ensure_quantization()

    def _ensure_payload_indices(self):
        """为高频过滤字段建 payload 索引；timestamp/importance 也支持 Range 查询。"""
        for fname, schema in [
            ("record_type", "keyword"),
            ("user_id", "keyword"),
            ("chat_id", "keyword"),
            ("chat_type", "keyword"),
            ("timestamp", "float"),
            ("importance", "float"),
        ]:
            try:
                self._client.create_payload_index(
                    collection_name=COLLECTION,
                    field_name=fname,
                    field_schema=schema,
                )
            except Exception:
                pass  # 已存在则忽略
        logger.info("记忆 | payload 索引就绪")

    def _ensure_quantization(self):
        """Fix 3: 配置 INT8 scalar quantization，降低向量内存占用约 4 倍。"""
        try:
            self._client.update_collection(
                collection_name=COLLECTION,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            )
            logger.info("记忆 | INT8 量化就绪")
        except Exception as e:
            logger.warning(f"记忆 | 量化配置跳过（版本不支持或已配置）: {e}")

    # ── Embedding ──

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("记忆 | 加载 Embedding 模型 bge-small-zh-v1.5 ...")
            self._model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            logger.info("记忆 | 模型加载完成")
        return self._model

    def _embed(self, text: str) -> list[float]:
        """Fix 4: 带 LRU cache 的单条向量化。"""
        if text not in self._embed_cache:
            result = self._load_model().encode(text, normalize_embeddings=True).tolist()
            if len(self._embed_cache) >= EMBED_CACHE_SIZE:
                # 淘汰最旧的 25%
                for k in list(self._embed_cache.keys())[:EMBED_CACHE_SIZE // 4]:
                    del self._embed_cache[k]
            self._embed_cache[text] = result
        return self._embed_cache[text]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Fix 5: 批量向量化，比多次单独 encode 更高效。"""
        return self._load_model().encode(texts, normalize_embeddings=True).tolist()

    # ── 存储 ──

    async def store(self, entry: MemoryEntry) -> None:
        """单条存储（向后兼容），实际调用 store_batch。"""
        await self.store_batch([entry])

    async def store_batch(self, entries: list[MemoryEntry]) -> None:
        """Fix 2+5: 批量向量化 + 单次 upsert，减少 IO 次数。"""
        if not entries:
            return

        loop = asyncio.get_running_loop()
        docs = [f"{e.message} {e.response}" for e in entries]

        # 单条走缓存路径，多条走批量 encode 路径
        if len(docs) == 1:
            vectors = [await loop.run_in_executor(None, self._embed, docs[0])]
        else:
            vectors = await loop.run_in_executor(None, self._embed_batch, docs)
            # 顺手更新缓存
            for doc, vec in zip(docs, vectors):
                self._embed_cache[doc] = vec

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "user_id": e.user_id,
                    "chat_type": e.chat_type,
                    "chat_id": e.chat_id,
                    "user_name": e.user_name,
                    "message": e.message,
                    "response": e.response,
                    "timestamp": e.timestamp,
                    "importance": e.importance,
                    "record_type": e.record_type,
                },
            )
            for e, vec in zip(entries, vectors)
        ]

        await loop.run_in_executor(
            None,
            lambda: self._client.upsert(collection_name=COLLECTION, points=points),
        )
        logger.debug(f"记忆 | 批量存储 {len(entries)} 条")

    # ── 检索 ──

    async def search(
        self,
        query: str,
        chat_id: str | None = None,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """检索相关历史记忆，三因子加权重排（相似度×0.6 + 时间×0.2 + 重要性×0.2）。"""
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, self._embed, query)

        must = [FieldCondition(key="record_type", match=MatchValue(value="dialog"))]
        if user_id:
            must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        elif chat_id:
            must.append(FieldCondition(key="chat_id", match=MatchValue(value=chat_id)))
        query_filter = Filter(must=must)

        raw_limit = max(limit * 3, 15)
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

        SCORE_THRESHOLD = 0.5
        IMPORTANCE_MIN = 0.2

        now = time.time()
        scored = []
        for r in results:
            if r.score < SCORE_THRESHOLD:
                continue
            p = r.payload
            importance = p.get("importance", 0.5)
            if importance < IMPORTANCE_MIN:
                continue
            days_ago = (now - p.get("timestamp", 0.0)) / 86400
            recency = math.exp(-0.1 * days_ago)
            combined = r.score * 0.6 + recency * 0.2 + importance * 0.2
            scored.append((combined, p))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            MemoryEntry(
                user_id=p["user_id"],
                chat_type=p["chat_type"],
                chat_id=p["chat_id"],
                user_name=p.get("user_name", ""),
                message=p["message"],
                response=p["response"],
                timestamp=p.get("timestamp", 0.0),
                importance=p.get("importance", 0.5),
            )
            for _, p in scored[:limit]
        ]

    async def recent(
        self,
        chat_id: str | None = None,
        user_id: str | None = None,
        limit: int = 25,
    ) -> list[MemoryEntry]:
        """取最近 N 条对话记录（时间正序），用 order_by 避免全表扫描。"""
        must = []
        if chat_id:
            must.append(FieldCondition(key="chat_id", match=MatchValue(value=chat_id)))
        elif user_id:
            must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
            must.append(FieldCondition(key="chat_type", match=MatchValue(value="private")))
        must.append(FieldCondition(key="record_type", match=MatchValue(value="dialog")))

        loop = asyncio.get_running_loop()
        results, _ = await loop.run_in_executor(
            None,
            lambda: self._client.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(must=must),
                limit=limit,
                with_payload=True,
                with_vectors=False,
                order_by=OrderBy(key="timestamp", direction=Direction.DESC),
            ),
        )

        entries = [
            MemoryEntry(
                user_id=p["user_id"],
                chat_type=p["chat_type"],
                chat_id=p["chat_id"],
                user_name=p.get("user_name", ""),
                message=p["message"],
                response=p["response"],
                timestamp=p.get("timestamp", 0.0),
                importance=p.get("importance", 0.5),
            )
            for r in results
            for p in [r.payload]
        ]
        entries.reverse()  # DESC → 时间正序
        return entries

    async def get_impressions(
        self,
        chat_id: str | None = None,
        user_id: str | None = None,
        recent_limit: int = 3,
        key_limit: int = 5,
    ) -> list[MemoryEntry]:
        """Fix 1: 两个并行 Qdrant 查询取分层印象，替代 fetch-100-sort-in-Python。
        短期：最近 N 条（timestamp DESC）。
        长期：importance >= 0.7 的关键印象（importance DESC）。
        """
        base_must = [FieldCondition(key="record_type", match=MatchValue(value="impression"))]
        if user_id:
            base_must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        elif chat_id:
            base_must.append(FieldCondition(key="chat_id", match=MatchValue(value=chat_id)))

        key_must = base_must + [
            FieldCondition(key="importance", range=Range(gte=0.7))
        ]

        loop = asyncio.get_running_loop()

        (recent_results, _), (key_results, _) = await asyncio.gather(
            loop.run_in_executor(None, lambda: self._client.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(must=base_must),
                limit=recent_limit,
                with_payload=True,
                with_vectors=False,
                order_by=OrderBy(key="timestamp", direction=Direction.DESC),
            )),
            loop.run_in_executor(None, lambda: self._client.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(must=key_must),
                limit=key_limit,
                with_payload=True,
                with_vectors=False,
                order_by=OrderBy(key="importance", direction=Direction.DESC),
            )),
        )

        def _to_entry(p: dict) -> MemoryEntry:
            return MemoryEntry(
                user_id=p["user_id"],
                chat_type=p["chat_type"],
                chat_id=p["chat_id"],
                user_name=p.get("user_name", ""),
                message=p["message"],
                response=p.get("response", ""),
                timestamp=p.get("timestamp", 0.0),
                importance=p.get("importance", 0.5),
                record_type="impression",
            )

        recent = [_to_entry(r.payload) for r in recent_results]
        key = [_to_entry(r.payload) for r in key_results]

        # 合并去重：关键在前，最近在后
        seen: set[float] = set()
        merged: list[MemoryEntry] = []
        for e in key:
            if e.timestamp not in seen:
                seen.add(e.timestamp)
                merged.append(e)
        for e in recent:
            if e.timestamp not in seen:
                seen.add(e.timestamp)
                merged.append(e)

        return merged

    async def cleanup_old_entries(self, days: int = 90, importance_threshold: float = 0.2) -> None:
        """Fix 7: 删除超过 N 天且 importance 低于阈值的 dialog，防止数据库无限增长。
        impression / diary / 高重要度记录不受影响。
        """
        cutoff = time.time() - days * 86400
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._client.delete(
                    collection_name=COLLECTION,
                    points_selector=Filter(must=[
                        FieldCondition(key="timestamp", range=Range(lt=cutoff)),
                        FieldCondition(key="importance", range=Range(lt=importance_threshold)),
                        FieldCondition(key="record_type", match=MatchValue(value="dialog")),
                    ]),
                ),
            )
            logger.info(f"记忆 | TTL 清理：删除 {days} 天前 importance<{importance_threshold} 的 dialog")
        except Exception as e:
            logger.warning(f"记忆 | TTL 清理失败: {e}")


# ── 格式化工具 ──

def _time_label(timestamp: float) -> str:
    """时间戳 → 可读标签，按本地自然日计算"""
    from datetime import datetime
    delta = (datetime.now().date() - datetime.fromtimestamp(timestamp).date()).days
    if delta == 0:
        return "今天"
    if delta == 1:
        return "昨天"
    if delta < 7:
        return f"{delta}天前"
    return f"{delta // 7}周前"


def format_memories(memories: list[MemoryEntry]) -> str:
    """将记忆列表格式化为可注入 system prompt 的文本"""
    if not memories:
        return ""
    lines = []
    for m in memories:
        t = _time_label(m.timestamp)
        msg = m.message[:60].replace("\n", " ")
        resp = m.response[:60].replace("\n", " ")
        where = f"群{m.chat_id}" if m.chat_type == "group" else "私聊"
        name = m.user_name or f"用户{m.user_id[-4:]}"
        lines.append(f"- [{t}][{where}] {name}说: {msg} → 我: {resp}")
    return "## 相关记忆\n" + "\n".join(lines)
