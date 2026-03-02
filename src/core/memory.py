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
import numpy as np
from qdrant_client.models import (
    Direction,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    PointIdsList,
    PointStruct,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
from fastembed import TextEmbedding

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
        self._model: TextEmbedding | None = None
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

    def _load_model(self) -> TextEmbedding:
        if self._model is None:
            logger.info("记忆 | 加载 Embedding 模型 bge-small-zh-v1.5 ...")
            self._model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
            logger.info("记忆 | 模型加载完成")
        return self._model

    def _embed(self, text: str) -> list[float]:
        """Fix 4: 带 LRU cache 的单条向量化。"""
        if text not in self._embed_cache:
            result = list(self._load_model().embed([text]))[0].tolist()
            if len(self._embed_cache) >= EMBED_CACHE_SIZE:
                # 淘汰最旧的 25%
                for k in list(self._embed_cache.keys())[:EMBED_CACHE_SIZE // 4]:
                    del self._embed_cache[k]
            self._embed_cache[text] = result
        return self._embed_cache[text]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Fix 5: 批量向量化，比多次单独 encode 更高效。"""
        return [v.tolist() for v in self._load_model().embed(texts)]

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
        for e in entries:
            logger.debug(f"记忆存储 | type={e.record_type} user={e.user_name} msg={e.message[:50]!r}")

    # ── 检索 ──

    async def search(
        self,
        query: str,
        chat_id: str | None = None,
        user_id: str | None = None,
        limit: int = 5,
        global_fallback: bool = True,
    ) -> list[MemoryEntry]:
        """检索相关历史记忆，三因子加权重排（相似度×0.6 + 时间×0.2 + 重要性×0.2）。

        当按 chat/user 范围搜索无结果时，若 global_fallback=True 则自动 fallback 到全局搜索。
        """
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, self._embed, query)

        scoped = chat_id is not None or user_id is not None
        entries = await self._search_with_vector(vector, chat_id, user_id, limit)

        # 全局 fallback：按 chat/user 搜不到时，去掉范围限制再搜一次
        if not entries and scoped and global_fallback:
            entries = await self._search_with_vector(vector, None, None, limit)
            if entries:
                logger.debug(f"记忆检索 | 范围内无结果，全局 fallback 命中 {len(entries)} 条")

        return entries

    async def _search_with_vector(
        self,
        vector: list[float],
        chat_id: str | None,
        user_id: str | None,
        limit: int,
    ) -> list[MemoryEntry]:
        """用已有向量执行一次检索 + 重排。"""
        loop = asyncio.get_running_loop()

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

    async def get_diary(
        self,
        query: str = "",
        recent_limit: int = 3,
        search_limit: int = 5,
        key_limit: int = 5,
    ) -> list[MemoryEntry]:
        """取日记：近期 N 条 + 相关日记（有 query 走向量搜索，无 query 走 importance≥0.7）。
        合并去重后按时间正序返回。
        """
        base_filter = [
            FieldCondition(key="record_type", match=MatchValue(value="diary")),
            FieldCondition(key="user_id", match=MatchValue(value="48")),
        ]
        loop = asyncio.get_running_loop()

        if query:
            # 有 query：近期 3 条 + 向量搜索 5 条（相关历史）
            vector = await loop.run_in_executor(None, self._embed, query)
            (recent_results, _), search_results = await asyncio.gather(
                loop.run_in_executor(None, lambda: self._client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=Filter(must=base_filter),
                    limit=recent_limit,
                    with_payload=True,
                    with_vectors=False,
                    order_by=OrderBy(key="timestamp", direction=Direction.DESC),
                )),
                loop.run_in_executor(None, lambda: self._client.query_points(
                    collection_name=COLLECTION,
                    query=vector,
                    query_filter=Filter(must=base_filter),
                    limit=search_limit,
                    with_payload=True,
                ).points),
            )
            secondary = [r for r in search_results if r.score >= 0.4]
        else:
            # 无 query：近期 3 条 + importance≥0.7 的关键日记
            key_filter = base_filter + [FieldCondition(key="importance", range=Range(gte=0.7))]
            (recent_results, _), (secondary_raw, _) = await asyncio.gather(
                loop.run_in_executor(None, lambda: self._client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=Filter(must=base_filter),
                    limit=recent_limit,
                    with_payload=True,
                    with_vectors=False,
                    order_by=OrderBy(key="timestamp", direction=Direction.DESC),
                )),
                loop.run_in_executor(None, lambda: self._client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=Filter(must=key_filter),
                    limit=key_limit,
                    with_payload=True,
                    with_vectors=False,
                    order_by=OrderBy(key="importance", direction=Direction.DESC),
                )),
            )
            secondary = secondary_raw

        def _to_entry(r) -> MemoryEntry:
            p = r.payload
            return MemoryEntry(
                user_id="48",
                chat_type=p.get("chat_type", "private"),
                chat_id="48",
                user_name="48",
                message=p["message"],
                response="",
                timestamp=p.get("timestamp", 0.0),
                importance=p.get("importance", 0.6),
                record_type="diary",
            )

        recent = [_to_entry(r) for r in recent_results]
        other = [_to_entry(r) for r in secondary]

        # 合并去重：recent 优先（最新状态），other 补充历史相关
        seen: set[float] = set()
        merged: list[MemoryEntry] = []
        for e in recent:
            if e.timestamp not in seen:
                seen.add(e.timestamp)
                merged.append(e)
        for e in other:
            if e.timestamp not in seen:
                seen.add(e.timestamp)
                merged.append(e)

        merged.sort(key=lambda x: x.timestamp)
        return merged

    async def is_diary_duplicate(self, text: str, threshold: float = 0.85, hours: int = 24) -> bool:
        """检查最近 N 小时内是否已存在语义相似的日记。"""
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, self._embed, text)
        cutoff = time.time() - hours * 3600

        results = await loop.run_in_executor(None, lambda: self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=Filter(must=[
                FieldCondition(key="record_type", match=MatchValue(value="diary")),
                FieldCondition(key="user_id", match=MatchValue(value="48")),
                FieldCondition(key="timestamp", range=Range(gte=cutoff)),
            ]),
            limit=1,
            with_payload=True,
        ).points)

        if results and results[0].score >= threshold:
            dup_msg = results[0].payload.get("message", "")
            logger.debug(f"日记去重 | 相似度 {results[0].score:.3f} >= {threshold} | 已有: {dup_msg!r}")
            return True
        return False

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

    async def dedup_diary(self, threshold: float = 0.85, window_min: float = 15.0) -> int:
        """日记语义去重：删除时间接近且语义相似的重复日记。

        条件同时满足才算重复：cosine >= threshold 且 时间间隔 <= window_min 分钟。
        返回删除条数。
        """
        loop = asyncio.get_running_loop()

        # 拉全部日记（时间正序）
        diary_filter = Filter(must=[
            FieldCondition(key="record_type", match=MatchValue(value="diary")),
            FieldCondition(key="user_id", match=MatchValue(value="48")),
        ])
        all_points = []
        offset = None
        while True:
            results, next_offset = await loop.run_in_executor(
                None,
                lambda o=offset: self._client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=diary_filter,
                    limit=100,
                    with_payload=True,
                    with_vectors=True,
                    order_by=OrderBy(key="timestamp", direction=Direction.ASC),
                    offset=o,
                ),
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset

        if len(all_points) < 2:
            return 0

        # 提取并归一化向量
        vectors = np.array([np.array(p.vector) for p in all_points])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        window_sec = window_min * 60
        ids_to_delete = []
        deleted = set()

        for i in range(1, len(all_points)):
            if i in deleted:
                continue
            ts_i = all_points[i].payload.get("timestamp", 0)

            for j in range(i - 1, -1, -1):
                ts_j = all_points[j].payload.get("timestamp", 0)
                if ts_i - ts_j > window_sec:
                    break
                if j in deleted:
                    continue

                sim = float(np.dot(vectors[i], vectors[j]))
                if sim >= threshold:
                    deleted.add(i)
                    ids_to_delete.append(all_points[i].id)
                    break

        if not ids_to_delete:
            logger.info(f"日记去重 | {len(all_points)} 条日记，无重复")
            return 0

        # 分批删除
        for batch_start in range(0, len(ids_to_delete), 100):
            batch = ids_to_delete[batch_start:batch_start + 100]
            await loop.run_in_executor(
                None,
                lambda b=batch: self._client.delete(
                    collection_name=COLLECTION,
                    points_selector=PointIdsList(points=b),
                ),
            )

        logger.info(
            f"日记去重 | {len(all_points)} 条日记，删除 {len(ids_to_delete)} 条重复，"
            f"剩余 {len(all_points) - len(ids_to_delete)} 条"
        )
        return len(ids_to_delete)

    async def consolidate_diary(self, min_entries: int = 8) -> int:
        """每日日记压缩：把今天的流水账合并成 3~6 段回忆。
        只在条目数 >= min_entries 时执行。
        返回删除的旧条目数。
        """
        from core.diary import consolidate_diary

        loop = asyncio.get_running_loop()
        cutoff = time.time() - 24 * 3600  # 最近 24h

        diary_filter = Filter(must=[
            FieldCondition(key="record_type", match=MatchValue(value="diary")),
            FieldCondition(key="user_id", match=MatchValue(value="48")),
            FieldCondition(key="timestamp", range=Range(gte=cutoff)),
        ])

        # 拉最近 24h 的日记
        all_points = []
        offset = None
        while True:
            results, next_offset = await loop.run_in_executor(
                None,
                lambda o=offset: self._client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=diary_filter,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    order_by=OrderBy(key="timestamp", direction=Direction.ASC),
                    offset=o,
                ),
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset

        if len(all_points) < min_entries:
            logger.debug(f"日记压缩 | 条目不足 {len(all_points)}/{min_entries}，跳过")
            return 0

        # 提取文本
        raw_entries = [p.payload.get("message", "") for p in all_points if p.payload.get("message")]
        if len(raw_entries) < min_entries:
            return 0

        # LLM 压缩
        consolidated = await consolidate_diary(raw_entries)
        if not consolidated:
            logger.warning("日记压缩 | LLM 返回空，保留原始日记")
            return 0

        # 删除旧条目
        old_ids = [p.id for p in all_points]
        for batch_start in range(0, len(old_ids), 100):
            batch = old_ids[batch_start:batch_start + 100]
            await loop.run_in_executor(
                None,
                lambda b=batch: self._client.delete(
                    collection_name=COLLECTION,
                    points_selector=PointIdsList(points=b),
                ),
            )

        # 写入压缩后的条目
        now = time.time()
        new_entries = [
            MemoryEntry(
                user_id="48",
                chat_type="private",
                chat_id="48",
                user_name="48",
                message=text,
                response="",
                timestamp=now,
                importance=0.7,  # 压缩后的回忆 importance 偏高
                record_type="diary",
            )
            for text in consolidated
        ]
        await self.store_batch(new_entries)

        logger.info(f"日记压缩 | {len(old_ids)} 条 → {len(consolidated)} 条")
        return len(old_ids)

    # ── 故事检索 / 存储 ──

    async def search_stories(self, query: str, limit: int = 2) -> list[MemoryEntry]:
        """检索相关的故事片段（背景故事 + 自创叙述）。
        纯靠 similarity + importance 双因子排序，不做时间 rerank。
        """
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, self._embed, query)

        query_filter = Filter(must=[
            FieldCondition(key="record_type", match=MatchValue(value="story")),
        ])

        raw_limit = max(limit * 2, 6)
        try:
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
        except Exception as e:
            logger.warning(f"故事检索失败 | {e}")
            return []

        SCORE_THRESHOLD = 0.55

        scored = []
        for r in results:
            if r.score < SCORE_THRESHOLD:
                continue
            p = r.payload
            importance = p.get("importance", 0.8)
            combined = r.score * 0.7 + importance * 0.3
            scored.append((combined, p))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            MemoryEntry(
                user_id=p.get("user_id", "48"),
                chat_type=p.get("chat_type", "story"),
                chat_id=p.get("chat_id", ""),
                user_name=p.get("user_name", "48"),
                message=p.get("message", ""),
                response=p.get("response", ""),
                timestamp=p.get("timestamp", 0.0),
                importance=p.get("importance", 0.8),
                record_type="story",
            )
            for _, p in scored[:limit]
        ]

    async def is_story_duplicate(self, text: str, threshold: float = 0.80, hours: int = 24) -> bool:
        """检查最近 N 小时内是否已存在语义相似的自创故事。"""
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, self._embed, text)
        cutoff = time.time() - hours * 3600

        try:
            results = await loop.run_in_executor(None, lambda: self._client.query_points(
                collection_name=COLLECTION,
                query=vector,
                query_filter=Filter(must=[
                    FieldCondition(key="record_type", match=MatchValue(value="story")),
                    FieldCondition(key="chat_type", match=MatchValue(value="narrative")),
                    FieldCondition(key="timestamp", range=Range(gte=cutoff)),
                ]),
                limit=1,
                with_payload=True,
            ).points)

            if results and results[0].score >= threshold:
                dup_msg = results[0].payload.get("message", "")[:50]
                logger.debug(f"故事去重 | 相似度 {results[0].score:.3f} >= {threshold} | 已有: {dup_msg!r}")
                return True
        except Exception as e:
            logger.warning(f"故事去重检查失败 | {e}")
        return False

    async def store_story(self, text: str, importance: float = 0.8) -> bool:
        """存储 48 自创的故事叙述。"""
        entry = MemoryEntry(
            user_id="48",
            chat_type="narrative",
            chat_id="48",
            user_name="48",
            message=text,
            response="",
            importance=importance,
            record_type="story",
        )
        try:
            await self.store(entry)
            logger.info(f"故事存储 | {len(text)} 字 | importance={importance}")
            return True
        except Exception as e:
            logger.warning(f"故事存储失败 | {e}")
            return False


# ── 格式化工具 ──

def time_label(timestamp: float) -> str:
    """时间戳 → 可读标签，按本地自然日计算。今天/昨天带 HH:MM。
    公开函数，供 prompt.py 等模块复用。
    """
    from datetime import datetime
    d = datetime.fromtimestamp(timestamp)
    delta = (datetime.now().date() - d.date()).days
    if delta == 0:
        return f"今天 {d.strftime('%H:%M')}"
    if delta == 1:
        return f"昨天 {d.strftime('%H:%M')}"
    if delta < 7:
        return f"{delta}天前"
    if delta < 30:
        return f"{d.month}月{d.day}日"
    return f"{delta // 7}周前"


def format_memories(memories: list[MemoryEntry]) -> str:
    """将记忆列表格式化为可注入 system prompt 的文本"""
    if not memories:
        return ""
    lines = []
    for m in memories:
        t = time_label(m.timestamp)
        msg = m.message[:60].replace("\n", " ")
        resp = m.response[:60].replace("\n", " ")
        where = f"群{m.chat_id}" if m.chat_type == "group" else "私聊"
        name = m.user_name or f"用户{m.user_id[-4:]}"
        lines.append(f"- [{t}][{where}] {name}说: {msg} → 我: {resp}")
    return "## 相关记忆\n" + "\n".join(lines)
