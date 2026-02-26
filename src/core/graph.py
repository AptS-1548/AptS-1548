"""
人物关系图谱：用 SurrealDB 存储人物身份（别名）和人物间关系。

数据模型：
  person 表 — 人物节点（name, code, aliases, role, qq_id）
  knows 边  — 人物间关系（type, description）
"""

from nonebot import logger
from surrealdb import AsyncSurreal


class CharacterGraph:
    def __init__(self):
        self._db: AsyncSurreal | None = None
        self._persons_cache: list[dict] | None = None  # 缓存全部人物，减少查询
        # 连接参数（重连用）
        self._url: str = ""
        self._namespace: str = ""
        self._database: str = ""
        self._username: str = ""
        self._password: str = ""

    async def connect(
        self,
        url: str = "ws://localhost:8000",
        namespace: str = "apts",
        database: str = "main",
        username: str = "root",
        password: str = "root",
    ):
        """连接 SurrealDB 并选择命名空间和数据库。"""
        self._url = url
        self._namespace = namespace
        self._database = database
        self._username = username
        self._password = password

        self._db = AsyncSurreal(url=url)
        await self._db.connect()
        await self._db.signin({"username": username, "password": password})
        await self._db.use(namespace, database)
        logger.info(f"图谱 | 已连接 SurrealDB {url} ({namespace}/{database})")
        await self._refresh_cache()

    async def close(self):
        if self._db:
            try:
                await self._db.close()
            except Exception:
                pass
            self._db = None
            logger.info("图谱 | SurrealDB 连接已关闭")

    async def _reconnect(self) -> bool:
        """尝试重连 SurrealDB。成功返回 True。"""
        if not self._url:
            return False
        try:
            self._db = AsyncSurreal(url=self._url)
            await self._db.connect()
            await self._db.signin({"username": self._username, "password": self._password})
            await self._db.use(self._namespace, self._database)
            await self._refresh_cache()
            logger.info("图谱 | 重连成功")
            return True
        except Exception as e:
            logger.warning(f"图谱 | 重连失败: {e}")
            self._db = None
            return False

    async def _query(self, sql: str, params: dict | None = None) -> list:
        """执行查询，连接断开时自动重连一次。"""
        for attempt in range(2):
            try:
                if params:
                    result = await self._db.query(sql, params)
                else:
                    result = await self._db.query(sql)
                return result if isinstance(result, list) else []
            except Exception as e:
                if attempt == 0 and self._url:
                    logger.warning(f"图谱 | 查询失败，尝试重连: {e}")
                    if await self._reconnect():
                        continue
                raise

    async def _refresh_cache(self):
        """刷新人物缓存。SDK 1.0.8 query 直接返回 list[dict]。"""
        result = await self._query("SELECT * FROM person")
        self._persons_cache = result if isinstance(result, list) else []
        logger.debug(f"图谱 | 缓存 {len(self._persons_cache)} 个人物")

    # ── 别名解析 ──

    async def resolve_name(self, alias: str) -> dict | None:
        """给定任意称呼，返回匹配的人物信息。"""
        if not alias:
            return None

        # 先查缓存
        if self._persons_cache:
            for p in self._persons_cache:
                if alias in p.get("aliases", []):
                    logger.debug(f"图谱 | 别名解析(缓存) {alias!r} → {p.get('name')}")
                    return p

        # 缓存没命中，查库
        result = await self._query(
            "SELECT * FROM person WHERE aliases CONTAINS $alias",
            {"alias": alias},
        )
        if result:
            logger.debug(f"图谱 | 别名解析(查库) {alias!r} → {result[0].get('name')}")
            return result[0]
        logger.debug(f"图谱 | 别名解析 {alias!r} → 未找到")
        return None

    async def find_in_text(self, text: str) -> list[dict]:
        """在文本中匹配已知人物。返回所有匹配到的人物信息。

        遍历所有人物的所有别名，检查是否出现在文本中。
        只匹配 >= 2 字的别名，避免单字误匹配。
        """
        if not text or not self._persons_cache:
            return []

        matched = []
        seen_ids = set()

        for p in self._persons_cache:
            pid = str(p.get("id", ""))
            if pid in seen_ids:
                continue
            for alias in p.get("aliases", []):
                if len(alias) >= 2 and alias in text:
                    matched.append(p)
                    seen_ids.add(pid)
                    break

        if matched:
            names = [m["name"] for m in matched]
            logger.debug(f"图谱 | 文本匹配 {text[:30]!r} → {names}")
        return matched

    # ── 关系查询 ──

    async def get_connections(self, person_id: str) -> list[dict]:
        """查某人的所有关系（出边 + 入边）。"""
        result = await self._query(
            "SELECT *, out.name AS to_name, in.name AS from_name "
            f"FROM knows WHERE in = person:{person_id} OR out = person:{person_id}",
        )
        conns = result if isinstance(result, list) else []
        logger.debug(f"图谱 | 关系查询 person:{person_id} → {len(conns)} 条")
        return conns

    async def get_relationship(self, person_a: str, person_b: str) -> list[dict]:
        """查两人之间的关系。"""
        result = await self._query(
            f"SELECT * FROM knows WHERE "
            f"(in = person:{person_a} AND out = person:{person_b}) "
            f"OR (in = person:{person_b} AND out = person:{person_a})",
        )
        return result if isinstance(result, list) else []

    # ── 写入 ──

    async def add_alias(self, person_id: str, alias: str) -> bool:
        """给人物添加新别名。返回是否新增成功。"""
        if not alias or len(alias) < 2:
            return False

        # 检查缓存中是否已存在
        if self._persons_cache:
            for p in self._persons_cache:
                if str(p.get("id", "")).endswith(person_id):
                    if alias in p.get("aliases", []):
                        return False
                    break

        await self._query(
            f"UPDATE person:{person_id} SET aliases += $alias",
            {"alias": alias},
        )
        logger.info(f"图谱 | 新别名 person:{person_id} += {alias!r}")
        await self._refresh_cache()
        return True

    async def add_relationship(
        self, from_id: str, to_id: str, rel_type: str, description: str = "",
    ):
        """添加人物间关系。"""
        await self._query(
            f"RELATE person:{from_id}->knows->person:{to_id} "
            f"SET type = $type, description = $desc",
            {"type": rel_type, "desc": description},
        )
        logger.info(f"图谱 | 新关系 {from_id} -{rel_type}-> {to_id}: {description}")

    # ── 工具方法 ──

    def qq_to_person(self, qq_id: str) -> dict | None:
        """通过 QQ ID 查找人物（走缓存）。"""
        if not self._persons_cache:
            return None
        for p in self._persons_cache:
            if p.get("qq_id") == qq_id:
                return p
        return None
