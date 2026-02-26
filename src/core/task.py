"""
48 的待办事项系统：从对话中提取待办，注入 prompt，追踪完成状态。
存储在 SurrealDB task 表，复用 graph 的连接。
支持主动执行（5 分钟扫描 → 发消息给 target）和完成回报。
"""

from nonebot import logger


EXPIRE_HOURS = 48  # 待办自动过期时间


class TaskManager:
    def __init__(self, graph):
        self._graph = graph
        self._table_ready = False

    @property
    def _db(self):
        return self._graph._db

    async def _ensure_table(self):
        """确保 task 表存在。"""
        if self._table_ready or not self._db:
            return
        try:
            await self._db.query("DEFINE TABLE IF NOT EXISTS task SCHEMAFULL")
            await self._db.query("DEFINE FIELD IF NOT EXISTS content ON task TYPE string")
            await self._db.query("DEFINE FIELD IF NOT EXISTS source_user ON task TYPE string")
            await self._db.query("DEFINE FIELD IF NOT EXISTS source_qq ON task TYPE string DEFAULT ''")
            await self._db.query("DEFINE FIELD IF NOT EXISTS target_qq ON task TYPE option<string>")
            await self._db.query("DEFINE FIELD IF NOT EXISTS status ON task TYPE string DEFAULT 'pending'")
            await self._db.query("DEFINE FIELD IF NOT EXISTS created_at ON task TYPE datetime DEFAULT time::now()")
            await self._db.query("DEFINE FIELD IF NOT EXISTS sent_at ON task TYPE option<datetime>")
            await self._db.query("DEFINE FIELD IF NOT EXISTS done_at ON task TYPE option<datetime>")
            await self._db.query("DEFINE FIELD IF NOT EXISTS done_result ON task TYPE option<string>")
            self._table_ready = True
            logger.info("待办 | task 表已就绪")
        except Exception as e:
            logger.warning(f"待办 | 建表失败: {e}")

    async def add(self, content: str, source_user: str, source_qq: str = "") -> str | None:
        """新增待办。语义去重：内容完全相同的 pending 不重复添加。返回 task id 或 None。"""
        if not self._db or not content:
            return None
        await self._ensure_table()

        # 简单去重：完全相同的 pending 任务不重复
        existing = await self._db.query(
            "SELECT * FROM task WHERE status = 'pending' AND content = $content",
            {"content": content},
        )
        if isinstance(existing, list) and existing:
            logger.debug(f"待办跳过 | 已存在: {content!r}")
            return None

        result = await self._db.query(
            "CREATE task SET content = $content, source_user = $source, "
            "source_qq = $source_qq, status = 'pending', created_at = time::now()",
            {"content": content, "source": source_user, "source_qq": source_qq},
        )
        if isinstance(result, list) and result:
            task_id = str(result[0]["id"])
        else:
            task_id = "?"
        logger.info(f"待办新增 | {task_id} {content!r} (来自 {source_user})")
        return task_id

    async def set_target(self, task_id: str, target_qq: str):
        """设置任务的目标 QQ ID。"""
        if not self._db or not task_id or not target_qq:
            return
        await self._ensure_table()
        try:
            await self._db.query(
                f"UPDATE {task_id} SET target_qq = $qq",
                {"qq": target_qq},
            )
            logger.info(f"待办目标 | {task_id} → target_qq={target_qq}")
        except Exception as e:
            logger.warning(f"待办目标设置失败 | {task_id} {e}")

    async def get_pending(self) -> list[dict]:
        """获取所有 pending 待办。超过 48h 的自动标记 expired。"""
        if not self._db:
            return []
        await self._ensure_table()

        try:
            # 先过期处理
            await self._db.query(
                "UPDATE task SET status = 'expired' "
                "WHERE status = 'pending' AND created_at < time::now() - 48h",
            )

            result = await self._db.query(
                "SELECT * FROM task WHERE status = 'pending' ORDER BY created_at ASC",
            )
            pending = result if isinstance(result, list) else []
            if pending:
                logger.debug(f"待办查询 | {len(pending)} 条 pending")
            return pending
        except Exception as e:
            logger.warning(f"待办查询失败 | {e}")
            return []

    async def get_actionable(self) -> list[dict]:
        """获取可主动执行的任务：pending + 有 target_qq + 没 sent_at。"""
        if not self._db:
            return []
        await self._ensure_table()

        try:
            result = await self._db.query(
                "SELECT * FROM task WHERE status = 'pending' "
                "AND target_qq IS NOT NONE AND sent_at IS NONE "
                "ORDER BY created_at ASC",
            )
            actionable = result if isinstance(result, list) else []
            if actionable:
                logger.debug(f"待办可执行 | {len(actionable)} 条")
            return actionable
        except Exception as e:
            logger.warning(f"待办可执行查询失败 | {e}")
            return []

    async def mark_sent(self, task_id: str):
        """标记已发送主动消息。"""
        if not self._db or not task_id:
            return
        await self._ensure_table()
        try:
            await self._db.query(
                f"UPDATE {task_id} SET sent_at = time::now()",
            )
            logger.info(f"待办已发送 | {task_id}")
        except Exception as e:
            logger.warning(f"待办发送标记失败 | {task_id} {e}")

    async def mark_done(self, content_hint: str, result: str = "") -> bool:
        """用关键词匹配 pending 任务并标记完成。可附带结果摘要。返回是否有匹配。"""
        if not self._db or not content_hint:
            return False
        await self._ensure_table()

        params = {"hint": content_hint}
        result_clause = ", done_result = $result" if result else ""
        if result:
            params["result"] = result[:100]

        matched = await self._db.query(
            "UPDATE task SET status = 'done', done_at = time::now()" + result_clause + " "
            "WHERE status = 'pending' AND content CONTAINS $hint "
            "RETURN BEFORE",
            params,
        )
        matched = matched if isinstance(matched, list) else []
        if matched:
            for t in matched:
                logger.info(f"待办完成 | {t.get('content', '?')!r} (匹配: {content_hint!r}) result={result!r}")
            return True
        return False

    async def get_done_for_source(self, source_qq: str) -> list[dict]:
        """获取最近 24h 内完成的、属于该 source_qq 的待办（用于完成回报）。"""
        if not self._db or not source_qq:
            return []
        await self._ensure_table()

        try:
            result = await self._db.query(
                "SELECT * FROM task WHERE status = 'done' "
                "AND source_qq = $qq AND done_at > time::now() - 24h "
                "ORDER BY done_at DESC",
                {"qq": source_qq},
            )
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.warning(f"待办完成查询失败 | {e}")
            return []

    async def format_for_prompt(self, current_user: str = "", current_qq: str = "") -> str:
        """格式化待办注入 prompt。包含 pending 和该用户的已完成待办。
        current_user: 当前对话对象的名字，如果待办涉及此人会标记提醒。
        current_qq: 当前对话对象的 QQ ID，用于查已完成待办。
        没有任何内容返回空字符串。
        """
        parts = []

        # ── pending 待办 ──
        try:
            pending = await self.get_pending()
        except Exception as e:
            logger.warning(f"待办格式化失败 | {e}")
            pending = []

        if pending:
            lines = ["## 待办事项", "别人拜托你的事，跟对方聊天时顺便提一下："]
            for t in pending:
                content = t.get("content", "?")
                source = t.get("source_user", "")
                created = t.get("created_at", "")
                age = self._format_age(created)
                tag = ""
                if current_user and any(part in content for part in current_user.split() if len(part) >= 2):
                    tag = " ← 你现在就在跟这个人聊，说一下"
                if source:
                    lines.append(f"- {content}（{source} {age}）{tag}")
                else:
                    lines.append(f"- {content}（{age}）{tag}")
            parts.append("\n".join(lines))

        # ── 已完成待办（回报给 source） ──
        if current_qq:
            try:
                done = await self.get_done_for_source(current_qq)
            except Exception:
                done = []

            if done:
                done_lines = ["## 已完成的待办", "你之前帮对方做的事，告诉对方结果："]
                for t in done:
                    content = t.get("content", "?")
                    result = t.get("done_result", "")
                    done_at = t.get("done_at", "")
                    age = self._format_age(done_at)
                    if result:
                        done_lines.append(f"- {content}：{result}（{age}完成）")
                    else:
                        done_lines.append(f"- {content}（{age}完成）")
                parts.append("\n".join(done_lines))

        return "\n\n".join(parts)

    @staticmethod
    def _format_age(created_at) -> str:
        """把 SurrealDB 的 datetime 转为 '20分钟前' '3小时前' 格式。"""
        if not created_at:
            return ""
        try:
            if isinstance(created_at, str):
                from datetime import datetime, timezone
                ct = created_at.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ct)
                now = datetime.now(timezone.utc)
                diff_sec = (now - dt).total_seconds()
            else:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                diff_sec = (now - created_at).total_seconds()

            if diff_sec < 60:
                return "刚刚"
            if diff_sec < 3600:
                return f"{int(diff_sec / 60)}分钟前"
            if diff_sec < 86400:
                return f"{int(diff_sec / 3600)}小时前"
            return f"{int(diff_sec / 86400)}天前"
        except Exception:
            return ""
