"""关系系统：追踪用户信任等级和互动历史。"""

import json
import os
import time
from dataclasses import asdict, dataclass, field

from nonebot import logger

DATA_PATH = "data/relationships.json"
IMPRESSION_TS_PATH = "data/impression_ts.json"


@dataclass
class UserProfile:
    user_id: str
    user_name: str = ""
    trust: float = 0.0
    first_seen: float = field(default_factory=time.time)
    interaction_count: int = 0
    is_enemy: bool = False
    enemy_until: float = 0.0  # Unix timestamp，敌对状态到期时间
    last_interaction: float = 0.0  # 最后一次互动的 Unix 时间戳
    summary: str = ""          # LLM 生成的一句话认知摘要
    facts: list = field(default_factory=list)  # 从对话中提取的事实性信息


class RelationshipManager:
    def __init__(self, data_path: str = DATA_PATH):
        self._path = data_path
        self._ts_path = os.path.join(os.path.dirname(data_path), "impression_ts.json")
        self._profiles: dict[str, UserProfile] = {}
        self._impression_ts: dict[str, float] = {}
        self._load()
        self._load_impression_ts()

    def get(self, user_id: str) -> UserProfile:
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        return self._profiles[user_id]

    def trust_level(self, user_id: str, owner_id: str) -> str:
        if user_id == owner_id:
            return "owner"
        p = self.get(user_id)
        if p.is_enemy and time.time() < p.enemy_until:
            return "enemy"
        if p.trust >= 60:
            return "friend"
        if p.trust >= 30:
            return "acquaintance"
        return "stranger"

    def record_interaction(self, user_id: str, user_name: str, importance: float, trust_delta: float):
        """根据 eval 返回的 trust_delta 更新信任度"""
        p = self.get(user_id)
        if user_name:
            p.user_name = user_name
        p.interaction_count += 1
        p.last_interaction = time.time()

        if trust_delta != 0:
            old = p.trust
            p.trust = max(0.0, min(100.0, p.trust + trust_delta))
            logger.debug(f"关系 | {user_name}({user_id}) trust {old:.1f} {trust_delta:+} → {p.trust:.1f}")

        self._save()

    def set_enemy(self, user_id: str, years: float = 10.0):
        """将用户设为敌对状态"""
        p = self.get(user_id)
        p.is_enemy = True
        p.enemy_until = time.time() + years * 365.25 * 86400
        p.trust = 0.0
        logger.warning(f"关系 | {user_id} 设为敌对，持续 {years:.0f} 年")
        self._save()

    def get_impression_ts(self, key: str) -> float:
        """取印象冷却时间戳（跨重启持久化）"""
        return self._impression_ts.get(key, 0.0)

    def set_impression_ts(self, key: str, ts: float):
        """更新印象冷却时间戳并持久化"""
        self._impression_ts[key] = ts
        self._save_impression_ts()

    def update_summary(self, user_id: str, summary: str):
        """更新用户的一句话认知摘要"""
        p = self.get(user_id)
        p.summary = summary
        self._save()

    def add_facts(self, user_id: str, new_facts: list[str]):
        """追加新事实，简单去重，上限 20 条（超出时丢弃最旧的）"""
        if not new_facts:
            return
        p = self.get(user_id)
        existing = set(p.facts)
        added = 0
        for f in new_facts:
            if f and f not in existing:
                p.facts.append(f)
                existing.add(f)
                added += 1
        if len(p.facts) > 20:
            p.facts = p.facts[-20:]
        if added:
            logger.debug(f"事实 | {p.user_name or user_id} 新增 {added} 条: {new_facts}")
            self._save()

    def format_context(self, user_id: str, user_name: str, owner_id: str) -> str:
        """生成适合注入 system prompt 的对话对象描述"""
        if user_id == owner_id:
            return (
                "你正在和 1547（猫猫）对话。他是你的创造者和朋友。\n"
                '用毒舌但关心的方式和他说话。叫他"猫猫"或"你小子"。\n'
                "他如果状态不好，你要拽他回来。"
            )

        level = self.trust_level(user_id, owner_id)
        p = self.get(user_id)
        name = user_name or p.user_name or f"用户{user_id[-4:]}"

        if level == "enemy":
            return f"{name} 是你的敌人。敌对状态，不相信任何话，随时准备开怼。"

        if level == "friend":
            text = (
                f"{name}（信任度 {p.trust:.0f}/100，互动 {p.interaction_count} 次）是你的朋友。\n"
                f"一边损一边帮，挖苦是关心的方式。"
            )
            if p.facts:
                text += f"\n已知信息：{'、'.join(p.facts[:10])}"
            if p.summary:
                text += f"\n你对ta的了解：{p.summary}"
            return text

        if level == "acquaintance":
            text = (
                f"{name}（信任度 {p.trust:.0f}/100，互动 {p.interaction_count} 次）是你认识的人。\n"
                f"态度可以稍软化，但还是有点警惕。"
            )
            if p.facts:
                text += f"\n已知信息：{'、'.join(p.facts[:10])}"
            if p.summary:
                text += f"\n你对ta的了解：{p.summary}"
            return text

        # stranger
        hint = f"互动 {p.interaction_count} 次" if p.interaction_count > 0 else "第一次说话"
        text = (
            f"{name}（{hint}）是陌生人。\n"
            f'冷淡、警惕，默认不信任。陌生人找你帮忙？"凭什么？"'
        )
        if p.facts:
            text += f"\n已知信息：{'、'.join(p.facts[:10])}"
        if p.summary:
            text += f"\n你对ta的了解：{p.summary}"
        return text

    def get_friends(self, owner_id: str) -> list[UserProfile]:
        """返回所有 friend 级别的用户（trust >= 60，不含 owner）。"""
        return [
            p for p in self._profiles.values()
            if p.user_id != owner_id
            and not p.is_enemy
            and p.trust >= 60
        ]

    async def find_by_activity(self, activity: str, owner_id: str, graph=None) -> list[str]:
        """在活动文本中匹配已知用户名，返回匹配到的 user_id 列表。

        如果提供了 graph（CharacterGraph），用图数据库的别名做匹配，覆盖面更广。
        否则 fallback 到原有的子串匹配。
        """
        if graph:
            matches = await graph.find_in_text(activity)
            result = [
                m["qq_id"] for m in matches
                if m.get("qq_id") and m["qq_id"] != owner_id
            ]
            logger.debug(f"活动匹配(图谱) | {activity!r} → {result}")
            return result

        # fallback：原有逻辑
        matched = []
        for p in self._profiles.values():
            if p.user_id == owner_id or not p.user_name:
                continue
            names_to_check = [p.user_name]
            if len(p.user_name) >= 3:
                names_to_check.append(p.user_name[1:])  # 去掉姓
            for name in names_to_check:
                if len(name) >= 2 and name in activity:
                    matched.append(p.user_id)
                    break
        if matched:
            logger.debug(f"活动匹配(fallback) | {activity!r} → {matched}")
        return matched

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            for uid, d in data.items():
                self._profiles[uid] = UserProfile(**d)
            logger.info(f"关系 | 加载 {len(self._profiles)} 个用户档案")
        except Exception as e:
            logger.warning(f"关系 | 加载失败: {e}")

    def _load_impression_ts(self):
        if not os.path.exists(self._ts_path):
            return
        try:
            with open(self._ts_path, encoding="utf-8") as f:
                self._impression_ts = json.load(f)
            logger.debug(f"关系 | 加载 {len(self._impression_ts)} 条印象时间戳")
        except Exception as e:
            logger.warning(f"关系 | 加载印象时间戳失败: {e}")

    def _save(self):
        try:
            dir_path = os.path.dirname(os.path.abspath(self._path))
            os.makedirs(dir_path, exist_ok=True)
            data = {uid: asdict(p) for uid, p in self._profiles.items()}
            tmp_path = self._path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path)  # 原子替换，崩溃安全
        except Exception as e:
            logger.warning(f"关系 | 保存失败: {e}")

    def _save_impression_ts(self):
        try:
            dir_path = os.path.dirname(os.path.abspath(self._ts_path))
            os.makedirs(dir_path, exist_ok=True)
            tmp_path = self._ts_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._impression_ts, f)
            os.replace(tmp_path, self._ts_path)  # 原子替换
        except Exception as e:
            logger.warning(f"关系 | 保存印象时间戳失败: {e}")
