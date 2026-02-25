"""
主动行为：48 在适当时机主动联系 owner 和朋友。

两种触发方式：
1. 周期性检查：owner + friend 级别用户，LLM 判断要不要说
2. 日程驱动：日程提到某人（"和沐川对设计"），到时间主动联系
"""

import datetime
import random
import time

from nonebot import logger

from core.llm import chat
from core.memory import Memory
from core.prompt import format_diary, format_impressions
from core.relationship import RelationshipManager
from core.schedule import get_current_activity, get_schedule, get_willingness

_SYSTEM_OWNER = """你是 48。现在你有一段空闲，在想要不要主动找猫猫说点什么。

你会根据以下信息决定：
- 你最近的状态和日记
- 上次和猫猫聊天是什么时候
- 你现在在做什么
- 你对她最近的印象

如果你觉得有话想说（担心她、想跟进之前的事、或者就是想找她说两句），就写一条消息。
像平时在QQ上给她发消息一样，一条，简短，自然。

规则：
- 不要自我介绍
- 不要解释你为什么发消息
- 就像你平时会发的那种消息
- 一条就够

如果没什么特别想说的，回复[不主动]。
不要为了主动而主动，没话说就不说。"""

_SYSTEM_FRIEND = """你是 48。现在你有一段空闲，在想要不要主动找{name}说点什么。

关于{name}：
{relationship_desc}

你会根据以下信息决定：
- 你最近的状态
- 上次和{name}聊天是什么时候
- 你现在在做什么

如果有话想说就写一条消息，像平时在QQ上发消息一样。
一条，简短，自然。不要自我介绍，不要解释为什么发消息。

如果没什么想说的，回复[不主动]。
朋友之间偶尔找一下很正常，但别没话找话。"""

_SYSTEM_SCHEDULE = """你是 48。你的日程上写了现在要"{activity}"。
你要给{name}发一条消息，开始这个事。

像平时在QQ上发消息一样，一条，简短，自然。
比如约了对设计就说"设计的事，你现在有空吗"之类的。"""

# ── 全局状态 ──
_last_proactive: dict[str, float] = {}  # user_id -> 上次主动时间
_proactive_count_today: int = 0
_proactive_date: str = ""
_triggered_entries: set[str] = set()  # "2025-01-01_14:00" 已触发的日程条目


def _reset_daily():
    """跨天重置。"""
    global _proactive_count_today, _proactive_date, _triggered_entries
    today = datetime.date.today().isoformat()
    if _proactive_date != today:
        _proactive_count_today = 0
        _triggered_entries.clear()
        _proactive_date = today


async def check_proactive(
    target_id: str,
    owner_id: str,
    memory: Memory,
    relationship: RelationshipManager,
    cooldown_sec: int = 21600,
    max_daily: int = 2,
    min_silence_hours: float = 2.0,
) -> str | None:
    """周期性主动检查：对 owner 或 friend 发起主动消息。

    硬性门槛：
    1. 日程意愿度 >= 0.3
    2. 该用户的冷却期
    3. 今日总次数未超限
    4. 至少 N 小时没聊过
    """
    global _proactive_count_today

    _reset_daily()

    is_owner = target_id == owner_id
    target = relationship.get(target_id)
    name = target.user_name or f"用户{target_id[-4:]}"
    label = "owner" if is_owner else name

    # 门槛 1：日程意愿度
    willingness = get_willingness()
    if willingness < 0.3:
        logger.debug(f"主动检查 | {label} 跳过：意愿度 {willingness:.2f} < 0.3")
        return None

    # 门槛 2：per-user 冷却
    user_cooldown = cooldown_sec if is_owner else cooldown_sec * 2
    last_ts = _last_proactive.get(target_id, 0)
    since_last = time.time() - last_ts
    if last_ts > 0 and since_last < user_cooldown:
        logger.debug(f"主动检查 | {label} 跳过：冷却中 {since_last:.0f}s < {user_cooldown}s")
        return None

    # 门槛 3：每日上限
    if _proactive_count_today >= max_daily:
        logger.debug(f"主动检查 | {label} 跳过：今日已主动 {_proactive_count_today}/{max_daily} 次")
        return None

    # 门槛 4：沉默时长（last_interaction=0 表示从未记录，视为很久没聊）
    if target.last_interaction > 0:
        silence_hours = (time.time() - target.last_interaction) / 3600
        if silence_hours < min_silence_hours:
            logger.debug(f"主动检查 | {label} 跳过：刚聊过 {silence_hours:.1f}h < {min_silence_hours}h")
            return None
    else:
        silence_hours = 999  # 从未互动，视为极长沉默

    logger.info(f"主动检查 | {label} 通过所有门槛 | willingness={willingness:.2f} silence={silence_hours:.1f}h last_interaction={target.last_interaction:.0f} → 调用 LLM")

    # ── 收集上下文 ──
    activity, _ = get_current_activity()
    diary_entries = await memory.get_diary(recent_limit=5)
    impressions = await memory.get_impressions(user_id=target_id, recent_limit=3, key_limit=2)

    diary_text = format_diary(diary_entries) if diary_entries else ""
    impression_text = format_impressions(impressions) if impressions else ""

    # ── 构造 prompt ──
    if is_owner:
        system = _SYSTEM_OWNER
    else:
        rel_desc = relationship.format_context(target_id, name, owner_id)
        system = _SYSTEM_FRIEND.format(name=name, relationship_desc=rel_desc)

    parts = [f"你已经 {silence_hours:.0f} 小时没和{('猫猫' if is_owner else name)}说话了。"]
    parts.append(f"你现在在：{activity}")
    if diary_text:
        parts.append(diary_text)
    if impression_text:
        parts.append(impression_text)

    user_msg = "\n\n".join(parts)

    # ── LLM 决策 ──
    try:
        result = await chat(
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=128,
            disable_thinking=True,
        )
    except Exception as e:
        logger.warning(f"主动检查 | {name} LLM 失败: {e}")
        return None

    result = result.strip()

    if "[不主动]" in result:
        logger.info(f"主动检查 | {name} LLM 决定不主动 | silence={silence_hours:.1f}h")
        return None

    _last_proactive[target_id] = time.time()
    _proactive_count_today += 1
    logger.info(f"主动检查 | → {name} | silence={silence_hours:.1f}h msg={result!r}")
    return result


async def check_schedule_proactive(
    relationship: RelationshipManager,
    owner_id: str,
) -> list[tuple[str, str]]:
    """日程驱动：检查当前日程是否提到某人，返回 [(user_id, activity), ...]。

    每个日程条目只触发一次（按 日期_时间 去重）。
    """
    schedule, schedule_date = get_schedule()

    _reset_daily()

    if not schedule:
        return []

    now = datetime.datetime.now()
    current_minutes = now.hour * 60 + now.minute
    results = []

    for entry in schedule:
        entry_minutes = entry.hour * 60 + entry.minute
        # 只看已经到了且在 30 分钟窗口内的条目
        if entry_minutes > current_minutes or current_minutes - entry_minutes > 30:
            continue

        entry_key = f"{schedule_date}_{entry.hour:02d}:{entry.minute:02d}"
        if entry_key in _triggered_entries:
            continue

        # 在活动文本里找人名
        matched = relationship.find_by_activity(entry.activity, owner_id)
        if matched:
            _triggered_entries.add(entry_key)
            for uid in matched:
                results.append((uid, entry.activity))

    return results


async def generate_schedule_message(
    target_id: str,
    activity: str,
    relationship: RelationshipManager,
    owner_id: str,
) -> str | None:
    """为日程驱动的主动联系生成消息。"""
    target = relationship.get(target_id)
    name = target.user_name or f"用户{target_id[-4:]}"

    system = _SYSTEM_SCHEDULE.format(activity=activity, name=name)

    try:
        result = await chat(
            system=system,
            messages=[{"role": "user", "content": f"给{name}发一条消息。"}],
            max_tokens=64,
            disable_thinking=True,
        )
    except Exception as e:
        logger.warning(f"日程主动 | {name} LLM 失败: {e}")
        return None

    result = result.strip()
    logger.info(f"日程主动 | → {name} | activity={activity!r} msg={result!r}")
    return result
