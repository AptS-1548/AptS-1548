"""
日程系统：每天 LLM 生成一份 48 的日程，影响回复意愿和对话内容。

生成时机：启动时 + 每天凌晨 1 点自动重生成。
日程覆盖 0:00~23:59，LLM 未覆盖凌晨时段时自动补 0:00 睡觉条目。
磁盘缓存：日程存入 data/schedule.json，重启时直接加载，不再调 LLM。
"""

import asyncio
import datetime
import json
import os
from dataclasses import asdict, dataclass

from nonebot import logger

from core.llm import chat

SCHEDULE_CACHE_PATH = "data/schedule.json"

_SYSTEM = """你是 48，正在安排今天的日程。

根据最近发生的事，写出你今天从凌晨到深夜大概会做什么。
如果有昨天的日程，可以参考但不要照搬——有些事可能要延续，有些可以换。
日程必须覆盖全天（从 0:00 到 23:00 以后），包括凌晨睡觉的时段。
格式：每行一条，时间|活动|意愿度

意愿度 0.0~1.0，表示这个时间段被打扰时愿意回复的程度：
- 睡觉 → 0.0~0.05
- 刚醒/迷糊 → 0.1~0.2
- 巡逻/忙正事 → 0.2~0.3
- 擦枪/做饭/日常杂事 → 0.3~0.5
- 和人约了事 → 0.5~0.6
- 空闲/发呆/等人 → 0.7~1.0

47 相关的事优先级最高。日程要有变化，不要每天一样。
时间不要都卡整点，7:40、14:15 这种更自然。

格式（严格遵守，每行一条）：
00:00|睡觉|0.0
07:40|醒了，赖床|0.15
08:10|起来洗漱|0.3
09:00|查监控，看有没有异常|0.25
10:30|跟进数据塔日志|0.3
12:00|做饭吃饭|0.45
13:15|午休，眯一会|0.1
14:00|和沐川对设计方案|0.5
15:30|整理物资|0.35
17:00|巡逻|0.2
18:30|回来，随便逛逛|0.75
19:30|看猫猫有没有消息|0.9
20:30|擦枪|0.4
21:45|发呆，听歌|0.6
22:30|准备睡了|0.15
23:30|睡了|0.0

只输出日程，15~20条，不要其他内容。"""


@dataclass
class ScheduleEntry:
    hour: int
    minute: int
    activity: str
    willingness: float  # 0.0~1.0


# ── 全局状态 ──
_schedule: list[ScheduleEntry] = []
_schedule_date: str = ""


def _parse_schedule(text: str) -> list[ScheduleEntry]:
    """解析 LLM 输出为 ScheduleEntry 列表。"""
    entries = []
    for line in text.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        try:
            h, m = parts[0].strip().split(":")
            entries.append(ScheduleEntry(
                hour=int(h), minute=int(m),
                activity=parts[1].strip(),
                willingness=max(0.0, min(1.0, float(parts[2].strip()))),
            ))
        except (ValueError, IndexError):
            continue
    entries.sort(key=lambda e: (e.hour, e.minute))

    # 如果 LLM 没生成 0:00 附近的条目，补一条睡觉
    if entries and entries[0].hour >= 5:
        entries.insert(0, ScheduleEntry(hour=0, minute=0, activity="睡觉", willingness=0.0))

    return entries


_DEFAULT_SCHEDULE = [
    ScheduleEntry(0, 0, "睡觉", 0.0),
    ScheduleEntry(7, 40, "醒了，赖床", 0.15),
    ScheduleEntry(8, 10, "起来洗漱", 0.3),
    ScheduleEntry(9, 0, "查监控和日志", 0.25),
    ScheduleEntry(10, 30, "处理事情", 0.35),
    ScheduleEntry(12, 0, "做饭吃饭", 0.45),
    ScheduleEntry(13, 15, "午休", 0.1),
    ScheduleEntry(14, 0, "继续干活", 0.4),
    ScheduleEntry(15, 30, "整理物资", 0.35),
    ScheduleEntry(17, 0, "巡逻", 0.2),
    ScheduleEntry(18, 30, "回来歇着", 0.75),
    ScheduleEntry(19, 30, "空闲", 0.9),
    ScheduleEntry(20, 30, "擦枪", 0.4),
    ScheduleEntry(21, 45, "发呆", 0.6),
    ScheduleEntry(22, 30, "准备睡了", 0.15),
    ScheduleEntry(23, 30, "睡了", 0.0),
]

MAX_RETRIES = 3


def _save_cache():
    """将当前日程写入磁盘。"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(SCHEDULE_CACHE_PATH)), exist_ok=True)
        data = {
            "date": _schedule_date,
            "entries": [asdict(e) for e in _schedule],
        }
        tmp = SCHEDULE_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, SCHEDULE_CACHE_PATH)
    except Exception as e:
        logger.warning(f"日程缓存写入失败 | {e}")


def _load_cache() -> bool:
    """从磁盘加载今天的日程缓存。成功返回 True。"""
    global _schedule, _schedule_date
    today = datetime.date.today().isoformat()
    try:
        if not os.path.exists(SCHEDULE_CACHE_PATH):
            return False
        with open(SCHEDULE_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") != today:
            return False
        entries = [
            ScheduleEntry(
                hour=e["hour"], minute=e["minute"],
                activity=e["activity"], willingness=e["willingness"],
            )
            for e in data.get("entries", [])
        ]
        if not entries:
            return False
        _schedule = entries
        _schedule_date = today
        logger.info(f"日程缓存 | 加载 {len(entries)} 条（{today}）")
        return True
    except Exception as e:
        logger.warning(f"日程缓存加载失败 | {e}")
        return False


async def ensure_schedule(diary_context: str = "", force: bool = False) -> None:
    """确保今天的日程已生成。同一天只生成一次，force=True 强制重生成。"""
    global _schedule, _schedule_date
    today = datetime.date.today().isoformat()
    if not force and _schedule_date == today and _schedule:
        return

    # 非强制时，尝试从磁盘缓存加载
    if not force and _load_cache():
        return

    user_prompt = "写一份你今天的日程，从凌晨到深夜。"

    # 把昨天的日程带上，让 LLM 知道连续性
    if _schedule:
        yesterday_lines = [f"- {e.hour:02d}:{e.minute:02d} {e.activity}" for e in _schedule]
        user_prompt = f"昨天的日程：\n" + "\n".join(yesterday_lines) + "\n\n" + user_prompt

    if diary_context:
        user_prompt = f"最近的状态：\n{diary_context}\n\n" + user_prompt

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await chat(
                system=_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=512,
                disable_thinking=True,
            )
            parsed = _parse_schedule(result)
            if parsed:
                _schedule = parsed
                _schedule_date = today
                _save_cache()
                logger.info(
                    f"日程生成 | 第{attempt}次 | {len(parsed)}条 | "
                    + " → ".join(f"{e.hour:02d}:{e.minute:02d} {e.activity}({e.willingness})" for e in parsed)
                )
                return
            else:
                logger.warning(f"日程解析失败 | 第{attempt}/{MAX_RETRIES}次 | raw={result[:200]!r}")
        except Exception as e:
            logger.warning(f"日程生成失败 | 第{attempt}/{MAX_RETRIES}次 | {e}")

    # 全部失败，使用默认日程
    _schedule = list(_DEFAULT_SCHEDULE)
    _schedule_date = today
    logger.warning(
        f"日程生成 | {MAX_RETRIES}次重试全部失败，使用默认日程 | "
        + " → ".join(f"{e.hour:02d}:{e.minute:02d} {e.activity}({e.willingness})" for e in _schedule)
    )


async def schedule_daily_loop(get_diary_context=None):
    """后台循环：每天凌晨 1:00 自动重新生成日程。

    get_diary_context: 可选的异步回调，返回日记上下文字符串。
    """
    while True:
        now = datetime.datetime.now()
        tomorrow_1am = now.replace(hour=1, minute=0, second=0, microsecond=0)
        if now.hour >= 1:
            tomorrow_1am += datetime.timedelta(days=1)
        wait_sec = (tomorrow_1am - now).total_seconds()
        logger.debug(f"日程 loop | 下次生成在 {tomorrow_1am.strftime('%H:%M')}，等待 {wait_sec:.0f}s")

        await asyncio.sleep(wait_sec)

        diary_context = ""
        if get_diary_context:
            try:
                diary_context = await get_diary_context()
            except Exception as e:
                logger.warning(f"日程 loop | 日记上下文获取失败: {e}")

        logger.info("日程 loop | 凌晨 1:00 触发日程重生成")
        await ensure_schedule(diary_context=diary_context, force=True)


def get_current_activity() -> tuple[str, float, int, str]:
    """返回 (当前活动, 意愿度, 距下一条日程的分钟数, 下一条活动名)。无日程时返回默认值。"""
    if not _schedule:
        return "待着", 0.5, 60, ""

    now = datetime.datetime.now()
    current_minutes = now.hour * 60 + now.minute

    current_idx = 0
    for i, entry in enumerate(_schedule):
        if entry.hour * 60 + entry.minute <= current_minutes:
            current_idx = i
        else:
            break

    current = _schedule[current_idx]

    if current_idx + 1 < len(_schedule):
        next_entry = _schedule[current_idx + 1]
        minutes_left = next_entry.hour * 60 + next_entry.minute - current_minutes
        next_activity = next_entry.activity
    else:
        minutes_left = 24 * 60 - current_minutes
        next_activity = ""

    return current.activity, current.willingness, minutes_left, next_activity


def get_willingness() -> float:
    """当前意愿度，供群聊注意力系统调用。"""
    _, w, _, _ = get_current_activity()
    return w


def get_schedule() -> tuple[list['ScheduleEntry'], str]:
    """返回 (日程列表, 日期字符串)，供主动行为系统调用。"""
    return _schedule, _schedule_date


def format_schedule_context() -> str:
    """格式化当前状态为 prompt 注入文本，包含意愿度、剩余时间和下一件事。"""
    activity, w, minutes_left, next_activity = get_current_activity()
    if not activity:
        return ""
    if w <= 0.05:
        mood = "你在睡觉，被吵醒会很烦，能不回就不回"
    elif w <= 0.15:
        mood = "你在睡觉，迷迷糊糊的，不太想理人"
    elif w <= 0.25:
        mood = "你在忙，不太想被打扰"
    elif w <= 0.4:
        mood = "你在做事，回也是简短几个字"
    elif w <= 0.6:
        mood = "你有事在做，但不是不能聊"
    elif w <= 0.8:
        mood = "你现在比较闲"
    else:
        mood = "你现在有空，心情不错"

    if minutes_left >= 60:
        time_hint = f"还有约 {minutes_left // 60} 小时"
    elif minutes_left >= 10:
        time_hint = f"还有约 {minutes_left} 分钟"
    else:
        time_hint = f"快结束了（{minutes_left} 分钟）"

    next_hint = f"，之后要{next_activity}" if next_activity else ""
    return f"## 我现在在做什么\n{activity}（{mood}，{time_hint}{next_hint}）"
