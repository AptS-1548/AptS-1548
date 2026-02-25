import asyncio
import contextvars
import datetime
import random
import time
from collections import defaultdict, deque

import nonebot
from nonebot import on_message, get_driver, get_plugin_config, logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    MessageEvent,
    GroupMessageEvent,
    PrivateMessageEvent,
    Message,
)

from core.llm import chat
from core.prompt import build_system_prompt, build_group_turns, format_impressions, format_diary
from core.guard import Guard
from core.memory import Memory, MemoryEntry, format_memories
from core.eval import evaluate_exchange, evaluate_batch
from core.relationship import RelationshipManager
from core.summarize import generate_user_summary
from core.diary import generate_diary_entry
from core.schedule import ensure_schedule, get_willingness, format_schedule_context, schedule_daily_loop
from core.proactive import check_proactive, check_schedule_proactive, generate_schedule_message

from .config import Config

plugin_config = get_plugin_config(Config)
plugin_config.owner_id = str(plugin_config.owner_id)
plugin_config.bot_name = str(plugin_config.bot_name)

logger.info(f"插件加载 | bot={plugin_config.bot_name} owner={plugin_config.owner_id}")
logger.info(f"允许群: {plugin_config.allowed_groups or '(无)'}")
logger.info(f"速率: {plugin_config.rate_per_minute}/min, 日限: {plugin_config.daily_limit}")

guard = Guard(
    rate_per_minute=plugin_config.rate_per_minute,
    daily_limit=plugin_config.daily_limit,
    cache_ttl=plugin_config.cache_ttl,
)

memory = Memory(qdrant_url=plugin_config.qdrant_url)
relationship = RelationshipManager()

# ── 私聊：多轮对话历史 ──
_private_histories: dict[str, list[dict]] = defaultdict(list)
_private_last_time: dict[str, float] = {}  # user_id -> 上次消息时间
MAX_PRIVATE_HISTORY = 20
TIME_GAP_THRESHOLD = 600  # 10分钟以上插入时间标记

# ── 群聊：滚动上下文（记录谁说了什么） ──
_group_context: dict[str, list[dict]] = defaultdict(list)
MAX_GROUP_CONTEXT = 50

# ── 群聊注意力系统 ──
_group_attention: dict[str, float] = defaultdict(float)  # group_id -> 0.0~1.0
_group_last_msg_time: dict[str, float] = {}
_group_last_reply_time: dict[str, float] = {}
_group_msg_times: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))

ATTENTION_DECAY = 0.7
ATTENTION_IDLE_SEC = 300
ATTENTION_AT = 1.0
ATTENTION_OWNER = 0.2
ATTENTION_REPLY_BOOST = 0.2
ATTENTION_QUOTED = 0.5

REPLY_COOLDOWN_SEC = 30   # 回复后沉默 30 秒（被@除外）
DENSITY_WINDOW_SEC = 20   # 密度检测窗口
DENSITY_THRESHOLD = 6     # 20 秒内超过 6 条算太吵
DENSITY_MULT_MIN = 0.2    # 再吵也最多压到 0.2 倍

# ── 防抖：避免连续消息触发多次 LLM ──
_pending: dict[str, asyncio.Task] = {}
_burst_active: set[str] = set()   # 被取消过、正在冷静的会话
_burst_texts: dict[str, list[str]] = defaultdict(list)  # burst 期间积累的消息文本
_rebuilt_private_users: set[str] = set()  # 已重建过私聊历史的用户

# ── eval buffer：攒多轮一起评估 ──
_eval_buffer: dict[str, list[dict]] = defaultdict(list)  # user_id -> [round_info, ...]
_eval_timers: dict[str, asyncio.Task] = {}  # user_id -> flush timer
EVAL_BATCH_SIZE = 3       # 攒满 N 轮触发 eval
EVAL_FLUSH_TIMEOUT = 120  # 超时 N 秒自动 flush

# ── 印象去重：同一用户/群 2 小时内只存一次印象（运行时 + 持久化通过 relationship）──
IMPRESSION_COOLDOWN_SEC = 2 * 3600

# ── Trace ID（每次 LLM 调用一个，贯穿整条链路）──
_trace_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def _tid() -> str:
    return _trace_ctx.get()


def _new_tid() -> str:
    return format(random.randint(0, 0xFFFFFF), "06x")
DEBOUNCE_SEC = 1.5
BURST_COOLDOWN_SEC = 3.0  # 被取消后的冷静期，等用户说完


def _update_attention(group_id: str, event: GroupMessageEvent, bot_id: str) -> float:
    now = time.time()
    att = _group_attention[group_id]

    last_msg = _group_last_msg_time.get(group_id, 0)
    if now - last_msg > ATTENTION_IDLE_SEC:
        att = 0.0

    _group_last_msg_time[group_id] = now

    if event.is_tome():
        att = ATTENTION_AT
        _group_attention[group_id] = att
        logger.debug(f"注意力 | {group_id} @我 → {att:.2f}")
        return att

    att *= ATTENTION_DECAY

    user_id = str(event.user_id)
    if user_id == plugin_config.owner_id:
        att += ATTENTION_OWNER
    if _is_reply_to_me(event, bot_id):
        att += ATTENTION_QUOTED

    last_reply = _group_last_reply_time.get(group_id, 0)
    if now - last_reply < ATTENTION_IDLE_SEC:
        att += ATTENTION_REPLY_BOOST

    att = max(0.0, min(att, 1.0))
    _group_attention[group_id] = att
    logger.debug(f"注意力 | {group_id} att={att:.2f}")
    return att


def _is_reply_to_me(event: GroupMessageEvent, bot_id: str) -> bool:
    """判断消息是否引用了 bot 发的消息（通过 reply 段的 sender_id）"""
    try:
        for seg in event.message:
            if seg.type == "reply":
                # reply 段的 data["id"] 是消息 ID，sender_id 是原消息发送者
                sender_id = str(seg.data.get("sender_id", ""))
                if sender_id == bot_id:
                    return True
    except Exception:
        pass
    return False


def _density_multiplier(group_id: str) -> float:
    """消息密度越高，回复意愿越低（社恐不想凑热闹）"""
    now = time.time()
    recent = sum(1 for t in _group_msg_times[group_id] if now - t <= DENSITY_WINDOW_SEC)
    if recent <= DENSITY_THRESHOLD:
        return 1.0
    excess = recent - DENSITY_THRESHOLD
    return max(DENSITY_MULT_MIN, 1.0 - excess * 0.1)


def _time_multiplier() -> float:
    """从日程系统获取当前意愿度，替代硬编码时段表。"""
    w = get_willingness()
    if w > 0:
        return w
    return 0.05  # 睡觉时保留极低概率（被@仍会回）


def _should_reply_group(att: float) -> bool:
    if att >= 0.99:  # 被@，无视时段
        return True
    effective = att * _time_multiplier()
    if effective <= 0.05:
        return False
    return random.random() < effective


async def _attention_decay_loop():
    """后台定时衰减，让注意力随时间自然降低"""
    while True:
        await asyncio.sleep(30)
        for group_id in list(_group_attention.keys()):
            if _group_attention[group_id] > 0.01:
                _group_attention[group_id] *= 0.85
                logger.debug(f"注意力衰减 | {group_id} att={_group_attention[group_id]:.2f}")
            else:
                _group_attention[group_id] = 0.0


driver = get_driver()


@driver.on_startup
async def _startup():
    asyncio.create_task(_attention_decay_loop())
    asyncio.create_task(_rebuild_context())
    asyncio.create_task(memory.cleanup_old_entries())
    asyncio.create_task(_init_schedule())
    asyncio.create_task(schedule_daily_loop(_get_diary_context))
    asyncio.create_task(_proactive_loop())
    if plugin_config.diary_dedup_enabled:
        asyncio.create_task(_diary_dedup_loop())


async def _get_diary_context() -> str:
    """拉取日记供日程生成使用（比对话时多拉）。"""
    diary_entries = await memory.get_diary(
        recent_limit=plugin_config.schedule_diary_limit,
        key_limit=plugin_config.schedule_diary_key_limit,
    )
    return format_diary(diary_entries, show_time=True) if diary_entries else ""


async def _init_schedule():
    """启动时拉取日记，生成今天的日程。"""
    await asyncio.sleep(10)
    logger.info("启动日程初始化...")
    try:
        diary_entries = await memory.get_diary(
            recent_limit=plugin_config.schedule_diary_limit,
            key_limit=plugin_config.schedule_diary_key_limit,
        )
        diary_context = format_diary(diary_entries, show_time=True) if diary_entries else ""
        logger.info(f"日程初始化 | 拉取 {len(diary_entries)} 条日记")
        if diary_context:
            logger.info(f"日程初始化 | 日记内容:\n{diary_context}")
    except Exception:
        diary_context = ""
    await ensure_schedule(diary_context=diary_context, force=True)


async def _send_proactive(bot: Bot, target_id: str, msg: str, label: str = ""):
    """发送主动消息的统一逻辑：推入历史 + 发送 + 存记忆。"""
    _push_private(target_id, "assistant", msg)
    try:
        await bot.send_private_msg(user_id=int(target_id), message=Message(msg))
    except Exception as e:
        logger.warning(f"主动消息发送失败 | {label} {e}")
        return
    await memory.store(MemoryEntry(
        user_id=target_id,
        chat_type="private",
        chat_id=target_id,
        user_name="48",
        message="(主动发起)",
        response=msg,
        importance=0.5,
        record_type="dialog",
    ))
    logger.info(f"主动消息 | → {label} | {msg!r}")


async def _diary_dedup_loop():
    """后台循环：每天 0:30 自动执行日记语义去重。"""
    import datetime as _dt
    while True:
        now = _dt.datetime.now()
        target = now.replace(hour=0, minute=30, second=0, microsecond=0)
        if now >= target:
            target += _dt.timedelta(days=1)
        wait_sec = (target - now).total_seconds()
        logger.debug(f"日记去重 loop | 下次执行 {target.strftime('%H:%M')}，等待 {wait_sec:.0f}s")
        await asyncio.sleep(wait_sec)

        try:
            deleted = await memory.dedup_diary(
                threshold=plugin_config.diary_dedup_threshold,
                window_min=plugin_config.diary_dedup_window,
            )
            if deleted:
                logger.info(f"日记去重 loop | 完成，删除 {deleted} 条")
        except Exception as e:
            logger.warning(f"日记去重 loop | 失败: {e}")


async def _proactive_loop():
    """后台定时检查，触发 48 主动联系 owner 和朋友。"""
    await asyncio.sleep(300)  # 启动后等 5 分钟再开始
    logger.info(f"主动 loop 启动 | 间隔={plugin_config.proactive_check_sec}s 冷却={plugin_config.proactive_cooldown_sec}s")

    while True:
        await asyncio.sleep(plugin_config.proactive_check_sec)
        try:
            bot: Bot = nonebot.get_bot()
        except ValueError:
            continue

        owner_id = plugin_config.owner_id

        # ── Part 1: 日程驱动 ──
        try:
            schedule_contacts = await check_schedule_proactive(relationship, owner_id)
            for target_id, activity in schedule_contacts:
                msg = await generate_schedule_message(target_id, activity, relationship, owner_id)
                if msg:
                    name = relationship.get(target_id).user_name or target_id
                    await _send_proactive(bot, target_id, msg, label=f"日程→{name}")
        except Exception as e:
            logger.warning(f"日程主动检查异常 | {e}")

        # ── Part 2: 周期性检查 owner ──
        try:
            msg = await check_proactive(
                target_id=owner_id,
                owner_id=owner_id,
                memory=memory,
                relationship=relationship,
                cooldown_sec=plugin_config.proactive_cooldown_sec,
                max_daily=plugin_config.proactive_max_daily,
            )
            if msg:
                await _send_proactive(bot, owner_id, msg, label="owner")
        except Exception as e:
            logger.warning(f"主动检查异常(owner) | {e}")

        # ── Part 3: 随机选一个 friend 检查 ──
        try:
            friends = relationship.get_friends(owner_id)
            if friends:
                friend = random.choice(friends)
                msg = await check_proactive(
                    target_id=friend.user_id,
                    owner_id=owner_id,
                    memory=memory,
                    relationship=relationship,
                    cooldown_sec=plugin_config.proactive_cooldown_sec,
                    max_daily=plugin_config.proactive_max_daily,
                    min_silence_hours=6.0,  # 朋友沉默阈值更高
                )
                if msg:
                    await _send_proactive(bot, friend.user_id, msg, label=friend.user_name or friend.user_id)
        except Exception as e:
            logger.warning(f"主动检查异常(friend) | {e}")


async def _rebuild_context():
    """从 Qdrant 拉最近对话，重建群聊上下文，降低重启后断层感。"""
    for group_id in plugin_config.allowed_groups:
        try:
            entries = await memory.recent(chat_id=group_id, limit=MAX_GROUP_CONTEXT // 2)
            for e in entries:
                _push_group(group_id, e.user_name, e.message)
                _push_group(group_id, plugin_config.bot_name, e.response)
            if entries:
                logger.info(f"重建上下文 | 群 {group_id} 载入 {len(entries)} 条")
        except Exception as ex:
            logger.warning(f"重建上下文失败 | 群 {group_id} {ex}")


def _get_sender_name(event: GroupMessageEvent) -> str:
    sender = event.sender
    return sender.card or sender.nickname or str(event.user_id)


def _push_private(user_id: str, role: str, content: str, ts: float = 0):
    if not content.strip():
        return
    now = ts or time.time()
    # 用户消息且间隔超过阈值，插入时间标记
    if role == "user":
        last = _private_last_time.get(user_id, 0)
        gap = now - last if last else 0
        if gap >= TIME_GAP_THRESHOLD:
            dt_now = datetime.datetime.fromtimestamp(now)
            time_str = dt_now.strftime("%-m月%-d日 %H:%M")
            _private_histories[user_id].append(
                {"role": "user", "content": f"[{time_str}]"}
            )
            _private_histories[user_id].append(
                {"role": "assistant", "content": "（继续对话）"}
            )
    _private_last_time[user_id] = now
    _private_histories[user_id].append({"role": role, "content": content})
    if len(_private_histories[user_id]) > MAX_PRIVATE_HISTORY:
        _private_histories[user_id] = _private_histories[user_id][-MAX_PRIVATE_HISTORY:]


def _push_group(group_id: str, name: str, content: str):
    if not content.strip():
        return
    _group_context[group_id].append({"name": name, "content": content})
    if len(_group_context[group_id]) > MAX_GROUP_CONTEXT:
        _group_context[group_id] = _group_context[group_id][-MAX_GROUP_CONTEXT:]


def _is_allowed_group(group_id: str) -> bool:
    return bool(plugin_config.allowed_groups) and group_id in plugin_config.allowed_groups


async def _send(bot: Bot, is_group: bool, group_id: str | None, user_id: str, text: str):
    if is_group:
        logger.info(f"[{_tid()}] 发送 | 群 {group_id} → {text!r}")
        await bot.send_group_msg(group_id=int(group_id), message=Message(text))
    else:
        logger.info(f"[{_tid()}] 发送 | 私聊 {user_id} → {text!r}")
        await bot.send_private_msg(user_id=int(user_id), message=Message(text))


async def _typing_delay(text: str):
    delay = max(2.0, len(text) * random.uniform(0.3, 0.6))
    if random.random() < 0.2:
        delay += random.uniform(1.5, 4.0)
    await asyncio.sleep(delay)


async def _update_user_summary(user_id: str, user_name: str):
    """异步生成并更新用户摘要（在 _evaluate_and_store 的 task 里调用）"""
    try:
        impressions = await memory.get_impressions(user_id=user_id, recent_limit=5, key_limit=10)
        p = relationship.get(user_id)
        summary = await generate_user_summary(
            user_name=user_name,
            impressions=impressions,
            trust=p.trust,
            interaction_count=p.interaction_count,
        )
        if summary:
            relationship.update_summary(user_id, summary)
            logger.info(f"摘要更新 | user={user_name}: {summary!r}")
    except Exception as e:
        logger.warning(f"摘要更新失败 | user={user_id} {e}")


async def _write_diary_entry(
    user_id: str,
    user_name: str,
    message: str,
    response: str,
    importance: float,
) -> None:
    """异步生成并存储 48 的日记条目。拉取最近印象作为上下文，让日记有更广的视野。"""
    trust = relationship.get(user_id).trust

    # 取最近印象作为背景，让 48 能结合近期发生的事来写日记
    recent_context = ""
    recent_diary = ""
    try:
        impressions = await memory.get_impressions(user_id=user_id, recent_limit=3, key_limit=2)
        if impressions:
            recent_context = "\n".join(f"- {m.message}" for m in impressions[:5])
        diary_entries = await memory.get_diary(limit=3)
        if diary_entries:
            recent_diary = "\n".join(f"- {e.message}" for e in diary_entries)
    except Exception:
        pass

    entry_text = await generate_diary_entry(user_name, message, response, importance, trust, recent_context, recent_diary)
    if not entry_text:
        return

    # 语义去重：和最近 24h 的日记对比，太像就不存
    if await memory.is_diary_duplicate(entry_text):
        logger.debug(f"日记跳过 | 语义重复: {entry_text!r}")
        return

    await memory.store(MemoryEntry(
        user_id="48",
        chat_type="private",
        chat_id="48",
        user_name="48",
        message=entry_text,
        response="",
        importance=0.6,
        record_type="diary",
    ))
    logger.debug(f"日记已存 | {entry_text!r}")


async def _evaluate_and_store(
    user_id: str, chat_type: str, chat_id: str,
    user_name: str, message: str, response: str,
):
    """将对话记录存入 Qdrant，eval 部分丢进 buffer 攒批处理。"""

    # owner 跳过 eval，直接存 + 写日记 + 更新互动时间
    if user_id == plugin_config.owner_id:
        relationship.record_interaction(user_id, user_name, 0.5, 0)
        await memory.store(MemoryEntry(
            user_id=user_id,
            chat_type=chat_type,
            chat_id=chat_id,
            user_name=user_name,
            message=message,
            response=response,
            importance=0.5,
            record_type="dialog",
        ))
        asyncio.create_task(_write_diary_entry(user_id, user_name, message, response, importance=0.7))
        return

    # 立即存 dialog（不等 eval），importance 暂用 0.5
    await memory.store(MemoryEntry(
        user_id=user_id,
        chat_type=chat_type,
        chat_id=chat_id,
        user_name=user_name,
        message=message,
        response=response,
        importance=0.5,
        record_type="dialog",
    ))

    # 把这一轮丢进 eval buffer
    _eval_buffer[user_id].append({
        "user_name": user_name,
        "message": message,
        "response": response,
        "chat_type": chat_type,
        "chat_id": chat_id,
    })

    buf_len = len(_eval_buffer[user_id])
    logger.debug(f"[{_tid()}] eval buffer | user={user_name} 累计{buf_len}轮")

    # 满 N 轮立即 flush
    if buf_len >= EVAL_BATCH_SIZE:
        _cancel_eval_timer(user_id)
        asyncio.create_task(_flush_eval(user_id))
    else:
        # 没满就启动/重置超时 timer
        _reset_eval_timer(user_id)


def _cancel_eval_timer(user_id: str):
    t = _eval_timers.pop(user_id, None)
    if t and not t.done():
        t.cancel()


def _reset_eval_timer(user_id: str):
    _cancel_eval_timer(user_id)

    async def _timer():
        await asyncio.sleep(EVAL_FLUSH_TIMEOUT)
        await _flush_eval(user_id)

    _eval_timers[user_id] = asyncio.create_task(_timer())


async def _flush_eval(user_id: str):
    """从 buffer 取出所有轮次，一次性调 eval，处理结果。"""
    rounds_info = _eval_buffer.pop(user_id, [])
    _cancel_eval_timer(user_id)

    if not rounds_info:
        return

    # 构造 eval 输入
    rounds = [(r["user_name"], r["message"], r["response"]) for r in rounds_info]
    user_name = rounds_info[-1]["user_name"]
    chat_type = rounds_info[-1]["chat_type"]
    chat_id = rounds_info[-1]["chat_id"]

    importance, impression, trust_delta, facts = await evaluate_batch(rounds)

    impression_key = f"{chat_type}_{chat_id}_{user_id}"
    cooldown_ok = (
        importance >= 0.8
        or time.time() - relationship.get_impression_ts(impression_key) >= IMPRESSION_COOLDOWN_SEC
    )
    should_store_impression = impression and importance >= 0.4 and cooldown_ok

    logger.debug(f"eval flush | user={user_name} {len(rounds)}轮 imp={importance:.2f} trust={trust_delta:+} facts={facts} impression={impression!r} {'→ 存印象' if should_store_impression else '→ 跳过印象'}")

    relationship.record_interaction(user_id, user_name, importance, trust_delta)
    if facts:
        relationship.add_facts(user_id, facts)

    if should_store_impression:
        relationship.set_impression_ts(impression_key, time.time())
        await memory.store(MemoryEntry(
            user_id=user_id,
            chat_type=chat_type,
            chat_id=chat_id,
            user_name=user_name,
            message=impression,
            response="",
            importance=importance,
            record_type="impression",
        ))

    p = relationship.get(user_id)
    should_summarize = (
        (p.interaction_count > 0 and p.interaction_count % 10 == 0)
        or (not p.summary and p.interaction_count >= 5)
        or abs(trust_delta) >= 3
        or importance >= 0.8
    )
    if should_summarize:
        asyncio.create_task(_update_user_summary(user_id, user_name))

    # diary：用最后一轮的内容
    last = rounds_info[-1]
    should_write_diary = importance >= 0.6 or abs(trust_delta) >= 3
    if should_write_diary:
        asyncio.create_task(_write_diary_entry(
            user_id, user_name, last["message"], last["response"], importance,
        ))


async def _do_reply(bot: Bot, is_group: bool, group_id: str | None, user_id: str, user_name: str, text: str, is_owner: bool, search_query: str = ""):
    """调用 LLM 并发送回复。search_query 是 burst 期间所有消息的拼合，用于记忆/日记检索；text 是最后一条，用于上下文和发送。"""
    if not search_query:
        search_query = text
    allowed, reject_msg = guard.check_rate(user_id, is_owner=is_owner)
    if not allowed:
        logger.warning(f"[{_tid()}] 速率限制 | user={user_id} msg={reject_msg}")
        await _send(bot, is_group, group_id, user_id, reject_msg)
        return

    cached = guard.get_cached(user_id, text)
    if cached:
        logger.info(f"[{_tid()}] 缓存命中 | user={user_id}")
        await _send(bot, is_group, group_id, user_id, cached)
        return

    chat_id = group_id if is_group else user_id

    # 懒加载：首次私聊时从 Qdrant 重建历史
    if not is_group and user_id not in _rebuilt_private_users:
        _rebuilt_private_users.add(user_id)
        try:
            entries = await memory.recent(user_id=user_id, limit=MAX_PRIVATE_HISTORY // 2)
            for e in entries:
                _push_private(user_id, "user", e.message, ts=e.timestamp)
                _push_private(user_id, "assistant", e.response, ts=e.timestamp)
            if entries:
                logger.info(f"[{_tid()}] 重建私聊历史 | user={user_id} 载入 {len(entries)} 条")
        except Exception as ex:
            logger.warning(f"[{_tid()}] 重建私聊历史失败 | user={user_id} {ex}")

    # 检索相关历史记忆 + 最近印象 + 48 的近况日记
    # 群聊按群隔离，私聊按人检索（群聊+私聊全捞）
    if is_group:
        memories, impressions, diary_entries = await asyncio.gather(
            memory.search(query=search_query, chat_id=chat_id, limit=plugin_config.memory_search_limit),
            memory.get_impressions(chat_id=chat_id, recent_limit=plugin_config.impression_recent_limit, key_limit=plugin_config.impression_key_limit),
            memory.get_diary(query=search_query, recent_limit=plugin_config.diary_limit),
        )
    else:
        memories, impressions, diary_entries = await asyncio.gather(
            memory.search(query=search_query, user_id=user_id, limit=plugin_config.memory_search_limit),
            memory.get_impressions(user_id=user_id, recent_limit=plugin_config.impression_recent_limit, key_limit=plugin_config.impression_key_limit),
            memory.get_diary(query=search_query, recent_limit=plugin_config.diary_limit),
        )
    logger.info(f"[{_tid()}] 记忆检索 | dialog={len(memories)} impression={len(impressions)} diary={len(diary_entries)}")

    # fallback: 短消息向量搜索无结果时，用最近几条兜底
    if not memories:
        fallback = await (
            memory.recent(chat_id=chat_id, limit=3) if is_group
            else memory.recent(user_id=user_id, limit=3)
        )
        if fallback:
            memories = fallback
            logger.debug(f"[{_tid()}] 记忆 fallback | 向量无结果，取最近 {len(fallback)} 条")
    memory_context = format_memories(memories)
    impression_context = format_impressions(impressions)
    diary_context = format_diary(diary_entries)

    await ensure_schedule(diary_context=diary_context)
    schedule_context = format_schedule_context()

    rel_level = relationship.trust_level(user_id, plugin_config.owner_id)
    rel_context = relationship.format_context(user_id, user_name, plugin_config.owner_id)
    logger.debug(f"[{_tid()}] 关系 | user={user_name} level={rel_level} trust={relationship.get(user_id).trust:.1f}")
    if diary_context:
        logger.debug(f"[{_tid()}] 注入日记 |\n{diary_context}")
    if memory_context:
        logger.debug(f"[{_tid()}] 注入记忆 |\n{memory_context}")
    if impression_context:
        logger.debug(f"[{_tid()}] 注入印象 |\n{impression_context}")

    system, system_dynamic = build_system_prompt(
        user_id=user_id,
        owner_id=plugin_config.owner_id,
        is_group=is_group,
        memory_context=memory_context,
        impression_context=impression_context,
        relationship_context=rel_context,
        diary_context=diary_context,
        schedule_context=schedule_context,
    )

    if is_group:
        messages = build_group_turns(context=list(_group_context[group_id]), bot_name=plugin_config.bot_name)
    else:
        _push_private(user_id, "user", text)
        messages = list(_private_histories[user_id])

    logger.info(f"[{_tid()}] LLM → | user={user_id} msgs={len(messages)} group={is_group}")
    t0 = time.time()
    try:
        response = await chat(system=system, messages=messages, system_dynamic=system_dynamic)
        elapsed = time.time() - t0
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[{_tid()}] LLM ✗ | user={user_id} time={elapsed:.1f}s error={e}")
        logger.exception(e)
        return

    logger.info(f"[{_tid()}] LLM ← | user={user_id} len={len(response)} time={elapsed:.1f}s")

    # 私聊：48 可以选择不回复（日程系统驱动）
    if not is_group and "[不回复]" in response:
        logger.info(f"[{_tid()}] 48 选择不回复 | user={user_id}")
        _push_private(user_id, "assistant", "(已读不回)")
        return

    if is_group:
        _push_group(group_id, plugin_config.bot_name, response)
        _group_last_reply_time[group_id] = time.time()
        _group_attention[group_id] = min(_group_attention[group_id] + ATTENTION_REPLY_BOOST, 1.0)
    else:
        _push_private(user_id, "assistant", response)

    guard.record(user_id, text, response)

    # 异步评估 + 存入长期记忆，不阻塞发消息
    asyncio.create_task(_evaluate_and_store(
        user_id=user_id,
        chat_type="group" if is_group else "private",
        chat_id=chat_id,
        user_name=user_name,
        message=text,
        response=response,
    ))

    lines = [line.strip() for line in response.split("\n") if line.strip()]
    if not lines:
        return

    for line in lines[:-1]:
        await _send(bot, is_group, group_id, user_id, line)
        await _typing_delay(line)

    await _send(bot, is_group, group_id, user_id, lines[-1])


matcher = on_message(priority=99, block=False)


@matcher.handle()
async def handle(bot: Bot, event: MessageEvent):
    text = event.get_plaintext().strip()
    if not text:
        return

    user_id = str(event.user_id)
    is_group = isinstance(event, GroupMessageEvent)
    group_id = None

    if isinstance(event, PrivateMessageEvent):
        sender_name = event.sender.nickname or user_id
        logger.info(f"私聊 | {sender_name}({user_id}) {text[:40]!r}")
    elif is_group:
        group_id = str(event.group_id)

        if not _is_allowed_group(group_id):
            return

        sender_name = _get_sender_name(event)
        _push_group(group_id, sender_name, text)
        _group_msg_times[group_id].append(time.time())

        is_at = event.is_tome()
        att = _update_attention(group_id, event, bot.self_id)

        if not is_at:
            att *= _density_multiplier(group_id)

        if not _should_reply_group(att):
            logger.debug(f"群聊跳过 | {sender_name}@{group_id} att={att:.2f}")
            return

        # 回复冷却：说完话静默一段时间，被@除外
        if not is_at:
            elapsed = time.time() - _group_last_reply_time.get(group_id, 0)
            if elapsed < REPLY_COOLDOWN_SEC:
                logger.debug(f"回复冷却 | {group_id} 剩余{REPLY_COOLDOWN_SEC - elapsed:.0f}s")
                return

        logger.info(f"群聊回复 | {sender_name}@{group_id} att={att:.2f}")
    else:
        return

    is_owner = user_id == plugin_config.owner_id

    # 防抖：有新消息来就取消上一个待发任务，重新计时
    # 群聊：等所有人说完再回；私聊：等用户一口气说完再回
    key = f"group_{group_id}" if is_group else f"private_{user_id}"
    cancelling = key in _pending and not _pending[key].done()
    if cancelling:
        _pending[key].cancel()
        _burst_active.add(key)
        logger.info(f"防抖取消 | {key} → burst 冷静期")

    # 积累 burst 期间所有消息，防抖触发时拼合为 search_query，让记忆/日记检索覆盖整个 burst
    _burst_texts[key].append(text)

    delay = BURST_COOLDOWN_SEC if key in _burst_active else DEBOUNCE_SEC
    logger.info(f"防抖等待 | {key} delay={delay:.1f}s")

    async def debounced():
        current_task = asyncio.current_task()
        _trace_ctx.set(_new_tid())
        try:
            await asyncio.sleep(delay)
            # 取出并清空缓冲区（task 被取消时不会走到这里，缓冲继续积累）
            search_query = " ".join(_burst_texts.pop(key, [text]))
            logger.info(f"[{_tid()}] 防抖触发 | {key} → 调用 LLM query={search_query!r}")
            await _do_reply(bot, is_group, group_id, user_id, sender_name, text, is_owner, search_query=search_query)
        finally:
            # 只有自己还是当前任务时才清理，避免取消后覆盖后继任务的槽位
            if _pending.get(key) is current_task:
                _pending.pop(key, None)
                _burst_active.discard(key)

    _pending[key] = asyncio.create_task(debounced())
