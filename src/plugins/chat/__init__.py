import asyncio
import contextvars
import datetime
import random
import time
from collections import defaultdict, deque

from nonebot import on_message, get_driver, get_plugin_config, logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    MessageEvent,
    GroupMessageEvent,
    PrivateMessageEvent,
    Message,
)

from core.llm import chat
from core.prompt import build_system_prompt, build_group_turns
from core.guard import Guard
from core.memory import Memory, MemoryEntry, format_memories

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

# ── 私聊：多轮对话历史 ──
_private_histories: dict[str, list[dict]] = defaultdict(list)
MAX_PRIVATE_HISTORY = 20

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

REPLY_COOLDOWN_SEC = 90   # 回复后沉默 90 秒（被@除外）
DENSITY_WINDOW_SEC = 20   # 密度检测窗口
DENSITY_THRESHOLD = 6     # 20 秒内超过 6 条算太吵
DENSITY_MULT_MIN = 0.2    # 再吵也最多压到 0.2 倍

# ── 防抖：避免连续消息触发多次 LLM ──
_pending: dict[str, asyncio.Task] = {}
_burst_active: set[str] = set()  # 被取消过、正在冷静的会话
_rebuilt_private_users: set[str] = set()  # 已重建过私聊历史的用户

# ── Trace ID（每次 LLM 调用一个，贯穿整条链路）──
_trace_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def _tid() -> str:
    return _trace_ctx.get()


def _new_tid() -> str:
    return format(random.randint(0, 0xFFFFFF), "06x")
DEBOUNCE_SEC = 1.5
BURST_COOLDOWN_SEC = 3.0  # 被取消后的冷静期，等用户说完


def _update_attention(group_id: str, event: GroupMessageEvent) -> float:
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
    if _is_reply_to_me(event):
        att += ATTENTION_QUOTED

    last_reply = _group_last_reply_time.get(group_id, 0)
    if now - last_reply < ATTENTION_IDLE_SEC:
        att += ATTENTION_REPLY_BOOST

    att = max(0.0, min(att, 1.0))
    _group_attention[group_id] = att
    logger.debug(f"注意力 | {group_id} att={att:.2f}")
    return att


def _is_reply_to_me(event: GroupMessageEvent) -> bool:
    try:
        for seg in event.message:
            if seg.type == "reply":
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
    """根据当前时段返回注意力乘数"""
    hour = datetime.datetime.now().hour
    if hour < 7:    return 0.2   # 深夜，基本不回
    if hour < 9:    return 0.5   # 早上，慢慢醒
    if hour < 18:   return 0.8   # 白天
    if hour < 23:   return 1.0   # 傍晚，最活跃
    return 0.5                   # 深夜前，开始懒了


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


async def _rebuild_context():
    """从 Qdrant 拉最近对话，重建群聊上下文，降低重启后断层感。"""
    for group_id in plugin_config.allowed_groups:
        try:
            entries = await memory.recent(chat_id=group_id, limit=MAX_GROUP_CONTEXT // 2)
            for e in entries:
                _push_group(group_id, e.user_name, e.message)
                _push_group(group_id, "48", e.response)
            if entries:
                logger.info(f"重建上下文 | 群 {group_id} 载入 {len(entries)} 条")
        except Exception as ex:
            logger.warning(f"重建上下文失败 | 群 {group_id} {ex}")


def _get_sender_name(event: GroupMessageEvent) -> str:
    sender = event.sender
    return sender.card or sender.nickname or str(event.user_id)


def _push_private(user_id: str, role: str, content: str):
    _private_histories[user_id].append({"role": role, "content": content})
    if len(_private_histories[user_id]) > MAX_PRIVATE_HISTORY:
        _private_histories[user_id] = _private_histories[user_id][-MAX_PRIVATE_HISTORY:]


def _push_group(group_id: str, name: str, content: str):
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


async def _do_reply(bot: Bot, is_group: bool, group_id: str | None, user_id: str, user_name: str, text: str, is_owner: bool):
    """调用 LLM 并发送回复"""
    allowed, reject_msg = guard.check_rate(user_id, is_owner=is_owner)
    if not allowed:
        logger.warning(f"速率限制 | user={user_id} msg={reject_msg}")
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
                _push_private(user_id, "user", e.message)
                _push_private(user_id, "assistant", e.response)
            if entries:
                logger.info(f"重建私聊历史 | user={user_id} 载入 {len(entries)} 条")
        except Exception as ex:
            logger.warning(f"重建私聊历史失败 | user={user_id} {ex}")

    # 检索相关历史记忆
    # 群聊按群隔离，私聊按人检索（群聊+私聊全捞）
    if is_group:
        memories = await memory.search(query=text, chat_id=chat_id, limit=4)
    else:
        memories = await memory.search(query=text, user_id=user_id, limit=4)
    logger.info(f"[{_tid()}] 记忆检索 | {len(memories)} 条 chat_id={chat_id}")
    memory_context = format_memories(memories)
    if memory_context:
        logger.debug(f"注入记忆 |\n{memory_context}")

    system, system_dynamic = build_system_prompt(
        user_id=user_id,
        owner_id=plugin_config.owner_id,
        is_group=is_group,
        memory_context=memory_context,
    )

    if is_group:
        messages = build_group_turns(context=list(_group_context[group_id]), bot_name=plugin_config.bot_name)
    else:
        messages = list(_private_histories[user_id])
        messages.append({"role": "user", "content": text})

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

    if is_group:
        _push_group(group_id, plugin_config.bot_name, response)
        _group_last_reply_time[group_id] = time.time()
        _group_attention[group_id] = min(_group_attention[group_id] + ATTENTION_REPLY_BOOST, 1.0)
    else:
        _push_private(user_id, "user", text)
        _push_private(user_id, "assistant", response)

    guard.record(user_id, text, response)

    # 异步存入长期记忆，不阻塞发消息
    asyncio.create_task(memory.store(MemoryEntry(
        user_id=user_id,
        chat_type="group" if is_group else "private",
        chat_id=chat_id,
        user_name=user_name,
        message=text,
        response=response,
    )))

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
        att = _update_attention(group_id, event)

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

    delay = BURST_COOLDOWN_SEC if key in _burst_active else DEBOUNCE_SEC
    logger.info(f"防抖等待 | {key} delay={delay:.1f}s")

    async def debounced():
        current_task = asyncio.current_task()
        _trace_ctx.set(_new_tid())
        try:
            await asyncio.sleep(delay)
            logger.info(f"[{_tid()}] 防抖触发 | {key} → 调用 LLM")
            await _do_reply(bot, is_group, group_id, user_id, sender_name, text, is_owner)
        finally:
            # 只有自己还是当前任务时才清理，避免取消后覆盖后继任务的槽位
            if _pending.get(key) is current_task:
                _pending.pop(key, None)
                _burst_active.discard(key)

    _pending[key] = asyncio.create_task(debounced())
