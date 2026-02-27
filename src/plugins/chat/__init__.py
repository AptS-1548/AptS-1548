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

from core.llm import chat, set_api_call_hook
from core.prompt import build_system_prompt, build_group_turns, format_impressions, format_diary, format_stories
from core.guard import Guard
from core.memory import Memory, MemoryEntry, format_memories
from core.eval import evaluate_batch, EvalResult
from core.relationship import RelationshipManager
from core.summarize import generate_user_summary
from core.diary import generate_diary_entry
from core.schedule import ensure_schedule, get_willingness, format_schedule_context, schedule_daily_loop
from core.proactive import check_proactive, check_schedule_proactive, generate_schedule_message, generate_task_message, generate_followup_message
from core.graph import CharacterGraph
from core.task import TaskManager
from core.story import generate_story
from core.vision import describe_image

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
graph = CharacterGraph()
task_mgr = TaskManager(graph)

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

# ── 传话追踪：前一批 eval 触发了第三方日记，后续 eval 继续写日记 ──
_relay_active: dict[str, float] = {}  # user_id -> 触发时间戳
RELAY_WINDOW_SEC = 600  # 10 分钟内持续追踪

# ── 印象去重：同一用户/群 2 小时内只存一次印象（运行时 + 持久化通过 relationship）──
IMPRESSION_COOLDOWN_SEC = 2 * 3600

# ── follow-up：对话跟进（内存存储，不持久化）──
# user_id -> {context: str, due_at: float, user_name: str}
_pending_followups: dict[str, dict] = {}

# ── Trace ID（每次 LLM 调用一个，贯穿整条链路）──
_trace_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def _tid() -> str:
    return _trace_ctx.get()


def _new_tid() -> str:
    return format(random.randint(0, 0xFFFFFF), "06x")
DEBOUNCE_SEC = 1.5
BURST_COOLDOWN_SEC = 3.0  # 被取消后的冷静期，等用户说完

# ── 后台任务追踪（graceful shutdown 用）──
_background_tasks: list[asyncio.Task] = []


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
    # 注册 API 调用计数回调
    set_api_call_hook(guard.record_api_call)

    _background_tasks.extend([
        asyncio.create_task(_attention_decay_loop()),
        asyncio.create_task(_rebuild_context()),
        asyncio.create_task(memory.cleanup_old_entries()),
        asyncio.create_task(_init_schedule()),
        asyncio.create_task(schedule_daily_loop(_get_diary_context)),
        asyncio.create_task(_proactive_loop()),
        asyncio.create_task(_task_loop()),
    ])
    if plugin_config.diary_dedup_enabled:
        _background_tasks.append(asyncio.create_task(_diary_dedup_loop()))
    try:
        await graph.connect(
            url=plugin_config.surrealdb_url,
            username=plugin_config.surrealdb_user,
            password=plugin_config.surrealdb_password,
        )
    except Exception as e:
        logger.warning(f"图谱 | SurrealDB 连接失败，别名功能不可用: {e}")


@driver.on_shutdown
async def _shutdown():
    logger.info("正在关闭...")

    # 刷完所有待处理的 eval buffer
    for user_id in list(_eval_buffer.keys()):
        try:
            await _flush_eval(user_id)
        except Exception as e:
            logger.warning(f"关闭 | flush eval 失败 user={user_id}: {e}")

    # 关闭 SurrealDB 连接
    try:
        await graph.close()
    except Exception:
        pass

    # 取消后台任务
    for task in _background_tasks:
        if not task.done():
            task.cancel()

    replies, api = guard.daily_usage[0], guard.api_calls_today
    logger.info(f"关闭完成 | 今日回复={replies} API调用={api}")


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
    except Exception as e:
        logger.warning(f"日程初始化 | 日记获取失败: {e}")
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
    """后台循环：每天 0:30 自动执行日记语义去重 + 压缩。"""
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

        # 去重后压缩：把流水账合并成回忆
        try:
            consolidated = await memory.consolidate_diary(min_entries=8)
            if consolidated:
                logger.info(f"日记压缩 loop | 完成，压缩 {consolidated} 条")
        except Exception as e:
            logger.warning(f"日记压缩 loop | 失败: {e}")


async def _proactive_loop():
    """后台定时检查，触发 48 主动联系 owner 和朋友。"""
    await asyncio.sleep(300)  # 启动后等 5 分钟再开始
    logger.info(f"主动 loop 启动 | 间隔={plugin_config.proactive_check_sec}s 冷却={plugin_config.proactive_cooldown_sec}s")

    first_run = True
    while True:
        if first_run:
            first_run = False
        else:
            await asyncio.sleep(plugin_config.proactive_check_sec)
        try:
            bot: Bot = nonebot.get_bot()
        except ValueError:
            continue

        owner_id = plugin_config.owner_id

        # ── Part 0: 对话跟进 ──
        now = time.time()
        for uid in list(_pending_followups):
            fu = _pending_followups[uid]
            if now < fu["due_at"]:
                continue
            # 到期了
            del _pending_followups[uid]
            name = fu["user_name"]
            try:
                msg = await generate_followup_message(name, fu["context"])
                if msg:
                    await _send_proactive(bot, uid, msg, label=f"跟进→{name}")
            except Exception as e:
                logger.warning(f"跟进发送异常 | {name} {e}")

        # ── Part 1: 日程驱动 ──
        try:
            schedule_contacts = await check_schedule_proactive(relationship, owner_id, graph=graph)
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


async def _task_loop():
    """每 5 分钟扫描 pending tasks，主动联系事件对象。"""
    await asyncio.sleep(120)  # 启动等 2 分钟
    logger.info("待办 loop 启动 | 间隔=300s")

    first_run = True
    while True:
        if first_run:
            first_run = False
        else:
            await asyncio.sleep(300)
        try:
            bot: Bot = nonebot.get_bot()
        except ValueError:
            continue

        try:
            actionable = await task_mgr.get_actionable()
        except Exception as e:
            logger.warning(f"待办 loop 查询失败 | {e}")
            continue

        for task in actionable:
            target_qq = task.get("target_qq")
            content = task.get("content", "")
            source = task.get("source_user", "")
            task_id = str(task.get("id", ""))

            if not target_qq or not task_id:
                continue

            target_person = graph.qq_to_person(target_qq)
            target_name = target_person["name"] if target_person else target_qq

            try:
                msg = await generate_task_message(content, source, target_name)
            except Exception as e:
                logger.warning(f"待办 loop 消息生成失败 | {target_name} {e}")
                continue

            if msg:
                await _send_proactive(bot, target_qq, msg, label=f"待办→{target_name}")
                await task_mgr.mark_sent(task_id)


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
    mentioned_names: list[str] | None = None,
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
        diary_entries = await memory.get_diary(recent_limit=3)
        if diary_entries:
            recent_diary = "\n".join(f"- {e.message}" for e in diary_entries)
    except Exception as e:
        logger.warning(f"日记上下文获取失败 | {e}")

    entry_text = await generate_diary_entry(user_name, message, response, importance, trust, recent_context, recent_diary, mentioned_names)
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


async def _write_story_entry(
    user_name: str,
    rounds: list[dict],
    response_text: str,
    importance: float,
) -> None:
    """为高 importance 事件生成一段完整叙述并存储。"""
    messages = "\n".join(f"{user_name}: {r.get('message', '')[:80]}" for r in rounds)
    resp = response_text[:200] if response_text else ""

    try:
        diary_entries = await memory.get_diary(recent_limit=3)
        diary_texts = [e.message for e in diary_entries]
    except Exception as e:
        logger.warning(f"故事上下文获取失败 | {e}")
        diary_texts = []

    text = await generate_story(user_name, messages, resp, diary_texts, importance)
    if not text:
        return

    # 语义去重：避免同一事件生成重复故事
    if await memory.is_story_duplicate(text):
        logger.debug(f"故事跳过 | 语义重复: {text[:50]!r}")
        return

    await memory.store_story(text, importance=importance)


async def _evaluate_and_store(
    user_id: str, chat_type: str, chat_id: str,
    user_name: str, message: str, response: str,
):
    """将对话记录存入 Qdrant，eval 部分丢进 buffer 攒批处理。"""

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
        try:
            await _flush_eval(user_id)
        except Exception as e:
            logger.warning(f"eval timer flush 失败 | user={user_id}: {e}")

    _eval_timers[user_id] = asyncio.create_task(_timer())
    logger.debug(f"eval timer 启动 | user={user_id} timeout={EVAL_FLUSH_TIMEOUT}s")


async def _flush_eval(user_id: str):
    """从 buffer 取出所有轮次，一次性调 eval，处理结果。"""
    rounds_info = _eval_buffer.pop(user_id, [])
    _cancel_eval_timer(user_id)

    if not rounds_info:
        logger.debug(f"eval flush | user={user_id} buffer 为空，跳过")
        return

    logger.info(f"eval flush 开始 | user={user_id} {len(rounds_info)}轮")

    # 构造 eval 输入
    rounds = [(r["user_name"], r["message"], r["response"]) for r in rounds_info]
    user_name = rounds_info[-1]["user_name"]
    chat_type = rounds_info[-1]["chat_type"]
    chat_id = rounds_info[-1]["chat_id"]

    # 取 pending tasks 给 eval 判断完成情况
    try:
        pending_contents = [t["content"] for t in await task_mgr.get_pending()]
    except Exception as e:
        logger.debug(f"eval 待办列表获取失败 | {e}")
        pending_contents = []

    # 组装 eval 上下文：对话对象信息 + 已知事实 + 最近日记
    eval_context_parts = []
    p = relationship.get(user_id)
    level = relationship.trust_level(user_id, plugin_config.owner_id)
    eval_context_parts.append(f"对话对象：{user_name}（{level}，信任度{p.trust:.0f}）")
    if p.facts:
        eval_context_parts.append("已知信息：" + "、".join(p.facts[:5]))
    try:
        recent_diary = await memory.get_diary(recent_limit=5)
        if recent_diary:
            diary_lines = [f"- {e.message}" for e in recent_diary[:5]]
            eval_context_parts.append("48最近的状态：\n" + "\n".join(diary_lines))
    except Exception as e:
        logger.warning(f"eval 日记上下文获取失败 | {e}")
    eval_context = "\n".join(eval_context_parts)

    ev = await evaluate_batch(
        rounds,
        pending_tasks=pending_contents,
        context=eval_context,
    )

    # 短对话跳过：塞回 buffer，等下次有新消息时一起评估
    if not ev.impression and ev.importance <= 0.2 and ev.trust_delta == 0:
        _eval_buffer[user_id] = rounds_info + _eval_buffer.get(user_id, [])
        _reset_eval_timer(user_id)
        logger.debug(f"eval flush | user={user_name} 短对话，{len(rounds_info)}轮塞回 buffer")
        return

    # 新待办写入 + 解析 target
    if ev.tasks:
        for t in ev.tasks:
            try:
                task_id = await task_mgr.add(t, user_name, source_qq=user_id)
                # 从任务内容中解析目标人物
                if task_id and graph._db:
                    mentioned = await graph.find_in_text(t)
                    source_person = graph.qq_to_person(user_id)
                    source_pid = str(source_person.get("id", "")) if source_person else ""
                    targets = [m for m in mentioned if str(m.get("id", "")) != source_pid]
                    if targets and targets[0].get("qq_id"):
                        await task_mgr.set_target(task_id, targets[0]["qq_id"])
            except Exception as e:
                logger.warning(f"待办写入失败 | {e}")

    # 待办完成标记（附带结果摘要）
    if ev.done_tasks:
        for d in ev.done_tasks:
            try:
                result = ""
                if ev.task_results:
                    for r in ev.task_results:
                        if d in r and "：" in r:
                            result = r.split("：", 1)[1]
                            break
                await task_mgr.mark_done(d, result=result)
            except Exception as e:
                logger.warning(f"待办完成标记失败 | {e}")

    # 动态写入新别名到图谱
    if ev.aliases and graph._db:
        for alias_pair in ev.aliases:
            try:
                alias, real_name = alias_pair.split("=", 1)
                person = await graph.resolve_name(real_name.strip())
                if person:
                    person_id = str(person["id"]).replace("person:", "")
                    await graph.add_alias(person_id, alias.strip())
            except Exception as e:
                logger.debug(f"图谱 | 别名写入跳过: {alias_pair!r} {e}")

    # follow-up：对话跟进（只对 owner 和 friend 生效）
    if ev.follow_up and ev.follow_up_minutes > 0:
        rel_level_for_fu = relationship.trust_level(user_id, plugin_config.owner_id)
        if rel_level_for_fu in ("owner", "friend"):
            _pending_followups[user_id] = {
                "context": ev.follow_up,
                "due_at": time.time() + ev.follow_up_minutes * 60,
                "user_name": user_name,
            }
            logger.info(
                f"跟进登记 | user={user_name} minutes={ev.follow_up_minutes} "
                f"context={ev.follow_up!r}"
            )

    impression_key = f"{chat_type}_{chat_id}_{user_id}"
    cooldown_ok = (
        ev.importance >= 0.8
        or time.time() - relationship.get_impression_ts(impression_key) >= IMPRESSION_COOLDOWN_SEC
    )
    should_store_impression = ev.impression and ev.importance >= 0.4 and cooldown_ok

    logger.debug(
        f"eval flush | user={user_name} {len(rounds)}轮 imp={ev.importance:.2f} trust={ev.trust_delta:+} "
        f"facts={ev.facts} impression={ev.impression!r} "
        f"{'→ 存印象' if should_store_impression else '→ 跳过印象'}"
    )

    relationship.record_interaction(user_id, user_name, ev.importance, ev.trust_delta)
    if ev.facts:
        relationship.add_facts(user_id, ev.facts)

    if should_store_impression:
        relationship.set_impression_ts(impression_key, time.time())
        await memory.store(MemoryEntry(
            user_id=user_id,
            chat_type=chat_type,
            chat_id=chat_id,
            user_name=user_name,
            message=ev.impression,
            response="",
            importance=ev.importance,
            record_type="impression",
        ))

    p = relationship.get(user_id)
    should_summarize = (
        (p.interaction_count > 0 and p.interaction_count % 10 == 0)
        or (not p.summary and p.interaction_count >= 5)
        or abs(ev.trust_delta) >= 3
        or ev.importance >= 0.8
    )
    if should_summarize:
        asyncio.create_task(_update_user_summary(user_id, user_name))

    # diary
    last = rounds_info[-1]
    should_write_diary = ev.importance >= 0.6 or abs(ev.trust_delta) >= 3
    is_relay = False  # 标记：是否涉及第三方（传话场景）
    relay_names: list[str] = []  # 第三方人物名，传给 diary LLM 做代词消歧

    # 涉及第三方人物的对话也写日记（传话、讨论别人的事）
    if not should_write_diary and graph._db:
        try:
            # 也把 impression 拼进去搜，LLM 可能把代词还原成名字
            conv_text = " ".join(f"{r['message']} {r['response']}" for r in rounds_info)
            if ev.impression:
                conv_text += " " + ev.impression
            mentioned = await graph.find_in_text(conv_text)
            # 排除对话对象自己，只看是否提到了"第三方"
            talker = graph.qq_to_person(user_id)
            talker_id = str(talker.get("id", "")) if talker else ""
            third_party = [m for m in mentioned if str(m.get("id", "")) != talker_id]
            if third_party:
                should_write_diary = True
                is_relay = True
                _relay_active[user_id] = time.time()
                relay_names = [m["name"] for m in third_party]
                logger.debug(f"日记触发 | 对话涉及第三方: {relay_names}")
        except Exception as e:
            logger.debug(f"日记第三方检测失败 | {e}")

    # 传话追踪：前一批 eval 触发了第三方日记，后续 eval 继续写日记
    if not should_write_diary and user_id in _relay_active:
        if time.time() - _relay_active[user_id] < RELAY_WINDOW_SEC:
            should_write_diary = True
            is_relay = True
            _relay_active[user_id] = time.time()  # 续期
            logger.debug(f"日记触发 | 传话追踪延续 user={user_name}")
        else:
            del _relay_active[user_id]

    if should_write_diary:
        # 传话场景：把所有轮次拼起来给 diary LLM，让它能看到完整上下文写出具体内容
        if is_relay and len(rounds_info) > 1:
            full_msg = "\n".join(f"{r['user_name']}：{r['message']}\n48：{r['response']}" for r in rounds_info)
            full_resp = ""
        else:
            full_msg = last["message"]
            full_resp = last["response"]
        # 非 relay 场景也尝试检测提到的人物，帮日记 LLM 消歧代词
        if not relay_names and graph._db:
            try:
                all_text = full_msg + " " + full_resp
                if ev.impression:
                    all_text += " " + ev.impression
                all_mentioned = await graph.find_in_text(all_text)
                relay_names = [m["name"] for m in all_mentioned]
            except Exception as e:
                logger.debug(f"日记人物检测失败 | {e}")
        asyncio.create_task(_write_diary_entry(
            user_id, user_name, full_msg, full_resp, ev.importance, relay_names or None,
        ))

    # 高 importance 事件 → 写故事（比 diary 丰富得多的完整叙述）
    if ev.importance >= 0.8:
        story_resp = full_resp if should_write_diary else last["response"]
        asyncio.create_task(_write_story_entry(user_name, rounds_info, story_resp, ev.importance))


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

    # 图谱别名展开：用户提到"沈老师"，展开为"沈老师 沈沐川"，让向量搜索覆盖更多命中
    if graph._db:
        try:
            matched_persons = await graph.find_in_text(search_query)
            expand_names = []
            for p in matched_persons:
                # 把该人物的所有别名中、不在原文里的那些补上
                for alias in p.get("aliases", []):
                    if len(alias) >= 2 and alias not in search_query:
                        expand_names.append(alias)
            if expand_names:
                search_query = search_query + " " + " ".join(expand_names)
                logger.debug(f"[{_tid()}] 图谱展开 | 补充别名: {expand_names}")
        except Exception as e:
            logger.debug(f"[{_tid()}] 图谱展开跳过 | {e}")

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
    rel_level = relationship.trust_level(user_id, plugin_config.owner_id)
    allow_global_fallback = rel_level not in ("stranger", "enemy")

    if is_group:
        memories, impressions, diary_entries, story_entries = await asyncio.gather(
            memory.search(query=search_query, chat_id=chat_id, limit=plugin_config.memory_search_limit),
            memory.get_impressions(chat_id=chat_id, recent_limit=plugin_config.impression_recent_limit, key_limit=plugin_config.impression_key_limit),
            memory.get_diary(query=search_query, recent_limit=plugin_config.diary_limit),
            memory.search_stories(query=search_query, limit=plugin_config.story_search_limit),
        )
    else:
        memories, impressions, diary_entries, story_entries = await asyncio.gather(
            memory.search(query=search_query, user_id=user_id, limit=plugin_config.memory_search_limit, global_fallback=allow_global_fallback),
            memory.get_impressions(user_id=user_id, recent_limit=plugin_config.impression_recent_limit, key_limit=plugin_config.impression_key_limit),
            memory.get_diary(query=search_query, recent_limit=plugin_config.diary_limit),
            memory.search_stories(query=search_query, limit=plugin_config.story_search_limit),
        )
    logger.info(f"[{_tid()}] 记忆检索 | dialog={len(memories)} impression={len(impressions)} diary={len(diary_entries)} story={len(story_entries)}")

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
    story_context = format_stories(story_entries)

    await ensure_schedule(diary_context=diary_context)
    schedule_context = format_schedule_context()

    # 待办事项
    task_context = await task_mgr.format_for_prompt(current_user=user_name, current_qq=user_id)

    rel_context = relationship.format_context(user_id, user_name, plugin_config.owner_id)
    logger.debug(f"[{_tid()}] 关系 | user={user_name} level={rel_level} trust={relationship.get(user_id).trust:.1f}")
    if diary_context:
        logger.debug(f"[{_tid()}] 注入日记 |\n{diary_context}")
    if memory_context:
        logger.debug(f"[{_tid()}] 注入记忆 |\n{memory_context}")
    if impression_context:
        logger.debug(f"[{_tid()}] 注入印象 |\n{impression_context}")
    if story_context:
        logger.debug(f"[{_tid()}] 注入故事 |\n{story_context}")
    if task_context:
        logger.debug(f"[{_tid()}] 注入待办 |\n{task_context}")

    system, system_dynamic = build_system_prompt(
        user_id=user_id,
        owner_id=plugin_config.owner_id,
        is_group=is_group,
        memory_context=memory_context,
        impression_context=impression_context,
        relationship_context=rel_context,
        diary_context=diary_context,
        schedule_context=schedule_context,
        task_context=task_context,
        story_context=story_context,
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
        # 私聊历史里已经 push 了 user 消息，补一条失败标记保持配对
        if not is_group:
            _push_private(user_id, "assistant", "(连接失败，没回上)")
        return

    logger.info(f"[{_tid()}] LLM ← | user={user_id} len={len(response)} time={elapsed:.1f}s")

    # 空响应（thinking 泄露过滤后为空、LLM 异常等）→ 当作不回复处理
    if not response.strip():
        logger.warning(f"[{_tid()}] 空响应 | user={user_id}，视为不回复")
        if not is_group:
            _push_private(user_id, "assistant", "(已读不回)")
        return

    # 48 可以选择不回复
    if "[不回复]" in response:
        logger.info(f"[{_tid()}] 48 选择不回复 | user={user_id}")
        if not is_group:
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
    logger.info(f"[{_tid()}] 发送完成 | user={user_id} lines={len(lines)}")


matcher = on_message(priority=99, block=False)


@matcher.handle()
async def handle(bot: Bot, event: MessageEvent):
    text = event.get_plaintext().strip()

    # 提取图片段，识别内容（有缓存的直接查 dict）
    for seg in event.message:
        if seg.type == "image":
            file_key = seg.data.get("file", "")
            url = seg.data.get("url", "")
            sub_type = int(seg.data.get("sub_type", 0))
            if file_key and url:
                is_sticker = sub_type == 1
                desc = await describe_image(url, file_key, is_sticker=is_sticker)
                tag = "表情包" if is_sticker else "图片"
                text = f"{text} [{tag}：{desc}]".strip()

    if not text:
        return

    user_id = str(event.user_id)
    is_group = isinstance(event, GroupMessageEvent)
    group_id = None

    if isinstance(event, PrivateMessageEvent):
        sender_name = event.sender.nickname or user_id
        logger.info(f"私聊 | {sender_name}({user_id}) {text[:40]!r}")

        # 私聊意愿门控：睡觉/忙时有概率不处理
        # owner 只有深度睡眠时极小概率不响应，其他人按意愿度判定
        w = get_willingness()
        if user_id == plugin_config.owner_id:
            if w <= 0.1 and random.random() < 0.1:
                logger.debug(f"私聊跳过 | owner 深度睡眠 willingness={w:.2f}")
                return
        elif w < 0.5 and random.random() > max(w, 0.05):
            logger.debug(f"私聊跳过 | {sender_name}({user_id}) willingness={w:.2f}")
            return

        # 对方主动发消息了，取消跟进（不需要再追问）
        if user_id in _pending_followups:
            logger.debug(f"跟进取消 | user={sender_name} 对方已主动发消息")
            del _pending_followups[user_id]
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
