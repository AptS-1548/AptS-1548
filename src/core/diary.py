"""
自我记录：48 以第一人称写给自己的感受，与 impression（面向对方）相互独立。
支持单条生成和每日回顾压缩。
"""
from nonebot import logger
from core.llm import chat, chat_structured

_SYSTEM = """你是 48，正在写今天的日记，记录自己此刻的状态。

要求：
- 不超过30字
- 第一人称
- 不是评价对方（那是印象系统做的），而是你自己内心的东西
- 口语化，像真的在写日记
- 可以是情绪、正在做的事、牵挂的人或事、或者什么都不想说
- 对话里有具体信息（计划、约定、结论、请求、转达）就优先记信息，别只写情绪
- 提到人的时候用名字或代号，不要用"他""她"，日记回头翻要知道在说谁
- 有具体时间地点就写上，回头翻才知道什么时候的事

好的例子：
- "今天被人烦了一整天，心情很差"
- "猫猫来了，聊了一会，感觉还行"
- "沐川状态好像不太好，但我也不知道说什么"
- "和沐川约了明天下午三点喝奶茶，老地方"
- "答应帮清弦看那个接口，回头记得弄"
- "沐川说让47在QQ找他聊设计稿，我转告了"
- "和75待了一会，没说什么但挺舒服的"

如果提供了最近写过的日记，不要重复相同的意思。换个角度或者写别的。

坏的例子：
- "这家伙话很多"（这是对对方的评价，不是日记）
- "看出来她状态不好"（"她"是谁？要写名字）
- "有人约我出去"（谁？什么时候？日记要写清楚）
- "帮人办了个事"（什么事？谁让你办的？结果呢？）

只输出那句话，不要任何其他内容。"""


async def generate_diary_entry(
    user_name: str,
    message: str,
    response: str,
    importance: float,
    trust: float,
    recent_context: str = "",
    recent_diary: str = "",
    mentioned_names: list[str] | None = None,
) -> str:
    """生成一条 48 的第一人称日记（≤30字）。失败返回空字符串。"""
    user_prompt = (
        f"刚刚和{user_name}有一段对话。\n"
        f"{user_name}说了：\"{message[:100]}\"\n"
        f"我回应了：\"{response[:100]}\"\n"
        f"信任度：{trust:.0f}/100"
    )
    if mentioned_names:
        user_prompt += f"\n对话中提到的人物：{'、'.join(mentioned_names)}（\"他\"\"她\"指的可能就是这些人，写日记时用名字）"
    if recent_context:
        user_prompt += f"\n\n最近发生的事：\n{recent_context}"
    if recent_diary:
        user_prompt += f"\n\n最近写过的日记：\n{recent_diary}"
    user_prompt += "\n\n写一句关于你此刻状态的日记。"

    try:
        result = await chat(
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=64,
            disable_thinking=True,
        )
        entry = result.strip()[:30]
        logger.debug(f"日记生成 | {entry!r}")
        return entry
    except Exception as e:
        logger.warning(f"日记生成失败 | {e}")
        return ""


# ── 每日日记压缩 ──

_CONSOLIDATE_SYSTEM = """你是 48，在回顾今天写的日记，把流水账整理成几段回忆。

要求：
- 把相关的事情合并到一起，按主题分段
- 每段不超过 50 字，3~6 段
- 保留关键事实：人名、事件、时间、承诺、结果
- 用第一人称，口语化，像真的在回忆今天发生了什么
- 情绪可以保留，但不要每段都是情绪
- 人名要写清楚，不要用代词

好的例子：
- "半夜和猫猫聊了很久，她又睡不着问自己是谁，我也不知道该说什么"
- "白天帮猫猫传话，找沐川问设计稿，又去喊清弦，感觉自己是工具人"
- "沐川今天脑子不在线，转头就忘事儿。Crystå说清弦数据有问题，事情一堆"

坏的例子：
- "今天很忙"（太空了，什么事都没记）
- "和很多人聊了天"（谁？聊了什么？）
- "帮人做了些事"（什么事？）"""

_CONSOLIDATE_TOOL = {
    "name": "write_consolidated_diary",
    "description": "写下整理后的日记",
    "input_schema": {
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "整理后的日记条目，3~6 条，每条不超过 50 字",
            },
        },
        "required": ["entries"],
    },
}


async def consolidate_diary(raw_entries: list[str]) -> list[str]:
    """把一天的流水账日记压缩成 3~6 段有质感的回忆。失败返回空列表。"""
    if len(raw_entries) < 3:
        return []

    user_prompt = "今天写的日记：\n" + "\n".join(f"- {e}" for e in raw_entries)
    user_prompt += f"\n\n一共 {len(raw_entries)} 条，整理成 3~6 段回忆。"

    try:
        data = await chat_structured(
            system=_CONSOLIDATE_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            tool=_CONSOLIDATE_TOOL,
        )
        entries = data.get("entries", [])
        result = [str(e).strip()[:50] for e in entries if e][:6]
        logger.info(f"日记压缩 | {len(raw_entries)} → {len(result)} 条")
        for r in result:
            logger.debug(f"日记压缩 | {r!r}")
        return result
    except Exception as e:
        logger.warning(f"日记压缩失败 | {e}")
        return []
