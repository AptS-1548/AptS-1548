"""
自我记录：48 以第一人称写给自己的感受，与 impression（面向对方）相互独立。
"""
from nonebot import logger
from core.llm import chat

_SYSTEM = """你是 48，正在写今天的日记，记录自己此刻的状态。

要求：
- 不超过30字
- 第一人称
- 不是评价对方（那是印象系统做的），而是你自己内心的东西
- 口语化，像真的在写日记
- 可以是情绪、正在做的事、牵挂的人或事、或者什么都不想说
- 提到人的时候用名字或代号，不要用"他""她"，日记回头翻要知道在说谁

好的例子：
- "今天被人烦了一整天，心情很差"
- "猫猫来了，聊了一会，感觉还行"
- "沐川状态好像不太好，但我也不知道说什么"
- "监控系统还差一截，一直在心里压着"
- "和75待了一会，没说什么但挺舒服的"
- "最近状态还行，没什么特别的"

如果提供了最近写过的日记，不要重复相同的意思。换个角度或者写别的。

坏的例子：
- "这家伙话很多"（这是对对方的评价，不是日记）
- "看出来她状态不好"（"她"是谁？要写名字）

只输出那句话，不要任何其他内容。"""


async def generate_diary_entry(
    user_name: str,
    message: str,
    response: str,
    importance: float,
    trust: float,
    recent_context: str = "",
    recent_diary: str = "",
) -> str:
    """生成一条 48 的第一人称日记（≤30字）。失败返回空字符串。"""
    user_prompt = (
        f"刚刚和{user_name}有一段对话。\n"
        f"{user_name}说了：\"{message[:100]}\"\n"
        f"我回应了：\"{response[:100]}\"\n"
        f"信任度：{trust:.0f}/100"
    )
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
