"""
用户关系摘要：周期性生成 48 对某人的认知总结，持久化到 UserProfile.summary。
每 10 次互动触发一次，用于跨对话保留对这个人的长期认知。
"""

from nonebot import logger

from core.llm import chat
from core.memory import MemoryEntry

_SYSTEM = """你是 48，正在整理对某人的认知，写一句话总结。

要求：
- 不超过50字
- 第一人称，带你自己的情绪和判断
- 涵盖：这人是什么样的人、你们关系如何、有什么值得记住的
- 口语化，不要分析

例子：
- "猫猫，我创造者，话少但状态经常拉垮，需要我时不时盯着他"
- "这人来了好几次，聊技术挺靠谱的，但我还不太了解他这个人"
- "老是来找我帮忙的陌生人，没什么交情，也没什么坏心思"

只输出那句话，不要任何其他内容。"""


async def generate_user_summary(
    user_name: str,
    impressions: list[MemoryEntry],
    trust: float,
    interaction_count: int,
) -> str:
    """生成对某用户的一句话摘要。返回空字符串表示失败或条件不足。"""
    if not impressions and interaction_count < 5:
        return ""

    lines = "\n".join(f"- {m.message[:60]}" for m in impressions[:10])
    user_prompt = f"用户：{user_name}\n信任度：{trust:.0f}/100，共互动 {interaction_count} 次"
    if lines:
        user_prompt += f"\n\n过往印象：\n{lines}"

    try:
        result = await chat(
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=128,
            disable_thinking=True,
        )
        summary = result.strip()[:100]
        logger.debug(f"摘要生成 | {user_name}: {summary!r}")
        return summary
    except Exception as e:
        logger.warning(f"摘要生成失败 | {user_name}: {e}")
        return ""
