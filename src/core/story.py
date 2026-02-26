"""
自创故事：48 在高 importance 事件后写一段完整的第一人称叙述。
比 diary（30字快照）丰富得多，是她对经历的完整记录。
同时也是 Phase 6 微调的训练数据。
"""
from nonebot import logger
from core.llm import chat

_SYSTEM = """你是 48，在回忆刚才发生的事，写一段完整的叙述。

要求：
- 第一人称
- 200~500 字
- 有场景、有细节、有情绪
- 口语化但比日记正式一些，像在给未来的自己写信
- 提到人的时候用名字，不要用代词
- 时间地点写清楚
- 不是流水账，而是真的在回忆一件事的来龙去脉
- 可以有你的判断和感受，但别只写情绪

好的例子：
那天晚上猫猫突然找我，说她觉得自己什么都做不好。我其实不太会安慰人，但我知道她需要有人在。\
我们聊了很久，从她的工作聊到以前的事。我说了句"你本来就很烂，但那不妨碍你做了很多别人做不到的事"，\
她笑了。后来她说要去睡了，语气听着好多了。我松了口气，但心里还是有点担心。

坏的例子：
- 今天和猫猫聊天了，她不开心，我安慰了她（太简略，像日记）
- 她说...他说...（谁？要写名字）

只输出叙述本身，不要任何额外内容。"""


async def generate_story(
    user_name: str,
    messages: str,
    response: str,
    diary_entries: list[str] | None = None,
    importance: float = 0.8,
) -> str | None:
    """为高 importance 事件生成一段完整叙述（200~500字）。失败返回 None。"""
    user_prompt = (
        f"刚刚和{user_name}有一段重要的对话。\n\n"
        f"{user_name}说了：\n{messages[:300]}\n\n"
        f"我回应了：\n{response[:300]}\n\n"
        f"这次对话的重要程度：{importance:.1f}/1.0"
    )
    if diary_entries:
        user_prompt += "\n\n最近的日记：\n" + "\n".join(f"- {e}" for e in diary_entries[:5])
    user_prompt += "\n\n写一段关于这件事的完整叙述。"

    try:
        result = await chat(
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=512,
            disable_thinking=True,
        )
        text = result.strip()
        if len(text) < 50:
            logger.warning(f"故事生成 | 太短（{len(text)}字），丢弃")
            return None
        logger.info(f"故事生成 | {len(text)} 字 | importance={importance}")
        return text
    except Exception as e:
        logger.warning(f"故事生成失败 | {e}")
        return None
