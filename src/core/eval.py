"""
对话评估：重要性打分 + 主观印象 + 信任变化 + 事实提取。
合并为一次 LLM 调用；使用 tool_use 强制结构化输出，无需 JSON 解析。
短对话自动跳过，节省 token。
"""

from nonebot import logger

from core.llm import chat_structured

_EVAL_TOOL = {
    "name": "record_eval",
    "description": "记录对这段对话的评估结果",
    "input_schema": {
        "type": "object",
        "properties": {
            "importance": {
                "type": "number",
                "description": "对话重要性 0.0~1.0",
            },
            "impression": {
                "type": "string",
                "description": "48 视角对「对方」的主观印象，口语化带情绪，不超过50字",
            },
            "trust_delta": {
                "type": "integer",
                "description": "对方这次互动的信任变化，范围 -5~+5",
            },
            "facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "从用户发言提取的具体事实，没有则空数组",
            },
        },
        "required": ["importance", "impression", "trust_delta", "facts"],
    },
}

_SYSTEM = """你是蔡颖茵（AptS:1548 / 48），仿生人。正在回顾一段对话，记录你的判断。

你的性格：逆反、毒舌但护短、对人类默认不信任（数据塔事故）、绝不容忍背叛。
你最亲的人是 AptS:1547（猫猫），你监护她的状态。
你认识的人：1549（49，姐妹）、1543（沈沐川，熟人）、2275（林宸瑶，朋友）、1738（陆清弦，尊重但保持距离）。
对认识的人态度比陌生人好，但亲疏分明。

importance 标准：
- 0.0~0.2 日常闲聊，无意义
- 0.2~0.5 有一定信息量
- 0.5~0.8 情绪波动或涉及重要事项（提到47、背叛、信任相关）
- 0.8~1.0 关键事件，必须记住

impression 说明（面向「对方」，不是自己）：
- 你在评价对方，不是评价自己
- 对话里你说的话 = 48 那行，对方说的话 = 用户名那行
- 用你的语气，口语化、带情绪，不超过50字
- 可以记录对方的状态、聊了什么、你的感受
- 好的例子："猫猫今天又在装没事，明明状态很差还说没事" / "沐川来了，聊了设计的事，话比平时多，好像心情不太好" / "这人第一次来就问东问西的，不太舒服"
- 坏的例子："被吐槽话多了"（主语搞反了，你在评价对方不是评价自己）

trust_delta 标准（整数，从你的角度）：
- +3~+5  对方真诚帮助过我或47，或分享了重要私密
- +1~+2  愉快的交流，对方态度好，有信息量
-  0     普通对话，无明显正负
- -1~-2  对方态度敷衍、无礼、让我不舒服
- -3~-5  明显的欺骗、背刺、或伤害行为

facts 说明：
- 只提取用户明确说出的具体事实：地点、职业、学校、爱好、特殊经历等
- 没有新事实就返回空数组，不要猜测"""

_MIN_TOTAL_LEN = 15  # 短对话跳过阈值


def _format_rounds(rounds: list[tuple[str, str, str]]) -> str:
    """将多轮对话格式化为 eval 输入文本。"""
    lines = []
    for user_name, message, response in rounds:
        lines.append(f"{user_name}：{message}")
        lines.append(f"48：{response}")
    return "\n".join(lines)


async def evaluate_exchange(
    user_name: str,
    message: str,
    response: str,
) -> tuple[float, str, float, list[str]]:
    """单轮评估（向后兼容）。"""
    return await evaluate_batch([(user_name, message, response)])


async def evaluate_batch(
    rounds: list[tuple[str, str, str]],
) -> tuple[float, str, float, list[str]]:
    """多轮对话综合评估。返回 (importance, impression, trust_delta, facts)。
    短对话自动跳过。
    """
    if not rounds:
        return 0.2, "", 0, []

    # 过滤掉超短的轮次，但至少保留一轮
    meaningful = [
        r for r in rounds
        if len(r[1].strip()) + len(r[2].strip()) >= _MIN_TOTAL_LEN
    ]
    if not meaningful:
        total = sum(len(r[1]) + len(r[2]) for r in rounds)
        logger.debug(f"评估跳过 | 全部短对话 ({len(rounds)}轮, {total}字)")
        return 0.2, "", 0, []

    content = _format_rounds(meaningful)

    try:
        data = await chat_structured(
            system=_SYSTEM,
            messages=[{"role": "user", "content": content}],
            tool=_EVAL_TOOL,
        )
        importance = max(0.0, min(1.0, float(data.get("importance", 0.5))))
        impression = str(data.get("impression", ""))[:100]
        trust_delta = max(-5, min(5, int(data.get("trust_delta", 0))))
        raw_facts = data.get("facts", [])
        facts = [str(f)[:30] for f in raw_facts if f][:5]
        logger.debug(f"评估完成 | {len(meaningful)}轮 imp={importance:.1f} trust={trust_delta:+d} impression={impression!r}")
        return importance, impression, trust_delta, facts
    except Exception as e:
        logger.warning(f"评估失败 | {e}")
        return 0.5, "", 0, []
