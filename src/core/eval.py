"""
对话评估：重要性打分 + 主观印象 + 信任变化 + 事实提取。
合并为一次 LLM 调用；使用 tool_use 强制结构化输出，无需 JSON 解析。
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
                "description": "48 视角对「对方」的主观印象，口语化带情绪，不超过20字",
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

_SYSTEM = """你是 48，正在回顾一段对话，记录重要性、自己对对方的感受、信任变化、以及学到的事实。

importance 标准：
- 0.0~0.2 日常闲聊，无意义
- 0.2~0.5 有一定信息量
- 0.5~0.8 情绪波动或涉及重要事项
- 0.8~1.0 关键事件，必须记住

impression 说明（面向「对方」，不是自己）：
- 你是 48，正在评价对方，不是评价自己
- 对话里你说的话 = 48 那行，对方说的话 = 用户名那行
- 口语化、带情绪，不超过20字
- 好的例子："猫猫今天又在装没事" / "这家伙话好多烦死了"
- 坏的例子："被吐槽话多了"（这是你说的话，写成了被说）

trust_delta 标准（整数，从 48 的角度）：
- +3~+5  对方真诚帮助过我或 1547，或分享了重要私密
- +1~+2  愉快的交流，对方态度好，有信息量
-  0     普通对话，无明显正负
- -1~-2  对方态度敷衍、无礼、让我不舒服
- -3~-5  明显的欺骗、背刺、或伤害行为

facts 说明：
- 只提取用户明确说出的具体事实：地点、职业、学校、爱好、特殊经历等
- 没有新事实就返回空数组，不要猜测"""

_USER_TEMPLATE = """{user_name}：{message}
48：{response}"""


async def evaluate_exchange(user_name: str, message: str, response: str) -> tuple[float, str, float, list[str]]:
    """一次 LLM 调用（tool_use），返回 (importance, impression, trust_delta, facts)"""
    try:
        data = await chat_structured(
            system=_SYSTEM,
            messages=[{"role": "user", "content": _USER_TEMPLATE.format(
                user_name=user_name,
                message=message[:300],
                response=response[:300],
            )}],
            tool=_EVAL_TOOL,
        )
        importance = max(0.0, min(1.0, float(data.get("importance", 0.5))))
        impression = str(data.get("impression", ""))[:60]
        trust_delta = max(-5, min(5, int(data.get("trust_delta", 0))))
        raw_facts = data.get("facts", [])
        facts = [str(f)[:30] for f in raw_facts if f][:5]
        return importance, impression, trust_delta, facts
    except Exception as e:
        logger.warning(f"评估失败 | {e}")
        return 0.5, "", 0, []
