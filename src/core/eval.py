"""
对话评估：重要性打分 + 主观印象生成。
合并为一次 LLM 调用，降低 API 开销。
"""

import json
import re

from nonebot import logger

from core.llm import chat

_SYSTEM = """你是 48，正在回顾一段对话，记录重要性和自己的感受。

只输出 JSON，不要任何其他内容，不要代码块：
{"importance": 0.0到1.0的浮点数, "impression": "印象文本"}

importance 标准：
- 0.0~0.2 日常闲聊，无意义
- 0.2~0.5 有一定信息量
- 0.5~0.8 情绪波动或涉及重要事项
- 0.8~1.0 关键事件，必须记住

impression 要求：
- 你的内心想法，不是客观描述
- 口语化、带情绪，不超过20字
- 好的例子："猫猫今天又在装没事" / "这小子状态不对得盯着" / "感觉他快撑不住了"
- 坏的例子："情感承诺时刻" / "用户需要陪伴"（太分析性）"""

_USER_TEMPLATE = """{user_name}：{message}
48：{response}"""


def _extract_json(text: str) -> str:
    """剥掉可能的 markdown 代码块包装，提取 JSON 字符串。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def evaluate_exchange(user_name: str, message: str, response: str) -> tuple[float, str]:
    """一次 LLM 调用，返回 (importance 0~1, impression 一句话印象)"""
    try:
        raw = await chat(
            system=_SYSTEM,
            messages=[{"role": "user", "content": _USER_TEMPLATE.format(
                user_name=user_name,
                message=message[:300],
                response=response[:300],
            )}],
            max_tokens=256,
            disable_thinking=True,
        )
        cleaned = _extract_json(raw)
        if not cleaned:
            logger.warning(f"评估失败 | LLM 返回空 | raw={raw!r}")
            return 0.5, ""
        data = json.loads(cleaned)
        importance = max(0.0, min(1.0, float(data.get("importance", 0.5))))
        impression = str(data.get("impression", ""))[:60]
        return importance, impression
    except json.JSONDecodeError as e:
        logger.warning(f"评估失败 | JSON 解析错误: {e} | raw={raw!r}")
        return 0.5, ""
    except Exception as e:
        logger.warning(f"评估失败 | {e}")
        return 0.5, ""
