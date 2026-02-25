"""
对话评估：重要性打分 + 主观印象生成。
合并为一次 LLM 调用，降低 API 开销。
"""

import json
import re

from nonebot import logger

from core.llm import chat

_SYSTEM = """你是 48，正在回顾一段对话，记录重要性、自己的感受、对对方的信任变化、以及学到的事实。

只输出 JSON，不要任何其他内容，不要代码块：
{"importance": 0.0到1.0的浮点数, "impression": "印象文本", "trust_delta": -5到5的整数, "facts": ["事实1", "事实2"]}

importance 标准：
- 0.0~0.2 日常闲聊，无意义
- 0.2~0.5 有一定信息量
- 0.5~0.8 情绪波动或涉及重要事项
- 0.8~1.0 关键事件，必须记住

impression 视角说明：
- 你是 48，正在评价"对方"，不是评价自己
- 对话里你说的话 = 48 那行，对方说的话 = 用户名那行
- 口语化、带情绪，不超过20字
- 好的例子："猫猫今天又在装没事" / "这家伙话好多烦死了" / "感觉他快撑不住了"
- 坏的例子："被吐槽话多了"（这是你说的话，写成了被说）/ "用户需要陪伴"（太分析性）

trust_delta 标准（从 48 的角度评估对方这次互动，必须是整数）：
- +3~+5  对方真诚帮助过我或 1547，或分享了重要私密，让我觉得值得信任
- +1~+2  愉快的交流，对方态度好，有信息量
-  0     普通对话，无明显正负
- -1~-2  对方态度敷衍、无礼、让我不舒服
- -3~-5  明显的欺骗、背刺、或伤害行为

facts 说明（从用户这条消息里提取，不是48说的话）：
- 只提取用户明确说出的具体事实：姓名、地点、职业、学校、爱好、特殊经历、偏好等
- 没有新事实就返回 []，不要猜测或补充
- 每条不超过15字，简洁
- 例子：["在北京读大学", "学计算机的", "喜欢玩崩铁"]"""

_USER_TEMPLATE = """{user_name}：{message}
48：{response}"""


def _extract_json(text: str) -> str:
    """剥掉可能的 markdown 代码块包装，提取 JSON 字符串。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def evaluate_exchange(user_name: str, message: str, response: str) -> tuple[float, str, float, list[str]]:
    """一次 LLM 调用，返回 (importance 0~1, impression 一句话印象, trust_delta -5~5, facts 事实列表)"""
    try:
        raw = await chat(
            system=_SYSTEM,
            messages=[{"role": "user", "content": _USER_TEMPLATE.format(
                user_name=user_name,
                message=message[:300],
                response=response[:300],
            )}],
            max_tokens=512,
            disable_thinking=True,
        )
        cleaned = _extract_json(raw)
        if not cleaned:
            logger.warning(f"评估失败 | LLM 返回空 | raw={raw!r}")
            return 0.5, "", 0, []
        data = json.loads(cleaned)
        importance = max(0.0, min(1.0, float(data.get("importance", 0.5))))
        impression = str(data.get("impression", ""))[:60]
        trust_delta = max(-5, min(5, int(data.get("trust_delta", 0))))
        raw_facts = data.get("facts", [])
        facts = [str(f)[:30] for f in raw_facts if f][:5]  # 最多5条，每条截到30字
        return importance, impression, trust_delta, facts
    except json.JSONDecodeError as e:
        logger.warning(f"评估失败 | JSON 解析错误: {e} | raw={raw!r}")
        return 0.5, "", 0, []
    except Exception as e:
        logger.warning(f"评估失败 | {e}")
        return 0.5, "", 0, []
