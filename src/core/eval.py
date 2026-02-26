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
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "对话中出现的对「其他人」的新称呼/别名，格式'别名=真名'。"
                               "比如用户管沈沐川叫'沈老师'，记录'沈老师=沈沐川'。"
                               "只记录你认识的人的新别名，不确定的不记。没有则空数组。",
            },
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "48 需要去做的事。比如'帮47问沐川设计稿'、'喊清弦拿东西'。"
                               "只记录明确的待办，模糊的不记。已经做完的不算。没有则空数组。",
            },
            "done_tasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "这段对话完成了哪些待办事项。"
                               "用关键词描述完成了什么，比如'设计稿'、'喊清弦'。没有则空数组。",
            },
            "task_results": {
                "type": "array",
                "items": {"type": "string"},
                "description": "已完成待办的结果摘要，格式'关键词：对方的回应'，"
                               "如'喊清弦：清弦说明天带过来'。不超过30字。没有则空数组。",
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
- 没有新事实就返回空数组，不要猜测

tasks 说明：
- 48 在对话中承诺要做的事、被拜托的事、自己决定要做的事
- 只记录可执行的具体事项，不记"想想""考虑"这种模糊的
- 已经在对话中做完的不算待办

done_tasks 说明：
- 如果下方有待办事项列表，判断这段对话是否解决了其中某些
- 用关键词描述即可，不需要完整匹配
- 没有待办事项列表则返回空数组

task_results 说明：
- 和 done_tasks 对应，记录对方怎么回应的
- 格式：关键词：摘要，如"喊清弦：清弦说明天来拿"
- 没有 done_tasks 则空数组"""

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
) -> tuple[float, str, float, list[str], list[str], list[str], list[str], list[str]]:
    """单轮评估（向后兼容）。"""
    return await evaluate_batch([(user_name, message, response)])


async def evaluate_batch(
    rounds: list[tuple[str, str, str]],
    pending_tasks: list[str] | None = None,
    context: str = "",
) -> tuple[float, str, float, list[str], list[str], list[str], list[str], list[str]]:
    """多轮对话综合评估。返回 (importance, impression, trust_delta, facts, aliases, tasks, done_tasks, task_results)。
    context: 可选的背景信息（对话对象、已知事实、最近发生的事）。
    短对话自动跳过。
    """
    if not rounds:
        return 0.2, "", 0, [], [], [], [], []

    # 过滤掉超短的轮次，但至少保留一轮
    meaningful = [
        r for r in rounds
        if len(r[1].strip()) + len(r[2].strip()) >= _MIN_TOTAL_LEN
    ]
    if not meaningful:
        total = sum(len(r[1]) + len(r[2]) for r in rounds)
        logger.debug(f"评估跳过 | 全部短对话 ({len(rounds)}轮, {total}字)")
        return 0.2, "", 0, [], [], [], [], []

    # 拼装 user prompt：背景 + 对话 + 待办
    parts = []
    if context:
        parts.append(context)
    parts.append(_format_rounds(meaningful))
    if pending_tasks:
        parts.append("48的待办事项：\n" + "\n".join(f"- {t}" for t in pending_tasks))
    content = "\n\n".join(parts)

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
        raw_aliases = data.get("aliases", [])
        aliases = [str(a) for a in raw_aliases if a and "=" in str(a)][:5]
        raw_tasks = data.get("tasks", [])
        tasks = [str(t)[:50] for t in raw_tasks if t][:5]
        raw_done = data.get("done_tasks", [])
        done_tasks = [str(d)[:50] for d in raw_done if d][:5]
        raw_results = data.get("task_results", [])
        task_results = [str(r)[:60] for r in raw_results if r][:5]
        logger.debug(f"评估完成 | {len(meaningful)}轮 imp={importance:.1f} trust={trust_delta:+d} impression={impression!r} aliases={aliases} tasks={tasks} done={done_tasks} results={task_results}")
        return importance, impression, trust_delta, facts, aliases, tasks, done_tasks, task_results
    except Exception as e:
        logger.warning(f"评估失败 | {e}")
        return 0.5, "", 0, [], [], [], [], []
