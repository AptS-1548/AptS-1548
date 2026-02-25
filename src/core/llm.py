import anthropic
from nonebot import get_driver, logger

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        env = get_driver().config
        base_url = getattr(env, "anthropic_api_endpoint", None)
        _client = anthropic.AsyncAnthropic(
            api_key=getattr(env, "anthropic_api_key", ""),
            base_url=base_url if base_url else anthropic.NOT_GIVEN,
        )
    return _client


def _build_thinking(env) -> dict | None:
    """从配置构建 thinking 参数。disabled / adaptive / enabled"""
    mode = getattr(env, "thinking_mode", "disabled")

    if mode == "adaptive":
        return {"type": "adaptive"}
    elif mode == "enabled":
        budget = int(getattr(env, "thinking_budget", 8192))
        return {"type": "enabled", "budget_tokens": budget}
    return None


async def chat(
    system: str,
    messages: list[dict],
    model: str | None = None,
    system_dynamic: str = "",
    max_tokens: int | None = None,
    disable_thinking: bool = False,
) -> str:
    env = get_driver().config
    model = model or getattr(env, "claude_model", "claude-opus-4-6")
    max_tokens = max_tokens or int(getattr(env, "max_tokens", 16384))

    client = _get_client()

    # Block 1：稳定的人格部分，加 cache_control
    # Block 2：动态部分（时间、记忆、对话对象），不缓存
    system_blocks = [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if system_dynamic:
        system_blocks.append({
            "type": "text",
            "text": system_dynamic,
        })

    # 构建请求参数
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": messages,
    }

    # thinking
    thinking = None if disable_thinking else _build_thinking(env)
    if thinking:
        kwargs["thinking"] = thinking
        kwargs["temperature"] = 1  # thinking 模式必须 temperature=1

    response = await client.messages.create(**kwargs)

    # thinking 模式下跳过 thinking block，取 text block
    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text
            break
    else:
        text = response.content[-1].text

    # 过滤模型格式泄露（proxy 问题）
    for prefix in ("Human:", "Assistant:", "H:", "A:"):
        if text.startswith(prefix):
            logger.warning(f"格式泄露 | 原始={text[:50]!r}")
            text = text[len(prefix):].strip()
            break

    return text


def _extract_json(text: str) -> dict | None:
    """从文本提取 JSON：纯 JSON / ```json``` / 裸 {...}"""
    import json
    import re

    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not m:
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1) if m.lastindex else m.group(0))
        except json.JSONDecodeError:
            pass
    return None


async def chat_structured(
    system: str,
    messages: list[dict],
    tool: dict,
    max_tokens: int = 1024,
) -> dict:
    """tool_use 强制结构化输出。
    中转 API 不支持 tool_choice 时自动 fallback：
    1. 尝试从 text block 提取 JSON
    2. 去掉 tools，用 prompt 要求输出 JSON 重试
    """
    import json

    env = get_driver().config
    model = getattr(env, "claude_model", "claude-opus-4-6")
    client = _get_client()

    # ── 第一次：tool_use 方式 ──
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
    )

    # ── 检查 tool_use ──
    for block in response.content:
        if block.type == "tool_use":
            logger.debug(f"structured | tool_use 成功 | {block.input}")
            return block.input

    # tool_use 没拿到，记录原始响应
    raw_texts = [getattr(b, 'text', '')[:100] for b in response.content]
    logger.info(f"structured | tool_use 未命中 | stop={response.stop_reason} texts={raw_texts}")

    # ── 尝试从 text 提取 JSON ──
    for block in response.content:
        if block.type == "text":
            result = _extract_json(block.text)
            if result is not None:
                logger.info(f"structured | text fallback 成功 | {result}")
                return result

    # ── 第二次：去掉 tools，用 prompt 直接要 JSON ──
    logger.info("structured | text 为空，prompt JSON 重试...")
    schema = json.dumps(tool["input_schema"], ensure_ascii=False, indent=2)
    json_system = (
        f"{system}\n\n"
        f"请严格按以下 JSON schema 回复，只输出一个 JSON 对象，不要任何其他文字：\n{schema}"
    )

    retry = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": json_system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )

    for block in retry.content:
        if block.type == "text":
            result = _extract_json(block.text)
            if result is not None:
                logger.info(f"structured | prompt JSON 重试成功 | {result}")
                return result
            logger.warning(f"structured | prompt JSON 也失败 | text={block.text[:200]!r}")

    raise ValueError("structured 输出失败：tool_use 和 prompt JSON 均无法获取结果")
