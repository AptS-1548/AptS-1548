import anthropic
from nonebot import get_driver

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
    for block in response.content:
        if block.type == "text":
            return block.text

    return response.content[-1].text
