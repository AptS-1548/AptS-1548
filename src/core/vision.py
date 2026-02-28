"""图片识别与缓存：首次见到的图片/表情包用 vision 识别，缓存描述供后续复用。

缓存键 = OneBot file 字段（MD5 哈希），同一张图只识别一次。
识别失败的图片短期跳过（10 分钟内不重试），避免反复打 API。
"""

import asyncio
import base64
import json
import os
import time

import httpx
from nonebot import get_driver, logger

CACHE_PATH = "data/image_cache.json"
DOWNLOAD_TIMEOUT = 15
VISION_TIMEOUT = 30
FAIL_COOLDOWN = 600  # 识别失败后 10 分钟内不重试

_cache: dict[str, str] = {}
_fail_ts: dict[str, float] = {}  # file_key -> 上次失败时间（内存，不持久化）


def _load_cache():
    global _cache
    if not os.path.exists(CACHE_PATH):
        return
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            _cache = json.load(f)
        logger.info(f"图片缓存 | 加载 {len(_cache)} 条")
    except Exception as e:
        logger.warning(f"图片缓存 | 加载失败: {e}")


def _save_cache():
    try:
        os.makedirs(os.path.dirname(os.path.abspath(CACHE_PATH)), exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False)
        os.replace(tmp, CACHE_PATH)
    except Exception as e:
        logger.warning(f"图片缓存 | 保存失败: {e}")


_load_cache()


def _guess_media_type(file_key: str) -> str:
    lower = file_key.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


async def describe_image(url: str, file_key: str, is_sticker: bool = False, summary: str = "") -> str:
    """识别图片内容。有缓存直接返回，否则下载 + vision 识别后缓存。
    summary: OneBot 提供的图片摘要，作为识别失败时的 fallback。
    """
    if file_key in _cache:
        logger.debug(f"图片缓存命中 | {file_key[:16]}… → {_cache[file_key]!r}")
        return _cache[file_key]

    fallback = summary or ("表情" if is_sticker else "图片")

    # 短期失败冷却：同一张图 10 分钟内不重试
    if file_key in _fail_ts and time.time() - _fail_ts[file_key] < FAIL_COOLDOWN:
        return fallback

    # 下载
    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            image_data = resp.content
    except Exception as e:
        logger.warning(f"图片下载失败 | {file_key[:16]}…: {type(e).__name__}: {e}")
        _fail_ts[file_key] = time.time()
        return fallback

    media_type = _guess_media_type(file_key)
    b64 = base64.b64encode(image_data).decode()

    # Vision 识别
    from core.llm import _get_client, _notify_api_call

    env = get_driver().config
    model = getattr(env, "claude_model", "claude-sonnet-4-6")

    if is_sticker:
        prompt = "用中文简短描述这个表情包/贴图的意思和情绪，10字以内，只输出描述。如果看不到图片，只回复：[无图片]"
    else:
        prompt = "用中文简短描述这张图片的内容，20字以内，只输出描述。如果看不到图片，只回复：[无图片]"

    try:
        response = await asyncio.wait_for(
            _get_client().messages.create(
                model=model,
                max_tokens=50,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            ),
            timeout=VISION_TIMEOUT,
        )
        _notify_api_call()
        desc = response.content[0].text.strip()

        # 检测无效响应（模型看不到图片）
        if "[无图片]" in desc or "请提供图片" in desc or "无法看到" in desc:
            logger.warning(f"图片识别 | {file_key[:16]}… 模型未收到图片，使用 fallback")
            _fail_ts[file_key] = time.time()
            return fallback
    except Exception as e:
        logger.warning(f"图片识别失败 | {file_key[:16]}…: {type(e).__name__}: {e}")
        _fail_ts[file_key] = time.time()
        return fallback

    # 缓存
    _cache[file_key] = desc
    _fail_ts.pop(file_key, None)  # 成功了就清掉失败记录
    _save_cache()
    logger.info(f"图片识别 | {file_key[:16]}… → {desc!r} sticker={is_sticker}")
    return desc
