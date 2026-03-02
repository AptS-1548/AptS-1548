# ── Stage 1: 安装依赖 ──
FROM python:3.11-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY src/pyproject.toml src/uv.lock ./

ENV UV_PYTHON_PREFERENCE=only-system
RUN uv sync --frozen --no-install-project --no-dev

# ── Stage 2: 预下载 embedding 模型 ──
FROM deps AS model

ENV HF_ENDPOINT=https://hf-mirror.com
ENV FASTEMBED_CACHE_PATH=/app/models
RUN .venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-zh-v1.5', cache_dir='/app/models')"

# ── Stage 3: 运行镜像 ──
FROM python:3.11-slim

WORKDIR /app

# 先复制源码，再覆盖 .venv（保证 deps 阶段的依赖不被本地 .venv 覆盖）
COPY src/ .
COPY --from=deps /app/.venv /app/.venv

# 预下载的模型缓存
COPY --from=model /app/models /app/models

ENV PATH="/app/.venv/bin:$PATH"
ENV FASTEMBED_CACHE_PATH=/app/models
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/data

EXPOSE 8080

CMD ["python", "bot.py"]
