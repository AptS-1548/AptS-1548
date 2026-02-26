# 部署指南

## 前置依赖

- Python >= 3.10
- Docker（跑 Qdrant + SurrealDB）
- NapCat（QQ 协议端，需要单独部署）

## 部署步骤

### 1. 启动数据库

```bash
# 在项目根目录（apts-1548/）
docker compose up -d
```

这会启动：
- **Qdrant**（向量数据库）→ `localhost:6333`
- **SurrealDB**（图数据库）→ `localhost:8000`

数据持久化在 `data/qdrant/` 和 `data/surrealdb/`。

### 2. 安装 Python 依赖

```bash
cd src
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

首次运行会下载 embedding 模型 `BAAI/bge-small-zh-v1.5`（~100MB）。

### 3. 配置 .env

```bash
cp .env.example .env
# 编辑 .env，填入必要配置
```

**必须填的**：
- `ANTHROPIC_API_KEY` — Claude API key
- `OWNER_ID` — 创造者的 QQ 号
- `ONEBOT_WS_URLS` — NapCat 的 WebSocket 地址
- `ALLOWED_GROUPS` — 允许发言的群号列表

**可选调整**：
- `ANTHROPIC_API_ENDPOINT` — 如果用 API 中转
- `CLAUDE_MODEL` — 默认 `claude-opus-4-6`
- `SURREALDB_USER` / `SURREALDB_PASSWORD` — 默认 root/root

### 4. 初始化人物图谱

```bash
cd src
python scripts/init_graph.py
```

写入 9 个人物节点 + 11 条关系边到 SurrealDB。只需跑一次。

### 5. 导入背景故事（可选）

```bash
# 预览
python scripts/import_stories.py --story-dir /path/to/story-website/data/stories/zh-CN

# 导入
python scripts/import_stories.py --apply --story-dir /path/to/story-website/data/stories/zh-CN
```

把 1548 视角的故事章节（52 个场景 chunk）写入 Qdrant。幂等，可重复运行。

没有 story-website 数据也能正常运行，只是 48 不记得背景故事细节。

### 6. 启动 Bot

```bash
cd src
python bot.py
```

启动后会：
1. 预加载 embedding 模型
2. 连接 Qdrant + SurrealDB
3. 生成今天的日程（调一次 LLM）
4. 重建群聊/私聊上下文
5. 启动后台 loop（日程刷新、主动行为、待办扫描）
6. 等待 NapCat WebSocket 连接

## 数据迁移

如果要从旧环境迁移：

### Qdrant（对话记忆、日记、印象、故事）
```bash
# 旧机器：创建快照
curl -X POST 'http://localhost:6333/collections/apts1548/snapshots'
# 复制 data/qdrant/snapshots/ 下的快照文件到新机器
# 新机器：恢复
curl -X PUT 'http://localhost:6333/collections/apts1548/snapshots/recover' \
  -H 'Content-Type: application/json' \
  -d '{"location": "file:///qdrant/snapshots/快照文件名"}'
```

### SurrealDB（人物图谱、待办）
```bash
# 复制 data/surrealdb/ 目录到新机器即可（surrealkv 文件级持久化）
# 或者重新跑 init_graph.py（待办会丢失）
```

### JSON 文件
```bash
# 复制 src/data/ 下的文件
cp -r data/relationships.json data/impression_ts.json data/schedule.json 新机器/src/data/
```

## 目录结构

```
src/
├── bot.py               # 入口
├── .env                 # 配置（不入库）
├── .env.example         # 配置模板
├── pyproject.toml       # 依赖
├── core/                # 核心模块
│   ├── memory.py        # 长期记忆（Qdrant 向量检索）
│   ├── eval.py          # 对话评估（importance / impression / trust）
│   ├── llm.py           # LLM 调用（Claude API，超时保护）
│   ├── prompt.py        # Prompt 构建 + PERSONALITY 定义
│   ├── relationship.py  # 关系系统（信任度 / 用户画像）
│   ├── guard.py         # 速率限制 + 日预算 + API 计数
│   ├── diary.py         # 48 的日记（自动生成 + 每日压缩）
│   ├── schedule.py      # 日程系统（影响回复意愿）
│   ├── proactive.py     # 主动行为（找 owner / 找朋友 / 日程驱动）
│   ├── graph.py         # 人物关系图谱（SurrealDB，自动重连）
│   ├── task.py          # 待办事项（提取 → 执行 → 回报）
│   ├── story.py         # 自创故事（高 importance 事件叙述）
│   └── summarize.py     # 用户认知摘要
├── plugins/chat/
│   ├── __init__.py      # 消息处理主逻辑
│   └── config.py        # 插件配置定义
├── scripts/
│   ├── init_graph.py    # 初始化人物图谱（跑一次）
│   ├── import_stories.py# 导入背景故事（跑一次）
│   └── dedup_diary.py   # 日记手动去重
└── data/                # 运行时数据（自动生成）
    ├── relationships.json
    ├── impression_ts.json
    └── schedule.json
```

## 排查

**Bot 没反应** — 检查 NapCat WebSocket 是否连上，`ALLOWED_GROUPS` 是否配了

**Connection error** — API endpoint 网络不通，检查 `ANTHROPIC_API_ENDPOINT`

**记忆检索 dialog=0** — Qdrant 没数据或没连上，检查 `docker compose ps`

**日程用默认** — LLM 调用失败 3 次，检查 API key 和网络

**SurrealDB 连接失败** — 不影响基本对话，人物图谱/待办功能降级
