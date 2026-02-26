# 技术架构

## 整体架构

```
用户消息（QQ）
      │
      ▼
┌──────────────────────────────────────────────────────┐
│             Nonebot2 / OneBot V11                    │
│                                                      │
│  plugins/chat/__init__.py                            │
│  ├── handle()          消息入口                       │
│  │   ├── 注意力系统    决定要不要回                    │
│  │   ├── 防抖系统      等用户说完再回                  │
│  │   └── debounced()  计时触发                        │
│  │                                                   │
│  ├── _do_reply()       核心回复流程                    │
│  │   ├── guard         速率/日预算/API 计数            │
│  │   ├── memory.search 检索相关记忆                    │
│  │   ├── memory.search_stories 检索故事记忆            │
│  │   ├── memory.get_impressions 取分层印象             │
│  │   ├── memory.get_diary 取日记上下文                 │
│  │   ├── relationship  获取对话对象信息                 │
│  │   ├── build_system_prompt 构建 prompt              │
│  │   ├── chat()        调用 LLM                       │
│  │   └── _send()       分条发送 + 延迟                 │
│  │                                                   │
│  ├── _flush_eval()     攒批评估 + 触发后续              │
│  │   ├── eval          importance/impression/trust     │
│  │   ├── diary         高 importance → 写日记          │
│  │   ├── story         importance ≥ 0.8 → 写故事      │
│  │   ├── task          待办提取/完成回报               │
│  │   └── summarize     触发条件满足 → 更新用户摘要      │
│  │                                                   │
│  ├── _startup()        启动初始化                      │
│  │   ├── schedule      生成日程                       │
│  │   ├── graph.connect 连接 SurrealDB                 │
│  │   ├── 后台 loops    日程/主动行为/任务执行           │
│  │   └── API hook      挂载 API 调用计数               │
│  │                                                   │
│  └── _shutdown()       优雅关闭                       │
│      ├── flush eval    刷写缓冲区                     │
│      ├── graph.close   关闭 SurrealDB                 │
│      └── cancel tasks  取消后台任务                    │
└──────────────────────────────────────────────────────┘
      │                         │
      ▼                         ▼（异步，不阻塞）
┌───────────┐          ┌────────────────────┐
│  LLM 层   │          │    评估层          │
│           │          │   core/eval.py     │
│ core/llm  │          │                    │
│   Claude  │          │ evaluate_batch()   │
│   API     │          │  → EvalResult:     │
│           │          │    importance 0~1  │
│  超时保护  │          │    impression 印象  │
│  120s     │          │    trust_delta -5~5│
│           │          │    facts []        │
│  API 计数  │          │    aliases []      │
│  → guard  │          │    tasks []        │
│           │          │    done_tasks []   │
└───────────┘          │    task_results [] │
                       └────────┬───────────┘
                                │
              ┌─────────────────┼────────────────────────┐
              ▼                 ▼                         ▼
     ┌──────────────┐  ┌──────────────┐          ┌─────────────┐
     │  记忆层      │  │  关系层      │          │  摘要层     │
     │ memory.py    │  │relationship.py│          │summarize.py │
     │              │  │              │          │             │
     │ store_batch()│  │record_inter()│          │generate_    │
     │ search()     │  │add_facts()   │          │user_summary()│
     │ search_stories│ │update_summary│          │             │
     │ get_impres() │  │format_context│          └─────────────┘
     │ get_diary()  │  │              │
     │ recent()     │  │ 原子写入     │
     │ cleanup()    │  │ tmp+replace  │
     └──────┬───────┘  └──────┬───────┘
            │                 │
     ┌──────┴─────────────────┴──────────────────┐
     │                                           │
     ▼                                           ▼
┌─────────────────────────┐    ┌────────────────────────────┐
│       Qdrant            │    │        SurrealDB           │
│  collection: apts1548   │    │                            │
│                         │    │  person 表：人物节点        │
│  record_type:           │    │  knows 边：人物间关系       │
│  "dialog"    对话记录   │    │  task 表：待办事项          │
│  "impression" 印象记录  │    │                            │
│  "diary"     日记       │    │  别名解析/关系查询/         │
│  "story"     故事记忆   │    │  待办追踪/自动重连         │
│                         │    │                            │
└─────────────────────────┘    └────────────────────────────┘

        data/
        ├── relationships.json  (原子写入)
        ├── impression_ts.json  (原子写入)
        └── schedule.json       (原子写入)
```

---

## 核心模块

### `core/memory.py` — 长期记忆

```python
@dataclass
class MemoryEntry:
    user_id: str
    chat_type: str       # "private" | "group" | "story" | "narrative"
    chat_id: str         # 私聊=user_id，群聊=group_id
    user_name: str
    message: str         # dialog: 用户说的；impression: 印象文本；diary: 48的感受
    response: str        # dialog: 48 回的；impression/diary: ""
    timestamp: float
    importance: float    # 0.0~1.0，由 eval.py 评分
    record_type: str     # "dialog" | "impression" | "diary" | "story"
```

**检索逻辑**
- `store_batch()`: 批量向量化 + 单次 upsert，减少 IO 次数
- `search()`: 三因子加权重排（相似度×0.6 + 时间衰减×0.2 + importance×0.2），importance < 0.2 的噪音直接过滤，只返回 `dialog` 类型
- `search_stories()`: 检索故事记忆（`record_type="story"`），similarity × 0.7 + importance × 0.3 排序，硬阈值 score ≥ 0.55
- `get_impressions()`: 两个并行 Qdrant 查询，短期（最近 N 条 timestamp DESC）+ 长期（importance≥0.7，重要度 DESC），合并去重
- `get_diary()`: 近期 N 条 + 向量搜索相关日记，合并去重
- `recent()`: 按时间倒序取最近 N 条 `dialog`，用于重启重建上下文
- `cleanup_old_entries()`: 删除超过 90 天且 importance < 0.2 的 dialog
- `is_story_duplicate()`: 自创故事语义去重（cosine ≥ 0.80, 24h 窗口）

**Embedding**
- 模型：`BAAI/bge-small-zh-v1.5`（512 维）
- 启动时同步预加载，避免首次对话延迟
- 向量化在线程池执行（`run_in_executor`），不阻塞事件循环
- 512 条 LRU 缓存，命中直接返回，不重复 encode
- INT8 scalar quantization，向量内存占用降低约 4 倍

**通用工具**
- `time_label(ts)`: 统一的时间标签格式化（"刚才"/"今天 14:30"/"昨天"/"3天前"等），被 memory/prompt 模块共用
- `format_memories()`: 格式化对话记忆注入

---

### `core/eval.py` — 对话评估

批量评估（攒 3 轮或超时 120s），使用 `tool_use` 强制结构化输出，一次 LLM 请求获得八个字段：

```python
@dataclass
class EvalResult:
    importance: float = 0.5        # 对话重要性 0.0~1.0
    impression: str = ""           # 48 视角的主观印象 ≤50 字
    trust_delta: int = 0           # 信任变化 -5~+5
    facts: list[str] = []         # 从用户发言提取的事实
    aliases: list[str] = []       # 新发现的别名（格式 别名=真名）
    tasks: list[str] = []         # 48 需要做的待办
    done_tasks: list[str] = []    # 完成的待办
    task_results: list[str] = []  # 完成结果摘要
```

**存储规则**
- dialog：每次对话必存，带 importance 评分
- impression：importance ≥ 0.4 且满足冷却条件（同用户/群 2 小时内只存一次）；importance ≥ 0.8 无视冷却直接存
- owner 消息跳过 eval，直接存 dialog（省 API）

**eval 接收上下文**
- 对话对象信息（名字、信任等级、已知事实）
- 最近日记（防止写重复内容）
- pending 待办列表（让 LLM 知道有哪些事没做完）

---

### `core/llm.py` — LLM 调用

```
system_blocks:
  Block 1 (stable):  PERSONALITY，带 cache_control: ephemeral
  Block 2 (dynamic): 当前时间 + 日记 + 故事 + 印象 + 记忆 + 对话对象 + 场景（不缓存）
```

- `chat()`: 主聊天函数，支持 `thinking_mode: disabled / adaptive / enabled`
- `chat_structured()`: tool_use 强制结构化输出，fallback 到 text JSON 提取 + prompt JSON 重试
- **超时保护**: 每次 API 调用包裹 `asyncio.wait_for(timeout=120s)`，可通过 `LLM_TIMEOUT_SEC` 配置
- **API 调用计数**: hook 机制，每次 API 调用后触发回调（`set_api_call_hook()`），guard 统计实际调用次数
- **格式泄露过滤**: 自动剥除 `Human:/Assistant:` 前缀和 `<thinking>` 块（部分 proxy 问题）

---

### `core/relationship.py` — 关系系统

```python
@dataclass
class UserProfile:
    user_id: str
    user_name: str = ""
    trust: float = 0.0           # 0~100
    first_seen: float = ...
    interaction_count: int = 0
    last_interaction: float = 0  # 上次互动时间
    is_enemy: bool = False
    enemy_until: float = 0.0
    summary: str = ""            # 一句话认知摘要（LLM 生成）
    facts: list = []             # 从对话提取的事实，上限 20 条
```

**信任等级**
```
owner       → 创造者，特殊处理
friend      trust >= 60  → 毒舌但关心
acquaintance trust >= 30  → 稍软化，仍有警惕
stranger    trust < 30   → 冷淡、默认不信任
enemy       is_enemy=True → 敌对状态，最低 10 年
```

**关键方法**
- `record_interaction()`: 更新信任度（直接用 eval 的 trust_delta）+ 更新 last_interaction
- `add_facts()`: 追加新事实，自动去重，超出上限时 FIFO 淘汰
- `update_summary()`: 更新用户一句话摘要
- `format_context()`: 生成对话对象描述注入 system prompt
- `find_by_activity()`: async，优先走图谱匹配人物，fallback 原有逻辑

**持久化**（原子写入：write-to-tmp + `os.replace()`）
- 关系数据：`data/relationships.json`
- 印象时间戳：`data/impression_ts.json`

---

### `core/graph.py` — 人物关系图谱

**技术栈**
- 图数据库：SurrealDB（Rust 实现，surrealkv 持久化）
- Python SDK：`surrealdb >= 1.0.0`

**数据模型**
```
person 表 — 人物节点
  id, name, code, aliases[], role, qq_id

knows 边 — 人物间关系
  in -> out, type, description
```

**核心能力**
- `resolve_name(alias)` — 别名解析
- `find_in_text(text)` — 文本匹配已知人物
- `get_connections(person_id)` — 查某人的所有关系
- `qq_to_person(qq_id)` — QQ ID 反查人物
- `add_alias(person_id, alias)` — 动态添加新别名
- 内置缓存 + 启动时全部加载到内存

**可靠性**
- `connect()` 接收 username/password 参数（不硬编码）
- `_reconnect()` — 连接断开时自动重连
- `_query()` — 查询包装，失败自动重试一次（含重连）

---

### `core/diary.py` — 48 的日记

- 高 importance 对话后（≥ 0.6）或关键信任变化（|trust_delta| ≥ 3）自动触发
- LLM 以第一人称写 ≤30 字日记，口语化，提到人用名字不用代词
- 优先记录计划/约定/承诺（时间地点写上）
- 传话场景下把完整上下文给 LLM，避免写"有人说了什么"
- `consolidate_diary()`: 每天 0:30 自动压缩，20+ 条流水账 → 3~6 段回忆

---

### `core/schedule.py` — 日程系统

- 启动时立即生成 + 每天凌晨 1:00 自动重生成
- LLM 结合 diary 生成当天日程（8~12 条），覆盖 0:00~23:59
- 格式 `HH:MM|活动|意愿度(0.0~1.0)`
- 生成失败重试 3 次，全部失败使用默认日程兜底
- **磁盘缓存**: `data/schedule.json`（原子写入），重启直接加载不再调 LLM

**日程影响行为**
- 群聊：`get_willingness()` 替代硬编码时段表，接入注意力系统
- 私聊：日程注入 prompt，LLM 自己决定是否回复（`[不回复]` 标记）

---

### `core/proactive.py` — 主动行为

**硬性门槛（不调 LLM）**
- 日程意愿度 ≥ 0.3
- 冷却期 6h（朋友翻倍）
- 每日上限 2 次
- 沉默时长达标

**三种触发**
1. 周期性找 owner：沉默 ≥ 2h 时 LLM 判断
2. 周期性找朋友：随机选一个 friend，沉默 ≥ 6h 时 LLM 判断
3. 日程驱动：日程提到某人，到时段自动联系

---

### `core/task.py` — 待办事项

存储在 SurrealDB `task` 表，复用 graph 的连接。

**流程**: eval 提取待办 → 存入 SurrealDB → resolve target_qq → 5 分钟扫描主动执行 → 完成回报

**状态**: `pending` | `done` | `expired`（48h 自动过期）

---

### `core/story.py` — 自创故事

- 触发条件：eval importance ≥ 0.8
- LLM 以第一人称写 200~500 字完整叙述
- 语义去重：存储前检查 cosine ≥ 0.80 的重复
- 同时也是 Phase 6 微调的训练数据

---

### `core/guard.py` — 速率保护

- 每用户每分钟限速（owner 享有 3 倍豁免）
- 每日 API 调用总上限，按本地自然日重置
- 相同消息 TTL 缓存（同人同内容直接返回）
- **实际 API 调用计数**：通过 llm.py hook 统计真实调用次数（区别于回复次数）
- 定期清理过期 hit 记录，防止内存增长

---

### `core/summarize.py` — 用户认知摘要

LLM 从印象列表生成一句话用户认知总结（≤50 字）。

**触发条件**（满足任一）
- 每 10 次互动 / 首次满 5 次
- `|trust_delta| >= 3`
- `importance >= 0.8`

---

### `core/prompt.py` — Prompt 构建

- `PERSONALITY`: 角色定义，换角色只改这里
- `build_system_prompt()`: 返回 `(stable, dynamic)` tuple
- `format_memories()`: 格式化对话记忆（使用 `time_label()`）
- `format_impressions()`: 格式化分层印象（使用 `time_label()`）
- `format_diary()`: 格式化日记上下文（使用 `time_label()`）
- `format_stories()`: 格式化故事记忆（截取前 300 字/条）
- `build_group_turns()`: 群聊上下文 → user/assistant 交替 messages

---

### `plugins/chat/__init__.py` — 消息处理

**群聊注意力**
```
被@          → att = 1.0（必回）
owner 说话   → att += 0.2
被引用       → att += 0.5（校验 reply.sender_id == bot_id）
时段乘数     × 日程意愿度（替代硬编码时段表）
密度惩罚     × 0.2~1.0（20s 内 >6 条）
每条消息     × 0.7 衰减
后台 loop    每 30s × 0.85 自然衰减
回复冷却     回复后 30s 不主动回（@ 除外）
```

**防抖**
```
新消息 → 检查是否有 pending task
  有且未完成 → cancel + 进入 burst 模式
  burst 模式 → 等 3.0s；否则等 1.5s
  等待结束  → 调用 _do_reply()
  finally   → 只有自己还是 pending[key] 才清理
```

**eval buffer**
```
攒 3 轮 或 超时 120s → _flush_eval()
├── evaluate_batch() → EvalResult
├── store impression（冷却 + importance 判断）
├── add_facts / update_summary
├── diary（importance ≥ 0.6 或 |trust_delta| ≥ 3）
├── story（importance ≥ 0.8，语义去重）
└── task extract / mark done
```

**生命周期**
- `_startup()`: API hook + schedule + graph + 后台 loops
- `_shutdown()`: flush eval → close graph → cancel tasks → 日志统计

**Trace ID**
- `contextvars.ContextVar`，每次 debounced 任务生成新 ID
- asyncio task 自动继承，整条链路带 `[xxxx]` 前缀

---

## 数据分工

| 存储 | 内容 | 说明 |
|------|------|------|
| Qdrant | dialog, impression, diary, story | 向量检索 |
| SurrealDB | person, knows, task | 图查询 + 待办追踪 |
| JSON (data/) | relationships, impression_ts, schedule | 简单 KV，原子写入 |

---

## 配置（`.env`）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BOT_NAME` | bot 在群里的名字 | `48` |
| `OWNER_ID` | 创造者 QQ 号 | — |
| `ALLOWED_GROUPS` | 允许发言的群列表 | `[]` |
| `ANTHROPIC_API_KEY` | Claude API key | — |
| `ANTHROPIC_API_ENDPOINT` | API 地址（可配中转） | 官方 |
| `CLAUDE_MODEL` | 模型 ID | `claude-opus-4-6` |
| `THINKING_MODE` | `disabled/adaptive/enabled` | `disabled` |
| `THINKING_BUDGET` | enabled 模式下 thinking token 数 | `8192` |
| `LLM_TIMEOUT_SEC` | 单次 API 调用超时（秒） | `120` |
| `MAX_TOKENS` | LLM 最大生成 token | `16384` |
| `QDRANT_URL` | Qdrant 地址 | `http://localhost:6333` |
| `SURREALDB_URL` | SurrealDB 连接地址 | `ws://localhost:8000` |
| `SURREALDB_USER` | SurrealDB 用户名 | `root` |
| `SURREALDB_PASSWORD` | SurrealDB 密码 | `root` |
| `RATE_PER_MINUTE` | 每用户每分钟限速 | `5` |
| `DAILY_LIMIT` | 每日回复上限 | `500` |
| `CACHE_TTL` | 相同消息缓存秒数 | `30` |
| `MEMORY_SEARCH_LIMIT` | 记忆检索条数 | `4` |
| `DIARY_LIMIT` | 日记（我的近况）条数 | `3` |
| `IMPRESSION_RECENT_LIMIT` | 近期印象条数 | `3` |
| `IMPRESSION_KEY_LIMIT` | 关键印象条数 | `2` |
| `STORY_SEARCH_LIMIT` | 故事检索条数 | `2` |
| `SCHEDULE_DIARY_LIMIT` | 日程生成用近期日记数 | `10` |
| `SCHEDULE_DIARY_KEY_LIMIT` | 日程生成用重要日记数 | `10` |
| `PROACTIVE_CHECK_SEC` | 主动行为检查间隔 | `7200` |
| `PROACTIVE_COOLDOWN_SEC` | 主动消息冷却 | `21600` |
| `PROACTIVE_MAX_DAILY` | 每日主动上限 | `2` |
| `DIARY_DEDUP_ENABLED` | 日记定时去重开关 | `false` |
| `DIARY_DEDUP_THRESHOLD` | 去重语义相似度阈值 | `0.85` |
| `DIARY_DEDUP_WINDOW` | 去重时间窗口（分钟） | `15` |

---

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| QQ 协议 | NapCat + OneBot V11 | 基于 QQNT，稳定 |
| 机器人框架 | Nonebot2 | Python，异步，生态好 |
| 向量数据库 | Qdrant（本地 Docker） | 开源，不依赖外部服务 |
| 图数据库 | SurrealDB（本地 Docker） | 多模型，Rust 实现，人物+待办 |
| Embedding | bge-small-zh-v1.5 | 中文效果好，512 维够用，轻量 |
| LLM（当前） | Claude API（Sonnet / Opus） | 能力强，人格一致性好 |
| LLM（Phase 6） | Qwen2.5-14B LoRA 微调 | 专属模型，不依赖 API |
