# 技术架构

## 整体架构

```
用户消息（QQ）
      │
      ▼
┌─────────────────────────────────────────┐
│            Nonebot2 / OneBot V11        │
│                                         │
│  plugins/chat/__init__.py               │
│  ├── handle()          消息入口          │
│  │   ├── 注意力系统    决定要不要回       │
│  │   ├── 防抖系统      等用户说完再回     │
│  │   └── debounced()  计时触发           │
│  │                                      │
│  ├── _do_reply()       核心回复流程      │
│  │   ├── guard         速率/日预算检查   │
│  │   ├── memory.search 检索相关记忆      │
│  │   ├── memory.get_impressions 取印象   │
│  │   ├── build_system_prompt 构建 prompt │
│  │   ├── chat()        调用 LLM          │
│  │   └── _send()       分条发送 + 延迟   │
│  │                                      │
│  └── _evaluate_and_store() 异步评估存储  │
└─────────────────────────────────────────┘
      │                         │
      ▼                         ▼（异步，不阻塞）
┌───────────┐          ┌────────────────────┐
│  LLM 层   │          │    评估层          │
│           │          │   core/eval.py     │
│ core/llm  │          │                    │
│   Claude  │          │ evaluate_exchange()│
│   API     │          │  → importance 0~1  │
│           │          │  → impression 印象  │
└───────────┘          └────────┬───────────┘
                                │
                                ▼
                       ┌────────────────────┐
                       │    记忆层          │
                       │  core/memory.py    │
                       │                    │
                       │  store()           │
                       │  search()          │
                       │  get_impressions() │
                       │  recent()          │
                       └────────┬───────────┘
                                │
                                ▼
                       ┌────────────────────┐
                       │     Qdrant         │
                       │  collection: apts1548 │
                       │                    │
                       │  record_type:      │
                       │  "dialog"          │
                       │  "impression"      │
                       └────────────────────┘
```

---

## 核心模块

### `core/memory.py` — 长期记忆

```python
@dataclass
class MemoryEntry:
    user_id: str
    chat_type: str       # "private" | "group"
    chat_id: str         # 私聊=user_id，群聊=group_id
    user_name: str
    message: str         # dialog: 用户说的；impression: 印象文本
    response: str        # dialog: 48 回的；impression: ""
    timestamp: float
    importance: float    # 0.0~1.0，由 eval.py 评分
    record_type: str     # "dialog" | "impression"
```

**检索逻辑**
- `search()`: 三因子加权重排（相似度×0.6 + 时间衰减×0.2 + importance×0.2），importance < 0.2 的噪音直接过滤，只返回 `dialog` 类型
- `get_impressions()`: 按时间倒序取最近 N 条 `impression`
- `recent()`: 按时间倒序取最近 N 条 `dialog`，用于重启重建上下文

**Embedding**
- 模型：`BAAI/bge-small-zh-v1.5`（512 维）
- 启动时同步预加载，避免首次对话延迟
- 向量化在线程池执行（`run_in_executor`），不阻塞事件循环

---

### `core/eval.py` — 对话评估

每次对话结束后异步调用，一次 LLM 请求同时获得：

- **importance**（0.0~1.0）：对话的重要性评分
- **impression**（≤25 字）：48 视角的主观印象，口语化带情绪

**存储规则**
- dialog：每次对话必存，带 importance 评分
- impression：importance >= 0.4 且满足冷却条件（同用户/群 2 小时内只存一次）才存；importance >= 0.8 的重要事件无视冷却直接存

---

### `core/llm.py` — LLM 调用

```
system_blocks:
  Block 1 (stable):  PERSONALITY，带 cache_control: ephemeral
  Block 2 (dynamic): 当前时间 + 印象 + 记忆 + 对话对象 + 场景（不缓存）
```

支持 `thinking_mode: disabled / adaptive / enabled`。

---

### `core/prompt.py` — Prompt 构建

- `PERSONALITY`: 角色定义，换角色只改这里
- `build_system_prompt()`: 返回 `(stable, dynamic)` tuple
- `format_memories()`: 格式化对话记忆注入，时间标签按本地自然日计算
- `format_impressions()`: 格式化印象注入，时间标签按本地自然日计算
- `build_group_turns()`: 群聊上下文 → user/assistant 交替 messages

---

### `core/guard.py` — 速率保护

- 每用户每分钟限速（owner 享有 3 倍豁免）
- 每日 API 调用总上限，按本地自然日重置（非滚动 24h）
- 相同消息 TTL 缓存（同人同内容直接返回缓存，不调 API）

---

### `plugins/chat/__init__.py` — 消息处理

**群聊注意力**
```
被@          → att = 1.0（必回）
owner 说话   → att += 0.2
被引用       → att += 0.5（校验 reply.sender_id == bot_id）
时段乘数     × 0.2~1.0
密度惩罚     × 0.2~1.0（20s 内 >6 条）
每条消息     × 0.7 衰减
后台 loop    每 30s × 0.85 自然衰减
回复冷却     回复后 90s 不主动回（@ 除外）
```

**防抖**
```
新消息 → 检查是否有 pending task
  有且未完成 → cancel + 进入 burst 模式
  burst 模式 → 等 3.0s；否则等 1.5s
  等待结束  → 调用 _do_reply()
  finally   → 只有自己还是 pending[key] 才清理
```

**Trace ID**
- `contextvars.ContextVar`，每次 debounced 任务生成新 ID
- asyncio task 自动继承，整条链路都带 `[xxxx]` 前缀

---

## 配置（`.env`）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BOT_NAME` | bot 在群里的名字 | `48` |
| `OWNER_ID` | 创造者 QQ 号，享有特殊待遇 | — |
| `ALLOWED_GROUPS` | 允许发言的群，空则不在任何群发言 | `[]` |
| `ANTHROPIC_API_KEY` | Claude API key | — |
| `ANTHROPIC_API_ENDPOINT` | API 地址，可配置中转 | 官方 |
| `CLAUDE_MODEL` | 模型 ID | `claude-opus-4-6` |
| `THINKING_MODE` | `disabled/adaptive/enabled` | `disabled` |
| `THINKING_BUDGET` | enabled 模式下 thinking token 数 | `8192` |
| `QDRANT_URL` | Qdrant 地址 | `http://localhost:6333` |
| `RATE_PER_MINUTE` | 每用户每分钟限速 | `5` |
| `DAILY_LIMIT` | 每日 API 总调用上限 | `500` |
| `CACHE_TTL` | 相同消息缓存秒数 | `30` |

---

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| QQ 协议 | NapCat + OneBot V11 | 基于 QQNT，稳定 |
| 机器人框架 | Nonebot2 | Python，异步，生态好 |
| 向量数据库 | Qdrant（本地 Docker） | 开源，不依赖外部服务 |
| Embedding | bge-small-zh-v1.5 | 中文效果好，512 维够用，轻量 |
| LLM（当前） | Claude API（Sonnet / Opus） | 能力强，人格一致性好 |
| LLM（Phase 6） | Qwen2.5-14B LoRA 微调 | 专属模型，不依赖 API |
