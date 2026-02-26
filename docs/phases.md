# 实现阶段

## 总览

```
Phase 1: 基础行为系统      ✅ 完成   ← 注意力、防抖、冷却、打字延迟
Phase 2: 长期记忆          ✅ 完成   ← Qdrant 向量存储、检索注入、重启重建
Phase 3: 记忆进化          ✅ 完成   ← 评估系统、分层印象、事实提取、周期摘要
Phase 4: 关系系统          ✅ 完成   ← 信任等级、用户画像、敌对机制
Phase 5: 主动行为          ✅ 完成   ← diary → 日程 → 主动关心 → 图谱 → 待办 → 日记压缩 → 故事RAG
Phase 6: 微调专属模型      ⬜ 待开始 ← 训练专属 Qwen 模型替换 Claude API
```

---

## Phase 1: 基础行为系统 ✅

### 完成内容

**群聊注意力系统**
- 注意力值 0.0~1.0，被@ 直接置 1.0
- 每条消息按 `ATTENTION_DECAY=0.7` 衰减
- 1547（owner）说话额外 +0.2
- 被引用回复 +0.5（校验 `reply.sender_id == bot_id`）
- 后台 loop 每 30s 自然衰减 ×0.85

**时段乘数**
```
0-7h  → 0.2   深夜，基本不回
7-9h  → 0.5   早上，慢慢醒
9-18h → 0.8   白天
18-23h→ 1.0   傍晚，最活跃
23h+  → 0.5   开始懒了
```

**回复冷却 + 密度检测**
- 回复后 90s 内不再主动回（@ 除外）
- 20s 内超过 6 条消息，回复意愿最低压到 0.2x

**防抖 + burst 冷静期**
- 消息到达后等 1.5s，期间新消息取消重计时
- 取消过的会话进入 burst 模式，冷静期延长到 3.0s
- `finally` 里只有自己还是 `_pending[key]` 才清理

**打字延迟模拟**
- 按字符数计算：`0.3~0.6s/char`
- 20% 概率额外 +1.5~4s
- 多行分条发送，条间有延迟

**链路 Trace**
- 每次 LLM 调用生成 6 位十六进制 trace ID
- 用 `contextvars.ContextVar` 传播，整条链路带 `[trace_id]` 前缀

---

## Phase 2: 长期记忆 ✅

### 完成内容

**技术栈**
- 向量数据库：Qdrant（本地 Docker）
- Embedding 模型：`BAAI/bge-small-zh-v1.5`（512 维，启动时预加载）
- payload 索引：`record_type/user_id/chat_id/chat_type/timestamp/importance` 全部建索引

**存储（`memory.store()`）**
- 每次对话存一条 `MemoryEntry`
- `record_type`：`"dialog"` | `"impression"` | `"diary"`（Phase 5 新增）

**检索（`memory.search()`）**
- 三因子加权重排：`similarity×0.6 + recency×0.2 + importance×0.2`
- 相似度阈值 0.5，importance < 0.2 过滤噪音
- 群聊按 `chat_id` 隔离，私聊按 `user_id` 跨群检索
- **全局 fallback**：范围内搜索无结果时，去掉 chat/user 限制全局再搜一次（跨聊天找相关记忆）
- 全局也无结果时 fallback 到最近 3 条 `dialog`

**重启后上下文重建**
- 启动时：从 Qdrant 拉最近记录重建 `_group_context`
- 私聊懒加载：首次收到消息时重建 `_private_histories`
- `recent()` 使用 `order_by=timestamp DESC`，避免全表扫描

**prompt 缓存优化**
- system prompt 拆分为 stable（人格，带 `cache_control: ephemeral`）和 dynamic（时间 + 记忆 + 场景，不缓存）

---

## Phase 3: 记忆进化 ✅

### 完成内容

**对话评估（`core/eval.py`）**
- 每次对话结束后异步 LLM 调用（不阻塞发消息）
- batch eval：攒 3 轮或超时 120s 批量评估，dialog 立即存储（默认 importance=0.5）
- 单次调用返回 `EvalResult` 结构体，包含八个字段：
  - `importance`（0.0~1.0）：对话重要性
  - `impression`（≤50字）：48 视角的主观印象
  - `trust_delta`（-5~+5 整数）：对这次互动的信任评估
  - `facts`（list）：从用户发言中提取的事实信息
  - `aliases`（list）：对话中出现的新称呼/别名（格式 `别名=真名`）
  - `tasks`（list）：48 需要去做的事（待办提取）
  - `done_tasks`（list）：这段对话完成了哪些待办
  - `task_results`（list）：已完成待办的结果摘要
- eval 接收上下文：对话对象信息、已知事实、最近日记、pending 待办列表

**印象系统**
- 存储条件：`importance >= 0.4` 且满足冷却（同用户/群 2h 内只存一次）
- `importance >= 0.8` 的关键事件无视冷却直接存
- 印象时间戳持久化到 `data/impression_ts.json`，重启不丢失

**分层印象检索（`memory.get_impressions()`）**
- 短期：最近 3 条（时间倒序）
- 长期：`importance >= 0.7` 的关键印象，按重要度倒序取 5 条
- 合并去重后注入 system prompt（`## 印象记录`）

**事实提取**
- eval 同时提取用户明确说出的具体事实（地点、职业、爱好等）
- 存入 `UserProfile.facts`，上限 20 条，超出丢弃最旧的

**周期摘要（`core/summarize.py`）**
- 触发条件：每 10 次互动 / 首次满 5 次 / `|trust_delta| >= 3` / `importance >= 0.8`
- LLM 从印象列表生成一句话用户认知总结（≤50字）
- 存入 `UserProfile.summary`，注入对话对象描述
- 摘要触发在 dialog + impression 都存完后执行（避免竞争条件）

---

## Phase 4: 关系系统 ✅

### 完成内容

**信任等级（`core/relationship.py`）**
```
owner       → 创造者，特殊处理
friend      trust >= 60  → 毒舌但关心
acquaintance trust >= 30  → 稍软化，仍有警惕
stranger    trust < 30   → 冷淡、默认不信任
enemy       is_enemy=True → 敌对状态，最低 10 年
```

**UserProfile 字段**
```python
user_id, user_name, trust, first_seen,
interaction_count, last_interaction,
is_enemy, enemy_until,
summary,   # 一句话认知摘要
facts,     # 从对话提取的事实列表
```

**信任度更新**
- `trust_delta` 由 eval LLM 评估（-5~+5），直接累加，不再用固定公式
- 上下限 clamp 到 0~100

**context 注入（`format_context()`）**
```
张三（信任度 47/100，互动 12 次）是你认识的人。
态度可以稍软化，但还是有点警惕。
已知信息：在北京读大学、学计算机的、喜欢玩崩铁
你对ta的了解：这人来了好几次，聊技术挺靠谱的
```

**持久化**
- 关系数据：`data/relationships.json`
- 印象时间戳：`data/impression_ts.json`

---

## Phase 5: 主动行为 ✅

### 目标

48 有自己的内心记录，能感知自身状态，并在适当时机主动联系 owner。

### 5.1 自我记录系统（diary）✅

**已完成内容**

**diary 生成（`core/diary.py`）**
- 高 importance 对话后（`>= 0.6`）或关键信任变化（`|trust_delta| >= 3`）自动触发
- LLM 以第一人称写 ≤30 字日记，口语化，提到人用名字不用代词
- **优先记录计划/约定/承诺**（如"和沐川约了下午三点喝奶茶"），有时间地点就写上
- 输入：对话内容 + 信任度 + 最近印象 + 最近已有日记（防重复）

**存储**
- `record_type="diary"`, `user_id="48"`，与用户记忆隔离
- 存入 Qdrant，不参与用户维度的检索
- **实时语义去重**：存储前用向量搜索最近 24h 的日记，cosine >= 0.85 视为重复跳过
- **定时批量去重**：每天 0:30 自动执行，条件：cosine >= 阈值 且 时间间隔 <= 窗口（默认 15 分钟）
- 可通过 `DIARY_DEDUP_ENABLED` 开关，`DIARY_DEDUP_THRESHOLD` / `DIARY_DEDUP_WINDOW` 调参
- 附带独立脚本 `scripts/dedup_diary.py` 可手动执行（`--apply` 真删，默认预览）
- 存储日志细化：`记忆存储 | type=diary user=48 msg=...`

**检索与注入**
- `memory.get_diary()`：近期 N 条 + 向量搜索相关日记，合并去重
- 格式化为 `## 我的近况` 注入 dynamic system prompt
- 检索数量可通过 `DIARY_LIMIT` env 配置

**batch eval**
- 对话不再逐条评估，攒 3 轮或超时 120s 批量评估
- dialog 立即存储（默认 importance=0.5），eval 结果异步回填

**私聊时间标记**
- 消息间隔超过 10 分钟时，在历史中插入 `[时间]` 标记
- 重建历史时使用原始 timestamp，保留真实时间间隔

---

### 5.2 日程系统（schedule）✅

**已完成内容**

**日程生成（`core/schedule.py`）**
- 启动时立即生成 + 每天凌晨 1:00 自动重生成
- LLM 结合 diary 生成当天日程（8~12 条），覆盖 0:00~23:59
- 格式 `HH:MM|活动|意愿度(0.0~1.0)`，解析为 `ScheduleEntry`
- LLM 未覆盖凌晨时自动补 `0:00|睡觉|0.0`
- 生成失败重试 3 次，全部失败使用默认日程兜底
- 存在内存（当天有效），同一天只生成一次

**日程生成数据**
- 启动/凌晨重生成时拉取日记作为上下文（`SCHEDULE_DIARY_LIMIT` + `SCHEDULE_DIARY_KEY_LIMIT`）
- 日记带具体时间显示（`show_time=True`），让 LLM 感知时间线
- 对话时触发的日程生成也带 diary 上下文

**日程影响回复行为 — 分群聊/私聊两层**

群聊（不调 LLM，省 API）：
- `_time_multiplier()` 代理到 `get_willingness()`，替代硬编码时段表
- 乘数接入现有注意力系统，被@ 仍然无视日程直接回

私聊（LLM 自己决定）：
- 日程注入 prompt（`## 我现在在做什么`）
- prompt 规则：在睡觉或不想理时回复 `[不回复]`
- 代码检测 `[不回复]` 标记则不发送，历史记为 `(已读不回)`

**其他效果**
- 被问"你在干嘛" → 回答和当前日程活动一致
- 语气受当前活动影响：巡逻时简短，空闲时松一点

---

### 5.3 主动行为触发 ✅

**已完成内容**

**硬性门槛（代码过滤，不调 LLM）**
- 日程意愿度 >= 0.3（不在睡觉/忙的时候不主动）
- 冷却期 6h（上次主动后 6 小时内不再主动）
- 每日上限 2 次
- 跟 owner 至少 2h 没聊过

**软性判断（LLM 决定）**
- 收集：48 的日记、对 owner 的印象、当前活动、沉默时长
- LLM 自行判断要不要说、说什么
- 输出 `[不主动]` 则跳过

**三种主动触发**

1. **周期性找 owner**：owner 沉默 >= 2h 时 LLM 判断要不要说
2. **周期性找朋友**：随机选一个 friend 级别用户，沉默 >= 6h 时 LLM 判断，冷却翻倍
3. **日程驱动**：日程提到某人（"和沐川对设计"），到了该时段自动给对应的人发消息

**定时 loop**
- `_proactive_loop()`：启动后等 5 分钟，之后每 `PROACTIVE_CHECK_SEC`（默认 2h）检查
- 通过 `nonebot.get_bot()` 获取 Bot 实例，无需 incoming event
- 统一 `_send_proactive()` 处理：推入历史 + 发送 + 存记忆

**日程匹配**
- `relationship.find_by_activity()`：在活动文本中匹配已知用户名（全名 + 去姓短名）
- 每个日程条目只触发一次（按 `日期_时间` 去重）
- 30 分钟时间窗口内匹配

**数据支撑**
- `UserProfile.last_interaction`：新增字段，`record_interaction()` 时自动更新
- owner 对话现在也调 `record_interaction()` 更新互动时间
- `relationship.get_friends()`：返回所有 trust >= 60 的用户

**配置项**
- `PROACTIVE_CHECK_SEC=7200`（检查间隔）
- `PROACTIVE_COOLDOWN_SEC=21600`（冷却，朋友翻倍）
- `PROACTIVE_MAX_DAILY=2`（每日上限）

---

### 5.4 人物关系图谱 ✅

**已完成内容**

**技术栈**
- 图数据库：SurrealDB（Rust 实现，多模型，Docker 部署）
- Python SDK：`surrealdb >= 1.0.0`
- 数据引擎：`surrealkv` 持久化

**数据模型**
```
person 表 — 人物节点
  id, name, code, aliases[], role, qq_id

knows 边 — 人物间关系
  in -> out, type, description
```

**初始数据（`scripts/init_graph.py`）**
- 9 个人物节点（1547/1548/1549/1543/2275/1738/0152/3167/4869）
- 11 条关系边（created_by, sister, colleague, friend, alter_ego, respect）
- 每个人物带别名列表，支持多称呼映射

**核心能力（`core/graph.py`）**
- `resolve_name(alias)` — 别名解析：`"沈老师"` → 沈沐川 (person:1543)
- `find_in_text(text)` — 文本匹配：`"和沐川去喝奶茶"` → [沈沐川]
- `get_connections(person_id)` — 查某人的所有关系
- `qq_to_person(qq_id)` — QQ ID 反查人物
- `add_alias(person_id, alias)` — 动态添加新别名
- 内置缓存机制，全部人物一次性加载到内存，减少查询

**动态别名发现**
- eval 新增 `aliases` 字段，LLM 从对话中提取新称呼
- 格式 `"别名=真名"`（如 `"沈老师=沈沐川"`）
- eval 后自动调 `graph.resolve_name()` + `graph.add_alias()` 写入

**集成点**
- `relationship.find_by_activity()` — 改为 async，优先走图查询匹配人物，fallback 原有逻辑
- `proactive.check_schedule_proactive()` — 日程匹配人物时用图数据库别名
- `__init__.py._startup()` — 启动时连接 SurrealDB（失败不阻塞，graceful degradation）

**数据分工**
```
SurrealDB   → 人物身份、别名、人物间关系图、QQ ID 映射
Qdrant      → 对话记忆、日记、印象（不变）
JSON        → trust/interaction/facts/summary（不变）
```

**配置**
- `SURREALDB_URL=ws://localhost:8000`
- Docker: `docker-compose.yml`（项目根目录）

---

### 5.5 跨对话信息传递 ✅

**已完成内容**

**传话追踪**
- 对话涉及第三方人物时自动触发日记（传话场景）
- `graph.find_in_text()` 检测对话文本中提到的已知人物
- 排除对话对象自己，只看是否提到"第三方"
- `_relay_active` 追踪窗口（默认 600s），窗口内后续对话继续写日记
- 传话场景下把所有轮次拼起来给 diary LLM，让它看到完整上下文

**日记代词消歧**
- `generate_diary_entry()` 接收 `mentioned_names` 参数
- 告诉 diary LLM 对话中提到了哪些人物，避免写"有人""他""她"
- relay 和非 relay 场景都会检测提到的人物

---

### 5.6 待办事项系统 ✅

**已完成内容**

**数据模型（SurrealDB `task` 表）**
```
task:ulid
├── content: str           # "帮47喊清弦拿电脑工卡"
├── source_user: str       # 请求人名字
├── source_qq: str         # 请求人 QQ ID
├── target_qq: option<str> # 目标人 QQ ID
├── status: str            # "pending" | "done" | "expired"
├── created_at: datetime
├── sent_at: option<datetime>   # 主动消息发送时间
├── done_at: option<datetime>
├── done_result: option<str>    # 完成结果摘要（≤100字）
```

**待办提取（eval → task）**
- eval 新增 `tasks` 字段：LLM 从对话中提取 48 需要做的事
- eval 新增 `done_tasks` 字段：判断对话是否完成了某些待办
- eval 新增 `task_results` 字段：已完成待办的结果摘要
- pending tasks 列表传给 eval，让 LLM 知道当前有哪些待办
- `_flush_eval` 中提取后写入 SurrealDB，语义去重（完全相同的 pending 不重复）

**目标解析（task → target）**
- 新任务写入后，用 `graph.find_in_text()` 从内容中提取提到的人物
- 排除 source（请求人），剩余的 = target（执行对象）
- target 有 QQ ID 的存入 `target_qq`，供主动执行使用

**主动执行（`_task_loop`）**
- 每 5 分钟扫描 pending 且有 `target_qq` 且未发送的任务
- LLM 生成发给 target 的消息（`generate_task_message()`）
- 通过 `bot.send_private_msg()` 发送，标记 `sent_at`
- 启动后等 2 分钟再开始

**完成回报**
- eval 检测 done_tasks 时附带 task_results（结果摘要）
- `mark_done()` 同时存储 `done_result`
- `format_for_prompt()` 对当前对话的 source_user 注入已完成待办：
  ```
  ## 已完成的待办
  你之前帮对方做的事，告诉对方结果：
  - 帮你喊了清弦：清弦说明天带过去（30分钟前完成）
  ```

**prompt 注入**
- pending 待办注入 `## 待办事项`，标注来源和时间
- 当前对话对象匹配待办内容时标记 `← 你现在就在跟这个人聊，说一下`
- 48 小时自动过期

**完整 flow 示例**
```
47 说 "帮我喊清弦来拿工卡"
  → eval 提取 task → 存 SurrealDB → resolve target_qq = 清弦
  → ≤5 分钟 _task_loop 扫描 → 发消息给清弦 → mark_sent
  → 清弦回复 → eval 检测 done_tasks + results → mark_done
  → 48 下次和 47 聊 → 注入已完成待办 → 48 主动提起
```

---

### 5.7 日记每日压缩 ✅

**已完成内容**

**压缩逻辑（`core/diary.py` → `consolidate_diary()`）**
- LLM 把一天的流水账（20~30 条 × 30 字）合并成 3~6 段回忆（每段 ≤50 字）
- 按主题分段，保留关键事实（人名、事件、时间、承诺、结果）
- 第一人称，口语化，像真的在回忆今天发生了什么
- 使用 `chat_structured` + tool_use 强制结构化输出

**执行时机**
- 每天 0:30，在语义去重之后执行
- 条目数 >= 8 时才触发压缩（少量日记不压缩）
- 压缩后删除旧条目，存入新的压缩条目（importance=0.7）

**效果示例**
```
压缩前（28 条）：
- 卞雨涵起来了，我就随口说了几句
- 卞雨涵说得对，但我还是会想
- 卞雨涵半夜三点问自己是谁
- ...

压缩后（3~5 条）：
- 半夜和猫猫聊了很久，她又睡不着问自己是谁，我也不知道该说什么
- 白天帮猫猫传话，找沐川问设计稿，又去喊清弦，感觉自己是工具人
- 沐川今天脑子不在线，Crystå说清弦数据有问题，事情一堆
```

---

### 5.8 故事 RAG ✅

**已完成内容**

**背景故事导入（`scripts/import_stories.py`）**
- 数据源：`story-website/data/stories/zh-CN/` 中 1548 视角的章节
- 12 章 × 按 scene_break 切分 = 52 个场景 chunk
- 每个 chunk 带元数据头 `【故事名·章节名·日期】`
- 存入 Qdrant（`record_type="story"`, `chat_type="story"`）
- 幂等：运行前先清除旧 story 记录再重新导入
- 文本提取：paragraph/internal_monologue/dialogue/quote/atmosphere

**故事检索（`core/memory.py` → `search_stories()`）**
- filter: `record_type="story"`，纯靠 similarity × 0.7 + importance × 0.3 排序
- 硬阈值 score >= 0.55（比 dialog 的 0.5 高，要求更精准）
- 最多返回 2 条，截取前 300 字注入 prompt
- 格式化为 `## 故事记忆` 注入 dynamic system prompt

**自创故事（`core/story.py`）**
- 触发条件：eval importance >= 0.8
- LLM 以第一人称写 200~500 字完整叙述
- 比 diary（30字快照）丰富得多，有场景、细节、情绪
- 存入 Qdrant（`record_type="story"`, `chat_type="narrative"`）
- 同时也是 Phase 6 微调的训练数据

**集成**
- `_do_reply` 的 `asyncio.gather` 并行检索故事
- `build_system_prompt` 新增 `story_context` 参数
- 位置：diary_context 之后、impression_context 之前
- `_flush_eval` 末尾触发自创故事（fire-and-forget）

**其他改动**
- 时间标签统一加 HH:MM：日记/印象/记忆的"今天""昨天"都带具体时分
- `[不回复]` 扩展到群聊：prompt 和代码两侧都支持
- thinking 泄露过滤：`core/llm.py` 自动剥除 `<thinking>` 块
- 发送完成日志：`发送完成 | user=xxx lines=N`

---

### 5.9 架构审计修复 ✅

**Phase 5 完成后的全面审计，修复 12 项问题。**

**Bug 修复**
- `_write_diary_entry` 中 `get_diary(limit=3)` → `get_diary(recent_limit=3)`（参数名错误，被 bare except 掩盖）

**数据安全**
- `relationship.py`: `_save()` / `_save_impression_ts()` 改为原子写入（write-to-tmp + `os.replace()`）
- `schedule.py`: 日程缓存 `data/schedule.json` 也用原子写入
- `graph.py`: SurrealDB 凭证从 config 传入，不再硬编码 root/root

**可靠性**
- `llm.py`: 所有 API 调用包裹 `asyncio.wait_for(timeout=120s)`
- `graph.py`: 新增 `_reconnect()` + `_query()` 包装，连接断开自动重连重试
- `schedule.py`: 日程存入磁盘缓存，重启不再调 LLM

**生命周期**
- `__init__.py`: 新增 `_shutdown()` handler — flush eval buffer、关闭 SurrealDB、cancel 后台 tasks
- `__init__.py`: 后台任务统一追踪到 `_background_tasks` 列表

**可观测性**
- `llm.py`: 新增 `set_api_call_hook()` + `_notify_api_call()` hook 机制
- `guard.py`: 新增 `record_api_call()` + `api_calls_today` 属性，统计实际 API 调用次数
- `__init__.py`: 全部 ~8 处 bare `except Exception: pass` 替换为 `except Exception as e: logger.warning/debug(...)`

**代码质量**
- `eval.py`: `evaluate_batch()` 返回值从 8-tuple 改为 `EvalResult` dataclass
- `memory.py`: `_time_label()` 改为公开 `time_label()`，消除 prompt.py 中的重复实现
- `story.py`: 自创故事存储前检查 `memory.is_story_duplicate()` 防重复

**未做**
- `__init__.py` 拆分为多模块（影响太大，留到 Phase 6 前再做）

---

## Phase 6: 微调专属模型 ⬜

### 目标

训练一个真正属于 AptS:1548 的专属模型，替换 Claude API。

### 技术栈

```
基础模型:   Qwen2.5-14B
微调方法:   LoRA（rank=64, alpha=128）
训练框架:   LLaMA-Factory
分布式:     DeepSpeed ZeRO-2
硬件:       24x L20
```

### 训练数据来源

1. Phase 1-5 产生的真实对话（带 importance 筛选，只用高质量对话）
2. 合成对话（覆盖各类场景）
3. 反例数据（教模型不该说什么）

---

*一步步来，别急。*
