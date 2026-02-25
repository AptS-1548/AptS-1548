# 实现阶段

## 总览

```
Phase 1: 基础行为系统      ✅ 完成   ← 注意力、防抖、冷却、打字延迟
Phase 2: 长期记忆          ✅ 完成   ← Qdrant 向量存储、检索注入、重启重建
Phase 3: 记忆进化          ✅ 完成   ← 评估系统、分层印象、事实提取、周期摘要
Phase 4: 关系系统          ✅ 完成   ← 信任等级、用户画像、敌对机制
Phase 5: 主动行为          🚧 进行中 ← diary → 日程 → 主动关心 → 日记去重 → 故事RAG
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
- 向量无结果时 fallback 到最近 3 条 `dialog`

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
- 单次调用返回四个字段：
  - `importance`（0.0~1.0）：对话重要性
  - `impression`（≤50字）：48 视角的主观印象
  - `trust_delta`（-5~+5 整数）：对这次互动的信任评估
  - `facts`（list）：从用户发言中提取的事实信息
- owner 的消息跳过 eval，直接存 dialog（省 API）

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

## Phase 5: 主动行为 🚧

### 目标

48 有自己的内心记录，能感知自身状态，并在适当时机主动联系 owner。

### 5.1 自我记录系统（diary）✅

**已完成内容**

**diary 生成（`core/diary.py`）**
- 高 importance 对话后（`>= 0.6`）或关键信任变化（`|trust_delta| >= 3`）自动触发
- LLM 以第一人称写 ≤30 字日记，口语化，提到人用名字不用代词
- 输入：对话内容 + 信任度 + 最近印象 + 最近已有日记（防重复）

**存储**
- `record_type="diary"`, `user_id="48"`，与用户记忆隔离
- 存入 Qdrant，不参与用户维度的检索
- **语义去重**：存储前用向量搜索最近 24h 的日记，cosine >= 0.85 视为重复跳过
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

### 5.4 故事 RAG ⬜

**现有故事检索**
- 数据源：`weare-website/story-website` 中的故事章节
- 故事切 chunk 存 Qdrant（`record_type="story"`）
- 对话涉及 48 经历的具体细节时，检索相关段落补充 context
- prompt 里写了大事件概要，RAG 补细节（数据塔具体过程、诞生那天的对话等）

**48 自创故事**
- 高 importance 事件后触发，48 用第一人称写一段完整叙述（几百字）
- 比 diary（30字快照）更丰富，是她对经历的完整记录
- 存入 Qdrant 作为 RAG 素材，同时也是 Phase 6 微调的训练数据

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
