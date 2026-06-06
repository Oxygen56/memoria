# 完整版（知乎/掘金）

## 我给 AI Agent 装了一个"海马体"——开源记忆系统 Memoria

> 人脑会遗忘，这不是 bug，而是 feature。

---

### 一个真实的痛点

你有没有遇到过这种情况：让 AI Agent 帮你管理日程、写代码、做数据分析，第二天再来，它完全不记得昨天聊了什么？

于是你开始用各种方案给 Agent 加"记忆"：往向量数据库里塞所有对话、用 JSON 存关键信息、维护一个无限增长的 summary……

**然后新的问题来了：**

- 记忆越积越多，检索越来越慢，噪声越来越大
- 三个月前的"我喜欢吃川菜"和昨天的"最近在减盐"互相矛盾，Agent 不知道该信哪个
- 一个简单的问候，Agent 把你三年前的所有对话都翻出来"参考"

本质上，**现有方案把记忆当成了数据库 CRUD 问题，但人类记忆从来不是这样工作的。**

---

### Memoria：让 AI 像人一样记忆

[Memoria](https://github.com/Oxygen56/memoria) 是一个开源的 AI Agent 记忆基础设施，核心理念是：**记忆应该有生命周期、会自然衰减、能主动浮现、可相互关联。**

它不是又一个向量数据库的封装，而是一套完整的认知记忆系统。

---

### 技术亮点

#### 1. Ebbinghaus 遗忘曲线：数学公式解决记忆膨胀

受认知心理学启发，Memoria 用艾宾浩斯遗忘曲线建模记忆衰减：

```
retention = e^(-t / half_life)
```

不同类型的记忆有不同的半衰期：

| 记忆类型 | 半衰期 | 说明 |
|---------|--------|------|
| FACT | 180 天 | 事实性知识，衰减慢 |
| EVENT | 90 天 | 事件记忆，中等衰减 |
| PREFERENCE | 365 天 | 偏好记忆，非常稳定 |
| SKILL | 270 天 | 技能记忆，较为持久 |

记忆强度低于阈值后自动进入 COLD → OBLIVION 状态，最终被回收。**不需要手动清理，系统自己"遗忘"。**

#### 2. 主动召回：不查询也能想起来

传统方案是"用户问 → 检索 → 返回"。但人脑不是这样的——你看到一首歌，会突然想起某个夏天。

Memoria 的 **Awareness Engine（感知引擎）** 实现了主动召回：分析当前对话上下文，将相关记忆主动注入，不需要显式查询。

这意味着 Agent 可以"灵光一现"——在你聊工作的时候主动提起"上次你提到的那个方案"。

#### 3. 知识图谱引擎：记忆不是孤岛

基于 NetworkX 构建实体关系图，自动从记忆中提取实体和关系，支持多跳 BFS 推理。

比如：记住"张三是你的同事"和"张三在做推荐系统"，当你问"有没有人做过推荐系统"时，Memoria 能通过图关系找到张三。

#### 4. 四引擎协同 + 矛盾检测

Orchestrator 协调四个引擎协同工作：

- **衰减引擎**：管理记忆生命周期
- **感知引擎**：主动召回
- **图引擎**：关联推理
- **矛盾检测**：支持 LLM / CrossEncoder / 启发式规则三种 Provider，自动发现并解决冲突记忆

#### 5. 三层热度存储 + 存储适配器

```
HOT（始终注入）→ WARM（按需检索）→ COLD（归档）→ OBLIVION（待删除）
```

存储层完全解耦，支持 InMemory / ChromaDB / PgVector 随意切换，零 vendor lock-in，作为 SDK 嵌入你的项目，无需额外部署 Server。

---

### 竞品对比

| 能力 | Memoria | Mem0 | Letta | Zep |
|------|---------|------|-------|-----|
| 遗忘衰减 | ✅ Ebbinghaus 数学模型 | ❌ | ❌ | ❌ |
| 主动召回 | ✅ 感知引擎 | ❌ | ❌ | 有限 |
| 知识图谱 | ✅ NetworkX 多跳 | ❌ | ❌ | ❌ |
| 矛盾检测 | ✅ 多 Provider | ❌ | ❌ | 有限 |
| 开源程度 | 全部开源 | 全部开源 | 全部开源 | 核心闭源 |
| 部署方式 | 嵌入式 SDK | 需要 Server | 需要 Server | SaaS |

---

### 5 行代码上手

```python
from memoria import Memoria

agent_memory = Memoria(user_id="zhangsan")

# 存入记忆
await agent_memory.remember("用户喜欢用 Python 写后端，偏好 FastAPI 框架")

# 召回相关记忆（自动带衰减权重和图关联加分）
memories = await agent_memory.recall("帮我搭建一个 Web 服务")
print(memories)  # → 返回相关记忆，按综合评分排序
```

---

### 适用场景

- 个人 AI 助手（长期记住用户偏好）
- 客服 Agent（记住客户历史，主动关联上下文）
- 代码助手（记住项目架构和编码习惯）
- 多 Agent 系统（共享记忆层）

---

### 写在最后

Memoria 目前处于早期阶段，核心功能已经可用。如果你也在做 AI Agent 相关的工作，欢迎：

⭐ **GitHub Star**：https://github.com/Oxygen56/memoria

我们相信，**记忆是 Agent 从"工具"进化为"伙伴"的关键一步**。好的记忆系统不是记住所有东西，而是像人一样——记住重要的，忘掉该忘的，在对的时候想起对的事。

欢迎在评论区讨论你对 AI 记忆的看法，或者你在实际项目中遇到的记忆管理痛点。

---
---

# V2EX 简短版

## 开源了一个 AI Agent 记忆系统 Memoria — 会遗忘的记忆才是好记忆

做 AI Agent 开发的朋友应该都遇到过记忆管理的问题：记忆无限膨胀、旧信息和新信息矛盾、检索噪声大。

我做了 [Memoria](https://github.com/Oxygen56/memoria)，一个 Python 开源项目，核心思路是**让 AI 像人一样记忆**：

**几个关键设计：**

1. **Ebbinghaus 遗忘曲线** — 记忆有半衰期，自动衰减回收，不用手动清理
2. **主动召回** — 感知引擎分析上下文，主动注入相关记忆，不需要显式查询
3. **知识图谱** — NetworkX 实体关系 + 多跳推理，记忆之间有关联
4. **矛盾检测** — 支持 LLM/CrossEncoder/启发式规则，自动发现冲突
5. **存储解耦** — InMemory/ChromaDB/PgVector 随意切换，嵌入式 SDK 无需部署 Server

**对比 Mem0/Letta/Zep：** 唯一同时支持遗忘衰减 + 主动召回 + 知识图谱的方案，且完全开源、嵌入式部署。

```python
from memoria import Memoria
memory = Memoria(user_id="test")
await memory.remember("用户偏好 FastAPI")
results = await memory.recall("搭建 Web 服务")
```

GitHub: https://github.com/Oxygen56/memoria

欢迎 Star、提 Issue、讨论。如果你也在做 Agent 记忆相关的事情，很想听听大家的思路。
