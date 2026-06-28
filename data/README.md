# 测试数据

> 用于产品演示和功能测试，**不是真实用户数据**。

## 文件说明

| 文件 | 用途 |
|------|------|
| `mock-conversations.json` | 5 组模拟 AI 对话，覆盖正常咨询、饮食、伤病安全、计划推荐、动作纠正等场景 |
| `sample-training.sql` | 一个虚拟用户「小王」的 2 周训练数据，包含用户档案、训练记录、训练反馈、身体测量、训练计划 |

## data/mock-conversations.json

包含 5 个场景的完整对话，每个场景标注了：
- 用户画像（身高/体重/目标/经验）
- 意图检测结果
- RAG 检索到的知识库文档
- 安全预检（如有触发）

**场景列表**：
1. 深蹲膝盖内扣 — 正常训练咨询
2. 增肌期饮食方案 — 营养咨询
3. 卧推肩膀疼 — 伤病安全（触发安全预检，不调 LLM）
4. 今天练什么 — 训练计划推荐（基于训练历史）
5. 引体向上标准姿势 — 动作纠正

## data/sample-training.sql

一个虚拟用户 2 周的完整数据，演示渐进超负荷的合理进展：
- 卧推：65kg → 67.5kg（+2.5kg）
- 深蹲：80kg → 85kg（+5kg）
- 引体向上：6个 → 7个（+1个）
- 体重：72.0kg → 73.0kg（+1kg，增肌期合理速度）

## 如何使用

### 本地测试对话

mock-conversations.json 的格式与前端 POST /api/chat 接口兼容，可直接用 curl 测试：

```bash
# 测试对话
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我深蹲膝盖内扣怎么办", "session_id": "test-001"}'

# 测试安全预检
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我卧推肩膀疼", "session_id": "test-002"}'
```

### 导入测试数据到数据库

```bash
# 导入到本地 PostgreSQL
psql -U fitness -d fitness -f sample-training.sql
```
