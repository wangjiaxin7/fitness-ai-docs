# API 设计

## 1. 概述

- **协议**：HTTP/HTTPS
- **格式**：JSON
- **认证**：Session（Cookie）
- **流式响应**：SSE（Server-Sent Events）用于 AI 对话

## 2. API 列表

### 2.1 用户相关

#### POST /login — 登录

**请求**：`application/x-www-form-urlencoded`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| username | string | ✅ | 用户名 |
| password | string | ✅ | 密码 |

**响应**：
- 成功：302 重定向到 /chat
- 失败：200 返回登录页面（含错误提示）

#### POST /register — 注册

**请求**：`application/x-www-form-urlencoded`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| username | string | ✅ | 用户名 |
| password | string | ✅ | 密码 |
| confirm | string | ✅ | 确认密码 |
| display_name | string | ❌ | 显示名称 |

**响应**：
- 成功：302 重定向到 /chat
- 失败：200 返回注册页面（含错误提示）

#### GET /logout — 登出

**响应**：302 重定向到 /login

---

### 2.2 AI 对话

#### POST /api/chat — 发送消息（流式）

**请求**：`application/json`

```json
{
  "query": "深蹲膝盖内扣怎么办",
  "conversation_id": "conv_abc123"
}
```

**响应**：`text/event-stream`（SSE）

```
data: {"answer": "膝", "conversation_id": "conv_abc123"}
data: {"answer": "盖内扣", "conversation_id": "conv_abc123"}
data: {"answer": "是一个", "conversation_id": "conv_abc123"}
...
data: [DONE]
```

**处理流程**：
1. 伤病关键词预检 → 命中则直接返回就医建议
2. 加载用户档案 → 拼接到 query 前缀
3. RAG 知识检索 → Top-3 相关文档
4. 构建 Prompt → 系统提示词 + RAG + 历史 + 档案
5. 调用 DeepSeek API（stream=True）
6. 流式转发到前端

#### GET /api/conversations — 获取对话列表

**响应**：
```json
[
  {
    "id": "conv_abc123",
    "title": "深蹲膝盖内扣",
    "created_at": "2026-06-01T10:00:00Z"
  }
]
```

#### GET /api/conversations/{id}/messages — 获取对话消息

**响应**：
```json
[
  {
    "role": "user",
    "content": "深蹲膝盖内扣怎么办",
    "created_at": "2026-06-01T10:00:00Z"
  },
  {
    "role": "assistant",
    "content": "膝盖内扣通常是臀中肌无力...",
    "created_at": "2026-06-01T10:00:01Z"
  }
]
```

---

### 2.3 训练记录

#### GET /api/workouts — 获取训练记录列表

**查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | integer | ❌ | 页码（默认 1） |
| per_page | integer | ❌ | 每页数量（默认 20） |
| exercise | string | ❌ | 按动作筛选 |
| start_date | string | ❌ | 开始日期（YYYY-MM-DD） |
| end_date | string | ❌ | 结束日期（YYYY-MM-DD） |

**响应**：
```json
{
  "records": [
    {
      "id": "rec_abc123",
      "exercise_name": "深蹲",
      "sets": 4,
      "reps": 10,
      "weight": 80.0,
      "duration": 30,
      "created_at": "2026-06-01T10:00:00Z"
    }
  ],
  "total": 50,
  "page": 1,
  "per_page": 20
}
```

#### POST /api/workouts — 新增训练记录

**请求**：`application/json`

```json
{
  "exercise_name": "深蹲",
  "sets": 4,
  "reps": 10,
  "weight": 80.0,
  "duration": 30,
  "notes": "今天状态不错"
}
```

**响应**：
```json
{
  "id": "rec_abc123",
  "message": "记录成功"
}
```

---

### 2.4 动作分析

#### POST /api/analyze — 上传视频分析

**请求**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| video | file | ✅ | 训练视频文件 |
| exercise | string | ❌ | 动作名称（不填则自动检测） |

**响应**：
```json
{
  "analysis_id": "ana_abc123",
  "exercise": "squat",
  "overall_score": 7.5,
  "dimensions": {
    "amplitude": {"score": 8.0, "comment": "下蹲幅度到位"},
    "stability": {"score": 7.0, "comment": "有轻微晃动"},
    "symmetry": {"score": 8.5, "comment": "左右均衡"},
    "rhythm": {"score": 6.5, "comment": "下蹲速度偏快"},
    "control": {"score": 7.5, "comment": "离心控制良好"}
  },
  "coach_comment": "整体动作质量不错，主要问题是下蹲节奏偏快...",
  "chart_url": "/static/analysis/ana_abc123_angles.png"
}
```

---

### 2.5 用户档案

#### GET /api/profile — 获取当前用户档案

**响应**：
```json
{
  "name": "王佳信",
  "height": 175,
  "weight": 70,
  "age": 22,
  "goal": "增肌",
  "experience": "1年",
  "equipment": "哑铃、杠铃、引体向上杆"
}
```

#### PUT /api/profile — 更新用户档案

**请求**：`application/json`

```json
{
  "height": 175,
  "weight": 72,
  "goal": "增肌减脂"
}
```

---

### 2.6 训练反馈

#### POST /api/feedback — 提交训练反馈

**请求**：`application/json`

```json
{
  "record_id": "rec_abc123",
  "fatigue_level": 7,
  "pain_level": 0,
  "satisfaction": 8,
  "notes": "今天深蹲感觉很好"
}
```

---

### 2.7 健康检查

#### GET /api/health — 服务健康检查

**响应**：
```json
{
  "status": "ok",
  "database": "connected",
  "deepseek_api": "reachable",
  "timestamp": "2026-06-01T10:00:00Z"
}
```

## 3. 错误响应格式

所有 API 错误返回统一格式：

```json
{
  "error": "错误描述信息"
}
```

| HTTP 状态码 | 含义 |
|------------|------|
| 200 | 成功 |
| 302 | 重定向 |
| 400 | 请求参数错误 |
| 401 | 未登录 |
| 403 | 无权限 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

## 4. 安全设计

| 措施 | 说明 |
|------|------|
| Session 认证 | 基于 Cookie 的 Session，HttpOnly + Secure |
| CSRF | 表单提交使用 POST 方法 |
| SQL 注入 | 所有 SQL 使用参数化查询 |
| XSS | 用户输入过滤和转义 |
| 密码存储 | SHA256 + 随机 salt |
| API Key | 环境变量管理，不硬编码 |
| 伤病安全 | 代码层关键词预检 + 提示词红线 |
