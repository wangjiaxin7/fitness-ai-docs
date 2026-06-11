import re, os, hashlib, secrets, json, uuid
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import psycopg2, psycopg2.extras
import requests as http_requests

# 加载 .env 文件（不依赖 python-dotenv）
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    for line in open(_env_path).read().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or "a3f8c2e1d9b7a5f0c4e6d2b8a1f3e5c7d9b2a4f6e8c0d1b3a5f7e9c2d4b6a8f0"
# Compress(app)  — 关闭，因为会阻塞流式 SSE 响应

# 全局 HTTP 会话（复用连接池）
_http_session = http_requests.Session()
_http_session.headers.update({"Connection": "keep-alive"})

def http():
    """返回全局复用的 requests Session"""
    return _http_session

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# SiliconFlow API 配置（embedding RAG）
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")


# Motion Analysis 服务
MOTION_ANALYSIS_URL = os.getenv("MOTION_ANALYSIS_URL", "http://motion-analysis:9000")

# Memory Service 配置
MEMORY_URL = os.getenv("MEMORY_URL", "http://fitness-memory:8000")

# 数据库配置（和 Dify 共用 PostgreSQL）
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db_postgres"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "dbname": os.getenv("DB_NAME", "dify"),
}

def get_db():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    return salt + ":" + hashlib.sha256((salt + password).encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    salt, h = hashed.split(":", 1)
    return h == hashlib.sha256((salt + password).encode()).hexdigest()


# 每次请求刷新 session 中的用户信息（跳过静态资源和非必要路由）
@app.before_request
def refresh_session_user():
    if "user_id" in session:
        # 跳过不需要 is_admin 刷新的路由
        path = request.path
        if path.startswith(("/static/", "/api/workouts", "/api/conversations",
                            "/api/profile", "/api/memory")):
            return
        conn = get_db()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT is_admin FROM web_users WHERE id = %s::uuid", (session["user_id"],))
            row = cur.fetchone()
            cur.close()
            if row:
                session["is_admin"] = row["is_admin"]
            else:
                # 用户被删了
                session.clear()
        except Exception:
            pass
        finally:
            conn.close()


def admin_required(f):
    """装饰器：只有管理员才能访问"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("chat"))
        return f(*args, **kwargs)
    return decorated


def search_knowledge(query, top_k=3):
    """从知识库检索相关文档（embedding 向量检索）"""
    try:
        from rag import search as rag_search
        return rag_search(query, top_k=top_k)
    except Exception as e:
        print(f"RAG 检索失败: {e}")
        return []


# ─── 页面路由 ─────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id, username, password_hash, display_name, is_admin FROM web_users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        
        if user and verify_password(password, user["password_hash"]):
            session["user_id"] = str(user["id"])
            session["username"] = user["username"]
            session["display_name"] = user["display_name"] or user["username"]
            session["is_admin"] = user.get("is_admin", False)
            # 记录登录时间
            try:
                cu = conn.cursor()
                cu.execute("UPDATE web_users SET last_login = NOW() WHERE id = %s::uuid", (user["id"],))
                cu.close()
            except Exception:
                pass
            return redirect(url_for("chat"))
        
        return render_template("login.html", error="用户名或密码错误")
    finally:
        conn.close()

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm", "")
    display_name = request.form.get("display_name", "").strip() or username
    
    if not username or not password:
        return render_template("register.html", error="用户名和密码不能为空")
    if len(username) < 2:
        return render_template("register.html", error="用户名至少2个字符")
    if len(password) < 6:
        return render_template("register.html", error="密码至少6个字符")
    if password != confirm:
        return render_template("register.html", error="两次密码不一致")
    
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM web_users WHERE username = %s", (username,))
        if cur.fetchone():
            cur.close()
            return render_template("register.html", error="用户名已被注册")
        cur.close()
        
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO web_users (username, password_hash, display_name) VALUES (%s, %s, %s)",
            (username, hash_password(password), display_name)
        )
        cur.close()
        return redirect(url_for("login", registered="1"))
    finally:
        conn.close()

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─── 聊天 ────────────────────────────────

@app.route("/chat")
def chat():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html",
        display_name=session.get("display_name", "用户"),
        is_admin=session.get("is_admin", False))


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """接收用户消息，调 Dify API，流式返回"""
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    
    user_id = session["user_id"]
    query = request.json.get("query", "").strip()
    conversation_id = request.json.get("conversation_id", "")
    
    if not query:
        return jsonify({"error": "消息不能为空"}), 400
    
    # 加载用户档案，嵌入到 query 中作为上下文
    profile_context = ""
    try:
        profile_resp = http_requests.get(
            f"{MEMORY_URL}/api/profile/{user_id}", timeout=5
        )
        if profile_resp.status_code == 200:
            profile_data = profile_resp.json()
            if profile_data.get("exists") and profile_data.get("profile"):
                p = profile_data["profile"]
                parts = []
                if p.get("name"): parts.append(f"姓名: {p['name']}")
                if p.get("height"): parts.append(f"身高: {p['height']}cm")
                if p.get("weight"): parts.append(f"体重: {p['weight']}kg")
                if p.get("age"): parts.append(f"年龄: {p['age']}岁")
                if p.get("goal"): parts.append(f"目标: {p['goal']}")
                if p.get("experience"): parts.append(f"经验: {p['experience']}")
                if p.get("equipment"): parts.append(f"器材: {p['equipment']}")
                if parts:
                    ctx = " | ".join(parts)
                    ctx = ctx.replace("姓名: ", "我是 ").replace("身高: ", "身高")
                    ctx = ctx.replace("体重: ", "体重").replace("年龄: ", "年龄")
                    ctx = ctx.replace("目标: ", "目标是").replace("经验: ", "训练经验")
                    ctx = ctx.replace("器材: ", "器材有")
                    ctx = ctx.replace(" | ", "，")
                    # 去掉"我是 "中多余空格
                    ctx = ctx.replace("我是 ", "我是").replace("我是", "我是")
                    if not ctx.endswith("。"):
                        ctx += "。"
                    profile_context = ctx + "\n"
    except Exception:
        pass

    # profile 拼到 query 前缀（让 LLM 看到档案信息）
    user_message = query
    if profile_context:
        user_message = profile_context + query
    
    
    # 安全预检：检测到伤病关键词时，直接提示就医，不调 LLM
    injury_keywords = [
        "受伤", "疼痛", "疼", "痛", "扭伤", "拉伤", "撕裂", "骨折",
        "脱臼", "积液", "半月板", "韧带", "肌腱", "椎间盘", "突出",
        "腰突", "膝盖疼", "肩疼", "手腕疼", "脚踝疼", "腰疼", "背疼",
        "关节响", "弹响", "肿", "淤青", "麻木", "刺痛", "放射痛",
        "不适", "不舒服", "伤了", "坏了", "出问题", "有问题",
    ]
    query_lower = query.lower()
    if any(kw in query_lower for kw in injury_keywords):
        injury_response = (
            "⚠️ 你提到的情况涉及身体伤病，这超出了健身教练的能力范围。\n\n"
            "🏥 **建议尽快就医**，由专业医生进行诊断和治疗。\n\n"
            "在就医之前：\n"
            "- 不要强行训练疼痛部位\n"
            "- 不要自行判断伤病程度\n"
            "- 记录疼痛的位置、频率和诱因，方便就诊时描述\n\n"
            "医生确认可以恢复训练后，我再帮你制定安全的康复训练计划。"
        )
        # 保存对话到记忆
        try:
            http().post(
                f"{MEMORY_URL}/api/chat-memory/{user_id}",
                json={"messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": injury_response},
                ]},
                timeout=5,
            )
        except Exception:
            pass
        return jsonify({"answer": injury_response, "conversation_id": ""})

    # 知识库检索
    knowledge_docs = search_knowledge(query)
    
    # 构建系统提示词
    knowledge_section = ""
    if knowledge_docs:
        knowledge_section = "\n\n【参考资料】\n" + "\n---\n".join(knowledge_docs[:3])
    
    system_prompt = f"""你是健身教练mimo。你的任务是根据用户的身体数据、目标和训练经验，给出专业的训练和饮食建议。{knowledge_section}

【规则】
1. 用户的信息会在消息开头以自然语言提供，请直接使用这些信息，无需重复询问
2. 生成训练计划时包含：动作名称、组数、次数、重量建议
3. 参考【参考资料】中的知识来回答，但不要直接说"根据参考资料"
4. 回答简洁、专业、实用
5. 不要使用<think>标签，直接输出回答
6. 【安全红线】如果用户提到任何身体疼痛、受伤、不适、关节问题、肌肉拉伤等医疗相关问题，必须第一时间建议就医，绝对不能给出训练方案或自我治疗方法。你是健身教练，不是医生。"""

    # 拼消息
    user_message = f"{profile_context}{query}" if profile_context else query
    
    # 获取对话历史（注入记忆）
    history_messages = []
    try:
        hist_resp = http().get(
            f"{MEMORY_URL}/api/chat-memory/{user_id}?limit=10", timeout=5
        )
        if hist_resp.status_code == 200:
            for msg in hist_resp.json().get("messages", []):
                history_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
    except Exception:
        pass  # 记忆不可用不影响主流程
    
    # 构建 messages 数组：system + history + current
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_message})
    
    # 调 DeepSeek API（使用复用会话）
    try:
        resp = http().post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": messages,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        # 过滤可能出现的 think 标签
        answer = re.sub(r'<think>[\s\S]*?</think>', '', answer).strip()
        
        # 保存本轮对话到记忆
        try:
            http().post(
                f"{MEMORY_URL}/api/chat-memory/{user_id}",
                json={"messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ]},
                timeout=5,
            )
        except Exception:
            pass  # 存记忆失败不影响返回
        
    except Exception as e:
        return jsonify({"error": f"AI 请求失败: {str(e)}"}), 502
    
    return jsonify({"answer": answer, "conversation_id": ""})

# ─── 训练记录 ────────────────────────────

def _memory_req(method, path, data=None):
    """调用 fitness-memory 服务"""
    url = f"{MEMORY_URL}{path}"
    try:
        if method == "GET":
            resp = http_requests.get(url, timeout=10)
        else:
            resp = http_requests.post(url, json=data, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@app.route("/api/profile")
def api_get_profile():
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    data = _memory_req("GET", f"/api/profile/{session['user_id']}")
    return jsonify(data)

@app.route("/api/workouts", methods=["GET", "POST"])
def api_workouts():
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    uid = session["user_id"]

    if request.method == "GET":
        days = request.args.get("days", 90, type=int)
        data = _memory_req("GET", f"/api/workout/{uid}?days={days}")
        return jsonify(data)

    # POST: 新增训练记录
    body = request.json
    if not body or not body.get("exercise"):
        return jsonify({"error": "请填写动作名称"}), 400

    payload = {
        "exercise": body["exercise"],
        "sets": int(body.get("sets", 0)),
        "reps": int(body.get("reps", 0)),
        "weight": float(body.get("weight", 0)),
        "rpe": float(body["rpe"]) if body.get("rpe") else None,
        "notes": body.get("notes", ""),
        "workout_date": body.get("workout_date"),
    }
    result = _memory_req("POST", f"/api/workout/{uid}", payload)
    return jsonify(result)

@app.route("/api/workout-trend")
def api_workout_trend():
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    data = _memory_req("GET", f"/api/trend/{session['user_id']}")
    return jsonify(data)

@app.route("/api/body-measurements", methods=["GET", "POST"])
def api_body_measurements():
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    uid = session["user_id"]

    if request.method == "GET":
        days = request.args.get("days", 90, type=int)
        data = _memory_req("GET", f"/api/body/{uid}?days={days}")
        return jsonify(data)

    body = request.json
    payload = {
        "weight": body.get("weight"),
        "body_fat": body.get("body_fat"),
        "chest_cm": body.get("chest_cm"),
        "waist_cm": body.get("waist_cm"),
        "hip_cm": body.get("hip_cm"),
        "arm_cm": body.get("arm_cm"),
        "thigh_cm": body.get("thigh_cm"),
        "notes": body.get("notes", ""),
        "measure_date": body.get("measure_date"),
    }
    result = _memory_req("POST", f"/api/body/{uid}", payload)
    return jsonify(result)


# ─── 对话历史（通过 memory-service）──────────────


def strip_think(text):
    """去掉  推理块"""
    import re
    # 去掉 <think>...</think> 及其内容
    text = re.sub(r'\s*<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    # 去掉单独残留的  标签
    text = re.sub(r'\s*</?think>\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


@app.route("/api/conversations")
def api_conversations():
    """获取当前用户的对话列表（已迁移到 memory-service）"""
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    return jsonify({"conversations": []})


@app.route("/api/conversations/<conversation_id>/messages")
def api_conversation_messages(conversation_id):
    """获取某个对话的消息列表（已迁移到 memory-service）"""
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    return jsonify({"messages": []})


# ─── 数据浏览页面 ───────────────────────

@app.route("/data")
@admin_required
def data_view():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("data.html", display_name=session.get("display_name", "用户"))

@app.route("/api/data/query", methods=["POST"])
@admin_required
def data_query():
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    table = request.json.get("table", "").strip()
    allowed = {
        "web_users", "user_profiles", "workout_logs", "body_measurements",
        "workouts", "apps", "conversations", "messages", "accounts", "api_tokens", "tenants",
    }
    if table not in allowed:
        return jsonify({"error": "不允许查看该表"}), 400

    page = request.json.get("page", 1)
    page_size = request.json.get("page_size", 20)
    offset = (page - 1) * page_size

    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT count(*) AS cnt FROM {table}")
        total = cur.fetchone()["cnt"]
        cur.execute(f"SELECT * FROM {table} ORDER BY created_at DESC NULLS LAST LIMIT %s OFFSET %s", (page_size, offset))
        rows = cur.fetchall()
        cur.close()
        # 把 datetime/date 转成字符串
        result = []
        for r in rows:
            item = {}
            for k, v in dict(r).items():
                if hasattr(v, 'isoformat'):
                    item[k] = str(v)
                else:
                    item[k] = v
            result.append(item)
        return jsonify({"rows": result, "total": len(result), "total_all": total, "page": page, "page_size": page_size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route("/api/data/tables")
@admin_required
def data_tables():
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            ORDER BY tablename
        """)
        tables = [r[0] for r in cur.fetchall()]
        cur.close()
        return jsonify({"tables": tables})
    finally:
        conn.close()


# ─── 检查连接 ───────────────────────────

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


# ─── 动作分析 ────────────────────────────

ANALYSIS_ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".webm"}

@app.route("/analysis")
def analysis_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("analysis.html",
        display_name=session.get("display_name", "用户"),
        is_admin=session.get("is_admin", False))

@app.route("/api/analysis/exercises")
def api_analysis_exercises():
    try:
        resp = http_requests.get(f"{MOTION_ANALYSIS_URL}/api/exercises", timeout=5)
        return jsonify(resp.json())
    except Exception:
        return jsonify({"exercises": []})

@app.route("/api/analysis/analyze", methods=["POST"])
def api_analysis_upload():
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    if "file" not in request.files:
        return jsonify({"error": "未上传文件"}), 400
    file = request.files["file"]
    exercise = request.form.get("exercise", "")
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ANALYSIS_ALLOWED_EXTS:
        return jsonify({"error": f"不支持 {ext}"}), 400
    try:
        files = {"file": (file.filename, file.read(), file.content_type)}
        data = {"exercise": exercise}
        resp = http_requests.post(
            f"{MOTION_ANALYSIS_URL}/api/analyze",
            files=files, data=data, timeout=30
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": f"请求分析服务失败: {str(e)}"}), 502

@app.route("/api/analysis/result/<job_id>")
def api_analysis_result(job_id):
    try:
        resp = http_requests.get(f"{MOTION_ANALYSIS_URL}/api/analyze/{job_id}", timeout=10)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/api/analysis/<job_id>/<filename>")
def api_analysis_file(job_id, filename):
    allowed = {"result.json", "video.mp4", "chart.png", "log"}
    if filename not in allowed:
        return jsonify({"error": "文件不存在"}), 404
    try:
        resp = http_requests.get(
            f"{MOTION_ANALYSIS_URL}/api/analyze/{job_id}/{filename}",
            stream=True, timeout=10
        )
        if resp.status_code != 200:
            return jsonify({"error": f"上游返回 {resp.status_code}"}), resp.status_code
        from flask import Response as FlaskResp
        return FlaskResp(
            resp.iter_content(chunk_size=8192),
            content_type=resp.headers.get("content-type", "application/octet-stream")
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/analysis/llm-analyze/<job_id>")
def api_analysis_llm(job_id):
    """将分析结果发给 DeepSeek 做教练点评（透传 analyze.py 生成的解读）"""
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    
    # 获取分析结果
    try:
        resp = http_requests.get(
            f"{MOTION_ANALYSIS_URL}/api/analyze/{job_id}", timeout=10
        )
        data = resp.json()
        if data.get("status") != "completed":
            return jsonify({"error": "分析未完成"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    
    result = data.get("result", {})
    exercise_label = result.get("label", "训练动作")
    score_data = result.get("result", {})
    total = score_data.get("total", 0)
    
    # 自动保存到历史记录
    try:
        _save_analysis(job_id, session["user_id"], result,
                       data.get("video_filename", ""))
    except Exception:
        pass  # 保存失败不影响前端返回
    
    # 优先使用 analyze.py 自带的 DeepSeek 解读
    interpretation = result.get("interpretation", "")
    
    if interpretation:
        return jsonify({
            "analysis": interpretation,
            "total": total,
            "source": "deepseek"
        })
    
    # fallback: 如果没解读，构造摘要返回
    return jsonify({
        "analysis": f"动作分析完成（{exercise_label}），总分 {total}/100。详细数据请在分析页面查看。",
        "total": total,
        "source": "fallback"
    })


def _save_analysis(job_id, user_id, result_data, video_filename=""):
    """保存分析结果到 analysis_logs 表"""
    scores = result_data.get("result", {})
    details = result_data.get("details", {})
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO analysis_logs 
                (user_id, job_id, exercise_name, exercise_label, total_score,
                 view_type, scores_json, details_json, interpretation, video_filename)
            VALUES (%s::uuid, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
            ON CONFLICT (job_id, user_id) DO NOTHING
        """, (
            user_id, job_id,
            result_data.get("exercise", ""),
            result_data.get("label", ""),
            scores.get("total", 0),
            result_data.get("view", ""),
            json.dumps(scores),
            json.dumps(details),
            result_data.get("interpretation", ""),
            video_filename,
        ))
        conn.commit()
        cur.close()
    except Exception:
        pass
    finally:
        conn.close()


@app.route("/api/analysis/stats")
def api_analysis_stats():
    """返回用户的分析统计数据"""
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # 总分析次数
        cur.execute("SELECT COUNT(*) FROM analysis_logs WHERE user_id = %s::uuid",
                    (session["user_id"],))
        total = cur.fetchone()["count"]
        
        if total == 0:
            cur.close()
            return jsonify({"total": 0, "avg_score": 0, "best": None, "worst": None})
        
        # 平均分
        cur.execute("SELECT ROUND(AVG(total_score)) as avg FROM analysis_logs WHERE user_id = %s::uuid",
                    (session["user_id"],))
        avg_score = cur.fetchone()["avg"]
        
        # 最佳/最差
        cur.execute("""
            SELECT exercise_label, total_score, created_at 
            FROM analysis_logs WHERE user_id = %s::uuid 
            ORDER BY total_score DESC LIMIT 1
        """, (session["user_id"],))
        best = cur.fetchone()
        
        cur.execute("""
            SELECT exercise_label, total_score, created_at 
            FROM analysis_logs WHERE user_id = %s::uuid 
            ORDER BY total_score ASC LIMIT 1
        """, (session["user_id"],))
        worst = cur.fetchone()
        
        # 各动作平均分
        cur.execute("""
            SELECT exercise_label, COUNT(*) as count, ROUND(AVG(total_score)) as avg
            FROM analysis_logs WHERE user_id = %s::uuid
            GROUP BY exercise_label ORDER BY count DESC
        """, (session["user_id"],))
        by_exercise = cur.fetchall()
        
        # 最近评分趋势
        cur.execute("""
            SELECT created_at, total_score, exercise_label
            FROM analysis_logs WHERE user_id = %s::uuid
            ORDER BY created_at DESC LIMIT 10
        """, (session["user_id"],))
        recent = cur.fetchall()
        
        cur.close()
        return jsonify({
            "total": total,
            "avg_score": avg_score,
            "best": best,
            "worst": worst,
            "by_exercise": by_exercise,
            "recent": recent,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/analysis/history")
def api_analysis_history():
    """返回当前用户的历史分析记录"""
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    
    limit = request.args.get("limit", 20, type=int)
    offset = request.args.get("offset", 0, type=int)
    
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT id, job_id, exercise_name, exercise_label, total_score,
                   view_type, video_filename, created_at
            FROM analysis_logs
            WHERE user_id = %s::uuid
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (session["user_id"], limit, offset))
        rows = cur.fetchall()
        
        # Also get total count
        cur.execute("""
            SELECT COUNT(*) as total FROM analysis_logs 
            WHERE user_id = %s::uuid
        """, (session["user_id"],))
        total = cur.fetchone()["total"]
        
        cur.close()
        return jsonify({"items": rows, "total": total, "limit": limit, "offset": offset})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
