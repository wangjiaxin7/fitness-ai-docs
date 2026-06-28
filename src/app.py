import re, os, hashlib, secrets, json, uuid, logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import psycopg2, psycopg2.extras
import psycopg2.pool
import requests as http_requests
import bcrypt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

# Rate limiting（防暴力破解）
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per minute"],
    storage_uri="memory://",
)

# 全局 HTTP 会话（复用连接池）
_http_session = http_requests.Session()
_http_session.headers.update({"Connection": "keep-alive"})

# Prompt 加载函数
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(name):
    """加载 prompts/ 目录下的 prompt 文件"""
    path = os.path.join(PROMPTS_DIR, f"{name}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def detect_intent(query):
    """简单意图检测：训练、饮食、通用"""
    q = query.lower()
    training_kw = ["训练", "计划", "增肌", "减脂", "健身", "练", "组", "次", "深蹲", "卧推", "硬拉", "弯举", "推举", "分化", "休息日", "热身", "拉伸", "肌肉", "力量", "耐力", "有氧"]
    diet_kw = ["吃", "饮食", "营养", "热量", "蛋白质", "碳水", "脂肪", "餐", "食谱", "食物", "喝", "补剂", "蛋白粉", "肌酸", "体重", "增重", "减重"]
    if any(kw in q for kw in training_kw):
        return "training"
    if any(kw in q for kw in diet_kw):
        return "diet"
    return "general"

def http():
    """返回全局复用的 requests Session"""
    return _http_session

# 连接池（避免每次请求都新建连接）
_pool = None
def get_db():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2, maxconn=10, **DB_CONFIG
        )
    conn = _pool.getconn()
    conn.autocommit = True
    return conn

def put_db(conn):
    global _pool
    if _pool:
        _pool.putconn(conn)

# 敏感词过滤（注册时使用）
SENSITIVE_WORDS = [
    "爹", "妈逼", "草泥", "操你", "傻逼", "狗逼", "贱", "煞笔",
    "sb", "nmsl", "fuck", "shit", "bitch",
]

def contains_sensitive_word(text: str) -> bool:
    text_lower = text.lower()
    return any(w in text_lower for w in SENSITIVE_WORDS)

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

# get_db 已移到上方（使用连接池）

def hash_password(password: str) -> str:
    """使用 bcrypt 哈希密码（自动加盐，故意设计慢以防止暴力破解）"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """验证密码，兼容旧的 SHA256 格式"""
    if hashed.startswith('$2b$') or hashed.startswith('$2a$'):
        # bcrypt 格式
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    elif ':' in hashed:
        # 旧的 SHA256 格式（兼容迁移期）
        salt, h = hashed.split(":", 1)
        return h == hashlib.sha256((salt + password).encode()).hexdigest()
    return False


# 缓存 is_admin 查询（避免每次请求都查数据库）
_admin_cache = {}  # {user_id: (is_admin, timestamp)}
_ADMIN_CACHE_TTL = timedelta(minutes=5)

@app.before_request
def refresh_session_user():
    if "user_id" not in session:
        return
    path = request.path
    if path.startswith(("/static/", "/favicon.ico")):
        return
    user_id = session["user_id"]
    now = datetime.now()
    # 检查缓存
    if user_id in _admin_cache:
        is_admin, ts = _admin_cache[user_id]
        if now - ts < _ADMIN_CACHE_TTL:
            session["is_admin"] = is_admin
            return
    # 缓存过期，查数据库
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT is_admin FROM web_users WHERE id = %s::uuid", (user_id,))
        row = cur.fetchone()
        cur.close()
        if row:
            session["is_admin"] = row["is_admin"]
            _admin_cache[user_id] = (row["is_admin"], now)
        else:
            session.clear()
            _admin_cache.pop(user_id, None)
    except Exception as e:
        logger.warning(f"刷新 session 失败: {e}")
    finally:
        put_db(conn)


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
        results = rag_search(query, top_k=top_k)
        # 返回格式: [{"text": ..., "title": ..., "source": ...}]
        return results
    except Exception as e:
        logger.warning(f"RAG 检索失败: {e}")
        return []


# ─── 页面路由 ─────────────────────────────

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"), "favicon.ico", mimetype="image/x-icon")

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")  # 防暴力破解
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
            # 更新密码哈希为 bcrypt（如果还是旧格式）
            if ':' in user["password_hash"] and not user["password_hash"].startswith('$2'):
                try:
                    cu = conn.cursor()
                    new_hash = hash_password(password)
                    cu.execute("UPDATE web_users SET password_hash = %s WHERE id = %s::uuid", (new_hash, user["id"]))
                    cu.close()
                    logger.info(f"用户 {username} 密码哈希已升级为 bcrypt")
                except Exception as e:
                    logger.warning(f"密码哈希升级失败: {e}")
            # 记录登录时间
            try:
                cu = conn.cursor()
                cu.execute("UPDATE web_users SET last_login = NOW() WHERE id = %s::uuid", (user["id"],))
                cu.close()
            except Exception as e:
                logger.warning(f"更新登录时间失败: {e}")
            logger.info(f"用户登录: {username}")
            return redirect(url_for("chat"))
        
        logger.warning(f"登录失败（密码错误）: {username}")
        return render_template("login.html", error="用户名或密码错误")
    except Exception as e:
        logger.error(f"登录异常: {e}")
        return render_template("login.html", error="登录失败，请稍后重试")
    finally:
        put_db(conn)

@app.route("/register", methods=["GET", "POST"])
@limiter.limit("5 per minute")  # 防刷注册
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
    if len(password) < 8:
        return render_template("register.html", error="密码至少8个字符")
    if password != confirm:
        return render_template("register.html", error="两次密码不一致")
    if contains_sensitive_word(username) or contains_sensitive_word(display_name):
        return render_template("register.html", error="用户名或昵称包含敏感词")
    
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
        logger.info(f"新用户注册: {username}")
        return redirect(url_for("login", registered="1"))
    except Exception as e:
        logger.error(f"注册失败: {e}")
        return render_template("register.html", error="注册失败，请稍后重试")
    finally:
        put_db(conn)

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
                if p.get("name"): parts.append(f"我是{p['name']}")
                if p.get("height"): parts.append(f"身高{p['height']}cm")
                if p.get("weight"): parts.append(f"体重{p['weight']}kg")
                if p.get("age"): parts.append(f"年龄{p['age']}岁")
                if p.get("goal"): parts.append(f"目标是{p['goal']}")
                if p.get("experience"): parts.append(f"训练经验{p['experience']}")
                if p.get("equipment"): parts.append(f"器材有{p['equipment']}")
                if parts:
                    profile_context = "，".join(parts) + "。\n"
    except Exception as e:
        logger.warning(f"加载用户档案失败: {e}")
    # profile 拼到 query 前缀（让 LLM 看到档案信息）
    user_message = query
    if profile_context:
        user_message = profile_context + query
    
    
    # 如果没有 conversation_id，创建新对话
    if not conversation_id:
        conn = get_db()
        try:
            cur = conn.cursor()
            title = query[:20] + ("..." if len(query) > 20 else "")
            cur.execute(
                "INSERT INTO conversations (user_id, title) VALUES (%s::uuid, %s) RETURNING id",
                (user_id, title)
            )
            conversation_id = str(cur.fetchone()[0])
            cur.close()
        except Exception as e:
            logger.error(f"创建对话失败: {e}")
            conversation_id = ""
        finally:
            put_db(conn)

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
        except Exception as e:
            logger.warning(f"保存伤病对话记忆失败: {e}")
        # 保存到 messages 表
        if conversation_id:
            conn = get_db()
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (%s::uuid, %s, %s)",
                    (conversation_id, "user", query)
                )
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (%s::uuid, %s, %s)",
                    (conversation_id, "assistant", injury_response)
                )
                cur.close()
            except Exception as e:
                logger.error(f"保存伤病消息失败: {e}")
            finally:
                put_db(conn)
        return jsonify({"answer": injury_response, "conversation_id": conversation_id})

    # 知识库检索
    knowledge_docs = search_knowledge(query)
    
    # 构建系统提示词（知识库内容作为上下文注入，不暴露内部标签）
    knowledge_section = ""
    if knowledge_docs:
        parts = []
        for doc in knowledge_docs[:3]:
            parts.append(doc['text'])
        knowledge_section = "\n\n以下是相关知识内容：\n" + "\n---\n".join(parts)
    

    # 根据意图加载对应prompt
    intent = detect_intent(query)
    base_prompt = load_prompt("base")
    if intent == "training":
        extra_prompt = load_prompt("training")
    elif intent == "diet":
        extra_prompt = load_prompt("diet")
    else:
        extra_prompt = ""

    system_prompt = f"{base_prompt}\n\n{extra_prompt}\n\n{knowledge_section}"
    
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
    except Exception as e:
        logger.warning(f"加载对话历史记忆失败: {e}")
    
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
        except Exception as e:
            logger.warning(f"保存对话记忆失败: {e}")

        # 保存到 messages 表
        if conversation_id:
            conn = get_db()
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (%s::uuid, %s, %s)",
                    (conversation_id, "user", query)
                )
                cur.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (%s::uuid, %s, %s)",
                    (conversation_id, "assistant", answer)
                )
                cur.close()
            except Exception as e:
                logger.error(f"保存消息失败: {e}")
            finally:
                put_db(conn)

    except Exception as e:
        return jsonify({"error": f"AI 请求失败: {str(e)}"}), 502

    return jsonify({"answer": answer, "conversation_id": conversation_id})

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
    """获取当前用户的对话列表"""
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, title as name, EXTRACT(EPOCH FROM created_at) as created_at FROM conversations WHERE user_id = %s::uuid ORDER BY created_at DESC",
            (session["user_id"],)
        )
        rows = cur.fetchall()
        cur.close()
        return jsonify({"data": rows})
    except Exception as e:
        logger.error(f"获取对话列表失败: {e}")
        return jsonify({"data": [], "error": str(e)})
    finally:
        put_db(conn)


@app.route("/api/conversations/<conversation_id>/messages")
def api_conversation_messages(conversation_id):
    """获取某个对话的消息列表"""
    if "user_id" not in session:
        return jsonify({"error": "未登录"}), 401
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # 验证对话属于当前用户
        cur.execute(
            "SELECT id FROM conversations WHERE id = %s::uuid AND user_id = %s::uuid",
            (conversation_id, session["user_id"])
        )
        if not cur.fetchone():
            cur.close()
            return jsonify({"data": [], "error": "对话不存在"})
        # 获取消息，按时间排序
        cur.execute(
            "SELECT role, content FROM messages WHERE conversation_id = %s::uuid ORDER BY created_at",
            (conversation_id,)
        )
        rows = cur.fetchall()
        cur.close()
        # 配对：user + assistant -> {query, answer}
        result = []
        pending_query = None
        for msg in rows:
            if msg["role"] == "user":
                pending_query = msg["content"]
            elif msg["role"] == "assistant" and pending_query is not None:
                result.append({"query": pending_query, "answer": msg["content"]})
                pending_query = None
        # 如果最后一条是用户消息还没回复
        if pending_query is not None:
            result.append({"query": pending_query, "answer": ""})
        return jsonify({"data": result})
    except Exception as e:
        return jsonify({"data": [], "error": str(e)})


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
        "conversations", "messages", "conversation_memory", "analysis_logs",
    }
    if table not in allowed:
        return jsonify({"error": "不允许查看该表"}), 400

    page = request.json.get("page", 1)
    page_size = request.json.get("page_size", 20)
    offset = (page - 1) * page_size

    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # 使用 psycopg2.sql 防止 SQL 注入
        from psycopg2 import sql
        cur.execute(sql.SQL("SELECT count(*) AS cnt FROM {}").format(sql.Identifier(table)))
        total = cur.fetchone()["cnt"]
        cur.execute(
            sql.SQL("SELECT * FROM {} ORDER BY created_at DESC NULLS LAST LIMIT %s OFFSET %s").format(sql.Identifier(table)),
            (page_size, offset)
        )
        rows = cur.fetchall()
        cur.close()
        # 把 datetime/date 转成 ISO 格式字符串（带 Z 表示 UTC）
        result = []
        for r in rows:
            item = {}
            for k, v in dict(r).items():
                if hasattr(v, 'isoformat'):
                    # 加 Z 后缀，让前端知道这是 UTC 时间
                    item[k] = v.isoformat() + 'Z'
                else:
                    item[k] = v
            result.append(item)
        return jsonify({"rows": result, "total": len(result), "total_all": total, "page": page, "page_size": page_size})
    except Exception as e:
        logger.error(f"数据查询失败: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        put_db(conn)

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
    except Exception as e:
        logger.error(f"获取表列表失败: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        put_db(conn)


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
    except Exception as e:
        logger.warning(f"获取动作列表失败: {e}")
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
        logger.error(f"视频分析请求失败: {e}")
        return jsonify({"error": f"请求分析服务失败: {str(e)}"}), 502

@app.route("/api/analysis/result/<job_id>")
def api_analysis_result(job_id):
    try:
        resp = http_requests.get(f"{MOTION_ANALYSIS_URL}/api/analyze/{job_id}", timeout=10)
        return jsonify(resp.json())
    except Exception as e:
        logger.error(f"获取分析结果失败: {e}")
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
        logger.error(f"获取分析文件失败: {e}")
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
        logger.error(f"获取LLM分析失败: {e}")
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
    except Exception as e:
        logger.error(f"保存分析结果失败: {e}")
    finally:
        put_db(conn)


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
        
        # 把 datetime 转成带 Z 后缀的 ISO 格式
        def fix_time(row):
            if row and row.get('created_at') and hasattr(row['created_at'], 'isoformat'):
                row['created_at'] = row['created_at'].isoformat() + 'Z'
            return row
        
        best = fix_time(best)
        worst = fix_time(worst)
        recent = [fix_time(r) for r in recent]
        
        return jsonify({
            "total": total,
            "avg_score": avg_score,
            "best": best,
            "worst": worst,
            "by_exercise": by_exercise,
            "recent": recent,
        })
    except Exception as e:
        logger.error(f"获取分析统计失败: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        put_db(conn)


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
        
        # 把 datetime 转成带 Z 后缀的 ISO 格式（表示 UTC 时间）
        for row in rows:
            if row.get('created_at') and hasattr(row['created_at'], 'isoformat'):
                row['created_at'] = row['created_at'].isoformat() + 'Z'
        
        return jsonify({"items": rows, "total": total, "limit": limit, "offset": offset})
    except Exception as e:
        logger.error(f"获取分析历史失败: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        put_db(conn)


# ─── 错误处理 ────────────────────────────

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "接口不存在"}), 404
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"服务器内部错误: {request.path} - {e}")
    if request.path.startswith("/api/"):
        return jsonify({"error": "服务器内部错误"}), 500
    return render_template("500.html"), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429
    return "请求过于频繁，请稍后再试", 429


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
