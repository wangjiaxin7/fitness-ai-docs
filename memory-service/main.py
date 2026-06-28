from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import psycopg2, psycopg2.extras
from datetime import date, datetime
import os

app = FastAPI(title="Fitness Memory Service")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db_postgres"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "difyai123456"),
    "dbname": os.getenv("DB_NAME", "dify"),
}

def get_db():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn

class UserProfileIn(BaseModel):
    name: str = ""
    height: float = 0
    weight: float = 0
    age: int = 0
    goal: str = ""
    experience: str = ""
    equipment: str = ""

class WorkoutIn(BaseModel):
    exercise: str
    sets: int = 0
    reps: int = 0
    weight: float = 0
    rpe: Optional[float] = None
    notes: str = ""
    workout_date: Optional[str] = None

class BodyIn(BaseModel):
    weight: Optional[float] = None
    body_fat: Optional[float] = None
    chest_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    arm_cm: Optional[float] = None
    thigh_cm: Optional[float] = None
    notes: str = ""
    measure_date: Optional[str] = None

def _to_dict(row):
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, (date, datetime)):
            d[k] = str(v)
    return d

@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return {"exists": False}
        return {"exists": True, "profile": _to_dict(row)}
    finally:
        conn.close()

@app.post("/api/profile/{user_id}")
async def save_profile(user_id: str, profile: UserProfileIn):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
        existing = cur.fetchone()
        cur.close()
        old = dict(existing) if existing else {}
        
        # Merge: keep old values when new is empty/0
        # 关键：用 "height in old" 判断是否数据库中有值，而不是用 0 判断
        name = profile.name if profile.name.strip() else (old.get("name", ""))
        height = profile.height if profile.height != 0 else (old["height"] if "height" in old else 0)
        weight = profile.weight if profile.weight != 0 else (old["weight"] if "weight" in old else 0)
        age = profile.age if profile.age != 0 else (old["age"] if "age" in old else 0)
        goal = profile.goal if profile.goal.strip() else (old.get("goal", ""))
        exp = profile.experience if profile.experience.strip() else (old.get("experience", ""))
        equip = profile.equipment if profile.equipment.strip() else (old.get("equipment", ""))
        
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_profiles (user_id, name, height, weight, age, goal, experience, equipment, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                name=EXCLUDED.name, height=EXCLUDED.height, weight=EXCLUDED.weight,
                age=EXCLUDED.age, goal=EXCLUDED.goal, experience=EXCLUDED.experience,
                equipment=EXCLUDED.equipment, updated_at=NOW()
        """, (user_id, name, height, weight, age, goal, exp, equip))
        cur.close()
        return {"status": "ok"}
    finally:
        conn.close()

@app.get("/api/workout/{user_id}")
async def get_workouts(user_id: str, days: int = 30):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM workout_logs WHERE user_id = %s AND workout_date >= CURRENT_DATE - %s ORDER BY workout_date DESC, created_at DESC", (user_id, days))
        rows = [_to_dict(r) for r in cur.fetchall()]
        cur.close()
        return {"workouts": rows}
    finally:
        conn.close()

@app.post("/api/workout/{user_id}")
async def add_workout(user_id: str, workout: WorkoutIn):
    conn = get_db()
    try:
        cur = conn.cursor()
        wd = workout.workout_date or date.today().isoformat()
        cur.execute("INSERT INTO workout_logs (user_id, workout_date, exercise, sets, reps, weight, rpe, notes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", 
                    (user_id, wd, workout.exercise, workout.sets, workout.reps, workout.weight, workout.rpe, workout.notes))
        cur.close()
        return {"status": "ok"}
    finally:
        conn.close()

@app.get("/api/body/{user_id}")
async def get_body_measurements(user_id: str, days: int = 90):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM body_measurements WHERE user_id = %s AND measure_date >= CURRENT_DATE - %s ORDER BY measure_date DESC", (user_id, days))
        rows = [_to_dict(r) for r in cur.fetchall()]
        cur.close()
        return {"measurements": rows}
    finally:
        conn.close()

@app.post("/api/body/{user_id}")
async def add_body_measurement(user_id: str, body: BodyIn):
    conn = get_db()
    try:
        cur = conn.cursor()
        md = body.measure_date or date.today().isoformat()
        cur.execute("INSERT INTO body_measurements (user_id, measure_date, weight, body_fat, chest_cm, waist_cm, hip_cm, arm_cm, thigh_cm, notes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                    (user_id, md, body.weight, body.body_fat, body.chest_cm, body.waist_cm, body.hip_cm, body.arm_cm, body.thigh_cm, body.notes))
        cur.close()
        return {"status": "ok"}
    finally:
        conn.close()

@app.get("/api/trend/{user_id}")
async def get_trend(user_id: str):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT exercise, weight, sets, reps, workout_date FROM workout_logs WHERE user_id = %s ORDER BY workout_date DESC LIMIT 20", (user_id,))
        workouts = [_to_dict(r) for r in cur.fetchall()]
        cur.execute("SELECT weight, measure_date FROM body_measurements WHERE user_id = %s AND weight IS NOT NULL ORDER BY measure_date DESC LIMIT 3", (user_id,))
        weights = [{"date": str(r["measure_date"]), "weight": float(r["weight"])} for r in cur.fetchall()]
        cur.close()
        return {"recent_workouts": workouts, "recent_weights": weights}
    finally:
        conn.close()


# ── 新增: Chat Memory ──────────────────────────────

class ChatMessageIn(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatHistoryBatch(BaseModel):
    messages: list[ChatMessageIn]


def ensure_conversation_table():
    """启动时自动建表"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id SERIAL PRIMARY KEY,
                user_id UUID NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_memory_user
            ON conversation_memory(user_id, created_at DESC)
        """)
        cur.close()
    finally:
        conn.close()


@app.on_event("startup")
async def startup():
    ensure_conversation_table()


@app.post("/api/chat-memory/{user_id}")
async def save_chat_memory(user_id: str, batch: ChatHistoryBatch):
    """批量保存一轮对话（user + assistant），超200条时清理旧记录"""
    conn = get_db()
    try:
        cur = conn.cursor()
        for msg in batch.messages:
            cur.execute(
                "INSERT INTO conversation_memory (user_id, role, content) VALUES (%s, %s, %s)",
                (user_id, msg.role, msg.content),
            )
        # 清理: 保留每个用户最近 200 条
        cur.execute("""
            DELETE FROM conversation_memory
            WHERE user_id = %s AND id NOT IN (
                SELECT id FROM conversation_memory
                WHERE user_id = %s ORDER BY created_at DESC LIMIT 200
            )
        """, (user_id, user_id))
        cur.close()
        return {"status": "ok"}
    finally:
        conn.close()


@app.get("/api/chat-memory/{user_id}")
async def get_chat_memory(user_id: str, limit: int = 20):
    """获取最近 N 条对话记录（时间正序）"""
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT role, content, created_at FROM conversation_memory "
            "WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
        rows = list(reversed([_to_dict(r) for r in cur.fetchall()]))
        cur.close()
        return {"messages": rows}
    finally:
        conn.close()

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 新增: Session Feedback ──────────────────────────

class FeedbackIn(BaseModel):
    workout_date: Optional[str] = None
    completion_rate: Optional[float] = None
    total_volume: Optional[float] = 0
    fatigue_level: Optional[int] = None
    mood_rating: Optional[int] = None
    sleep_quality: Optional[int] = None
    notes: str = ""


@app.post("/api/feedback/{user_id}")
async def add_feedback(user_id: str, fb: FeedbackIn):
    conn = get_db()
    try:
        wd = fb.workout_date or date.today().isoformat()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO session_feedback
                (user_id, workout_date, completion_rate, total_volume,
                 fatigue_level, mood_rating, sleep_quality, notes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (user_id, workout_date) DO UPDATE SET
                completion_rate=EXCLUDED.completion_rate,
                total_volume=EXCLUDED.total_volume,
                fatigue_level=EXCLUDED.fatigue_level,
                mood_rating=EXCLUDED.mood_rating,
                sleep_quality=EXCLUDED.sleep_quality,
                notes=EXCLUDED.notes
        """, (user_id, wd, fb.completion_rate, fb.total_volume,
              fb.fatigue_level, fb.mood_rating, fb.sleep_quality, fb.notes))
        cur.close()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()


@app.get("/api/feedback/{user_id}")
async def get_feedback(user_id: str, days: int = 30):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM session_feedback WHERE user_id = %s AND workout_date >= CURRENT_DATE - %s ORDER BY workout_date DESC",
                    (user_id, days))
        rows = [_to_dict(r) for r in cur.fetchall()]
        cur.close()
        return {"feedbacks": rows}
    finally:
        conn.close()


# ── 新增: Decision Log ──────────────────────────────

class DecisionIn(BaseModel):
    decision_source: str = "rule"
    decision_action: str = "hold"
    rule_signals: dict = {}
    trend_analysis: dict = {}
    explanation_data: dict = {}
    user_message: str = ""


@app.post("/api/decision/{user_id}")
async def add_decision(user_id: str, dec: DecisionIn):
    import json
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO decision_log
                (user_id, evaluate_date, decision_source, decision_action,
                 rule_signals, trend_analysis, explanation_data, user_message)
            VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s)
        """, (user_id, dec.decision_source, dec.decision_action,
              json.dumps(dec.rule_signals), json.dumps(dec.trend_analysis),
              json.dumps(dec.explanation_data), dec.user_message))
        cur.close()
        return {"status": "ok"}
    finally:
        conn.close()


@app.get("/api/decision/{user_id}")
async def get_decisions(user_id: str, limit: int = 10):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM decision_log WHERE user_id = %s ORDER BY evaluate_date DESC LIMIT %s",
                    (user_id, limit))
        rows = [_to_dict(r) for r in cur.fetchall()]
        cur.close()
        return {"decisions": rows}
    finally:
        conn.close()


# ── 新增: Training Plan ─────────────────────────────

class PlanIn(BaseModel):
    plan_name: str = ""
    phase: str = ""
    start_date: str
    end_date: Optional[str] = None
    plan_detail: dict = {}


@app.post("/api/plan/{user_id}")
async def save_plan(user_id: str, plan: PlanIn):
    import json
    conn = get_db()
    try:
        # 先取消之前的活跃计划
        cur = conn.cursor()
        cur.execute("UPDATE training_plan SET is_active = false WHERE user_id = %s AND is_active = true", (user_id,))
        cur.execute("""
            INSERT INTO training_plan
                (user_id, plan_name, phase, start_date, end_date, plan_detail, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, true)
        """, (user_id, plan.plan_name, plan.phase, plan.start_date,
              plan.end_date, json.dumps(plan.plan_detail)))
        cur.close()
        return {"status": "ok"}
    finally:
        conn.close()


@app.get("/api/plan/{user_id}/active")
async def get_active_plan(user_id: str):
    conn = get_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM training_plan WHERE user_id = %s AND is_active = true ORDER BY created_at DESC LIMIT 1", (user_id,))
        row = cur.fetchone()
        cur.close()
        if row:
            return {"exists": True, "plan": _to_dict(row)}
        return {"exists": False}
    finally:
        conn.close()
