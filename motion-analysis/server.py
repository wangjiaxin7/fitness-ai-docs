"""
API 服务: 动作视频分析
POST /api/analyze          → 上传视频, 返回 job_id
GET  /api/analyze/{id}     → 查询结果（轮询用）
GET  /api/analyze/{id}/result.json  → 下载评分结果
GET  /api/analyze/{id}/video.mp4    → 下载标注视频
GET  /api/analyze/{id}/chart.png    → 下载角度图表
"""

import os, sys, uuid, json, shutil, subprocess, threading, re, time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Header, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="动作分析服务")
app.add_middleware(CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5000").split(","),
    allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(os.environ.get("ANALYSIS_DATA_DIR", "/data"))
ANALYZE_SCRIPT = "/app/analyze.py"
MODEL_PATH = "/data/pose_landmarker.task"
ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MAX_SIZE_MB = 200
API_TOKEN = os.getenv("ANALYSIS_API_TOKEN", "")  # 空=不校验

DATA_DIR.mkdir(parents=True, exist_ok=True)

# 存储任务状态（内存缓存，磁盘持久化）
jobs = {}

def _job_dir(job_id: str) -> Path:
    """校验 job_id 格式，防止路径穿越"""
    if not re.match(r'^[a-f0-9]{12}$', job_id):
        return None
    return DATA_DIR / job_id

def _persist_job(job_id: str, info: dict):
    """持久化任务状态到磁盘"""
    jobs[job_id] = info
    try:
        status_path = DATA_DIR / job_id / "status.json"
        status_path.write_text(json.dumps(info, default=str))
    except Exception:
        pass

def _recover_job(job_id: str) -> dict:
    """从磁盘恢复任务状态"""
    status_path = DATA_DIR / job_id / "status.json"
    result_path = DATA_DIR / job_id / "score_result.json"
    if status_path.exists():
        try:
            info = json.loads(status_path.read_text())
            jobs[job_id] = info
            return info
        except Exception:
            pass
    if result_path.exists():
        info = {
            "status": "completed",
            "result": json.loads(result_path.read_text()),
            "has_video": (DATA_DIR / job_id / "analyzed_output.mp4").exists(),
            "has_chart": (DATA_DIR / job_id / "angle_chart.png").exists(),
        }
        jobs[job_id] = info
        return info
    return None

# 启动时清理超过24小时的旧任务目录
def _cleanup_old_jobs():
    try:
        cutoff = time.time() - 86400
        for d in DATA_DIR.iterdir():
            if d.is_dir() and re.match(r'^[a-f0-9]{12}$', d.name):
                if d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    jobs.pop(d.name, None)
    except Exception:
        pass

_cleanup_old_jobs()

async def verify_token(authorization: str = Header(default="")):
    """简单 token 校验（ANALYSIS_API_TOKEN 为空时不校验）"""
    if not API_TOKEN:
        return
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="未授权")

def run_analysis(job_id, video_path, exercise_name):
    """后台运行分析"""
    out_dir = DATA_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = out_dir / "run.log"
    result_path = out_dir / "score_result.json"
    analyzed_video = out_dir / "analyzed_output.mp4"
    chart = out_dir / "angle_chart.png"
    
    try:
        env = os.environ.copy()
        env.update({
            "OUTPUT_DIR": str(out_dir),
            "POSE_MODEL": MODEL_PATH,
        })
        
        cmd = [sys.executable, ANALYZE_SCRIPT, video_path]
        if exercise_name:
            cmd.append(exercise_name)
        
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=600
            )
        
        if result_path.exists():
            result = json.loads(result_path.read_text())
            _persist_job(job_id, {
                "status": "completed",
                "result": result,
                "has_video": analyzed_video.exists(),
                "has_chart": chart.exists(),
                "video_path": str(analyzed_video) if analyzed_video.exists() else None,
                "chart_path": str(chart) if chart.exists() else None,
                "log": log_path.read_text()[-2000:],
            })
        else:
            _persist_job(job_id, {
                "status": "error",
                "error": "分析未生成结果文件",
                "log": log_path.read_text()[-2000:],
            })
    except subprocess.TimeoutExpired:
        _persist_job(job_id, {"status": "error", "error": "分析超时(>10分钟)"})
    except Exception as e:
        _persist_job(job_id, {"status": "error", "error": str(e)})
    
    # 清理上传的视频文件节省空间
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except:
        pass


@app.post("/api/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    exercise: str = Form(default=""),
    _=Depends(verify_token),
):
    """上传视频并开始分析"""
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_EXTS:
        return JSONResponse(
            {"error": f"不支持的文件格式: {ext}，支持: {', '.join(ALLOWED_EXTS)}"},
            status_code=400
        )
    
    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = str(job_dir / f"input{ext}")
    
    # 流式写入 + 大小检查（不先把整个文件读进内存）
    size = 0
    max_bytes = MAX_SIZE_MB * 1024 * 1024
    with open(video_path, "wb") as f:
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                f.close()
                os.remove(video_path)
                return JSONResponse({"error": f"文件过大(>{MAX_SIZE_MB}MB)"}, status_code=400)
            f.write(chunk)
    
    _persist_job(job_id, {"status": "processing"})
    
    threading.Thread(
        target=run_analysis,
        args=(job_id, video_path, exercise.strip()),
        daemon=True
    ).start()
    
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/analyze/{job_id}")
async def get_result(job_id: str):
    """查询分析结果（轮询用），也支持重建已完成的磁盘结果"""
    job_path = _job_dir(job_id)
    if job_path is None:
        return JSONResponse({"error": "无效的任务ID"}, status_code=400)
    
    if job_id in jobs:
        return jobs[job_id]
    
    # 从磁盘恢复
    info = _recover_job(job_id)
    if info:
        return info
    
    return JSONResponse({"error": "任务不存在"}, status_code=404)


@app.get("/api/analyze/{job_id}/result.json")
async def download_result(job_id: str):
    job_path = _job_dir(job_id)
    if job_path is None:
        return JSONResponse({"error": "无效的任务ID"}, status_code=400)
    path = job_path / "score_result.json"
    if not path.exists():
        return JSONResponse({"error": "结果不存在"}, status_code=404)
    return FileResponse(str(path), media_type="application/json")


@app.get("/api/analyze/{job_id}/video.mp4")
async def download_video(job_id: str):
    job_path = _job_dir(job_id)
    if job_path is None:
        return JSONResponse({"error": "无效的任务ID"}, status_code=400)
    path = job_path / "analyzed_output.mp4"
    if not path.exists():
        return JSONResponse({"error": "视频不存在"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4")


@app.get("/api/analyze/{job_id}/chart.png")
async def download_chart(job_id: str):
    job_path = _job_dir(job_id)
    if job_path is None:
        return JSONResponse({"error": "无效的任务ID"}, status_code=400)
    path = job_path / "angle_chart.png"
    if not path.exists():
        return JSONResponse({"error": "图表不存在"}, status_code=404)
    return FileResponse(str(path), media_type="image/png")


@app.get("/api/analyze/{job_id}/log")
async def get_log(job_id: str):
    job_path = _job_dir(job_id)
    if job_path is None:
        return JSONResponse({"error": "无效的任务ID"}, status_code=400)
    path = job_path / "run.log"
    if not path.exists():
        return JSONResponse({"error": "日志不存在"}, status_code=404)
    return FileResponse(str(path), media_type="text/plain")


@app.get("/api/exercises")
async def list_exercises():
    """返回支持的动作列表"""
    return {
        "exercises": [
            {"name": "squat", "label": "深蹲"},
            {"name": "bench-press", "label": "杠铃卧推"},
            {"name": "incline-db-press", "label": "上斜哑铃卧推"},
            {"name": "pull-up", "label": "引体向上"},
            {"name": "lat-pulldown", "label": "高位下拉"},
            {"name": "chest-press", "label": "胸部推举"},
            {"name": "back-pull", "label": "背部拉力"},
            {"name": "shoulder-press", "label": "肩部推举"},
            {"name": "dumbbell-fly", "label": "哑铃飞鸟"},
            {"name": "bicep-curl", "label": "二头弯举"},
            {"name": "leg-press", "label": "腿举"},
            {"name": "romanian-deadlift", "label": "罗马尼亚硬拉"},
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
