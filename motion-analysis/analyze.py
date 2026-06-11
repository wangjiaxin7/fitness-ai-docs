#!/usr/bin/env python3
"""
动作分析框架: 多种训练动作的 MediaPipe Pose 分析与评分
支持: 深蹲 / 杠铃卧推 / 上斜哑铃卧推 / 引体向上 / 高位下拉 / 通用胸背肩 / 哑铃飞鸟 / 二头弯举 / 腿举

用法:
  python3 analyze.py <视频路径> [动作名称]
  python3 analyze.py --list          # 列出支持的动作
  python3 analyze.py --demo          # 合成深蹲视频跑demo

动作名称(不指定则自动检测):
  squat, bench-press, incline-db-press, pull-up, lat-pulldown,
  dumbbell-fly, bicep-curl, leg-press
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    PoseLandmarksConnections,
)
from mediapipe import Image as MpImage
from mediapipe import ImageFormat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json, math, sys, os, urllib.request

MODEL_PATH = os.environ.get("POSE_MODEL", "/tmp/pose_landmarker.task")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.getcwd())

# ─── MediaPipe 关键点索引 ─────────────────────────────
# 简写方便使用
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


# ═══════════════════════════════════════════════════════
# Part 1: 动作定义
# ═══════════════════════════════════════════════════════

def calc_angle(a, b, c):
    """三点夹角 (b 为顶点), 返回度或 None"""
    if a is None or b is None or c is None:
        return None
    v1 = np.array([a[0]-b[0], a[1]-b[1]])
    v2 = np.array([c[0]-b[0], c[1]-b[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return math.degrees(math.acos(max(-1, min(1, cos_a))))


class ExerciseDef:
    """动作定义: 关键点、角度计算、评分规则"""
    
    def __init__(self, name, label, landmarks_needed, angle_configs, score_rules,
                 phase_inverted=False, recommended_view=None, view_warning=""):
        self.name = name
        self.label = label
        self.landmarks_needed = landmarks_needed
        self.angle_configs = angle_configs      # [(name, vertex, a, c), ...]
        self.score_rules = score_rules          # [{"name", "weight", "fn"}, ...]
        self.phase_inverted = phase_inverted    # True for pulling exercises (pull-up, lat-pulldown)
        self.recommended_view = recommended_view  # "front"/"side"/"back"/None
        self.view_warning = view_warning        # 拍摄角度不匹配时的提示
    
    def extract_angles(self, lm, w, h):
        """从关键点提取该动作的角度"""
        def xy(idx):
            if idx is None or lm[idx] is None or lm[idx].visibility < 0.3:
                return None
            return (lm[idx].x * w, lm[idx].y * h)
        
        result = {}
        for cfg in self.angle_configs:
            a = xy(cfg.get("a"))
            b = xy(cfg.get("b"))
            c = xy(cfg.get("c"))
            d = xy(cfg.get("d")) if "d" in cfg else None
            e = xy(cfg.get("e")) if "e" in cfg else None
            f_pt = xy(cfg.get("f")) if "f" in cfg else None
            
            if cfg["type"] == "three_point":
                val = calc_angle(a, b, c)
            elif cfg["type"] == "depth_ratio":
                # 深度比: 肘-肩垂直距离 / 肩-髋垂直距离
                # 肘在肩下方为正, 归一化到 0-100%
                if a and b and c:
                    elbow_drop = b[1] - a[1]  # y向下为+, 肘在肩下为正
                    torso_len = abs(c[1] - a[1])  # 躯干长度
                    val = max(0, min(100, elbow_drop / max(torso_len, 1) * 100))
                else:
                    val = None
                # 也保存左右独立值
                if a and b:
                    lv = b[1] - a[1]
                    result[f"{cfg['name']}_left_raw"] = lv
                if d and e:
                    rv = e[1] - d[1]
                    result[f"{cfg['name']}_right_raw"] = rv
            elif cfg["type"] == "vertical_angle":
                # a-b 连线与垂直方向夹角
                if a and b:
                    dx = b[0] - a[0]
                    dy = b[1] - a[1]
                    val = abs(math.degrees(math.atan2(dx, -dy))) if dy != 0 else 0
                else:
                    val = None
            elif cfg["type"] == "left_right":
                # 左右平均值
                lv = calc_angle(a, b, c)
                rv = calc_angle(d, e, f_pt) if d and e and f_pt else None
                val = (lv + rv) / 2 if lv is not None and rv is not None else (lv or rv)
            else:
                val = None
            
            result[cfg["name"]] = val
            # 也保存左右独立值用于对称分析
            if cfg["type"] == "left_right":
                lv = calc_angle(a, b, c)
                rv = calc_angle(d, e, f_pt) if d and e and f_pt else None
                result[f"{cfg['name']}_left"] = lv
                result[f"{cfg['name']}_right"] = rv
        
        return result
    
    def score(self, angle_sequence):
        """对整组角度时序评分，返回 dict"""
        scores = {}
        details = {}
        
        # 空序列或无效数据，直接返回 0
        if not angle_sequence or not any(
            k for f in angle_sequence for k in f if not k.startswith("_")
        ):
            return {"total": 0, "_details": {}, "error": "无有效数据"}
        
        for rule in self.score_rules:
            fn = rule["fn"]
            try:
                val = fn(angle_sequence)
                scores[rule["name"]] = val.get("score", 0)
                details[rule["name"]] = val
            except (ValueError, ZeroDivisionError, TypeError):
                scores[rule["name"]] = 0
                details[rule["name"]] = {"score": 0, "error": "计算异常"}
        
        total = sum(scores.values())
        max_possible = sum(r.get("weight", 25) for r in self.score_rules)
        total = round(total / max_possible * 100) if max_possible > 0 else 0
        
        return {"total": min(100, max(0, total)), **scores, "_details": details}


# ─── 动作注册表 ──────────────────────────────────────
EXERCISES = {}

def register(ex):
    exercise = ex()
    EXERCISES[exercise.name] = exercise
    return ex


# ─── 工具函数 ────────────────────────────────────────

def get_col(seq, name):
    """从 angle_sequence 中提取同名角度序列"""
    return [f.get(name) for f in seq if name in f and f[name] is not None]

def filter_none(vals):
    return [v for v in vals if v is not None]

def percentile_min(vals, p=5):
    """取第 p 百分位数代替 min，抗单帧抖动"""
    v = filter_none(vals)
    if not v:
        return None
    return np.percentile(v, p)

def percentile_max(vals, p=95):
    """取第 p 百分位数代替 max，抗单帧抖动"""
    v = filter_none(vals)
    if not v:
        return None
    return np.percentile(v, p)

def min_max_avg(vals):
    v = filter_none(vals)
    if not v:
        return None, None, None
    return min(v), max(v), sum(v) / len(v)


# ─── 帧间平滑 ────────────────────────────────────────
def ema_smooth(vals, alpha=0.25):
    """指数移动平均平滑"""
    result = []
    prev = None
    for v in vals:
        if v is None:
            result.append(None)
            prev = None
        elif prev is None:
            result.append(v)
            prev = v
        else:
            smoothed = alpha * v + (1 - alpha) * prev
            result.append(round(smoothed, 1))
            prev = smoothed
    return result

def smooth_angle_sequence(seq, alpha=0.25):
    """对角度时序逐列应用 EMA 平滑"""
    if not seq:
        return seq
    keys = set()
    for f in seq:
        keys.update(k for k, v in f.items() 
                    if not k.startswith('_') and isinstance(v, (int, float)))
    for key in keys:
        vals = [f.get(key) for f in seq]
        smoothed = ema_smooth(vals, alpha)
        for i, f_val in enumerate(seq):
            if smoothed[i] is not None:
                f_val[key] = smoothed[i]
    return seq

def angle_to_score(val, ideal_min, ideal_max, max_score,
                   warn_min=None, warn_max=None):
    """角度→分数: ideal 范围内满分, 超出逐级扣"""
    if val is None:
        return 0
    if warn_min is None: warn_min = ideal_min - 15
    if warn_max is None: warn_max = ideal_max + 15
    if ideal_min <= val <= ideal_max:
        return max_score
    elif warn_min <= val < ideal_min:
        ratio = (val - warn_min) / (ideal_min - warn_min)
        return round(max_score * max(0, ratio))
    elif ideal_max < val <= warn_max:
        ratio = (warn_max - val) / (warn_max - ideal_max)
        return round(max_score * max(0, ratio))
    return 0


def lower_is_better_score(val, full_threshold, fail_threshold, max_score):
    """数值越小越好: <= full_threshold 满分, >= fail_threshold 为 0。"""
    if val is None:
        return 0
    if val <= full_threshold:
        return max_score
    if val >= fail_threshold:
        return 0
    ratio = (fail_threshold - val) / max(1e-6, fail_threshold - full_threshold)
    return round(max_score * max(0, min(1, ratio)))


def moving_average(vals, window=5):
    """轻度平滑, 降低关键点抖动对极值检测的干扰。"""
    v = filter_none(vals)
    if len(v) < 3 or window <= 1:
        return v
    half = window // 2
    out = []
    for i in range(len(v)):
        start = max(0, i - half)
        end = min(len(v), i + half + 1)
        out.append(sum(v[start:end]) / (end - start))
    return out


def find_local_minima(vals, min_gap=8, max_val=None):
    """检测动作底部极小值, 用于重复一致性和节奏评估。
    max_val: 只保留值 ≤ max_val 的极小值，过滤掉顶部平段的假谷底。
    """
    v = moving_average(vals, window=5)
    if len(v) < 3:
        return []
    minima = []
    last_idx = -min_gap
    for i in range(1, len(v) - 1):
        if v[i] <= v[i - 1] and v[i] <= v[i + 1] and (i - last_idx) >= min_gap:
            if max_val is None or v[i] <= max_val:
                minima.append((i, v[i]))
                last_idx = i
    return minima


def find_local_maxima(vals, min_gap=8, min_val=None):
    """检测动作顶部极大值, 与 find_local_minima 对应。"""
    v = moving_average(vals, window=5)
    if len(v) < 3:
        return []
    maxima = []
    last_idx = -min_gap
    for i in range(1, len(v) - 1):
        if v[i] >= v[i - 1] and v[i] >= v[i + 1] and (i - last_idx) >= min_gap:
            if min_val is None or v[i] >= min_val:
                maxima.append((i, v[i]))
                last_idx = i
    return maxima


def analyze_phases(vals, min_gap=8):
    """将角度时序分割为逐次重复的阶段数据。
    返回: [{"rep": n, "bottom_angle", "bottom_frame",
             "eccentric_frames", "concentric_frames",
             "eccentric_speed", "concentric_speed", "top_angle"}, ...]
    """
    raw = filter_none(vals)
    if len(raw) < 10:
        return []

    # 利用现有平滑和阈值找到有效底部/顶部
    bottom_threshold = np.percentile(raw, 25)
    top_threshold = np.percentile(raw, 75)
    
    bottoms = find_local_minima(raw, min_gap=min_gap, max_val=bottom_threshold)
    tops = find_local_maxima(raw, min_gap=min_gap, min_val=top_threshold)

    if len(bottoms) < 1:
        return []

    reps = []
    for i, (b_idx, b_val) in enumerate(bottoms):
        # 找这个底部之前的最近顶部（下放起点）
        prev_top = max((idx for idx, val in tops if idx < b_idx), default=None)
        # 找这个底部之后的最近顶部（推起终点）
        next_top = min((idx for idx, val in tops if idx > b_idx), default=None)

        eccentric_frames = (b_idx - prev_top) if prev_top is not None else None
        concentric_frames = (next_top - b_idx) if next_top is not None else None

        rep_data = {
            "rep": i + 1,
            "bottom_angle": round(b_val, 1),
            "bottom_frame": b_idx,
        }
        if prev_top is not None:
            rep_data["eccentric_frames"] = eccentric_frames
            for t_idx, t_val in tops:
                if t_idx == prev_top:
                    rep_data["top_angle_entry"] = round(t_val, 1)
                    break
        if next_top is not None:
            rep_data["concentric_frames"] = concentric_frames
            for t_idx, t_val in tops:
                if t_idx == next_top:
                    rep_data["top_angle_exit"] = round(t_val, 1)
                    break

        reps.append(rep_data)

    return reps


def phases_to_deepseek_text(exercise_label, reps, view_type, phase_inverted=False):
    """将阶段分析数据转成给 DeepSeek 的结构化文本
    phase_inverted=True 时标签反转(拉类动作: 角度最小=顶端, 最大=底端)
    """
    if not reps or len(reps) < 1:
        return ""

    # 拉类动作标签反转
    if phase_inverted:
        bottom_label = "顶端收缩角"      # 角度最小值 = 手臂最弯曲 = 顶端
        top_label = "底端伸展角"          # 角度最大值 = 手臂最伸直 = 底端
        deep_label = "最充分收缩"
        shallow_label = "收缩最浅"
    else:
        bottom_label = "底部肘角"
        top_label = "顶端肘角"
        deep_label = "最深"
        shallow_label = "最浅"

    lines = [f"\n【逐次重复分析（共{len(reps)}次）】"]
    
    bottom_angles = [r["bottom_angle"] for r in reps]
    avg_bottom = sum(bottom_angles) / len(bottom_angles)
    lines.append(f"平均{bottom_label}: {avg_bottom:.1f}°" +
                 (f" (波动 {max(bottom_angles)-min(bottom_angles):.1f}°)" if len(bottom_angles) > 1 else ""))

    for r in reps:
        details = []
        details.append(f"{bottom_label} {r['bottom_angle']}°")
        if "eccentric_frames" in r:
            details.append(f"下放{r['eccentric_frames']}帧")
        if "concentric_frames" in r:
            details.append(f"推起{r['concentric_frames']}帧")
        if "top_angle_exit" in r:
            details.append(f"{top_label}{r['top_angle_exit']}°")
        lines.append(f"  第{r['rep']}次: {' | '.join(details)}")

    # 找出有问题的重复
    if len(reps) >= 2:
        deepest = min(reps, key=lambda r: r["bottom_angle"])
        shallowest = max(reps, key=lambda r: r["bottom_angle"])
        if deepest["bottom_angle"] != shallowest["bottom_angle"]:
            lines.append(f"  差异: 第{deepest['rep']}次{deep_label}({deepest['bottom_angle']}°), "
                        f"第{shallowest['rep']}次{shallow_label}({shallowest['bottom_angle']}°)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════

def bottom_consistency_score(vals, max_score, good_spread=8, fail_spread=30):
    """看各次重复底部肘角是否稳定, 而不是惩罚大 ROM。
    使用数据动态阈值（P25），只承认真正底部的极小值。
    """
    raw = filter_none(vals)
    if len(raw) < 3:
        return {"score": 0, "error": "数据不足"}

    # 动态阈值: 只取数据中最低的 25% 作为底部候选
    bottom_threshold = np.percentile(raw, 25)
    minima = find_local_minima(raw, max_val=bottom_threshold)
    bottoms = [v for _, v in minima]
    if len(bottoms) >= 2:
        spread = max(bottoms) - min(bottoms)
        avg_bottom = sum(bottoms) / len(bottoms)
        rep_count = len(bottoms)
    else:
        sample = sorted(raw)[:max(3, min(8, max(3, len(raw) // 8)))]
        if len(sample) < 2:
            return {"score": 0, "error": "数据不足"}
        spread = max(sample) - min(sample)
        avg_bottom = sum(sample) / len(sample)
        rep_count = 0

    if spread <= good_spread:
        score = max_score
    elif spread >= fail_spread:
        score = 0
    else:
        ratio = (fail_spread - spread) / max(1e-6, fail_spread - good_spread)
        score = round(max_score * max(0, min(1, ratio)))

    result = {
        "score": score,
        "底部肘角波动": round(spread, 1),
        "平均底部肘角": round(avg_bottom, 1),
    }
    if rep_count:
        result["检测到底部次数"] = rep_count
    return result


def tempo_ratio_details(vals, min_gap=8):
    """估计离心/向心时长比; 卧推类理想是离心略慢于向心。
    使用 P25 阈值过滤假谷底。
    """
    v = moving_average(vals, window=5)
    if len(v) < 6:
        return None

    bottom_threshold = np.percentile(v, 25) if v else 999
    minima = [idx for idx, _ in find_local_minima(v, min_gap=min_gap, max_val=bottom_threshold)]
    if not minima:
        return None

    maxima = []
    last_idx = -min_gap
    for i in range(1, len(v) - 1):
        if v[i] >= v[i - 1] and v[i] >= v[i + 1] and (i - last_idx) >= min_gap:
            maxima.append(i)
            last_idx = i

    ratios = []
    for bottom in minima:
        prev_max = max((i for i in maxima if i < bottom), default=None)
        next_max = min((i for i in maxima if i > bottom), default=None)
        if prev_max is None or next_max is None:
            continue
        eccentric = bottom - prev_max
        concentric = next_max - bottom
        if eccentric >= 2 and concentric >= 2:
            ratios.append(eccentric / concentric)

    if not ratios:
        return None
    ratios.sort()
    mid = len(ratios) // 2
    ratio = ratios[mid] if len(ratios) % 2 == 1 else (ratios[mid - 1] + ratios[mid]) / 2
    return {"ratio": ratio, "cycles": len(ratios)}


def tempo_score(vals, max_score, ideal_min=1.2, ideal_max=2.5, fail_min=0.6, fail_max=4.0):
    details = tempo_ratio_details(vals)
    if not details:
        return {"score": 0, "error": "数据不足"}

    ratio = details["ratio"]
    if ideal_min <= ratio <= ideal_max:
        score = max_score
    elif fail_min <= ratio < ideal_min:
        frac = (ratio - fail_min) / max(1e-6, ideal_min - fail_min)
        score = round(max_score * max(0, min(1, frac)))
    elif ideal_max < ratio <= fail_max:
        frac = (fail_max - ratio) / max(1e-6, fail_max - ideal_max)
        score = round(max_score * max(0, min(1, frac)))
    else:
        score = 0

    return {
        "score": score,
        "离心/向心比": round(ratio, 2),
        "有效节奏周期": details["cycles"],
    }


# ═══════════════════════════════════════════════════════
# Part 2: 各动作定义
# ═══════════════════════════════════════════════════════

# ─── 深蹲 Squat ───────────────────────────────────────
@register
def squat():
    return ExerciseDef(
        name="squat",
        label="深蹲 Squat",
        landmarks_needed=[L_HIP, L_KNEE, L_ANKLE, L_SHOULDER, R_SHOULDER, R_HIP, R_KNEE, R_ANKLE],
        angle_configs=[
            {"name": "knee", "type": "left_right",
             "a": L_HIP, "b": L_KNEE, "c": L_ANKLE,
             "d": R_HIP, "e": R_KNEE, "f": R_ANKLE},
            {"name": "hip", "type": "left_right",
             "a": L_SHOULDER, "b": L_HIP, "c": L_KNEE,
             "d": R_SHOULDER, "e": R_HIP, "f": R_KNEE},
            {"name": "back", "type": "vertical_angle",
             "a": L_SHOULDER, "b": L_HIP},  # 近似用左肩-左髋
        ],
        score_rules=[
            {"name": "深度", "weight": 40,
             "fn": lambda seq: {
                 "score": angle_to_score(
                     min(filter_none(get_col(seq, "knee")) or [180]),
                     70, 90, 40, 120, 160),
                 "最低膝角": round(min(filter_none(get_col(seq, "knee")) or [180]), 1)
             }},
            {"name": "对称性", "weight": 20,
             "fn": lambda seq: _symmetry_score(seq, weight=20, col="knee", mult=2)},
            {"name": "背角控制", "weight": 20,
             "fn": lambda seq: {
                 "score": max(0, 20 - max(0,
                     (max(filter_none(get_col(seq, "back"))) or 0) -
                     (min(filter_none(get_col(seq, "back"))) or 0) - 10) * 1.5),
                 "背角范围": round(
                     (max(filter_none(get_col(seq, "back"))) or 0) -
                     (min(filter_none(get_col(seq, "back"))) or 0), 1)
             }},
            {"name": "髋膝同步", "weight": 20,
             "fn": lambda seq: {
                 "score": 20 if not filter_none(get_col(seq, "hip")) else
                     min(20, (min(filter_none(get_col(seq, "hip")) + [180]) /
                              max(1, 180)) * 25),
                 "髋活动范围": round(
                     (max(filter_none(get_col(seq, "hip"))) or 0) -
                     (min(filter_none(get_col(seq, "hip"))) or 0), 1) if
                     filter_none(get_col(seq, "hip")) else 0
             }},
        ]
    )

# ─── 杠铃卧推 Bench Press ────────────────────────────
@register
def bench_press():
    return ExerciseDef(
        name="bench-press",
        label="杠铃卧推 Bench Press",
        landmarks_needed=[L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
                          L_HIP, R_HIP],
        angle_configs=[
            # 肘角: 肩→肘→腕, 下放越深肘角越小
            {"name": "elbow", "type": "left_right",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_WRIST},
            # 肩外展角: 髋-肩-肘, 正面更适合看大臂外张程度
            {"name": "abduction", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
            # 保留 depth 仅用于兼容旧图表/调试, 不再参与卧推评分
            {"name": "depth", "type": "depth_ratio",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_HIP,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_HIP},
        ],
        score_rules=[
            {"name": "幅度", "weight": 30,
             "fn": lambda seq: {
                 "score": lower_is_better_score(
                     percentile_min(get_col(seq, "elbow"), 5) or 180,
                     70, 125, 30),
                 "最小肘角(P5)": round(percentile_min(get_col(seq, "elbow"), 5) or 0, 1)
             }},
            {"name": "左右对称", "weight": 20,
             "fn": lambda seq: {
                 "score": max(0, 20 - (np.percentile(
                     [abs(l-r) for l, r in zip(
                         filter_none(get_col(seq, "elbow_left")),
                         filter_none(get_col(seq, "elbow_right")))],
                     95) if any(True for _ in zip(
                         filter_none(get_col(seq, "elbow_left")),
                         filter_none(get_col(seq, "elbow_right")))) else 0) * 1.5),
                 "左右差异(P95)": round(np.percentile(
                     [abs(l-r) for l, r in zip(
                         filter_none(get_col(seq, "elbow_left")),
                         filter_none(get_col(seq, "elbow_right")))],
                     95), 1) if any(True for _ in zip(
                         filter_none(get_col(seq, "elbow_left")),
                         filter_none(get_col(seq, "elbow_right")))) else 0
             }},
            {"name": "稳定性", "weight": 20,
             "fn": lambda seq: {
                 "score": angle_to_score(
                     percentile_max(get_col(seq, "abduction"), 90) or 0,
                     45, 75, 20, 20, 90),
                 "外展角(P90)": round(percentile_max(get_col(seq, "abduction"), 90) or 0, 1)
             }},
            {"name": "控制力", "weight": 15,
             "fn": lambda seq: bottom_consistency_score(get_col(seq, "elbow"), 15, 8, 28)},
            {"name": "动作节奏", "weight": 15,
             "fn": lambda seq: tempo_score(get_col(seq, "elbow"), 15, 1.2, 2.5, 0.6, 4.0)},
        ]
    )

# ─── 上斜哑铃卧推 Incline Dumbbell Press ────────────
@register
def incline_db_press():
    return ExerciseDef(
        name="incline-db-press",
        label="上斜哑铃卧推 Incline DB Press",
        landmarks_needed=[L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
                          L_HIP, R_HIP],
        angle_configs=[
            # 肘角: 肩→肘→腕, 下放越深肘角越小
            {"name": "elbow", "type": "left_right",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_WRIST},
            # 推举角: 髋→肩→肘, 上斜角度过高/过低都说明轨迹偏
            {"name": "press_angle", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
            # 肩外展角: 正面更适合判断肘部是否外张
            {"name": "abduction", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
            # 保留 depth 仅用于兼容旧图表/调试, 不再参与上斜评分
            {"name": "depth", "type": "depth_ratio",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_HIP,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_HIP},
        ],
        score_rules=[
            {"name": "幅度", "weight": 25,
             "fn": lambda seq: {
                 "score": lower_is_better_score(
                     percentile_min(get_col(seq, "elbow"), 5) or 180,
                     75, 130, 25),
                 "最小肘角(P5)": round(percentile_min(get_col(seq, "elbow"), 5) or 0, 1)
             }},
            {"name": "左右对称", "weight": 20,
             "fn": lambda seq: {
                 "score": max(0, 20 - (np.percentile(
                     [abs(l-r) for l, r in zip(
                         filter_none(get_col(seq, "elbow_left")),
                         filter_none(get_col(seq, "elbow_right")))],
                     95) if any(True for _ in zip(
                         filter_none(get_col(seq, "elbow_left")),
                         filter_none(get_col(seq, "elbow_right")))) else 0) * 1.5),
                 "左右差异(P95)": round(np.percentile(
                     [abs(l-r) for l, r in zip(
                         filter_none(get_col(seq, "elbow_left")),
                         filter_none(get_col(seq, "elbow_right")))],
                     95), 1) if any(True for _ in zip(
                         filter_none(get_col(seq, "elbow_left")),
                         filter_none(get_col(seq, "elbow_right")))) else 0
             }},
            {"name": "稳定性", "weight": 20,
             "fn": lambda seq: {
                 "score": angle_to_score(
                     percentile_max(get_col(seq, "abduction"), 90) or 0,
                     45, 75, 20, 20, 90),
                 "外展角(P90)": round(percentile_max(get_col(seq, "abduction"), 90) or 0, 1)
             }},
            {"name": "推举角", "weight": 10,
             "fn": lambda seq: {
                 "score": angle_to_score(
                     percentile_max(get_col(seq, "press_angle"), 90) or 0,
                     50, 75, 10, 35, 90),
                 "最大推举角(P90)": round(percentile_max(get_col(seq, "press_angle"), 90) or 0, 1)
             }},
            {"name": "控制力", "weight": 15,
             "fn": lambda seq: bottom_consistency_score(get_col(seq, "elbow"), 15, 10, 30)},
            {"name": "动作节奏", "weight": 10,
             "fn": lambda seq: tempo_score(get_col(seq, "elbow"), 10, 1.2, 2.8, 0.6, 4.0)},
        ]
    )


# ─── 引体向上 Pull-up ────────────────────────────────
def _symmetry_score(seq, weight=20, col="elbow", mult=2, use_p95=True):
    """通用对称性评分: P95代替max, 避免几帧检测噪声扣光分"""
    diffs = [abs(l-r) for l, r in
             zip(filter_none(get_col(seq, f"{col}_left")),
                 filter_none(get_col(seq, f"{col}_right")))]
    if not diffs:
        return {"score": 0, "左右差异": 0}
    val = (percentile_max(diffs, p=95) or 0) if use_p95 else max(diffs)
    return {"score": max(0, weight - val * mult), "左右差异": round(val, 1)}

def _symmetry_score_pullup(seq):
    """引体向上对称性（兼容旧调用）"""
    return _symmetry_score(seq, weight=20, col="elbow", mult=1.0)

@register
def pull_up():
    return ExerciseDef(
        name="pull-up",
        label="引体向上 Pull-up",
        landmarks_needed=[L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
                          L_HIP, R_HIP, NOSE],
        angle_configs=[
            # 肘角: 底端~170°, 顶端~30°(完全收缩)
            {"name": "elbow", "type": "left_right",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_WRIST},
            # 肩角: 肩关节活动范围
            {"name": "shoulder_flex", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
            # 身体垂直度
            {"name": "tilt", "type": "vertical_angle",
             "a": L_HIP, "b": L_SHOULDER},
        ],
        score_rules=[
            {"name": "幅度", "weight": 40,
             "fn": lambda seq: {"score": angle_to_score(
                 min(filter_none(get_col(seq,"elbow")) or [180]),
                 20, 50, 40, 10, 80),
                 "最小肘角": round(min(filter_none(get_col(seq,"elbow")) or [180]),1),
                 "最大肘角": round(max(filter_none(get_col(seq,"elbow")) or [0]),1)}},
            {"name": "底端伸展", "weight": 25,
             "fn": lambda seq: {"score": angle_to_score(
                 max(filter_none(get_col(seq,"elbow")) or [0]),
                 160, 180, 25, 130, 180),
                 "最大肘角": round(max(filter_none(get_col(seq,"elbow")) or [0]),1)}},
            {"name": "对称性", "weight": 20,
             "fn": lambda seq: _symmetry_score_pullup(seq)},
            {"name": "身体稳定", "weight": 15,
             "fn": lambda seq: {"score": max(0, 15 - (max(filter_none(get_col(seq,"tilt")) or [0]) -
                                                      min(filter_none(get_col(seq,"tilt")) or [0])) * 0.7),
                                "倾斜变化": round((max(filter_none(get_col(seq,"tilt")) or [0]) -
                                                   min(filter_none(get_col(seq,"tilt")) or [0])),1)}},
        ],
        phase_inverted=True,   # 拉类动作: 角度最小值=顶端, 最大值=底端
        recommended_view="back",
        view_warning="建议从背面拍摄引体向上，能看到完整的肘部弯曲和身体姿态",
    )

# ─── 高位下拉 Lat Pulldown ───────────────────────────
@register
def lat_pulldown():
    return ExerciseDef(
        name="lat-pulldown",
        label="高位下拉 Lat Pulldown",
        landmarks_needed=[L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
                          L_HIP, R_HIP],
        angle_configs=[
            {"name": "elbow", "type": "left_right",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_WRIST},
            {"name": "shoulder_ext", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
            {"name": "tilt", "type": "vertical_angle",
             "a": L_HIP, "b": L_SHOULDER},
        ],
        score_rules=[
            {"name": "幅度", "weight": 35,
             "fn": lambda seq: {"score": angle_to_score(
                 min(filter_none(get_col(seq,"elbow")) or [180]),
                 30, 60, 35, 15, 90),
                 "最小肘角": round(min(filter_none(get_col(seq,"elbow")) or [180]),1)}},
            {"name": "顶端伸展", "weight": 25,
             "fn": lambda seq: {"score": angle_to_score(
                 max(filter_none(get_col(seq,"elbow")) or [0]),
                 160, 180, 25, 130, 180),
                 "最大肘角": round(max(filter_none(get_col(seq,"elbow")) or [0]),1)}},
            {"name": "对称性", "weight": 25,
             "fn": lambda seq: _symmetry_score(seq, weight=25, col="elbow", mult=3)},
            {"name": "上身稳定", "weight": 15,
             "fn": lambda seq: {"score": max(0, 15 - max(0, max(filter_none(get_col(seq,"tilt")) or [0]) - 15) * 0.8),
                                "最大后倾": round(max(filter_none(get_col(seq,"tilt")) or [0]),1)}},
        ],
        phase_inverted=True,   # 拉类动作
    )


# ─── 通用: 胸部推举 ─────────────────────────────────
@register
def chest_press():
    return ExerciseDef(
        name="chest-press",
        label="胸部推举 Chest Press (通用)",
        landmarks_needed=[L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
                          L_HIP, R_HIP],
        angle_configs=[
            {"name": "elbow", "type": "left_right",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_WRIST},
            {"name": "abduction", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
        ],
        score_rules=[
            {"name": "幅度", "weight": 40,
             "fn": lambda seq: {"score": angle_to_score(
                 min(filter_none(get_col(seq,"elbow")) or [180]),
                 70, 100, 40, 50, 140)}},
            {"name": "对称性", "weight": 35,
             "fn": lambda seq: _symmetry_score(seq, weight=35, col="elbow", mult=3)},
            {"name": "肘部控制", "weight": 25,
             "fn": lambda seq: {"score": angle_to_score(
                 max(filter_none(get_col(seq,"abduction")) or [0]),
                 60, 90, 25, 40, 110)}},
        ]
    )


# ─── 通用: 背部划船/下拉 ────────────────────────────
@register
def back_pull():
    return ExerciseDef(
        name="back-pull",
        label="背部拉力 Back Pull (通用)",
        landmarks_needed=[L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
                          L_HIP, R_HIP],
        angle_configs=[
            {"name": "elbow", "type": "left_right",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_WRIST},
            {"name": "shoulder_retract", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
        ],
        score_rules=[
            {"name": "收缩幅度", "weight": 40,
             "fn": lambda seq: {"score": angle_to_score(
                 min(filter_none(get_col(seq,"elbow")) or [180]),
                 30, 60, 40, 15, 90)}},
            {"name": "伸展幅度", "weight": 30,
             "fn": lambda seq: {"score": angle_to_score(
                 max(filter_none(get_col(seq,"elbow")) or [0]),
                 160, 180, 30, 130, 180)}},
            {"name": "对称性", "weight": 30,
             "fn": lambda seq: _symmetry_score(seq, weight=30, col="elbow", mult=3)},
        ],
        phase_inverted=True,   # 拉类动作
    )


# ─── 通用: 肩部推举 ─────────────────────────────────
@register
def shoulder_press():
    return ExerciseDef(
        name="shoulder-press",
        label="肩部推举 Shoulder Press (通用)",
        landmarks_needed=[L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
                          L_HIP, R_HIP],
        angle_configs=[
            {"name": "elbow", "type": "left_right",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_WRIST},
            {"name": "overhead", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
        ],
        score_rules=[
            {"name": "幅度", "weight": 40,
             "fn": lambda seq: {"score": angle_to_score(
                 min(filter_none(get_col(seq,"elbow")) or [180]),
                 40, 70, 40, 20, 100)}},
            {"name": "顶端锁定", "weight": 30,
             "fn": lambda seq: {"score": angle_to_score(
                 min(filter_none(get_col(seq,"overhead")) or [180]),
                 160, 180, 30, 130, 180)}},
            {"name": "对称性", "weight": 30,
             "fn": lambda seq: _symmetry_score(seq, weight=30, col="elbow", mult=3)},
        ]
    )


# ─── 哑铃飞鸟 Dumbbell Fly ───────────────────────────
@register
def dumbbell_fly():
    return ExerciseDef(
        name="dumbbell-fly",
        label="哑铃飞鸟 Dumbbell Fly",
        landmarks_needed=[L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
                          L_HIP, R_HIP],
        angle_configs=[
            # 肩外展角: 髋→肩→肘, 反映手臂张开程度
            # 底端(手臂张开): ~120-160°, 顶端(手臂合拢): ~30-60°
            {"name": "abduction", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
            # 肘角: 保持微弯不锁死
            {"name": "elbow", "type": "left_right",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_WRIST},
        ],
        score_rules=[
            {"name": "幅度", "weight": 40,
             "fn": lambda seq: {"score": angle_to_score(
                 max(filter_none(get_col(seq, "abduction")) or [0]),
                 120, 160, 40, 90, 180),
                 "最大外展角": round(max(filter_none(get_col(seq, "abduction")) or [0]), 1)}},
            {"name": "肘部保护", "weight": 30,
             "fn": lambda seq: {"score": angle_to_score(
                 min(filter_none(get_col(seq, "elbow")) or [180]),
                 140, 170, 30, 120, 180),
                 "最小肘角": round(min(filter_none(get_col(seq, "elbow")) or [180]), 1)}},
            {"name": "对称性", "weight": 30,
             "fn": lambda seq: _symmetry_score(seq, weight=30, col="abduction", mult=2)},
        ]
    )


# ─── 二头弯举 Bicep Curl ─────────────────────────────
@register
def bicep_curl():
    return ExerciseDef(
        name="bicep-curl",
        label="二头弯举 Bicep Curl",
        landmarks_needed=[L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
                          L_HIP, R_HIP],
        angle_configs=[
            # 肘角: 主要指标
            # 底端(手臂伸直): ~160-180°, 顶端(完全弯曲): ~20-50°
            {"name": "elbow", "type": "left_right",
             "a": L_SHOULDER, "b": L_ELBOW, "c": L_WRIST,
             "d": R_SHOULDER, "e": R_ELBOW, "f": R_WRIST},
            # 肩前倾角: 肩→肘 与垂直方向夹角, 检测耸肩借力
            {"name": "shoulder_swing", "type": "left_right",
             "a": L_HIP, "b": L_SHOULDER, "c": L_ELBOW,
             "d": R_HIP, "e": R_SHOULDER, "f": R_ELBOW},
            # 身体摆动: 垂直度
            {"name": "tilt", "type": "vertical_angle",
             "a": L_HIP, "b": L_SHOULDER},
        ],
        score_rules=[
            {"name": "收缩幅度", "weight": 35,
             "fn": lambda seq: {"score": angle_to_score(
                 min(filter_none(get_col(seq, "elbow")) or [180]),
                 20, 50, 35, 10, 80),
                 "最小肘角": round(min(filter_none(get_col(seq, "elbow")) or [180]), 1),
                 "最大肘角": round(max(filter_none(get_col(seq, "elbow")) or [0]), 1)}},
            {"name": "底端伸展", "weight": 25,
             "fn": lambda seq: {"score": angle_to_score(
                 max(filter_none(get_col(seq, "elbow")) or [0]),
                 155, 180, 25, 130, 180),
                 "最大肘角": round(max(filter_none(get_col(seq, "elbow")) or [0]), 1)}},
            {"name": "身体稳定", "weight": 20,
             "fn": lambda seq: {"score": max(0, 20 - (max(filter_none(get_col(seq, "tilt")) or [0]) -
                                                      min(filter_none(get_col(seq, "tilt")) or [0])) * 1.5),
                                "倾斜变化": round((max(filter_none(get_col(seq, "tilt")) or [0]) -
                                                   min(filter_none(get_col(seq, "tilt")) or [0])), 1)}},
            {"name": "对称性", "weight": 20,
             "fn": lambda seq: _symmetry_score(seq, weight=20, col="elbow", mult=2)},
        ],
        recommended_view="side",
        view_warning="建议从侧面拍摄二头弯举，正面拍无法准确看到肘部弯曲角度",
    )


# ─── 腿举 Leg Press ──────────────────────────────────
@register
def leg_press():
    return ExerciseDef(
        name="leg-press",
        label="腿举 Leg Press",
        landmarks_needed=[L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE,
                          L_SHOULDER, R_SHOULDER],
        angle_configs=[
            # 膝角: 主要指标
            # 顶端(伸直): ~160-180°, 底端(弯曲): ~70-100°
            {"name": "knee", "type": "left_right",
             "a": L_HIP, "b": L_KNEE, "c": L_ANKLE,
             "d": R_HIP, "e": R_KNEE, "f": R_ANKLE},
            # 髋角: 髋关节弯曲程度
            {"name": "hip", "type": "left_right",
             "a": L_SHOULDER, "b": L_HIP, "c": L_KNEE,
             "d": R_SHOULDER, "e": R_HIP, "f": R_KNEE},
        ],
        score_rules=[
            {"name": "下蹲深度", "weight": 35,
             "fn": lambda seq: {"score": angle_to_score(
                 min(filter_none(get_col(seq, "knee")) or [180]),
                 70, 100, 35, 50, 130),
                 "最小膝角": round(min(filter_none(get_col(seq, "knee")) or [180]), 1)}},
            {"name": "顶端伸展", "weight": 25,
             "fn": lambda seq: {"score": angle_to_score(
                 max(filter_none(get_col(seq, "knee")) or [0]),
                 155, 180, 25, 130, 180),
                 "最大膝角": round(max(filter_none(get_col(seq, "knee")) or [0]), 1)}},
            {"name": "对称性", "weight": 25,
             "fn": lambda seq: _symmetry_score(seq, weight=25, col="knee", mult=2)},
            {"name": "髋膝同步", "weight": 15,
             "fn": lambda seq: {"score": angle_to_score(
                 (min(filter_none(get_col(seq, "hip")) or [180]) +
                  max(filter_none(get_col(seq, "hip")) or [0])) / 2,
                 80, 120, 15, 50, 150),
                 "平均髋角": round(
                     (min(filter_none(get_col(seq, "hip")) or [180]) +
                      max(filter_none(get_col(seq, "hip")) or [0])) / 2, 1)}},
        ],
        recommended_view="side",
        view_warning="建议从侧面拍摄腿举，正面拍膝盖活动范围不明显",
    )


# ─── 通用关键点提取（不依赖具体动作配置）───────────
def extract_universal(lm, w, h):
    """从原始关键点提取通用特征：身体倾斜角、肘角、膝角"""
    def xy(idx):
        if lm[idx] is None or lm[idx].visibility < 0.3:
            return None
        return (lm[idx].x * w, lm[idx].y * h)
    
    result = {}
    
    ls, rs = xy(L_SHOULDER), xy(R_SHOULDER)
    lh, rh = xy(L_HIP), xy(R_HIP)
    
    # 身体倾斜角：肩中点→髋中点 与垂直方向夹角
    if ls and rs and lh and rh:
        mid_s = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        mid_h = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
        dx = mid_h[0] - mid_s[0]
        dy = mid_h[1] - mid_s[1]
        result["tilt"] = abs(math.degrees(math.atan2(dx, -dy))) if dy != 0 else 0
        result["body_vertical"] = dy  # 正=肩在上髋在下(站立), 负=躺姿
    
    # 肘角（左右平均）
    le = xy(L_ELBOW)
    re = xy(R_ELBOW)
    lw = xy(L_WRIST)
    rw = xy(R_WRIST)
    
    if ls and le and lw:
        result["elbow_left"] = calc_angle(ls, le, lw)
    if rs and re and rw:
        result["elbow_right"] = calc_angle(rs, re, rw)
    
    # 膝角（左右平均）
    lk = xy(L_KNEE)
    rk = xy(R_KNEE)
    la = xy(L_ANKLE)
    ra = xy(R_ANKLE)
    
    if lh and lk and la:
        result["knee_left"] = calc_angle(lh, lk, la)
    if rh and rk and ra:
        result["knee_right"] = calc_angle(rh, rk, ra)
    
    # 腕相对于肩的高度（用于判断悬挂/推举）
    if ls and lw:
        result["wrist_over_shoulder"] = ls[1] - lw[1]  # 正=腕在肩上
    if rs and rw:
        result["wrist_over_shoulder_r"] = rs[1] - rw[1]
    
    return result


# ─── 动作检测（根据身体朝向+角度模式推断）────────────
def detect_exercise(angle_sequence, prefer=None, frame_count=0):
    """三阶段智能检测:
    1. 指定了动作 → 直接返回
    2. 少于30帧 → 用体态判断（站立/躺/悬挂）
    3. 30帧后 → 用角度时序精细区分
    """
    if prefer and prefer in EXERCISES:
        return EXERCISES[prefer]
    
    # 提取通用特征
    tilts = filter_none(get_col(angle_sequence, "tilt"))
    body_v = filter_none(get_col(angle_sequence, "body_vertical"))
    elbows_l = filter_none(get_col(angle_sequence, "elbow_left"))
    elbows_r = filter_none(get_col(angle_sequence, "elbow_right"))
    knees = [calc_angle(
        None, None, None) for _ in range(len(get_col(angle_sequence, "knee_left")))]
    knees_l = filter_none(get_col(angle_sequence, "knee_left"))
    knees_r = filter_none(get_col(angle_sequence, "knee_right"))
    wrist_up = filter_none(get_col(angle_sequence, "wrist_over_shoulder"))
    
    avg_tilt = sum(tilts) / len(tilts) if tilts else 0
    elbow_range = (max(elbows_l + elbows_r + [0]) - min(elbows_l + elbows_r + [180])) if (elbows_l or elbows_r) else 0
    
    # ─── 阶段1: 身体朝向判别 ───
    if frame_count < 30 or len(tilts) < 5:
        avg_bv = sum(body_v) / len(body_v) if body_v else 1
        if avg_tilt > 50:
            return EXERCISES["bench-press"]
        if avg_bv < 0:
            return EXERCISES["pull-up"]
        return EXERCISES["squat"]  # 默认站立
    
    # ─── 阶段2: 精细分类（30帧+）───
    # 躺姿系列: 身体倾斜角区分杠铃卧推 vs 上斜哑铃卧推
    if avg_tilt > 50:
        if avg_tilt > 70:
            return EXERCISES["bench-press"]      # 几乎平躺 → 杠铃卧推
        else:
            return EXERCISES["incline-db-press"]  # 45-70° → 上斜哑铃卧推
    
    # 站立系列
    knee_range = (max(knees_l + knees_r + [0]) - min(knees_l + knees_r + [180])) if (knees_l or knees_r) else 0
    
    # 悬挂系列：腕在肩上 + 肘角范围大
    avg_wu = sum(wrist_up) / len(wrist_up) if wrist_up else 0
    if avg_wu > 50:  # 腕明显在肩上
        min_elbow = min(elbows_l + elbows_r + [180])
        if min_elbow < 60:
            return EXERCISES["pull-up"]
        else:
            return EXERCISES["lat-pulldown"]
    
    # 下肢动作
    if knee_range > 40:
        # 腿举: 坐姿/斜躺 + 膝盖活动大, 躯干不是直立
        if avg_tilt > 20:
            return EXERCISES["leg-press"]
        return EXERCISES["squat"]
    
    # 上肢动作（站姿）
    avg_elbow_l = sum(elbows_l) / len(elbows_l) if elbows_l else 0
    avg_elbow_r = sum(elbows_r) / len(elbows_r) if elbows_r else 0
    avg_elbow = (avg_elbow_l + avg_elbow_r) / 2
    
    min_elbow = min(elbows_l + elbows_r + [180])
    max_elbow = max(elbows_l + elbows_r + [0])
    
    # 二头弯举: 肘角范围大 + 最小肘角很小(完全弯曲) + 肩部相对不动
    if elbow_range > 80 and min_elbow < 40:
        return EXERCISES["bicep-curl"]
    
    if max_elbow - min_elbow > 60 and max_elbow > 150:
        if min_elbow < 60:
            return EXERCISES["shoulder-press"]
        else:
            return EXERCISES["chest-press"]
    elif max_elbow - min_elbow > 30:
        return EXERCISES["back-pull"]
    
    return EXERCISES["squat"]


# ─── 视角检测 ───────────────────────────────────────
def detect_view_type(angle_sequence):
    """判断视频是侧面还是正面拍摄。返回 'side' 或 'front'
    
    基于三个特征综合判断:
    1. 左右肘角差异均值 — 侧拍因透视投影差异大，正面左右对称差异小
    2. 左右肘角相关性 — 正面两侧同步运动相关性极高(~0.99)，侧拍极低(~0.3)
    3. 左右有效帧数比 — 侧拍远侧臂被遮挡时检测帧少，正面两侧相当
    """
    left_e = filter_none(get_col(angle_sequence, "elbow_left"))
    right_e = filter_none(get_col(angle_sequence, "elbow_right"))
    
    if not left_e or not right_e:
        return "side"
    
    # 特征1: 左右有效帧数比 — 一侧明显被遮挡即侧拍
    lc = len(left_e)
    rc = len(right_e)
    ratio = min(lc, rc) / max(lc, rc, 1)
    if ratio < 0.6:
        return "side"
    
    # 特征2: 左右肘角差异
    diffs = [abs(l-r) for l, r in zip(left_e, right_e)]
    if not diffs:
        return "side"
    avg_diff = sum(diffs) / len(diffs)
    
    if avg_diff > 25:
        return "side"
    
    # 特征3: 左右肘角曲线相关性（最可靠）
    # 正面两侧同步运动，相关性极高(>0.95)；侧拍因透视变形，相关性低(<0.6)
    min_len = min(len(left_e), len(right_e))
    if min_len > 15:
        corr = float(np.corrcoef(left_e[:min_len], right_e[:min_len])[0, 1])
        if abs(corr) < 0.75:
            return "side"
    
    return "front"


# ─── 视角自适应评分 ───────────────────────────────────
def score_with_view(exercise, angle_sequence, view_type):
    """根据拍摄视角调整评分权重"""
    raw = exercise.score(angle_sequence)
    if "error" in raw:
        return raw
    details = raw.pop("_details", {})

    # 卧推类按视角启用不同指标
    # 注意: 正面拍摄时 2D 投影会夸大肘外展(实际手肘向前移动被误判为外张)
    #       控制力/节奏是时间序列指标，与视角无关，正侧都可以用
    if exercise.name == "bench-press":
        if view_type == "side":
            weights = {
                "幅度": 1.0,
                "控制力": 1.0,
                "动作节奏": 1.0,
                "左右对称": 0.0,
                "稳定性": 0.0,
            }
        else:
            weights = {
                "幅度": 1.0,
                "左右对称": 1.0,
                "控制力": 1.0,
                "动作节奏": 1.0,
                "稳定性": 0.0,      # 正面2D外展角不可信
            }
    elif exercise.name == "incline-db-press":
        if view_type == "side":
            weights = {
                "幅度": 1.0,
                "控制力": 1.0,
                "动作节奏": 1.0,
                "左右对称": 0.0,
                "稳定性": 0.0,
                "推举角": 0.0,
            }
        else:
            weights = {
                "幅度": 1.0,
                "左右对称": 1.0,
                "控制力": 1.0,
                "动作节奏": 1.0,
                "稳定性": 0.0,      # 正面2D外展角不可信
                "推举角": 0.0,      # 正面2D推举角不可信
            }
    elif view_type == "side":
        weights = {}
        for rule in exercise.score_rules:
            n = rule["name"]
            if "对称" in n or "左右" in n:
                weights[n] = 0.0
            elif "外展" in n or "稳定" in n:
                weights[n] = 0.0
            elif "幅度" in n or "深度" in n or "节奏" in n:
                weights[n] = 2.0
            elif "控制" in n:
                weights[n] = 1.5
            else:
                weights[n] = 1.0
    else:
        weights = {rule["name"]: 1.0 for rule in exercise.score_rules}

    active_rules = [
        (r, weights.get(r["name"], 1.0))
        for r in exercise.score_rules
        if weights.get(r["name"], 1.0) > 0
    ]

    raw_total = 0
    max_possible = 0
    for rule, w_mult in active_rules:
        n = rule["name"]
        score = details.get(n, {}).get("score", 0)
        w = rule.get("weight", 25)
        raw_total += score * w_mult
        max_possible += w * w_mult

    total = round(raw_total / max(max_possible, 1) * 100)
    total = min(100, max(0, total))

    scores = {
        n: details.get(n, {}).get("score", 0)
        for n, w_mult in weights.items() if w_mult > 0
    }
    return {"total": total, "_view": view_type, "_details": details, **scores}

# ═══════════════════════════════════════════════════════
# Part 3: 分析引擎
# ═══════════════════════════════════════════════════════

def analyze_video(video_path, exercise_name=None):
    """返回 (angles_sequence, output_video_path)"""
    print(f"\n▶ 分析视频: {video_path}")
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  分辨率: {w}x{h}, {fps:.1f}fps, {total_frames}帧")
    
    out_path = str(Path(OUTPUT_DIR) / "analyzed_output.mp4")
    out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    angle_sequence = []
    frame_count = 0
    detected_count = 0
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            timestamp_ms = int(frame_count / fps * 1000)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            frame_angles = {"frame": frame_count - 1}
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                detected_count += 1
                lm = result.pose_landmarks[0]
                
                # 通用特征提取（独立于动作，用于自动分类）
                universal = extract_universal(lm, w, h)
                frame_angles.update(universal)
                
                # 动作检测（先存1帧→再决定）
                exercise = detect_exercise(angle_sequence, exercise_name, frame_count) if angle_sequence else \
                    (EXERCISES.get(exercise_name) if exercise_name else EXERCISES["squat"])
                
                frame_angles.update(exercise.extract_angles(lm, w, h))
                frame_angles["_exercise"] = exercise.name
                
                # 画关键点
                connections = PoseLandmarksConnections.POSE_LANDMARKS
                for conn in connections:
                    s, e_ = lm[conn.start], lm[conn.end]
                    if s.visibility > 0.3 and e_.visibility > 0.3:
                        cv2.line(frame,
                                 (int(s.x*w), int(s.y*h)),
                                 (int(e_.x*w), int(e_.y*h)),
                                 (0, 255, 0), 2)
                for pt in lm:
                    if pt.visibility > 0.3:
                        cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 3, (0,0,255), -1)
                
                # 显示角度
                y_off = 30
                for k, v in frame_angles.items():
                    if v is not None and not k.startswith("_") and isinstance(v, (int, float)):
                        cv2.putText(frame, f"{k}: {v:.1f}°", (10, y_off),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        y_off += 22
                        if y_off > h - 20:
                            break
            
            out_video.write(frame)
            if frame_count % 30 == 0:
                print(f"  处理: {frame_count}/{total_frames} ({100*frame_count//total_frames}%)  "
                      f"检出: {detected_count}帧", end="\r")
            
            if frame_angles:
                angle_sequence.append(frame_angles)
    
    cap.release()
    out_video.release()
    
    print(f"\n✅ 分析完成: {frame_count}帧, {detected_count}帧检出人体")
    print(f"   标注视频: {out_path}")

    # 对角度时序应用 EMA 帧间平滑
    angle_sequence = smooth_angle_sequence(angle_sequence, alpha=0.25)

    return angle_sequence, out_path


# ═══════════════════════════════════════════════════════
# Part 4: 合成视频 (Demo)
# ═══════════════════════════════════════════════════════

def create_demo_video(path, num_frames=180, fps=30, exercise="squat"):
    """为指定动作生成合成测试视频（含完整人体，保证 MediaPipe 能检测）"""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    mid_x = w // 2
    
    def smooth(t):
        return 0.5 * (1 - math.cos(2 * math.pi * t))
    
    def draw_arms(canvas, shoulder, hip_y, phase, width=30):
        """画手臂（对检测很重要）"""
        arm_len = int(80 + 20 * phase)
        elbow_y = shoulder[1] + arm_len
        hand_y = elbow_y + int(60 + 20 * phase * 0.5)
        # 左臂
        ex = shoulder[0] - width
        cv2.line(canvas, shoulder, (ex, elbow_y), (80, 100, 160), 8)
        cv2.line(canvas, (ex, elbow_y), (ex - 10, hand_y), (80, 100, 160), 7)
        # 右臂
        ex2 = shoulder[0] + width
        cv2.line(canvas, shoulder, (ex2, elbow_y), (80, 100, 160), 8)
        cv2.line(canvas, (ex2, elbow_y), (ex2 + 10, hand_y), (80, 100, 160), 7)
    
    def draw_head(canvas, pos):
        cv2.ellipse(canvas, pos, (22, 26), 0, 0, 360, (70, 90, 140), -1)
        cv2.ellipse(canvas, pos, (22, 26), 0, 0, 360, (180, 200, 230), 2)
    
    for i in range(num_frames):
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 220
        cycle = num_frames // 3
        t = (i % cycle) / cycle
        phase = smooth(t)
        
        if exercise == "squat":
            shoulder = (mid_x, 130)
            hip_y = int(230 + 80 * phase)
            hip_x = int(mid_x + 15 * phase)
            knee_x = int(mid_x - 10 + 60 * phase - 10 * max(0, phase - 0.5) * 2)
            knee_y = int(360 - 90 * phase)
            ankle = (int(mid_x - 20 + 20 * phase), 450)
            torso = np.array([[mid_x-25,130],[mid_x+25,130],[hip_x+20,hip_y],[hip_x-20,hip_y]], np.int32)
            cv2.fillPoly(canvas, [torso], (80, 120, 180))
            cv2.line(canvas, (hip_x,hip_y), (knee_x,knee_y), (100,100,180), 18)
            cv2.line(canvas, (knee_x,knee_y), ankle, (130,130,200), 14)
            draw_head(canvas, (mid_x, 80))
            draw_arms(canvas, shoulder, hip_y, phase)
            for pt, label in [((hip_x,hip_y),"髋"),((knee_x,knee_y),"膝"),(ankle,"踝"),(shoulder,"肩")]:
                cv2.circle(canvas, pt, 5, (0,0,255), -1)
                cv2.putText(canvas, label, (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        elif exercise in ("bench-press", "chest-press"):
            # 仰卧杠铃卧推（侧视图）
            shoulder = (mid_x, 200)
            elbow_x = int(mid_x - 40 + 70 * phase)
            elbow_y = int(290 - 50 * phase)
            wrist_x = int(mid_x - 60 + 60 * phase)
            wrist_y = int(340 - 110 * phase)
            hip = (mid_x, 350)
            # 凳面
            cv2.line(canvas, (mid_x-120,180), (mid_x+120,180), (80,80,80), 15)
            cv2.line(canvas, (mid_x-120,180), (mid_x-130,210), (80,80,80), 8)
            cv2.line(canvas, (mid_x+120,180), (mid_x+130,210), (80,80,80), 8)
            # 身体
            cv2.line(canvas, shoulder, hip, (80,120,180), 16)  # 躯干
            cv2.line(canvas, shoulder, (elbow_x,elbow_y), (100,100,180), 12)  # 上臂
            cv2.line(canvas, (elbow_x,elbow_y), (wrist_x,wrist_y), (100,100,180), 10)  # 前臂
            # 杠铃
            cv2.line(canvas, (wrist_x-10,wrist_y-5), (wrist_x+10,wrist_y-5), (60,60,60), 6)
            draw_head(canvas, (mid_x-10, 160))
            for pt, label in [(shoulder,"肩"),((elbow_x,elbow_y),"肘"),((wrist_x,wrist_y),"腕")]:
                cv2.circle(canvas, pt, 5, (0,0,255), -1)
                cv2.putText(canvas, label, (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        elif exercise == "incline-db-press":
            # 上斜哑铃卧推
            shoulder = (mid_x, 180)
            # 左臂
            l_el_x = int(mid_x - 60 + 50 * phase)
            l_el_y = int(250 - 60 * phase)
            l_wr_x = int(mid_x - 80 + 40 * phase)
            l_wr_y = int(290 - 100 * phase)
            # 右臂
            r_el_x = int(mid_x + 60 - 50 * phase)
            r_el_y = int(250 - 60 * phase)
            r_wr_x = int(mid_x + 80 - 40 * phase)
            r_wr_y = int(290 - 100 * phase)
            hip = (mid_x, 340)
            cv2.line(canvas, (mid_x-130,170), (mid_x+130,170), (80,80,80), 15)  # 凳面
            cv2.line(canvas, (mid_x-130,170), (mid_x-140,200), (80,80,80), 8)
            cv2.line(canvas, (mid_x+130,170), (mid_x+140,200), (80,80,80), 8)
            cv2.line(canvas, shoulder, hip, (80,120,180), 16)
            cv2.line(canvas, shoulder, (l_el_x,l_el_y), (100,100,180), 10)  # 左上臂
            cv2.line(canvas, (l_el_x,l_el_y), (l_wr_x,l_wr_y), (100,100,180), 9)  # 左前臂
            cv2.line(canvas, shoulder, (r_el_x,r_el_y), (100,100,180), 10)  # 右上臂
            cv2.line(canvas, (r_el_x,r_el_y), (r_wr_x,r_wr_y), (100,100,180), 9)  # 右前臂
            # 哑铃
            for wx,wy in [(l_wr_x,l_wr_y),(r_wr_x,r_wr_y)]:
                cv2.rectangle(canvas, (wx-12,wy-18), (wx+12,wy-6), (60,60,60), -1)
            draw_head(canvas, (mid_x, 150))
            for pt,label in [(shoulder,"肩"),((l_el_x,l_el_y),"左肘"),((r_el_x,r_el_y),"右肘")]:
                cv2.circle(canvas, pt, 4, (0,0,255), -1)
                cv2.putText(canvas, label, (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
        
        elif exercise in ("pull-up", "lat-pulldown"):
            bar_h = 100
            cv2.line(canvas, (mid_x-120, bar_h), (mid_x+120, bar_h), (100,100,100), 8)
            s = 1 - phase  # 0=底, 1=顶
            head = (mid_x, int(bar_h + 40 + 70 * s))
            shoulder = (mid_x, int(bar_h + 80 + 80 * s))
            elbow_x = int(mid_x - 15 + 15 * (1 - s))
            elbow_y = int(bar_h + 130 + 80 * s)
            wrist = (int(mid_x - 25 + 20 * (1 - s)), int(bar_h + 180 + 50 * s))
            hip = (mid_x, int(bar_h + 180 + 100 * s))
            # 身体
            cv2.line(canvas, shoulder, hip, (80,120,180), 16)
            cv2.line(canvas, shoulder, (elbow_x,elbow_y), (100,100,180), 10)
            cv2.line(canvas, (elbow_x,elbow_y), wrist, (100,100,180), 8)
            # 右臂对称
            r_el_x = int(mid_x + 15 - 15 * (1 - s))
            r_el_y = elbow_y
            r_wr = (int(mid_x + 25 - 20 * (1 - s)), wrist[1])
            cv2.line(canvas, shoulder, (r_el_x,r_el_y), (100,100,180), 10)
            cv2.line(canvas, (r_el_x,r_el_y), r_wr, (100,100,180), 8)
            draw_head(canvas, head)
            for pt,label in [(head,"头"),(shoulder,"肩"),((elbow_x,elbow_y),"肘"),(hip,"髋")]:
                cv2.circle(canvas, pt, 4, (0,0,255), -1)
                cv2.putText(canvas, label, (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            # 握杠
            cv2.line(canvas, (mid_x-30,bar_h+5), (mid_x+30,bar_h+5), (180,130,80), 4)
        
        elif exercise in ("shoulder-press",):
            shoulder = (mid_x, 180)
            # 推起时手臂上举
            s = 1 - phase  # 0=底, 1=顶
            el_x = int(mid_x - 30 + 20 * s)
            el_y = int(260 - 80 * s)
            wr_x = int(mid_x - 25 + 15 * s)
            wr_y = int(340 - 160 * s)
            hip = (mid_x, 350)
            cv2.line(canvas, (mid_x-20,160), (mid_x+20,160), (80,80,80), 12)  # 凳面
            cv2.line(canvas, shoulder, hip, (80,120,180), 16)
            cv2.line(canvas, shoulder, (el_x,el_y), (100,100,180), 10)
            cv2.line(canvas, (el_x,el_y), (wr_x,wr_y), (100,100,180), 8)
            draw_head(canvas, (mid_x, 150))
            # 杠铃片
            cv2.circle(canvas, (wr_x,wr_y-10), 10, (60,60,60), -1)
            cv2.line(canvas, (wr_x-5,wr_y-10), (wr_x+5,wr_y-10), (100,100,100), 4)
            for pt,label in [(shoulder,"肩"),((el_x,el_y),"肘"),((wr_x,wr_y),"腕")]:
                cv2.circle(canvas, pt, 4, (0,0,255), -1)
                cv2.putText(canvas, label, (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
        
        elif exercise in ("back-pull",):
            # 背部划船
            shoulder = (mid_x, 170)
            s = 1 - phase  # 0=伸展, 1=收缩
            el_x = int(mid_x + 40 * s)
            el_y = int(220 + 30 * s - 40 * s)
            wr_x = int(mid_x + 100 * s)
            wr_y = int(290 + 20 * s - 30 * s)
            hip = (mid_x, 350)
            cv2.line(canvas, shoulder, hip, (80,120,180), 16)
            cv2.line(canvas, shoulder, (el_x,el_y), (100,100,180), 10)
            cv2.line(canvas, (el_x,el_y), (wr_x,wr_y), (100,100,180), 8)
            draw_head(canvas, (mid_x-5, 140))
            for pt,label in [(shoulder,"肩"),((el_x,el_y),"肘"),((wr_x,wr_y),"腕")]:
                cv2.circle(canvas, pt, 4, (0,0,255), -1)
                cv2.putText(canvas, label, (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
        
        label = EXERCISES.get(exercise, squat()).label if exercise in EXERCISES else exercise
        cv2.putText(canvas, f"{label} Demo", (10, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
        cv2.putText(canvas, f"帧 {i+1}/{num_frames}", (w-110, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)
        out.write(canvas)
    
    out.release()
    print(f"合成视频: {path} ({num_frames}帧, {fps}fps)")


def plot_angles(angle_sequence, exercise):
    """绘制所有配置的角度时序图"""
    angle_names = list(dict.fromkeys(
        k for f in angle_sequence for k in f if not k.startswith("_") and isinstance(f[k], (int, float))
    ))
    
    n = min(len(angle_names), 4)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3*n+1), facecolor='white')
    if n == 1:
        axes = [axes]
    
    frames = [f["frame"] for f in angle_sequence]
    colors = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800','#00BCD4']
    
    for i, name in enumerate(angle_names[:4]):
        ax = axes[i]
        vals = [f.get(name) for f in angle_sequence]
        ax.plot(frames, vals, color=colors[i % len(colors)], lw=1.5)
        ax.set_ylabel(f"{name} (°)", fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    if n > 1:
        axes[-1].set_xlabel("帧", fontsize=10)
    
    plt.tight_layout()
    chart_path = Path(OUTPUT_DIR) / "angle_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"角度图表: {chart_path}")
    return chart_path


# ═══════════════════════════════════════════════════════
# Part 5: DeepSeek 教练解读
# ═══════════════════════════════════════════════════════

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL",
                                   "https://api.deepseek.com/v1/chat/completions")

def interpret_with_deepseek(exercise_label, total_score, scores, details, view_type, phase_text=""):
    """将评分数据发给 DeepSeek，返回自然语言教练反馈"""
    if not DEEPSEEK_API_KEY:
        return "（未配置 DeepSeek API Key，无法生成教练解读）"

    # 构建评分摘要
    metrics_lines = []
    for k, v in scores.items():
        if k in ("total", "_view", "_details"):
            continue
        d = details.get(k, {})
        extra = " | ".join(f"{sk}: {sv}" for sk, sv in d.items() if sk != "score")
        score_str = f"  - {k}: {v}分"
        if extra:
            score_str += f" ({extra})"
        metrics_lines.append(score_str)

    summary = (
        f"【动作分析结果】\n"
        f"动作: {exercise_label}\n"
        f"拍摄视角: {'侧面' if view_type == 'side' else '正面'}\n"
        f"总分: {total_score}/100\n"
        f"详细评分:\n" + "\n".join(metrics_lines) +
        (f"\n{phase_text}" if phase_text else "")
    )

    system_prompt = (
        "你是一名退役特种兵健身教练，风格毒舌但专业。"
        "你的特点是：直接指出问题、不留情面、但每句话都有干货。"
        "用户来找你是为了变强，不是来听好话的。"
        "要求：\n"
        "1. 先给总体评价（1-2句话），可以带讽刺\n"
        "2. 然后列出2-4条具体问题，每条包含："
        "问题描述 → 为什么不对 → 怎么改，语气要狠\n"
        "3. 如果提供了【逐次重复分析】，必须用数据打脸"
        "（如：第4次比第1次浅7度，你在偷懒？）\n"
        "4. 最后给1个下次训练的硬性要求\n"
        "5. 可以用比喻、讽刺、激将法，但不能人身攻击\n"
        "6. 如果数据不完整，直接说'角度没拍全，重拍'\n"
        "7. 不要使用 Markdown 格式，纯文字\n"
        "8. 适当使用口语化表达，像真人在说话"
    )

    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary}
        ],
        "temperature": 0.7,
        "max_tokens": 800,
    }).encode("utf-8")

    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"（教练解读生成失败: {str(e)}）"


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def list_exercises():
    print("\n支持的动作:")
    print(f"  {'名称':<20} {'标签'}")
    print(f"  {'-'*20} {'-'*30}")
    for name, ex in sorted(EXERCISES.items()):
        print(f"  {name:<20} {ex.label}")
    print()

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    args = [a for a in sys.argv[1:] if a]
    
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        list_exercises()
        return
    
    if args[0] == "--list":
        list_exercises()
        return
    
    if args[0] == "--demo":
        exercise = args[1] if len(args) > 1 and args[1] in EXERCISES else "squat"
        video_path = f"{OUTPUT_DIR}/synthetic_{exercise}.mp4"
        print(f"Demo: {EXERCISES[exercise].label}")
        create_demo_video(video_path, exercise=exercise)
        angle_sequence, out_video = analyze_video(video_path, exercise_name=exercise)
    else:
        video_path = args[0]
        if not os.path.exists(video_path):
            print(f"❌ 文件不存在: {video_path}")
            sys.exit(1)
        
        exercise_name = args[1] if len(args) > 1 and args[1] in EXERCISES else None
        if exercise_name:
            print(f"指定动作: {EXERCISES[exercise_name].label}")
        else:
            print("动作: 自动检测")
        
        angle_sequence, out_video = analyze_video(video_path, exercise_name)
    
    if not angle_sequence:
        print("❌ 未检测到人体")
        print("可能原因: 视频角度不正、画面模糊、或非人体动作")
        return
    
    # 检查是否有实际的关键点数据
    has_data = any(k for f in angle_sequence for k in f 
                   if not k.startswith("_") and f[k] is not None)
    if not has_data:
        print("⚠️  检测到人体但未能提取角度数据（可能是非标准姿态）")
        # 仍然尝试评分（会得到 0 分）
    
    # 确定最终使用的动作
    detected = {}
    for f in angle_sequence:
        ex = f.get("_exercise")
        if ex:
            detected[ex] = detected.get(ex, 0) + 1
    best_ex = max(detected, key=detected.get) if detected else "squat"
    exercise = EXERCISES.get(best_ex, squat())
    print(f"  最终识别动作: {exercise.label}")
    
    # 视角自适应评分
    view_type = detect_view_type(angle_sequence)
    print(f"  拍摄视角: {'侧面' if view_type == 'side' else '正面'}")
    
    # 视角匹配检查
    view_warning = ""
    if exercise.recommended_view:
        # 正面拍摄时 "back" 视角也算正面
        actual = view_type  # "front" or "side"
        expected = exercise.recommended_view
        if expected == "back" and actual == "front":
            pass  # 引体向上正面拍是常态，不警告
        elif expected == "side" and actual == "front":
            view_warning = exercise.view_warning
        elif expected == "front" and actual == "side":
            view_warning = exercise.view_warning
    
    if view_warning:
        print(f"\n  ⚠️  {view_warning}")
    print("\n" + "=" * 56)
    print(f"  评分: {exercise.label}")
    print("=" * 56)
    result = score_with_view(exercise, angle_sequence, view_type)
    details = result.pop("_details", {})
    view_info = result.pop("_view", view_type)
    
    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │  总分: {result['total']:>3}/100{' ' * (36 - len(str(result['total'])))}{'│'}")
    print(f"  ├──────────────────────────────────────┤")
    for k, v in result.items():
        if k == "total": continue
        extra = details.get(k, {})
        extra_str = " | ".join(f"{sk}: {sv}" for sk, sv in extra.items() if sk != "score")
        score_val = round(v) if isinstance(v, float) else v
        print(f"  │  {score_val:<4} {k:<28}{'│'}")
        if extra_str:
            print(f"  │  {'':>4} {extra_str:<28}{'│'}")
    print(f"  └──────────────────────────────────────┘")
    
    # 教练解读 + 阶段分析
    print("\n" + "=" * 56)
    print("  教练解读")
    print("=" * 56)
    print()
    
    # 获取肘角数据做逐次重复分析（不影响评分）
    phase_text = ""
    try:
        elbow_vals = get_col(angle_sequence, "elbow")
        if elbow_vals and len(filter_none(elbow_vals)) > 10:
            reps = analyze_phases(elbow_vals)
            if reps:
                phase_text = phases_to_deepseek_text(exercise.label, reps, view_type,
                                                      phase_inverted=exercise.phase_inverted)
                print(f"  [逐次重复分析: {len(reps)}次]")
    except Exception:
        pass

    interpretation = interpret_with_deepseek(
        exercise.label, result.get("total", 0),
        result, details, view_type, phase_text
    )
    print(interpretation)
    print()
    
    # 图表
    print("\n" + "=" * 56)
    print("  角度时序图")
    print("=" * 56)
    chart = plot_angles(angle_sequence, exercise)
    
    # 保存
    result_path = Path(OUTPUT_DIR) / "score_result.json"
    clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
    output = {"exercise": exercise.name, "label": exercise.label,
              "result": clean_result, "details": {k: v for k,v in details.items()},
              "view": view_info, "interpretation": interpretation}
    if view_warning:
        output["view_warning"] = view_warning
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  {'=' * 52}")
    print(f"  输出文件:")
    print(f"    输入视频:    {video_path}")
    print(f"    标注视频:    {out_video}")
    print(f"    角度图表:    {chart}")
    print(f"    评分结果:    {result_path}")
    print(f"  {'=' * 52}")
    print("\n  ✅ 完成！")


if __name__ == "__main__":
    main()
