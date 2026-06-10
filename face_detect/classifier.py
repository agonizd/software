# face_detect/classifier.py
"""
脸型分类器 — 根据面部比例判断 6 种脸型。

6 种脸型: 鹅蛋脸(OVAL)、圆脸(ROUND)、方脸(SQUARE)、
         长脸(LONG)、心形脸(HEART)、菱形脸(DIAMOND)

阈值来源: contracts.py 中 FaceFeatures 的阈值参考表
         （来自面部美学文献 + 实测调参）

注意: 阈值可以在这里调整，不要硬编码在 detector.py 里。
"""

import sys
import os

# 确保能导入项目根目录的 contracts.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contracts import FaceShape


# ── 阈值配置 ──────────────────────────────────────────────────
# 调参时只改这里！不要动判断逻辑。

THRESHOLDS = {
    # face_ratio（脸长/脸宽）阈值
    "LONG_FACE_RATIO":    1.70,   # > 1.70 → 长脸
    "ROUND_FACE_RATIO":   1.20,   # < 1.20 且 jaw_cheek_ratio > 0.80 → 圆脸
    "OVAL_RATIO_LOW":     1.30,   # 鹅蛋脸范围下限
    "OVAL_RATIO_HIGH":    1.60,   # 鹅蛋脸范围上限

    # jaw_cheek_ratio（下颌宽/颧骨宽）阈值
    "SQUARE_JAW_CHEEK":   0.90,   # > 0.90 → 方脸
    "ROUND_JAW_CHEEK":    0.80,   # 圆脸时 > 0.80
    "HEART_JAW_CHEEK":    0.70,   # < 0.70 → 心形脸/菱形脸候选

    # forehead_ratio（额头宽/颧骨宽）阈值
    "HEART_FOREHEAD":     1.05,   # > 1.05 → 心形脸候选（额头宽）
    "DIAMOND_FOREHEAD":   0.95,   # < 0.95 → 菱形脸候选（额头窄）
}


def classify_face_shape(features: dict) -> tuple:
    """
    根据面部比例判断脸型。

    参数:
        features: {
            "face_ratio":      float,
            "jaw_cheek_ratio": float,
            "forehead_ratio":  float,
        }

    返回:
        (FaceShape, confidence: float)
        confidence 范围 0.0~1.0

    判断优先级（由高到低，避免模糊地带）:
        1. 长脸  — face_ratio > 1.70
        2. 圆脸  — face_ratio < 1.20 且 jaw_cheek_ratio > 0.80
                   （先判断圆脸，避免低 face_ratio 被误判为方脸）
        3. 方脸  — jaw_cheek_ratio > 0.90 且 face_ratio ≥ 1.20
        4. 心形脸 — forehead_ratio > 1.05 且 jaw_cheek_ratio < 0.70
        5. 菱形脸 — jaw_cheek_ratio < 0.70 且 forehead_ratio < 0.95
        6. 鹅蛋脸 — 默认（黄金比例兜底）
    """
    fr  = features["face_ratio"]
    jcr = features["jaw_cheek_ratio"]
    fhr = features["forehead_ratio"]

    T = THRESHOLDS  # 缩写，方便写条件

    # ── 1. 长脸：脸长明显大于脸宽 ──
    if fr > T["LONG_FACE_RATIO"]:
        # face_ratio 越大，置信度越高
        conf = min(0.95, 0.70 + (fr - T["LONG_FACE_RATIO"]) * 0.5)
        return FaceShape.LONG, round(conf, 2)

    # ── 2. 圆脸：脸宽接近脸长，下颌宽 ──
    #    先判断圆脸，避免低 face_ratio 的圆脸被误判为方脸
    if fr < T["ROUND_FACE_RATIO"] and jcr > T["ROUND_JAW_CHEEK"]:
        conf = min(0.92, 0.70 + (T["ROUND_FACE_RATIO"] - fr) * 1.5)
        return FaceShape.ROUND, round(conf, 2)

    # ── 3. 方脸：下颌角宽，接近颧骨宽，且脸不太短 ──
    if jcr > T["SQUARE_JAW_CHEEK"] and fr >= T["ROUND_FACE_RATIO"]:
        conf = min(0.95, 0.70 + (jcr - T["SQUARE_JAW_CHEEK"]) * 2.0)
        return FaceShape.SQUARE, round(conf, 2)

    # ── 4. 心形脸：额头宽，下巴尖（下颌窄）──
    if fhr > T["HEART_FOREHEAD"] and jcr < T["HEART_JAW_CHEEK"]:
        conf = min(0.88, 0.70 + (fhr - T["HEART_FOREHEAD"]) * 1.5)
        return FaceShape.HEART, round(conf, 2)

    # ── 5. 菱形脸：颧骨最宽，额头和下巴都窄 ──
    if jcr < T["HEART_JAW_CHEEK"] and fhr < T["DIAMOND_FOREHEAD"]:
        conf = min(0.80, 0.65 + (T["HEART_JAW_CHEEK"] - jcr) * 1.0)
        return FaceShape.DIAMOND, round(conf, 2)

    # ── 6. 鹅蛋脸（默认，黄金比例）──
    if T["OVAL_RATIO_LOW"] <= fr <= T["OVAL_RATIO_HIGH"]:
        # 在鹅蛋脸最佳范围内，置信度最高
        conf = 0.90
    else:
        # 不在最佳范围但也不属于其他类型，置信度稍低
        conf = 0.72

    return FaceShape.OVAL, round(conf, 2)
