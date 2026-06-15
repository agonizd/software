"""
scoring.py — C 模块：多因子打分逻辑

score = face_match × 0.50
      + style_cosine_similarity × 0.30
      + popularity_score × 0.20
"""

from __future__ import annotations
import math
from typing import List
from contracts import (
    FaceReport, StyleVector, Hairstyle, FaceShape,
    ALL_STYLE_DIMS,
)


# 三因子权重（可调参）
WEIGHT_FACE = 0.50
WEIGHT_STYLE = 0.30
WEIGHT_POPULARITY = 0.20


def calc_face_match(face_report: FaceReport, hairstyle: Hairstyle) -> float:
    """
    脸型匹配得分。
    发型.suitable_shapes 包含用户脸型 → 1.0，否则 → 0.0。
    如果 face_report.confidence < 0.85，可以打折（可选增强）。
    """
    if face_report.face_shape in hairstyle.suitable_shapes:
        # 高置信度不打折，中等置信度 × confidence
        if face_report.is_reliable():
            return 1.0
        else:
            return face_report.confidence  # 如 0.7 → 0.7
    return 0.0


def calc_style_similarity(user_vec: StyleVector, hair_vec: StyleVector) -> float:
    """
    余弦相似度：用户风格向量 vs 发型风格向量。
    返回 0.0 ~ 1.0（负值 clip 到 0）。
    """
    a = user_vec.to_list()
    b = hair_vec.to_list()

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na < 1e-9 or nb < 1e-9:
        return 0.0

    cosine = dot / (na * nb)
    return max(0.0, cosine)


def calc_total_score(face_match: float, style_sim: float, popularity: float) -> float:
    """三因子加权求和"""
    return face_match * WEIGHT_FACE + style_sim * WEIGHT_STYLE + popularity * WEIGHT_POPULARITY
