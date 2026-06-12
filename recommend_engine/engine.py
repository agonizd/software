"""
engine.py — C 模块核心：推荐引擎

对外暴露: recommend(face_report, preferences, top_n=5) -> List[Recommendation]

评分流程:
  1. 如果 preferences.natural_language 非空且 style_vector 为空
     → 先调用 infer_style_vector() 转为向量
  2. 调用 B 模块的 search_hairstyles(face_shape, style_vector, length, curl)
  3. 对返回的每条发型计算综合评分:
     score = face_match * 0.50 + style_similarity * 0.30 + popularity * 0.20
  4. 按 score 降序 → 取前 top_n → 生成推荐理由(reasons)
"""

from __future__ import annotations
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts import (
    FaceReport, StylePreferences, StyleVector,
    Hairstyle, Recommendation, FaceShape,
)
from recommend_engine.scoring import (
    calc_face_match, calc_style_similarity, calc_total_score,
)

# ─────────────────────────────────────────────────────────────
# B 模块接口：B 交付后把下面这行取消注释，注释掉 Mock 那行
# from hairstyle_db.db import search_hairstyles
from contracts import mock_search_hairstyles as search_hairstyles
# ─────────────────────────────────────────────────────────────


def recommend(
    face_report: FaceReport,
    preferences: StylePreferences,
    top_n: int = 5,
) -> List[Recommendation]:
    """
    综合脸型 + 风格偏好，返回 Top-N 发型推荐。

    参数:
        face_report:  A 模块的输出
        preferences:  用户偏好（风格向量 or 自然语言）
        top_n:        返回 Top-N（默认 5, 最大 10）

    返回:
        推荐列表（按 score 降序），可能为空列表（无匹配）
    """
    if top_n < 1 or top_n > 10:
        raise ValueError(f"top_n 必须在 1~10 之间,实际: {top_n}")

    # 1. 如果有自然语言但无向量 → 先反向推理
    sv = preferences.style_vector
    if sv is None and preferences.natural_language is not None:
        from recommend_engine.reverse_infer import infer_style_vector
        sv = infer_style_vector(preferences.natural_language)

    # 2. 从 B 拉取候选发型
    candidates: List[Hairstyle] = search_hairstyles(
        face_shape=face_report.face_shape,
        style_vector=sv,
        length=preferences.preferred_length,
        curl=preferences.preferred_curl,
        limit=50,
    )

    # 降级：候选为空时忽略脸型，纯风格推荐
    fallback_mode = False
    if not candidates:
        candidates = search_hairstyles(
            style_vector=sv,
            length=preferences.preferred_length,
            curl=preferences.preferred_curl,
            limit=50,
        )
        fallback_mode = True

    if not candidates:
        return []

    # 3. 对每个候选发型打分
    scored = []
    for hairstyle in candidates:
        face_match = calc_face_match(face_report, hairstyle)
        style_sim = calc_style_similarity(sv, hairstyle.style_vector) if sv else 0.0
        popularity = hairstyle.popularity

        total = calc_total_score(face_match, style_sim, popularity)
        reasons = _build_reasons(
            face_report, hairstyle, face_match, style_sim, sv, fallback_mode
        )
        scored.append((total, hairstyle, reasons))

    # 4. 降序排序，取 Top-N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    # 5. 归一化 score 到 0~1
    max_score = top[0][0] if top else 1.0
    if max_score == 0:
        max_score = 1.0

    results = []
    for score, hairstyle, reasons in top:
        results.append(Recommendation(
            hairstyle=hairstyle,
            score=round(score / max_score, 4),
            reasons=reasons,
            details={
                "face_match": round(
                    calc_face_match(face_report, hairstyle) *
                    0.50, 4
                ),
                "style_similarity": round(
                    calc_style_similarity(sv, hairstyle.style_vector) *
                    0.30 if sv else 0.0, 4
                ),
                "popularity": hairstyle.popularity,
            },
        ))
    return results


def _build_reasons(
    face_report: FaceReport,
    hairstyle: Hairstyle,
    face_match: float,
    style_sim: float,
    sv: StyleVector = None,
    fallback_mode: bool = False,
) -> List[str]:
    """生成至少 2 条推荐理由"""
    reasons = []

    # 理由 1：脸型
    if fallback_mode:
        reasons.append("当前脸型候选较少，以风格匹配为主要依据")
    elif face_match == 1.0:
        reasons.append(
            f"你的{face_report.face_shape.value}非常适合{hairstyle.name}"
        )

    # 理由 2：风格契合度
    if style_sim > 0.7 and sv is not None:
        best_dim = sv.dominant_dim()
        reasons.append(
            f"该发型的{best_dim}风格与你偏好一致(相似度{style_sim:.0%})"
        )
    elif style_sim > 0.4 and sv is not None:
        reasons.append("与你的风格偏好较为契合")

    # 理由 3：热度
    if hairstyle.popularity > 0.8:
        reasons.append("该发型当前很受欢迎，好评度高")
    elif hairstyle.popularity > 0.6:
        reasons.append("口碑稳定的经典款式")

    # 理由 4：避雷提示
    if hairstyle.warnings:
        reasons.append(f"提示：{'; '.join(hairstyle.warnings)}")

    # 兜底：保证至少 2 条
    if len(reasons) < 2:
        reasons.append(
            f"综合评分 {round((face_match * 0.5 + style_sim * 0.3 + hairstyle.popularity * 0.2) * 100):.0f} 分"
        )

    return reasons
