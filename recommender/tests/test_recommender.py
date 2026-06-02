"""
test_recommender.py — C 模块单元测试
运行: cd D:/work工作空间 && $env:PYTHONPATH="D:\\work工作空间"; python -m pytest recommender/tests/test_recommender.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contracts import (
    FaceReport, FaceShape, StylePreferences, StyleVector,
    HairLength, HairCurl, FaceFeatures,
    ALL_STYLE_DIMS,
)
from recommender.engine import recommend
from recommender.reverse_infer import infer_style_vector, _infer_by_keywords


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def oval_face():
    return FaceReport(
        face_shape=FaceShape.OVAL, confidence=0.92,
        features=FaceFeatures(face_ratio=1.48, jaw_cheek_ratio=0.78, forehead_ratio=1.05),
    )

@pytest.fixture
def round_face():
    return FaceReport(
        face_shape=FaceShape.ROUND, confidence=0.88,
        features=FaceFeatures(face_ratio=1.05, jaw_cheek_ratio=0.92, forehead_ratio=0.98),
    )

@pytest.fixture
def diamond_face():
    return FaceReport(
        face_shape=FaceShape.DIAMOND, confidence=0.65,
        features=FaceFeatures(face_ratio=1.42, jaw_cheek_ratio=0.55, forehead_ratio=0.82),
    )

@pytest.fixture
def elegant_prefs():
    return StylePreferences(
        style_vector=StyleVector(优雅=0.9, 复古=0.6),
    )

@pytest.fixture
def crisp_prefs():
    return StylePreferences(
        style_vector=StyleVector(干练=0.9, 优雅=0.3),
    )

@pytest.fixture
def natural_language_prefs():
    return StylePreferences(natural_language="干练通勤风")


# ─────────────────────────────────────────────────────────────
# 测试 1：正常流程 - 返回 Top-5 且按分数降序
# ─────────────────────────────────────────────────────────────

def test_recommend_returns_top5_sorted(oval_face, crisp_prefs):
    result = recommend(oval_face, crisp_prefs, top_n=5)
    assert len(result) == 5, f"期望 5 条推荐，实际 {len(result)} 条"
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True), "推荐列表未按分数降序排列"


# ─────────────────────────────────────────────────────────────
# 测试 2：分数归一化到 0~1
# ─────────────────────────────────────────────────────────────

def test_recommend_scores_normalized(round_face, elegant_prefs):
    result = recommend(round_face, elegant_prefs, top_n=5)
    assert len(result) > 0, "推荐结果为空"
    for r in result:
        assert 0.0 <= r.score <= 1.0, f"score={r.score} 超出 [0, 1] 范围"


# ─────────────────────────────────────────────────────────────
# 测试 3：每条推荐至少 2 条理由 + 包含 details
# ─────────────────────────────────────────────────────────────

def test_recommend_has_reasons_and_details(oval_face, crisp_prefs):
    result = recommend(oval_face, crisp_prefs, top_n=5)
    for r in result:
        assert len(r.reasons) >= 2, f"「{r.hairstyle.name}」只有 {len(r.reasons)} 条理由"
        assert len(r.details) == 3, f"details 应包含 3 个因子，实际 {len(r.details)}"


# ─────────────────────────────────────────────────────────────
# 测试 4：关键词推理 - "干练通勤" 应使干练维度最高
# ─────────────────────────────────────────────────────────────

def test_infer_keywords_crisp_dominates():
    vec = _infer_by_keywords("我想要干练通勤风")
    assert vec.干练 > 0.5, f"干练={vec.干练:.2f}，期望 > 0.5"
    dims = {d: getattr(vec, d) for d in ALL_STYLE_DIMS}
    max_dim = max(dims, key=dims.get)
    assert max_dim == "干练", f"最高维度应为「干练」，实际为「{max_dim}」"


# ─────────────────────────────────────────────────────────────
# 测试 5：反向推理输入校验
# ─────────────────────────────────────────────────────────────

def test_infer_style_vector_empty_raises():
    with pytest.raises(ValueError, match="不能为空"):
        infer_style_vector("")

def test_infer_style_vector_too_long_raises():
    with pytest.raises(ValueError, match="超过 200 字"):
        infer_style_vector("很长的文字" * 100)


# ─────────────────────────────────────────────────────────────
# 测试 6：菱形脸（数据稀少）降级，不崩溃
# ─────────────────────────────────────────────────────────────

def test_recommend_fallback_when_rare_face(diamond_face, elegant_prefs):
    result = recommend(diamond_face, elegant_prefs, top_n=5)
    assert isinstance(result, list), "返回值应为列表"
    for r in result:
        assert isinstance(r, type(result[0])) if result else True


# ─────────────────────────────────────────────────────────────
# 测试 7：自然语言偏好 → 自动调用 infer_style_vector
# ─────────────────────────────────────────────────────────────

def test_recommend_with_natural_language(oval_face, natural_language_prefs):
    result = recommend(oval_face, natural_language_prefs, top_n=5)
    assert isinstance(result, list), "自然语言偏好应正常工作"


# ─────────────────────────────────────────────────────────────
# 测试 8：推荐结果使用官方数据类（Hairstyle 含 length/curl）
# ─────────────────────────────────────────────────────────────

def test_recommend_hairstyle_has_length_curl(oval_face, crisp_prefs):
    result = recommend(oval_face, crisp_prefs, top_n=5)
    for r in result:
        assert hasattr(r.hairstyle, "length"), "Hairstyle 应包含 length 字段"
        assert hasattr(r.hairstyle, "curl"), "Hairstyle 应包含 curl 字段"
        assert isinstance(r.hairstyle.length, HairLength)
        assert isinstance(r.hairstyle.curl, HairCurl)
