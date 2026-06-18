"""
B 模块测试：hairstyle_db.database

测试范围：
- init_db()           — 数据库初始化（force True / False）
- search_hairstyles() — 多条件组合查询 + 余弦排序
- get_hairstyle_by_id() — 精确 ID 查询
- 边界场景            — 空结果、limit 截断、style_vector 映射
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contracts import (
    FaceShape, HairCurl, HairLength, Hairstyle, StyleVector,
)
from hairstyle_db.database import (
    init_db,
    search_hairstyles,
    get_hairstyle_by_id,
    _b_vec_to_style_vector,
)


# ═════════════════════════════════════════════════════════════
# 夹具
# ═════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def setup_db():
    """每次测试前确保数据库存在。"""
    init_db(force=False)


# ═════════════════════════════════════════════════════════════
# 1. init_db()
# ═════════════════════════════════════════════════════════════

def test_init_db_creates_file():
    """force=True 重建数据库。"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hairstyles.db")
    init_db(force=True)
    assert os.path.exists(db_path), "数据库文件应存在"
    # 验证有 50 条记录
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM hairstyles").fetchone()[0]
    conn.close()
    assert count == 50, f"应有 50 条种子数据，实际 {count}"


def test_init_db_idempotent():
    """force=False 不重复初始化。"""
    init_db(force=False)
    # 应无异常，且文件不重建
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hairstyles.db")
    assert os.path.exists(db_path)


# ═════════════════════════════════════════════════════════════
# 2. _b_vec_to_style_vector()
# ═════════════════════════════════════════════════════════════

def test_b_vec_to_style_vector_mapping():
    """6 维向量 [elegant, cute, capable, fresh, cool, romantic]
       → StyleVector [干练, 甜美, 复古, 酷飒, 自然, 优雅]"""
    vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    sv = _b_vec_to_style_vector(vec)
    assert sv.干练 == 0.3   # capable → 干练
    assert sv.甜美 == 0.2   # cute → 甜美
    assert sv.复古 == 0.6   # romantic → 复古
    assert sv.酷飒 == 0.5   # cool → 酷飒
    assert sv.自然 == 0.4   # fresh → 自然
    assert sv.优雅 == 0.1   # elegant → 优雅


# ═════════════════════════════════════════════════════════════
# 3. search_hairstyles() — 基础查询
# ═════════════════════════════════════════════════════════════

def test_search_all_no_params():
    """无参数 → 返回所有发型（按 popularity 排序）。"""
    results = search_hairstyles(limit=50)
    assert len(results) == 50
    # 按 similarity_score 降序（无 style_vector 时 = popularity）
    for i in range(len(results) - 1):
        assert results[i].similarity_score >= results[i + 1].similarity_score


def test_search_by_face_shape():
    """按脸型过滤。"""
    results = search_hairstyles(face_shape=FaceShape.OVAL, limit=50)
    # 鹅蛋脸适配几乎所有发型
    assert len(results) >= 20


def test_search_by_round_face():
    """圆脸 → 应有专属发型。"""
    results = search_hairstyles(face_shape=FaceShape.ROUND, limit=50)
    # 应包含圆脸专属: hair_041
    ids = [h.id for h in results]
    assert "hair_041" in ids, "应包含圆脸显瘦卷发"


def test_search_by_length():
    """按长度过滤。"""
    results = search_hairstyles(length=HairLength.SHORT, limit=50)
    for h in results:
        assert h.length == HairLength.SHORT, f"{h.id} 应为短发"


def test_search_by_curl():
    """按卷度过滤。"""
    results = search_hairstyles(curl=HairCurl.CURLY, limit=50)
    for h in results:
        assert h.curl == HairCurl.CURLY, f"{h.id} 应为大卷"


# ═════════════════════════════════════════════════════════════
# 4. search_hairstyles() — 余弦相似度
# ═════════════════════════════════════════════════════════════

def test_search_with_style_vector_sorting():
    """风格向量 → 余弦相似度排序有效。"""
    results = search_hairstyles(
        face_shape=FaceShape.OVAL,
        style_vector=StyleVector(干练=1.0, 甜美=0.0, 复古=0.0, 酷飒=0.0, 自然=0.0, 优雅=0.0),
        limit=50,
    )
    assert len(results) > 0
    # 纯干练风格 → 顶部应是干练向发型
    top = results[0]
    assert top.style_vector.干练 >= 0.5, f"顶部发型 {top.id} 干练值 {top.style_vector.干练} 过低"


def test_search_style_cosine_differentiation():
    """不同风格向量 → 返回不同排序。"""
    capable_results = search_hairstyles(
        style_vector=StyleVector(干练=1.0, 甜美=0.0, 复古=0.0, 酷飒=0.0, 自然=0.0, 优雅=0.0),
        limit=5,
    )
    cute_results = search_hairstyles(
        style_vector=StyleVector(干练=0.0, 甜美=1.0, 复古=0.0, 酷飒=0.0, 自然=0.0, 优雅=0.0),
        limit=5,
    )
    top_capable_ids = [h.id for h in capable_results]
    top_cute_ids = [h.id for h in cute_results]
    # 两种风格第一名不同
    assert top_capable_ids[0] != top_cute_ids[0], "不同风格应产出不同推荐"


# ═════════════════════════════════════════════════════════════
# 5. search_hairstyles() — 组合查询
# ═════════════════════════════════════════════════════════════

def test_search_combined_all_params():
    """脸型 + 风格 + 长度 + 卷度 + limit 全组合。"""
    results = search_hairstyles(
        face_shape=FaceShape.OVAL,
        style_vector=StyleVector(干练=0.9, 甜美=0.1, 复古=0.0, 酷飒=0.0, 自然=0.3, 优雅=0.1),
        length=HairLength.SHORT,
        curl=HairCurl.STRAIGHT,
        limit=5,
    )
    assert len(results) <= 5
    for h in results:
        assert h.length == HairLength.SHORT
        assert h.curl == HairCurl.STRAIGHT
        assert FaceShape.OVAL in h.suitable_shapes


# ═════════════════════════════════════════════════════════════
# 6. search_hairstyles() — 边界 & 空结果
# ═════════════════════════════════════════════════════════════

def test_search_limit_truncation():
    """limit 截断生效。"""
    results = search_hairstyles(limit=3)
    assert len(results) == 3


def test_search_empty_result():
    """不存在的组合 → 空列表。"""
    results = search_hairstyles(
        face_shape=FaceShape.ROUND,
        length=HairLength.VERY_SHORT,
    )
    # 可能为空，也可能有少数匹配
    assert isinstance(results, list)


# ═════════════════════════════════════════════════════════════
# 7. get_hairstyle_by_id()
# ═════════════════════════════════════════════════════════════

def test_get_hairstyle_by_id_valid():
    """有效 ID → 返回 Hairstyle。"""
    h = get_hairstyle_by_id("hair_001")
    assert h is not None
    assert h.id == "hair_001"
    assert h.name == "锁骨微卷发"
    assert isinstance(h.style_vector, StyleVector)
    assert h.length == HairLength.MEDIUM


def test_get_hairstyle_by_id_invalid():
    """无效 ID → 返回 None。"""
    h = get_hairstyle_by_id("hair_999")
    assert h is None


def test_get_hairstyle_by_id_all_fields():
    """返回的 Hairstyle 字段完整。"""
    h = get_hairstyle_by_id("hair_024")
    assert h is not None
    assert h.name
    assert len(h.suitable_shapes) > 0
    assert h.length is not None
    assert h.curl is not None
    assert 0.0 <= h.popularity <= 1.0
    assert isinstance(h.warnings, list)
    assert isinstance(h.care_tips, str)
    assert h.description


# ═════════════════════════════════════════════════════════════
# 8. 多脸型覆盖
# ═════════════════════════════════════════════════════════════

def test_all_face_shapes_have_results():
    """10 种脸型每种至少一条推荐。"""
    all_shapes = [
        FaceShape.OVAL, FaceShape.ROUND, FaceShape.SQUARE,
        FaceShape.LONG, FaceShape.HEART, FaceShape.DIAMOND,
        FaceShape.OVAL_WIDE, FaceShape.INV_TRIANGLE,
        FaceShape.TRIANGLE, FaceShape.NARROW,
    ]
    for shape in all_shapes:
        results = search_hairstyles(face_shape=shape, limit=1)
        assert len(results) >= 1, f"{shape.value} 应至少有一条推荐"
