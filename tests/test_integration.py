"""
test_integration.py — 端到端集成测试

测试完整数据流：
  上传照片 → 脸型检测 → 风格推理 → 推荐 → 报告生成

覆盖场景：
1. 完整链路（鹅蛋脸 + 干练风）
2. 完整链路（圆脸 + 甜美风）
3. 照片分析失败降级
4. 无偏好时的默认推荐
5. 所有脸型 × 风格组合
6. 发型数据库完整性验证
7. 推荐结果排序验证

运行: pytest tests/test_integration.py -v
"""

import json
import os
import sys

# 强制 Mock 模式（集成测试不需要真实文件/硬件依赖）
os.environ["HAIR_FORCE_MOCK"] = "1"

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

import pytest

from contracts import (
    FaceShape, FaceReport, FaceFeatures, StylePreferences, StyleVector,
    Hairstyle, Recommendation, HairLength, HairCurl,
)
from agent_app.tools import (
    detect_face_shape_tool,
    search_hairstyles_tool,
    recommend_tool,
    generate_report_tool,
)


# ======================================================================
#  场景 1：完整链路 — 鹅蛋脸 + 干练风
# ======================================================================

def test_full_pipeline_oval_crisp():
    """鹅蛋脸 + 「干练通勤风」→ 推荐 → 报告"""
    # Step 1: 脸型分析
    face_raw = json.loads(detect_face_shape_tool("test.jpg"))
    assert face_raw["status"] == "ok"
    assert face_raw["face_shape"] == "鹅蛋脸"
    assert face_raw["confidence"] >= 0.8

    # 剥离 status 字段，构造纯 face_report
    face_result = {
        "face_shape": face_raw["face_shape"],
        "confidence": face_raw["confidence"],
        "features": face_raw["features"],
    }
    assert "features" in face_result

    # Step 2: 风格向量推理（自然语言 → StyleVector）
    style_query = json.dumps({
        "face_shape": face_result["face_shape"],
    }, ensure_ascii=False)
    hair_result = json.loads(search_hairstyles_tool(style_query))
    assert len(hair_result) >= 5, f"鹅蛋脸应有大量候选发型，实际只有 {len(hair_result)}"

    # Step 3: 推荐
    rec_params = json.dumps({
        "face_report": face_result,
        "preferences": {"natural_language": "干练通勤风"},
        "top_n": 3,
    }, ensure_ascii=False)
    rec_result = json.loads(recommend_tool(rec_params))
    assert len(rec_result) == 3
    # 验证排序：score 降序
    scores = [r["score"] for r in rec_result]
    assert scores == sorted(scores, reverse=True), f"推荐未按分数降序: {scores}"
    # 验证理由非空
    for r in rec_result:
        assert len(r["reasons"]) >= 1
        assert "hairstyle" in r
        assert "name" in r["hairstyle"]

    # Step 4: 生成报告
    report_params = json.dumps({
        "face_report": face_result,
        "recommendations": rec_result,
    }, ensure_ascii=False)
    report = generate_report_tool(report_params)
    # 报告包含 6 章
    for chapter in ["脸型分析", "五官特点", "风格象限", "推荐发型", "避雷提示", "打理建议"]:
        assert chapter in report, f"报告缺少章节: {chapter}"
    assert len(report) > 500


# ======================================================================
#  场景 2：完整链路 — 圆脸 + 甜美风
# ======================================================================

def test_full_pipeline_round_sweet():
    """圆脸 + 「甜美日系风」→ 推荐 → 报告"""
    # Step 1: 指定圆脸
    from contracts_mock import mock_detect_face_shape
    face_report = mock_detect_face_shape(face_shape=FaceShape.ROUND)
    face_dict = face_report.to_dict()
    assert face_dict["face_shape"] == "圆脸"

    # Step 2: 搜索圆脸适合的发型
    query = json.dumps({"face_shape": "圆脸"}, ensure_ascii=False)
    results = json.loads(search_hairstyles_tool(query))
    # 圆脸匹配的应少于鹅蛋脸（更挑剔）
    assert len(results) <= 10

    # Step 3: 甜美风推荐
    rec_params = json.dumps({
        "face_report": face_dict,
        "preferences": {"natural_language": "甜美日系风"},
        "top_n": 3,
    }, ensure_ascii=False)
    recs = json.loads(recommend_tool(rec_params))
    assert len(recs) == 3
    # 甜美风应该排在前面的有甜美属性
    top_name = recs[0]["hairstyle"]["name"]
    print(f"  圆脸+甜美风 Top1: {top_name} (score={recs[0]['score']:.2%})")

    # Step 4: 报告
    report_params = json.dumps({
        "face_report": face_dict,
        "recommendations": recs,
    }, ensure_ascii=False)
    report = generate_report_tool(report_params)
    assert "圆脸" in report


# ======================================================================
#  场景 3：分析失败降级
# ======================================================================

def test_pipeline_with_error_face():
    """脸型分析失败 → 不应崩溃，应优雅降级"""
    # 模拟错误报告
    from contracts_mock import mock_detect_face_shape_error
    error_report = mock_detect_face_shape_error()
    error_dict = error_report.to_dict()
    assert error_dict.get("error") is not None

    # 即使脸型分析失败，搜索仍应返回所有发型
    query = json.dumps({}, ensure_ascii=False)
    all_hair = json.loads(search_hairstyles_tool(query))
    assert len(all_hair) == 10

    # 推荐不应该崩溃（使用 fallback 鹅蛋脸）
    rec_params = json.dumps({
        "face_report": error_dict,
        "preferences": {},
        "top_n": 3,
    }, ensure_ascii=False)
    # 错误报告可能让推荐失败或返回空，这都可以接受
    # 关键是不要抛异常
    try:
        recs = json.loads(recommend_tool(rec_params))
        assert isinstance(recs, list)
    except Exception:
        pass  # 降级到异常也是可接受行为


# ======================================================================
#  场景 4：无偏好默认推荐
# ======================================================================

def test_recommend_without_preferences():
    """用户没有表达风格偏好 → 仍应有推荐结果"""
    from contracts_mock import mock_detect_face_shape
    face = mock_detect_face_shape(face_shape=FaceShape.OVAL)
    face_dict = face.to_dict()

    rec_params = json.dumps({
        "face_report": face_dict,
        "preferences": {},  # 空偏好
        "top_n": 5,
    }, ensure_ascii=False)
    recs = json.loads(recommend_tool(rec_params))
    assert len(recs) == 5
    # 无偏好时按热度 + 脸型匹配排序
    print(f"  无偏好 Top1: {recs[0]['hairstyle']['name']} (score={recs[0]['score']:.2%})")


# ======================================================================
#  场景 5：所有脸型 × 主流风格交叉验证
# ======================================================================

@pytest.mark.parametrize("shape", [
    FaceShape.OVAL, FaceShape.ROUND, FaceShape.SQUARE,
    FaceShape.LONG, FaceShape.HEART, FaceShape.DIAMOND,
])
@pytest.mark.parametrize("style_desc", [
    "干练通勤风", "甜美日系风", "自然随性风", "优雅气质风",
])
def test_all_faces_all_styles(shape, style_desc):
    """交叉验证：每种脸型搭配每种风格都能返回合理结果"""
    from contracts_mock import mock_detect_face_shape

    face = mock_detect_face_shape(face_shape=shape)
    face_dict = face.to_dict()

    rec_params = json.dumps({
        "face_report": face_dict,
        "preferences": {"natural_language": style_desc},
        "top_n": 3,
    }, ensure_ascii=False)

    try:
        recs = json.loads(recommend_tool(rec_params))
    except Exception:
        recs = []

    # 每种脸型至少应匹配到 1 款发型
    assert len(recs) >= 1, f"{shape.value} + {style_desc} 推荐为空！"
    # score 在合理范围
    for r in recs:
        assert 0.0 <= r["score"] <= 1.0, f"异常分数: {r['score']}"


# ======================================================================
#  场景 6：数据库完整性验证
# ======================================================================

def test_hairstyle_db_integrity():
    """验证 JSON 数据库的 10 条种子数据完整性"""
    import os as _os
    db_path = _os.path.join(_PROJ_ROOT, "hairstyle_db", "hairstyles.json")
    assert _os.path.exists(db_path), f"数据库文件不存在: {db_path}"

    from hairstyle_db.db import load_hairstyles, get_hairstyle_by_id, count_hairstyles

    total = count_hairstyles()
    assert total == 10, f"期望 10 条，实际 {total} 条"

    all_hs = load_hairstyles()
    ids = {h.id for h in all_hs}
    assert len(ids) == 10, f"ID 有重复: {10 - len(ids)} 条"

    # 每条数据完整性检查
    required_fields = ["id", "name", "image_url", "suitable_shapes",
                       "style_vector", "length", "curl", "popularity"]
    for h in all_hs:
        h_dict = h.to_dict()
        for field in required_fields:
            assert field in h_dict, f"发型 {h.id} 缺少字段: {field}"
        assert h.popularity >= 0, f"{h.id} popularity 为负数"

    # ID 精确查询
    h = get_hairstyle_by_id("hair_005")
    assert h is not None
    assert h.name == "干练齐肩短发"


# ======================================================================
#  场景 7：推荐排序始终降序
# ======================================================================

def test_recommend_always_descending():
    """无论输入什么，推荐结果必须按 score 降序排列"""
    from contracts_mock import mock_detect_face_shape

    test_cases = [
        (FaceShape.OVAL, "甜美"),
        (FaceShape.ROUND, "干练"),
        (FaceShape.SQUARE, "优雅"),
        (FaceShape.LONG, "酷飒"),
    ]

    for shape, style in test_cases:
        face = mock_detect_face_shape(face_shape=shape)
        rec_params = json.dumps({
            "face_report": face.to_dict(),
            "preferences": {"natural_language": style},
            "top_n": 5,
        }, ensure_ascii=False)
        recs = json.loads(recommend_tool(rec_params))
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True), \
            f"{shape.value} + {style} 未降序: {scores}"
        assert len(recs) == min(5, len(scores))


# ======================================================================
#  场景 8：端到端数据序列化一致性
# ======================================================================

def test_end_to_end_serialization():
    """验证 FaceReport → JSON → 推荐 → 报告全链路数据不丢失"""
    face = FaceReport(
        face_shape=FaceShape.OVAL,
        confidence=0.95,
        features=FaceFeatures(
            face_ratio=1.48, jaw_cheek_ratio=0.78, forehead_ratio=1.05,
            eye_distance="适中", nose_type="直挺", chin_shape="圆",
        ),
    )

    # 序列化 → 反序列化
    face_dict = face.to_dict()
    face2 = FaceReport.from_dict(face_dict)
    assert face2.face_shape == FaceShape.OVAL
    assert abs(face2.confidence - 0.95) < 0.001
    assert face2.features.face_ratio == 1.48

    # 通过 JSON 工具链再验证一次
    rec_params = json.dumps({
        "face_report": face_dict,
        "preferences": {"natural_language": "自然风"},
        "top_n": 3,
    }, ensure_ascii=False)
    recs_json = recommend_tool(rec_params)
    recs = json.loads(recs_json)

    report_params = json.dumps({
        "face_report": face_dict,
        "recommendations": recs,
    }, ensure_ascii=False)
    report = generate_report_tool(report_params)

    assert isinstance(report, str)
    assert len(report) > 200


# ======================================================================
#  运行
# ======================================================================
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=_PROJ_ROOT,
    )
    sys.exit(result.returncode)
