"""
test_agent.py — Agent 模块单元测试

测试覆盖：
1. 4 个工具函数的基本调用
2. Mock Agent 的完整决策流程
3. Session 状态同步
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_app.tools import (
    detect_face_shape_tool,
    search_hairstyles_tool,
    recommend_tool,
    generate_report_tool,
)
from agent_app.agent import MockAgent


# ============================================================
# 工具函数测试
# ============================================================

def test_detect_face_shape():
    """脸型识别工具"""
    result = json.loads(detect_face_shape_tool("test.jpg"))
    assert result["status"] == "ok"
    assert result["face_shape"] in ["鹅蛋脸", "圆脸", "方脸", "长脸", "心形脸", "菱形脸"]
    assert 0 <= result["confidence"] <= 1
    assert "features" in result


def test_search_hairstyles():
    """发型搜索工具"""
    query = json.dumps({"face_shape": "鹅蛋脸", "limit": 3}, ensure_ascii=False)
    result = json.loads(search_hairstyles_tool(query))
    assert isinstance(result, list)
    assert len(result) == 3
    for h in result:
        assert "name" in h
        assert "suitable_shapes" in h


def test_recommend():
    """推荐工具"""
    params = json.dumps({
        "face_report": {
            "face_shape": "鹅蛋脸", "confidence": 0.92,
            "features": {"face_ratio": 1.48, "jaw_cheek_ratio": 0.78, "forehead_ratio": 1.05,
                         "eye_distance": "适中", "nose_type": "直挺", "chin_shape": "圆"},
        },
        "preferences": {"natural_language": "干练通勤风"},
        "top_n": 3,
    }, ensure_ascii=False)
    result = json.loads(recommend_tool(params))
    assert len(result) == 3
    # 验证降序排列
    for i in range(len(result) - 1):
        assert result[i]["score"] >= result[i + 1]["score"]
    # 验证每条包含必要字段
    for r in result:
        assert "hairstyle" in r
        assert "score" in r
        assert "reasons" in r
        assert len(r["reasons"]) >= 1


def test_generate_report():
    """报告生成工具"""
    # 先获取推荐结果
    recs_params = json.dumps({
        "face_report": {
            "face_shape": "鹅蛋脸", "confidence": 0.92,
            "features": {"face_ratio": 1.48, "jaw_cheek_ratio": 0.78, "forehead_ratio": 1.05,
                         "eye_distance": "适中", "nose_type": "直挺", "chin_shape": "圆"},
        },
        "preferences": {"natural_language": "甜美"},
        "top_n": 3,
    }, ensure_ascii=False)
    recs = json.loads(recommend_tool(recs_params))

    # 生成报告
    params = json.dumps({
        "face_report": {
            "face_shape": "鹅蛋脸", "confidence": 0.92,
            "features": {"face_ratio": 1.48, "jaw_cheek_ratio": 0.78, "forehead_ratio": 1.05,
                         "eye_distance": "适中", "nose_type": "直挺", "chin_shape": "圆"},
        },
        "recommendations": recs,
    }, ensure_ascii=False)
    report = generate_report_tool(params)

    # 报告是 Markdown 字符串
    assert isinstance(report, str)
    assert len(report) > 500
    # 包含 6 章
    chapters = ["脸型分析", "五官特点", "风格象限", "推荐发型", "避雷提示", "打理建议"]
    for ch in chapters:
        assert ch in report, f"报告缺少章节: {ch}"


# ============================================================
# Mock Agent 测试
# ============================================================

def test_mock_agent_welcome():
    """开始对话 → 返回欢迎语"""
    agent = MockAgent()
    resp = agent.process("你好")
    assert "专属发型顾问" in resp
    assert "正面照" in resp


def test_mock_agent_face_analysis():
    """上传照片 → 分析脸型"""
    agent = MockAgent()
    resp = agent.process("帮我看看脸型", image_path="test.jpg")
    assert "分析完成" in resp
    assert "脸型是" in resp
    assert agent.session["stage"] == "face_analyzed"
    assert agent.session["face_report"] is not None


def test_mock_agent_recommend_after_face():
    """分析脸型后 → 描述风格 → 返回推荐"""
    agent = MockAgent()
    # Step 1: 分析脸型
    agent.process("", image_path="test.jpg")
    assert agent.session["stage"] == "face_analyzed"

    # Step 2: 风格推荐
    resp = agent.process("我想要干练通勤风")
    assert "推荐" in resp
    assert agent.session["stage"] == "recommended"
    assert len(agent.session["recommendations"]) == 3


def test_mock_agent_generate_report():
    """推荐后 → 生成报告"""
    agent = MockAgent()
    agent.process("", image_path="test.jpg")
    agent.process("我想要甜美风")

    resp = agent.process("生成报告")
    assert "脸型分析" in resp
    assert "风格象限" in resp


def test_mock_agent_style_guidance():
    """有脸型后 → 用户没说风格 → 引导选择"""
    agent = MockAgent()
    agent.process("", image_path="test.jpg")
    resp = agent.process("嗯，知道了")
    assert "风格偏好" in resp.lower() or "干练" in resp or "甜美" in resp


# ============================================================
# 运行测试
# ============================================================
if __name__ == "__main__":
    tests = [
        ("脸型识别工具", test_detect_face_shape),
        ("发型搜索工具", test_search_hairstyles),
        ("推荐工具", test_recommend),
        ("报告生成工具", test_generate_report),
        ("Mock Agent 欢迎", test_mock_agent_welcome),
        ("Mock Agent 脸型分析", test_mock_agent_face_analysis),
        ("Mock Agent 推荐流程", test_mock_agent_recommend_after_face),
        ("Mock Agent 报告生成", test_mock_agent_generate_report),
        ("Mock Agent 风格引导", test_mock_agent_style_guidance),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("  agent_app 单元测试")
    print("=" * 60)

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    if failed == 0:
        print(f"  🎉 全部 {passed}/{passed} 通过！")
    else:
        print(f"  ⚠️  {passed}/{passed+failed} 通过，{failed} 失败")
    print(f"{'='*60}")
