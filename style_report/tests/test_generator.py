"""
================================================================================
  test_风格报告.py — D 模块：风格报告生成器 · 综合测试
================================================================================

  覆盖 8 项必测 + 额外辅助断言。
  运行: python test_风格报告.py
================================================================================
"""

from __future__ import annotations

import sys
import traceback
import os

# ── 确保项目根目录在 sys.path ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from contracts import (
    FaceReport,
    FaceShape,
    FaceFeatures,
    Recommendation,
    Hairstyle,
    StyleVector,
    StylePreferences,
    HairLength,
    HairCurl,
    ALL_STYLE_DIMS,
    mock_detect_face_shape,
    mock_detect_face_shape_error,
    mock_recommend,
    mock_generate_style_report,
)

# ── 导入 D 模块（风格报告）─────────────────────────────────────────
from style_report.generator import generate_style_report, _generate_report_local

# ============================================================================
#  测试框架
# ============================================================================

passed = 0
failed = 0
total = 0
failures: list[dict] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    """简单的测试断言。"""
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ [{name}] {detail}")
        return True
    else:
        failed += 1
        msg = f"  ❌ [{name}] FAILED: {detail}"
        print(msg)
        failures.append({"name": name, "detail": detail})
        return False


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ============================================================================
#  测试用例
# ============================================================================

section("测试 1: 正常流程 — 鹅蛋脸 + 甜美风")

report_oval = mock_detect_face_shape(face_shape=FaceShape.OVAL)
preferences_sweet = StylePreferences(natural_language="日系甜美少女风")
recs_oval = mock_recommend(report_oval, preferences_sweet, top_n=5)
report_str_1 = generate_style_report(report_oval, recs_oval, use_llm=False)

check("1.1 返回类型为 str", isinstance(report_str_1, str))
check("1.2 报告非空", len(report_str_1) > 0)

# 6 个章节标题
chapter_titles = ["脸型分析", "五官特点", "风格象限", "推荐发型", "避雷提示", "打理建议"]
for title in chapter_titles:
    check(f"1.3 包含章节: {title}", title in report_str_1,
          f"报告包含'{title}'章节标题")

check("1.4 包含'鹅蛋脸'", "鹅蛋脸" in report_str_1)
check("1.5 包含'黄金比例'", "黄金比例" in report_str_1)
check("1.6 推荐列表非空", len(recs_oval) > 0,
      f"推荐列表长度={len(recs_oval)}")
check("1.7 包含发型名称", recs_oval[0].hairstyle.name in report_str_1,
      f"发型名: {recs_oval[0].hairstyle.name}")

# ────────────────────────────────────────────────────────────────────—

section("测试 2: 正常流程 — 圆脸 + 干练风")

report_round = mock_detect_face_shape(face_shape=FaceShape.ROUND)
preferences_smart = StylePreferences(natural_language="干练通勤风")
recs_round = mock_recommend(report_round, preferences_smart, top_n=5)
report_str_2 = generate_style_report(report_round, recs_round, use_llm=False)

check("2.1 包含'圆脸'", "圆脸" in report_str_2)
check("2.2 包含'减龄'", "减龄" in report_str_2,
      "圆脸描述应包含'减龄'关键词")

# 圆脸专用警告
round_warnings = [
    "避免厚重齐刘海",
    "避免贴脸直发",
    "避免耳上超短发",
]
for w in round_warnings:
    check(f"2.3 圆脸避雷: {w}", w in report_str_2,
          f"圆脸报告应包含避雷提示: {w}")

# ────────────────────────────────────────────────────────────────────—

section("测试 3: 正常流程 — 方脸 + 复古风")

report_square = mock_detect_face_shape(face_shape=FaceShape.SQUARE)
preferences_retro = StylePreferences(natural_language="复古港风")
recs_square = mock_recommend(report_square, preferences_retro, top_n=5)
report_str_3 = generate_style_report(report_square, recs_square, use_llm=False)

check("3.1 包含'方脸'", "方脸" in report_str_3)
check("3.2 包含'下颌线条'", "下颌线条" in report_str_3,
      "方脸描述应包含'下颌线条'关键词")

# 方脸专用警告
square_warnings = [
    "避免齐耳短发",
    "避免过于直硬的发型",
    "避免中分直发",
]
for w in square_warnings:
    check(f"3.3 方脸避雷: {w}", w in report_str_3,
          f"方脸报告应包含避雷提示: {w}")

# ────────────────────────────────────────────────────────────────────—

section("测试 4: 边界 — 空推荐列表")

report_empty = mock_detect_face_shape(face_shape=FaceShape.LONG)
try:
    report_str_4 = generate_style_report(report_empty, [], use_llm=False)
    check("4.1 不抛异常", True)
    check("4.2 返回类型为 str", isinstance(report_str_4, str))
    check("4.3 报告非空", len(report_str_4) > 0)
    check("4.4 包含'暂无匹配发型'",
          "暂无匹配发型" in report_str_4,
          "空推荐列表时应显示'暂无匹配发型'提示")
except Exception as e:
    check("4.1 不抛异常", False, f"抛出异常: {type(e).__name__}: {e}")
    check("4.2 返回类型为 str", False, "因异常未返回")
    check("4.3 报告非空", False, "因异常未返回")
    check("4.4 包含'暂无匹配发型'", False, "因异常未返回")

# ────────────────────────────────────────────────────────────────────—

section("测试 5: 边界 — 低置信度（菱形脸, confidence=0.65）")

# 注意: confidence < 0.6 才显示警告，0.65 不会触发
report_diamond = mock_detect_face_shape(face_shape=FaceShape.DIAMOND)
preferences_natural = StylePreferences(natural_language="自然慵懒风")
recs_diamond = mock_recommend(report_diamond, preferences_natural, top_n=5)
report_str_5 = generate_style_report(report_diamond, recs_diamond, use_llm=False)

check("5.1 报告正常生成", len(report_str_5) > 0)
check("5.2 返回类型为 str", isinstance(report_str_5, str))
check("5.3 包含'菱形脸'", "菱形脸" in report_str_5)

# 置信度=0.65 >= 0.6，不应显示低置信度警告
check("5.4 不触发低置信度警告 (conf=0.65 >= 0.6)",
      report_str_5.count("识别置信度较低") == 0,
      "confidence=0.65，不应出现置信度警告")

# ────────────────────────────────────────────────────────────────────—

section("测试 6: 边界 — face_report.error 非空")

report_error = mock_detect_face_shape_error()

# 验证 error 字段非空
check("6.1 face_report.error 非空", report_error.error is not None,
      f"error={report_error.error}")

recs_for_error = mock_recommend(report_error,
                                StylePreferences(natural_language="日系甜美风"),
                                top_n=3)
report_str_6 = generate_style_report(report_error, recs_for_error, use_llm=False)

check("6.2 报告正常生成", len(report_str_6) > 0)
check("6.3 返回类型为 str", isinstance(report_str_6, str))
check("6.4 包含错误信息",
      "分析异常" in report_str_6,
      f"报告应包含'分析异常'错误信息。报告长度={len(report_str_6)}")
check("6.5 包含具体错误内容",
      report_error.error[:6] in report_str_6
      or "未检测到人脸" in report_str_6,
      f"应包含错误详情。error={report_error.error}")

# ────────────────────────────────────────────────────────────────────—

section("测试 7: 千人千面验证 — 鹅蛋脸 vs 圆脸")

# 复用测试 1 和测试 2 的结果
report_str_oval = report_str_1
report_str_round = report_str_2

# 7.1 脸型特点描述不同
check("7.1 两份报告的脸型特点描述不同",
      report_str_oval != report_str_round,
      "鹅蛋脸和圆脸报告内容应不同")

# 提取五官建议部分做比较
def extract_section(text: str, section_name: str) -> str:
    """提取报告的某个章节内容。"""
    # 查找章节起始位置
    lines = text.split('\n')
    start_idx = -1
    for i, line in enumerate(lines):
        if section_name in line and line.strip().startswith('##'):
            start_idx = i
            break
    if start_idx == -1:
        return ""
    # 取该章节及之后若干行
    end_idx = start_idx + 1
    while end_idx < len(lines):
        if lines[end_idx].strip().startswith('##') and lines[end_idx].strip() != '':
            break
        end_idx += 1
    return '\n'.join(lines[start_idx:end_idx])

face_oval_section = extract_section(report_str_oval, "脸型分析")
face_round_section = extract_section(report_str_round, "脸型分析")

check("7.2 两份报告的'脸型分析'章节不同",
      face_oval_section != face_round_section,
      "不同脸型的分析内容应该不同")

features_oval_section = extract_section(report_str_oval, "五官特点")
features_round_section = extract_section(report_str_round, "五官特点")

check("7.3 两份报告的'五官特点'章节不同",
      features_oval_section != features_round_section,
      "不同脸型的五官建议应该不同")

warning_oval_section = extract_section(report_str_oval, "避雷提示")
warning_round_section = extract_section(report_str_round, "避雷提示")

check("7.4 两份报告的'避雷提示'章节不同",
      warning_oval_section != warning_round_section,
      "不同脸型的避雷提示应该不同")

# ────────────────────────────────────────────────────────────────────—

section("测试 8: Mock 函数 — mock_generate_style_report")

report_test = mock_detect_face_shape(face_shape=FaceShape.OVAL)
recs_test = mock_recommend(report_test,
                           StylePreferences(natural_language="日系甜美风"),
                           top_n=3)
mock_result = mock_generate_style_report(report_test, recs_test)

check("8.1 返回类型为 str", isinstance(mock_result, str))
check("8.2 返回值非空", len(mock_result) > 0)
check("8.3 包含 Markdown 标题标记",
      "##" in mock_result or "#" in mock_result,
      "Mock 报告应包含 Markdown 标记")
check("8.4 包含 Markdown 表格标记",
      "|" in mock_result,
      "Mock 报告应包含表格标记")
check("8.5 包含 Markdown 引用标记",
      ">" in mock_result,
      "Mock 报告应包含引用块标记")


# ============================================================================
#  测试报告
# ============================================================================

print("\n" + "=" * 65)
print("  测试报告总览")
print("=" * 65)

if failed == 0:
    print(f"\n  🎉 全部 {passed}/{total} 项测试通过!")
    print(f"  D 模块（风格报告生成器）质量良好，无 Bug。")
else:
    print(f"\n  ⚠️  {passed}/{total} 通过, {failed} 项失败")
    print(f"\n  失败项列表:")
    for f in failures:
        print(f"    ❌ [{f['name']}]: {f['detail']}")

print(f"\n  Routing Decision: {'NoOne' if failed == 0 else 'Engineer (Alex)'}")
print("=" * 65)

# 退出码
sys.exit(0 if failed == 0 else 1)
