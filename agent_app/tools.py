"""
tools.py — LangChain 工具包装层

把 A/B/C/D 四个模块的函数包装成 Agent 可调用的 @tool。
每个 tool 遵循三个原则：
1. 输入是字符串（Agent 只能传 str）
2. 输出是字符串（Agent 只能读 str）
3. docstring 是 Agent 的"使用说明书"——它靠这个决定调哪个 tool

动态导入策略：
- 优先导入真实模块（各团队交付后自动生效）
- 真实模块不可用时，fallback 到 contracts_mock 的 Mock 实现
"""

import json
import sys
import os
import importlib.util

# ── 项目根目录 ──────────────────────────────────────────────────
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

# 导入数据类（纯合约无依赖）
from contracts import (
    FaceShape, HairLength, HairCurl,
    FaceReport, StyleVector, StylePreferences,
)


# ============================================================
#  动态导入模块函数（真实 > Mock fallback）
# ============================================================

# 环境变量 FORCE_MOCK=1 强制使用所有 Mock 函数（测试/演示场景）
_FORCE_MOCK = os.environ.get("HAIR_FORCE_MOCK", "").strip() == "1"

def _try_import(module_path: str, *names: str):
    """
    尝试从 module_path 导入 names。
    如果 FORCE_MOCK 或导入失败 → 返回 (False, None)
    成功 → 返回 (True, tuple of objects)
    """
    if _FORCE_MOCK:
        return (False, None)
    try:
        mod = importlib.import_module(module_path)
        results = tuple(getattr(mod, n) for n in names)
        return (True, results)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return (False, None)


# ── A 模块：脸型识别 ──
_a_ok, _a = _try_import("face_detect.detector", "detect_face_shape")
if _a_ok:
    (detect_face_shape_real,) = _a
else:
    detect_face_shape_real = None

# ── B 模块：发型搜索 ──
_b_real_ok, _b = _try_import("hairstyle_db.db", "search_hairstyles", "get_hairstyle_by_id")
if _b_real_ok:
    (search_hairstyles_real, get_hairstyle_by_id_real) = _b
else:
    search_hairstyles_real = None
    get_hairstyle_by_id_real = None

# ── C 模块：推荐引擎 ──
_c_ok, _c = _try_import("recommend_engine.engine", "recommend")
(_recommend_real,) = _c if _c_ok else (None,)
_c2_ok, _c2 = _try_import("recommend_engine.reverse_infer", "infer_style_vector")
(_infer_style_vector_real,) = _c2 if _c2_ok else (None,)

# ── D 模块：风格报告 ──
_d_ok, _d = _try_import("style_report.generator", "generate_style_report")
(_generate_style_report_real,) = _d if _d_ok else (None,)

# ── Mock fallback ──
from contracts_mock import (
    mock_detect_face_shape,
    mock_search_hairstyles,
    mock_get_hairstyle_by_id,
    mock_recommend,
    mock_infer_style_vector,
    mock_generate_style_report,
)


# ============================================================
#  实际使用的函数（真实优先，Mock 兜底）
# ============================================================

_use_detect_face_shape = detect_face_shape_real or mock_detect_face_shape
_use_search_hairstyles = search_hairstyles_real or mock_search_hairstyles
_use_get_hairstyle_by_id = get_hairstyle_by_id_real or mock_get_hairstyle_by_id
_use_recommend = _recommend_real or mock_recommend
_use_infer_style_vector = _infer_style_vector_real or mock_infer_style_vector
_use_generate_style_report = _generate_style_report_real or mock_generate_style_report

# 模块可用性报告（调试用）
_MODULE_STATUS = {
    "A (face_detect)": detect_face_shape_real is not None,
    "B (hairstyle_db)": search_hairstyles_real is not None,
    "C (recommend_engine)": _recommend_real is not None,
    "D (style_report)": _generate_style_report_real is not None,
}


# ============================================================
# Tool 1: 脸型识别（对应 A 模块）
# ============================================================
def detect_face_shape_tool(image_path: str) -> str:
    """
    分析用户上传的人像照片，识别脸型（鹅蛋脸/圆脸/方脸/长脸/心形脸/菱形脸）。

    什么时候用：
    - 用户上传了一张照片
    - 用户问"我是什么脸型"
    - 任何需要知道脸型的场景

    参数 image_path：照片的本地文件路径（如 "uploaded_photos/face.jpg"）

    返回：JSON 字符串，包含 face_shape（脸型中文名）、confidence（置信度 0~1）、
          features（face_ratio/jaw_cheek_ratio/forehead_ratio/eye_distance/nose_type/chin_shape）
    """
    try:
        report = _use_detect_face_shape(image_path)

        if report.error:
            return json.dumps({
                "status": "error",
                "message": report.error,
            }, ensure_ascii=False)

        return json.dumps({
            "status": "ok",
            "face_shape": report.face_shape.value,
            "confidence": report.confidence,
            "features": {
                "face_ratio": report.features.face_ratio,
                "jaw_cheek_ratio": report.features.jaw_cheek_ratio,
                "forehead_ratio": report.features.forehead_ratio,
                "eye_distance": report.features.eye_distance,
                "nose_type": report.features.nose_type,
                "chin_shape": report.features.chin_shape,
            },
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"脸型识别失败：{str(e)}",
        }, ensure_ascii=False)


# ============================================================
# Tool 2: 发型搜索（对应 B 模块）
# ============================================================
def search_hairstyles_tool(query_json: str) -> str:
    """
    从发型数据库中搜索匹配的发型。

    什么时候用：
    - 用户描述了风格偏好（如"甜美日系风"）
    - 用户指定了长度或卷度（如"短发"、"大卷"）
    - 需要给用户展示候选发型列表

    参数 query_json：JSON 字符串，格式如下：
    {
        "face_shape": "鹅蛋脸",       // 可选，适合的脸型
        "style_vector": {            // 可选，风格偏好向量
            "干练": 0.9, "甜美": 0.1, "复古": 0.0,
            "酷飒": 0.0, "自然": 0.3, "优雅": 0.1
        },
        "length": "中发",            // 可选，"短发"/"中发"/"长发"
        "curl": "微卷",              // 可选，"直发"/"微卷"/"大卷"
        "limit": 10                  // 可选，最多返回几条
    }

    返回：JSON 数组，每项包含 id/name/image_url/suitable_shapes/style_vector/
          length/curl/popularity/warnings/care_tips
    """
    try:
        q = json.loads(query_json)

        face_shape = FaceShape(q["face_shape"]) if q.get("face_shape") else None
        length = HairLength(q["length"]) if q.get("length") else None
        curl = HairCurl(q["curl"]) if q.get("curl") else None
        limit = int(q.get("limit", 10))

        style_vector = None
        if q.get("style_vector"):
            style_vector = StyleVector(**q["style_vector"])

        results = _use_search_hairstyles(
            face_shape=face_shape,
            style_vector=style_vector,
            length=length,
            curl=curl,
            limit=limit,
        )

        return json.dumps([h.to_dict() for h in results], ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"发型搜索失败：{str(e)}",
        }, ensure_ascii=False)


# ============================================================
# Tool 3: 发型推荐（对应 C 模块）
# ============================================================
def recommend_tool(params_json: str) -> str:
    """
    根据脸型 + 风格偏好，综合推荐 Top-N 发型。

    什么时候用：
    - 用户提供了脸型 + 风格描述（如"我是鹅蛋脸，想要干练风"）
    - 需要给用户排序后的推荐列表

    参数 params_json：JSON 字符串，格式如下：
    {
        "face_report": {                    // 必填，脸型分析结果
            "face_shape": "鹅蛋脸",
            "confidence": 0.92,
            "features": { ... }
        },
        "preferences": {                    // 选填，用户偏好
            "natural_language": "干练通勤风",  // 自然语言描述
            "preferred_length": "中发",        // 可选
            "preferred_curl": "直发"           // 可选
        },
        "top_n": 5                          // 可选，默认 5
    }

    返回：JSON 数组，每项包含 hairstyle（发型详情）+ score（评分 0~1）+
          reasons（推荐理由列表）+ details（各因子得分）
    """
    try:
        p = json.loads(params_json)

        # 解析 FaceReport
        fr_data = p["face_report"]
        face_report = FaceReport.from_dict(fr_data)

        # 解析 StylePreferences
        pref_data = p.get("preferences", {})
        preferences = StylePreferences()
        if pref_data.get("natural_language"):
            preferences.natural_language = pref_data["natural_language"]
        if pref_data.get("preferred_length"):
            preferences.preferred_length = HairLength(pref_data["preferred_length"])
        if pref_data.get("preferred_curl"):
            preferences.preferred_curl = HairCurl(pref_data["preferred_curl"])

        top_n = int(p.get("top_n", 5))

        results = _use_recommend(face_report, preferences, top_n=top_n)

        return json.dumps([r.to_dict() for r in results], ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"推荐失败：{str(e)}",
        }, ensure_ascii=False)


# ============================================================
# Tool 4: 风格报告生成（对应 D 模块）
# ============================================================
def generate_report_tool(params_json: str) -> str:
    """
    生成完整的个人风格分析报告（Markdown 格式）。

    什么时候用：
    - 用户有脸型分析结果 + 推荐列表，想看到详细分析
    - 用户问"为什么这些发型适合我"
    - 用户想看完整报告

    参数 params_json：JSON 字符串，格式如下：
    {
        "face_report": { ... },         // 必填，脸型分析结果
        "recommendations": [ ... ]      // 必填，推荐列表
    }

    返回：Markdown 字符串（6 章：脸型分析/五官特点/风格象限/推荐发型/避雷/打理）
    """
    try:
        p = json.loads(params_json)

        face_report = FaceReport.from_dict(p["face_report"])

        # 解析推荐列表
        from contracts import Hairstyle, Recommendation
        recommendations = []
        for r_data in p.get("recommendations", []):
            hs_data = r_data["hairstyle"]
            hs = Hairstyle.from_dict(hs_data)
            rec = Recommendation(
                hairstyle=hs,
                score=r_data["score"],
                reasons=r_data["reasons"],
                details=r_data.get("details", {}),
            )
            recommendations.append(rec)

        report_md = _use_generate_style_report(face_report, recommendations)

        return report_md

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"报告生成失败：{str(e)}",
        }, ensure_ascii=False)


# ============================================================
# 工具注册表（供 Agent 初始化使用）
# ============================================================

# LangChain 风格的 tool 列表（如果用 LangChain）
# 每个 tool 的 description 就是上面的 docstring
TOOL_DEFINITIONS = [
    {
        "name": "detect_face_shape",
        "function": detect_face_shape_tool,
        "description": (
            "分析用户上传的人像照片，识别脸型（鹅蛋脸/圆脸/方脸/长脸/心形脸/菱形脸）。"
            "参数 image_path 是照片的本地文件路径。"
            "返回 JSON：face_shape（脸型名）、confidence（置信度）、features（面部比例）。"
            "当用户上传照片或询问脸型时，必须先调用此工具。"
        ),
    },
    {
        "name": "search_hairstyles",
        "function": search_hairstyles_tool,
        "description": (
            "从发型数据库搜索匹配的发型。"
            "参数 query_json 是 JSON 字符串，可包含 face_shape/style_vector/length/curl/limit。"
            "返回匹配发型的 JSON 数组。"
            "当用户描述风格偏好（如'甜美日系'）或指定长度/卷度时使用。"
        ),
    },
    {
        "name": "recommend",
        "function": recommend_tool,
        "description": (
            "根据脸型 + 风格偏好综合推荐 Top-N 发型。"
            "参数 params_json 包含 face_report（脸型分析JSON）+ preferences（风格偏好）+ top_n。"
            "返回排序后的推荐列表，每项含 score（评分）、reasons（推荐理由）、details（各因子得分）。"
            "当用户提供了脸型信息并想要发型推荐时使用，这是核心推荐工具。"
        ),
    },
    {
        "name": "generate_style_report",
        "function": generate_report_tool,
        "description": (
            "生成完整的个人风格分析报告（Markdown 格式，6 章）。"
            "参数 params_json 包含 face_report + recommendations。"
            "返回 Markdown 字符串，可直接渲染展示。"
            "当用户想看详细分析报告、问'为什么适合'、或需要完整风格建议时使用。"
        ),
    },
]


# ── 辅助函数：方便 Agent 直接调用 ──

def call_tool_by_name(name: str, args: str) -> str:
    """根据工具名调用对应函数"""
    for tool in TOOL_DEFINITIONS:
        if tool["name"] == name:
            return tool["function"](args)
    return json.dumps({
        "status": "error",
        "message": f"未知工具：{name}",
    }, ensure_ascii=False)


if __name__ == "__main__":
    # 快速自测：逐个调用工具
    print("=" * 60)
    print("  tools.py 自检")
    print("=" * 60)

    # Test 1: 脸型识别
    print("\n── Tool 1: detect_face_shape ──")
    result = detect_face_shape_tool("test.jpg")
    data = json.loads(result)
    print(f"  脸型: {data['face_shape']}, 置信度: {data['confidence']}")
    print(f"  ✅ 通过" if data["status"] == "ok" else f"  ❌ 失败")

    # Test 2: 发型搜索
    print("\n── Tool 2: search_hairstyles ──")
    query = json.dumps({"face_shape": "鹅蛋脸", "limit": 3}, ensure_ascii=False)
    result = search_hairstyles_tool(query)
    data = json.loads(result)
    print(f"  返回 {len(data)} 条: {[h['name'] for h in data]}")
    print(f"  ✅ 通过" if len(data) == 3 else f"  ❌ 失败")

    # Test 3: 推荐
    print("\n── Tool 3: recommend ──")
    params = json.dumps({
        "face_report": {
            "face_shape": "鹅蛋脸", "confidence": 0.92,
            "features": {"face_ratio": 1.48, "jaw_cheek_ratio": 0.78, "forehead_ratio": 1.05,
                         "eye_distance": "适中", "nose_type": "直挺", "chin_shape": "圆"},
        },
        "preferences": {"natural_language": "干练通勤风"},
        "top_n": 3,
    }, ensure_ascii=False)
    result = recommend_tool(params)
    data = json.loads(result)
    names_and_scores = [(r["hairstyle"]["name"], r["score"]) for r in data]
    print(f"  返回 {len(data)} 条: {names_and_scores}")
    print(f"  ✅ 通过" if len(data) == 3 else f"  ❌ 失败")

    # Test 4: 报告生成
    print("\n── Tool 4: generate_style_report ──")
    recs_data = json.loads(result)
    params = json.dumps({
        "face_report": {
            "face_shape": "鹅蛋脸", "confidence": 0.92,
            "features": {"face_ratio": 1.48, "jaw_cheek_ratio": 0.78, "forehead_ratio": 1.05,
                         "eye_distance": "适中", "nose_type": "直挺", "chin_shape": "圆"},
        },
        "recommendations": recs_data,
    }, ensure_ascii=False)
    report = generate_report_tool(params)
    contains_all = all(
        s in report for s in ["脸型分析", "五官特点", "风格象限", "推荐发型", "避雷提示", "打理建议"]
    )
    print(f"  报告长度: {len(report)} 字符")
    print(f"  包含 6 章: {contains_all}")
    print(f"  {'✅ 通过' if contains_all else '❌ 失败'}")

    print("\n" + "=" * 60)
    print("  自检完成！")
    print("=" * 60)
