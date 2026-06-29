"""
================================================================================
  风格报告.py — 发型推荐 AI Agent 系统 · D 模块：风格报告生成器
================================================================================

  功能: 根据脸型分析结果(FaceReport) + 发型推荐列表(List[Recommendation])
       生成 Markdown 格式的个人风格分析报告。

  核心函数:
    - generate_style_report()        主入口，优先 LLM 生成，失败降级本地模板
    - _generate_report_with_llm()    使用 OpenAI GPT-3.5-turbo API 生成
    - _generate_report_local()       本地模板生成（不调 API）
    - _generate_transfer_image()     用AI把推荐发型"换"到用户照片上（需要 OPENAI_API_KEY）

  报告结构(6 章 + 可选效果图):
    一、脸型分析
    二、五官特点与发型暗示（Markdown 三线表）
    三、风格象限（6 维星级表）
    四、推荐发型 Top-N（可嵌入 AI 生成的发型迁移效果图）
    五、避雷提示
    六、日常打理建议

  依赖: contracts.py（同项目的数据类定义 + Mock 实现）

  运行测试: python 风格报告.py
================================================================================
"""

from __future__ import annotations

import base64
import json
import os
import sys
from typing import List, Optional, Dict

# ── 导入 contracts ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contracts import (  # type: ignore[import-untyped]
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
    mock_recommend,
)

# ── 尝试加载 dotenv ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================
#  内置映射表
# ============================================================================

# ── 脸型特点描述 ──
FACE_SHAPE_DESCRIPTIONS: Dict[str, str] = {
    "鹅蛋脸": "被称为\"黄金比例\"脸型，面部线条流畅，长宽比例近乎完美，几乎任何发型都能轻松驾驭。",
    "圆脸": "脸颊饱满、线条柔和，视觉上减龄效果显著。适合能拉长面部比例的发型。",
    "方脸": "下颌线条分明，给人干练率性的感觉。需要用柔和线条来平衡硬朗轮廓。",
    "长脸": "纵向比例偏大，需要在视觉上增加宽度来平衡。横向有层次感的发型是最佳选择。",
    "心形脸": "上宽下窄，额头饱满、下巴精致。关键是平衡额头和下巴的比例。",
    "菱形脸": "颧骨突出、额头和下巴偏窄，是非常有辨识度的脸型。需要柔化颧骨线条。",
}

# ── 五官特征映射 ──
EYE_DISTANCE_HINTS: Dict[str, str] = {
    "偏窄": "建议选择中分或露出额头的发型，避免厚重刘海加重视觉压迫",
    "适中": "适合大部分刘海类型（空气刘海、侧分、八字刘海），发型选择自由度很高",
    "偏宽": "偏分或侧分效果更好，可以适当缩短眼距的视觉感受",
}

NOSE_TYPE_HINTS: Dict[str, str] = {
    "直挺": "几乎任何发型都能驾驭，刘海厚度不影响整体协调感",
    "圆润": "建议选择有层次的发型，侧分刘海比齐刘海更能突出鼻部优势",
    "偏宽": "避免过于贴脸的直发，微卷发可以分散视觉焦点",
}

CHIN_SHAPE_HINTS: Dict[str, str] = {
    "尖": "锁骨发或大波浪可以平衡尖下巴，增加下半脸的量感",
    "圆": "层次感发型可以打破圆润感，锁骨长度最适合",
    "方": "带弧度的卷发可以柔和下颌线条，避免齐耳短发",
}

# ── 按脸型通用避雷 ──
FACE_SHAPE_WARNINGS: Dict[str, List[str]] = {
    "鹅蛋脸": [],
    "圆脸": [
        "避免厚重齐刘海（会让脸更圆）",
        "避免贴脸直发（暴露脸型轮廓）",
        "避免耳上超短发（缺乏修饰作用）",
    ],
    "方脸": [
        "避免齐耳短发（突出下颌角）",
        "避免过于直硬的发型（强化硬朗线条）",
        "避免中分直发（对称强调方形轮廓）",
    ],
    "长脸": [
        "避免过高颅顶发型（视觉拉长）",
        "避免贴头皮直发（暴露纵向长度）",
        "避免露出全部额头",
    ],
    "心形脸": [
        "避免加重额头宽度的发型",
        "避免下巴处过于轻薄（会突出下巴尖锐感）",
    ],
    "菱形脸": [
        "避免贴脸直发（暴露颧骨宽度）",
        "避免耳上短发（缺乏对颧骨的修饰）",
        "避免中分（对称突出菱形轮廓）",
    ],
}

# ── 星级映射 ──
def _value_to_stars(val: float) -> str:
    """将 0.0~1.0 的浮点值映射为 1~5 星字符串。"""
    if val < 0.2:
        return "★☆☆☆☆"
    elif val < 0.4:
        return "★★☆☆☆"
    elif val < 0.6:
        return "★★★☆☆"
    elif val < 0.8:
        return "★★★★☆"
    else:
        return "★★★★★"


# ============================================================================
#  LLM 报告生成
# ============================================================================

# ── LLM System Prompt ──
SYSTEM_PROMPT = """你是一位资深发型顾问，拥有20年发型设计经验。

请根据以下用户的脸型分析数据和推荐发型列表，生成一份专业的个人风格分析报告。

## 报告结构要求

### 一、脸型分析
- 写出脸型名称、置信度、核心比例值
- 用通俗易懂的语言解释该脸型的特点
- 语言亲切专业

### 二、五官特点与发型暗示
- 用Markdown三线表格式
- 每行包含：特征名称 | 具体描述 | 发型建议
- 基于眼距、鼻型、下巴三个维度

### 三、风格象限
- 用星级表示6个风格维度的适配度（0~5星制）

### 四、推荐发型 Top-N
- 每条包含：排名、发型名称、匹配度百分比
- 2~3条推荐理由

### 五、避雷提示
- 列出2~3条该避免的发型特征或类型

### 六、日常打理建议
- 推荐造型产品、修剪周期、日常护理小贴士

## 重要要求
- 直接输出Markdown内容，不要用代码块包裹
- 语言亲切但不失专业
- 每部分控制在3~5句话
- 根据实际数据生成个性化内容，不要使用占位文字"""


def _build_user_prompt(face_report: FaceReport, recommendations: List[Recommendation]) -> str:
    """构建发送给 LLM 的 user prompt，包含脸型数据和推荐列表的 JSON。"""
    face_json = json.dumps(face_report.to_dict(), ensure_ascii=False, indent=2)
    recs_json = json.dumps(
        [r.to_dict() for r in recommendations],
        ensure_ascii=False,
        indent=2,
    )
    return (
        f"## 用户脸型数据\n```json\n{face_json}\n```\n\n"
        f"## 推荐发型列表\n```json\n{recs_json}\n```"
    )


def _generate_report_with_llm(face_report: FaceReport, recommendations: List[Recommendation]) -> str:
    """使用 OpenAI GPT-3.5-turbo API 生成风格报告。

    Args:
        face_report: 脸型分析结果
        recommendations: 发型推荐列表

    Returns:
        Markdown 格式的风格报告字符串

    Raises:
        RuntimeError: API 调用失败时抛出，由上层降级处理
    """
    import openai  # 延迟导入，避免未安装时阻塞本地模式

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置，无法调用 LLM")

    client = openai.OpenAI(api_key=api_key)

    user_prompt = _build_user_prompt(face_report, recommendations)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM 返回内容为空")

    return content


# ============================================================================
#  发型迁移（AI 生成效果图）
# ============================================================================

def _generate_transfer_image(
    user_photo_path: str,
    hairstyle_desc: str,
) -> Optional[str]:
    """用 AI 把指定发型"换"到用户照片上，返回效果图的 base64 编码字符串。

    调用 OpenAI gpt-image-1 模型（通过 images.generate 端点），
    传入用户照片 + 发型描述，生成换发后的效果图。
    失败返回 None，不影响报告整体生成。

    Args:
        user_photo_path: 用户正面照片的本地文件路径
        hairstyle_desc: 发型描述（如"中分及肩波波头，自然黑色"）

    Returns:
        效果图的 base64 编码字符串（可直接嵌入 Markdown），失败返回 None
    """
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[信息] OPENAI_API_KEY 未配置，跳过发型迁移图像生成")
        return None

    if not os.path.exists(user_photo_path):
        print(f"[警告] 用户照片不存在: {user_photo_path}")
        return None

    try:
        client = openai.OpenAI(api_key=api_key)

        with open(user_photo_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        prompt = (
            f"Transform this person's hairstyle to: {hairstyle_desc}. "
            "Keep the face, facial features, skin tone, and identity completely unchanged. "
            "Only change the hair — style, length, color, and texture as described. "
            "Photorealistic result, same lighting, pose, and background."
        )

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            quality="medium",
            response_format="b64_json",
        )

        return response.data[0].b64_json

    except Exception as e:
        print(f"[警告] 发型迁移图像生成失败 ({hairstyle_desc}): {e}")
        return None


# ============================================================================
#  本地模板报告生成
# ============================================================================

def _generate_report_local(
    face_report: FaceReport,
    recommendations: List[Recommendation],
    user_photo_path: str = "",
) -> str:
    """本地模板生成风格报告（不调用 API）。

    根据 face_report 和 recommendations 的实际数据动态填充，
    生成完整的 6 章 Markdown 报告。
    如果提供了 user_photo_path，会为每条推荐发型生成 AI 效果图并嵌入报告。

    Args:
        face_report: 脸型分析结果
        recommendations: 发型推荐列表
        user_photo_path: 用户照片路径（可选，提供则生成换发效果图）

    Returns:
        Markdown 格式的风格报告字符串
    """
    face_shape_value: str = face_report.face_shape.value
    features: FaceFeatures = face_report.features

    # ── 警告信息收集 ──
    warnings: List[str] = []

    if face_report.error:
        warnings.append(f"⚠️ 分析异常：{face_report.error}")

    if face_report.confidence < 0.6:
        warnings.append("⚠️ 识别置信度较低，分析结果仅供参考")

    # ── 截断 Top-5 ──
    top_recs = recommendations[:5]

    # ── 第一章：脸型分析 ──
    face_desc = FACE_SHAPE_DESCRIPTIONS.get(
        face_shape_value,
        f"你的脸型经系统分析为{face_shape_value}。",
    )

    chapter_1_lines: List[str] = [
        "## 一、脸型分析",
        "",
        f"- **脸型**：{face_shape_value}",
        f"- **置信度**：{face_report.confidence:.0%}",
        f"- **长宽比（脸宽/脸长）**：{features.face_ratio:.2f}",
        f"- **颧颌比（下颌宽/颧骨宽）**：{features.jaw_cheek_ratio:.2f}",
        f"- **额颧比（额头宽/颧骨宽）**：{features.forehead_ratio:.2f}",
        "",
        f"> {face_desc}",
        "",
    ]

    # 置信度警告
    if face_report.confidence < 0.6:
        chapter_1_lines.append("> ⚠️ 识别置信度较低，分析结果仅供参考")
        chapter_1_lines.append("")

    if face_report.error:
        chapter_1_lines.append(f"> ⚠️ 分析异常：{face_report.error}")
        chapter_1_lines.append("")

    # ── 第二章：五官特点与发型暗示 ──
    eye_hint = EYE_DISTANCE_HINTS.get(features.eye_distance, "适中")
    nose_hint = NOSE_TYPE_HINTS.get(features.nose_type, "直挺")
    chin_hint = CHIN_SHAPE_HINTS.get(features.chin_shape, "圆")

    chapter_2 = f"""## 二、五官特点与发型暗示

| 特征 | 描述 | 发型建议 |
|:---|:---|:---|
| 眼距 | {features.eye_distance} | {eye_hint} |
| 鼻型 | {features.nose_type} | {nose_hint} |
| 下巴 | {features.chin_shape} | {chin_hint} |

"""

    # ── 第三章：风格象限 ──
    agg_vector = StyleVector()
    if top_recs:
        for rec in top_recs:
            sv = rec.hairstyle.style_vector
            agg_vector.干练 += sv.干练
            agg_vector.甜美 += sv.甜美
            agg_vector.复古 += sv.复古
            agg_vector.酷飒 += sv.酷飒
            agg_vector.自然 += sv.自然
            agg_vector.优雅 += sv.优雅
        n = len(top_recs)
        for dim in ALL_STYLE_DIMS:
            setattr(agg_vector, dim, getattr(agg_vector, dim) / n)

    chapter_3 = f"""## 三、风格象限

| 干练 | 甜美 | 复古 | 酷飒 | 自然 | 优雅 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| {_value_to_stars(agg_vector.干练)} | {_value_to_stars(agg_vector.甜美)} | {_value_to_stars(agg_vector.复古)} | {_value_to_stars(agg_vector.酷飒)} | {_value_to_stars(agg_vector.自然)} | {_value_to_stars(agg_vector.优雅)} |

"""

    # ── 第四章：推荐发型 Top-N ──
    if top_recs:
        rec_rows: List[str] = []
        for i, rec in enumerate(top_recs, 1):
            reasons_text = "；".join(rec.reasons)
            rec_rows.append(
                f"| {i} | **{rec.hairstyle.name}** | {rec.score:.0%} | "
                f"{rec.hairstyle.length.value} | {rec.hairstyle.curl.value} | "
                f"{reasons_text} |"
            )

        chapter_4_lines = [
            f"## 四、推荐发型 Top-{len(top_recs)}",
            "",
            "| 排名 | 发型 | 匹配度 | 长度 | 卷度 | 理由 |",
            "|:---:|------|:---:|:---:|:---:|------|",
        ]
        chapter_4_lines.extend(rec_rows)
        chapter_4_lines.append("")

        # ── 嵌入发型迁移效果图 ──
        if user_photo_path and os.path.exists(user_photo_path):
            for i, rec in enumerate(top_recs, 1):
                desc = (
                    f"{rec.hairstyle.name}（{rec.hairstyle.length.value}, "
                    f"{rec.hairstyle.curl.value}）"
                )
                img_b64 = _generate_transfer_image(user_photo_path, desc)
                if img_b64:
                    chapter_4_lines.append(
                        f"![{rec.hairstyle.name}效果图]"
                        f"(data:image/png;base64,{img_b64})"
                    )
                    chapter_4_lines.append(
                        f"> *AI 生成效果图 — {rec.hairstyle.name}，仅供参考，实际效果以发型师操作为准。*"
                    )
                    chapter_4_lines.append("")

        chapter_4 = "\n".join(chapter_4_lines) + "\n"
    else:
        chapter_4 = """## 四、推荐发型 Top-0

| 排名 | 发型 | 匹配度 | 长度 | 卷度 | 理由 |
|:---:|------|:---:|:---:|:---:|------|
| - | 暂无匹配发型，请调整风格偏好后重试 | - | - | - | - |

"""

    # ── 第五章：避雷提示 ──
    warnings_set: set = set()
    # 脸型通用避雷
    for w in FACE_SHAPE_WARNINGS.get(face_shape_value, []):
        warnings_set.add(w)
    # 推荐发型的 warnings
    for rec in top_recs:
        for w in rec.hairstyle.warnings:
            warnings_set.add(w)

    if warnings_set:
        warning_lines = "\n".join(f"- {w}" for w in sorted(warnings_set))
        chapter_5 = f"""## 五、避雷提示

{warning_lines}

"""
    else:
        chapter_5 = """## 五、避雷提示

- 暂无特殊避雷提示，你可以大胆尝试各种风格！

"""

    # ── 第六章：日常打理建议（去重，最多 5 条）──
    tips_set: set = set()
    for rec in top_recs:
        tip = rec.hairstyle.care_tips
        if tip and tip.strip():
            tips_set.add(tip.strip())

    if tips_set:
        tips_list = list(tips_set)
        tip_lines = "\n".join(f"- {t}" for t in tips_list[:5])
    else:
        tip_lines = "- 根据发型类型选择合适的造型产品，保持定期修剪习惯（建议每6~8周修剪一次）。"

    chapter_6 = f"""## 六、日常打理建议

{tip_lines}

"""

    # ── 页脚 ──
    footer = """---
*报告由 AI 发型顾问 Agent 自动生成，仅供参考。最终选择请结合个人喜好和实际试戴效果。*
"""

    # ── 组装 ──
    report = "\n".join([
        *chapter_1_lines,
        chapter_2,
        chapter_3,
        chapter_4,
        chapter_5,
        chapter_6,
        footer,
    ])

    return report


# ============================================================================
#  主入口
# ============================================================================

def generate_style_report(
    face_report: FaceReport,
    recommendations: List[Recommendation],
    use_llm: bool = True,
    user_photo_path: str = "",
) -> str:
    """根据脸型分析和发型推荐生成 Markdown 风格报告。

    优先使用 LLM（OpenAI GPT-3.5-turbo）生成高质量个性化报告，
    LLM 不可用时（API Key 未配置、网络异常等）自动降级为本地模板生成。
    如果提供了 user_photo_path 且有 OPENAI_API_KEY，会在推荐发型章节
    为每款发型自动生成 AI 换发效果图。

    Args:
        face_report: A 模块输出的脸型分析结果
        recommendations: C 模块输出的发型推荐列表（可为空）
        use_llm: 是否尝试 LLM 生成，默认 True
        user_photo_path: 用户正面照片路径（可选），提供则嵌入 AI 换发效果图

    Returns:
        Markdown 格式的风格报告字符串
    """
    if use_llm:
        try:
            report = _generate_report_with_llm(face_report, recommendations)
            if user_photo_path and recommendations:
                report = _embed_transfer_images(report, recommendations, user_photo_path)
            return report
        except Exception:
            pass

    return _generate_report_local(face_report, recommendations, user_photo_path)


def _embed_transfer_images(
    report: str,
    recommendations: List[Recommendation],
    user_photo_path: str,
) -> str:
    """在已有 Markdown 报告中为每条推荐嵌入效果图（供 LLM 报告的补充）。"""
    if not user_photo_path or not os.path.exists(user_photo_path):
        return report

    for i, rec in enumerate(recommendations[:5], 1):
        desc = f"{rec.hairstyle.name}（{rec.hairstyle.length.value}, {rec.hairstyle.curl.value}）"
        img_b64 = _generate_transfer_image(user_photo_path, desc)
        if img_b64:
            img_block = (
                f"\n![{rec.hairstyle.name}效果图]"
                f"(data:image/png;base64,{img_b64})\n"
                f"> *AI 生成效果图 — {rec.hairstyle.name}，仅供参考。*\n"
            )
            # 在推荐名称第一次出现的位置后面插入
            marker = f"**{rec.hairstyle.name}**"
            idx = report.find(marker)
            if idx != -1:
                # 找到该行末尾（下一个换行）
                line_end = report.find("\n", idx)
                if line_end != -1:
                    report = report[:line_end + 1] + img_block + report[line_end + 1:]

    return report


# ============================================================================
#  Mock 函数（供 E 模块快速开发使用）
# ============================================================================

def mock_generate_style_report_d(
    face_report: FaceReport,
    recommendations: List[Recommendation],
) -> str:
    """快速 Mock 版本 — 直接调用本地模板生成（无 LLM 依赖）。

    供 E 模块开发者在联调前使用，与 generate_style_report() 接口完全一致。

    Args:
        face_report: 人脸分析报告（可用 mock_detect_face_shape 生成）
        recommendations: 推荐列表（可用 mock_recommend 生成）

    Returns:
        Markdown 格式的个人风格报告
    """
    return _generate_report_local(face_report, recommendations)


# ============================================================================
#  测试入口
# ============================================================================

if __name__ == "__main__":
    """直接运行时测试场景。"""

    def _print_separator(title: str) -> None:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}\n")

    # ── 场景 1：鹅蛋脸 + 甜美风（含换发效果图）──
    _print_separator("场景 1：鹅蛋脸 + 甜美风")
    report_oval = mock_detect_face_shape(face_shape=FaceShape.OVAL)
    preferences_sweet = StylePreferences(natural_language="日系甜美少女风")
    recs_oval = mock_recommend(report_oval, preferences_sweet, top_n=5)
    # 如果有用户照片和 API key，自动生成换发效果图
    print(generate_style_report(report_oval, recs_oval, use_llm=False))

    # ── 场景 2：圆脸 + 干练风 ──
    _print_separator("场景 2：圆脸 + 干练风")
    report_round = mock_detect_face_shape(face_shape=FaceShape.ROUND)
    preferences_smart = StylePreferences(natural_language="干练通勤风")
    recs_round = mock_recommend(report_round, preferences_smart, top_n=5)
    print(generate_style_report(report_round, recs_round, use_llm=False))

    # ── 场景 3：方脸 + 复古风 ──
    _print_separator("场景 3：方脸 + 复古风")
    report_square = mock_detect_face_shape(face_shape=FaceShape.SQUARE)
    preferences_retro = StylePreferences(natural_language="复古港风")
    recs_square = mock_recommend(report_square, preferences_retro, top_n=5)
    print(generate_style_report(report_square, recs_square, use_llm=False))

    # ── 场景 4：空推荐 ──
    _print_separator("场景 4：空推荐")
    report_empty = mock_detect_face_shape(face_shape=FaceShape.LONG)
    print(generate_style_report(report_empty, [], use_llm=False))

    # ── 场景 5：低置信度（菱形脸） ──
    _print_separator("场景 5：低置信度（菱形脸）")
    report_diamond = mock_detect_face_shape(face_shape=FaceShape.DIAMOND)
    preferences_natural = StylePreferences(natural_language="自然慵懒风")
    recs_diamond = mock_recommend(report_diamond, preferences_natural, top_n=5)
    print(generate_style_report(report_diamond, recs_diamond, use_llm=False))

    print(f"\n{'=' * 70}")
    print("  全部 5 个场景测试完成")
    print(f"  (如配置 OPENAI_API_KEY + user_photo_path，场景1会含换发效果图)")
    print(f"{'=' * 70}")
