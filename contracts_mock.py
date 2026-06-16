"""
===============================================================================
  contracts_mock.py — Mock 实现层 + 种子数据
===============================================================================

  从 contracts.py 拆分出来，职责单一：
  - 种子数据：Mock 脸型报告、10 款发型数据
  - Mock 函数：所有模块的 mock_* 实现
  - 真实模块已交付时，逐步替换为本文件的实现即可

  依赖：contracts.py（纯数据类 + 枚举）
===============================================================================
"""

from __future__ import annotations

import math
from typing import List, Optional, Dict

from contracts import (
    FaceShape, HairLength, HairCurl, StyleVector,
    FaceReport, FaceFeatures, Hairstyle, StylePreferences,
    Recommendation, ALL_STYLE_DIMS,
)


# ============================================================================
#  A Mock: 6 种脸型报告
# ============================================================================

_MOCK_FACE_REPORTS: Dict[FaceShape, FaceReport] = {
    FaceShape.OVAL: FaceReport(
        face_shape=FaceShape.OVAL, confidence=0.92,
        features=FaceFeatures(face_ratio=1.48, jaw_cheek_ratio=0.78, forehead_ratio=1.05,
                               eye_distance="适中", nose_type="直挺", chin_shape="圆"),
    ),
    FaceShape.ROUND: FaceReport(
        face_shape=FaceShape.ROUND, confidence=0.88,
        features=FaceFeatures(face_ratio=1.05, jaw_cheek_ratio=0.92, forehead_ratio=0.98,
                               eye_distance="偏宽", nose_type="圆润", chin_shape="圆"),
    ),
    FaceShape.SQUARE: FaceReport(
        face_shape=FaceShape.SQUARE, confidence=0.85,
        features=FaceFeatures(face_ratio=1.22, jaw_cheek_ratio=0.95, forehead_ratio=1.02,
                               eye_distance="适中", nose_type="偏宽", chin_shape="方"),
    ),
    FaceShape.LONG: FaceReport(
        face_shape=FaceShape.LONG, confidence=0.86,
        features=FaceFeatures(face_ratio=1.78, jaw_cheek_ratio=0.72, forehead_ratio=0.90,
                               eye_distance="偏窄", nose_type="直挺", chin_shape="尖"),
    ),
    FaceShape.HEART: FaceReport(
        face_shape=FaceShape.HEART, confidence=0.80,
        features=FaceFeatures(face_ratio=1.35, jaw_cheek_ratio=0.62, forehead_ratio=1.12,
                               eye_distance="适中", nose_type="直挺", chin_shape="尖"),
    ),
    FaceShape.DIAMOND: FaceReport(
        face_shape=FaceShape.DIAMOND, confidence=0.65,
        features=FaceFeatures(face_ratio=1.42, jaw_cheek_ratio=0.55, forehead_ratio=0.82,
                               eye_distance="偏窄", nose_type="直挺", chin_shape="尖"),
    ),
}


def mock_detect_face_shape(image_path: str = "", face_shape: FaceShape = FaceShape.OVAL) -> FaceReport:
    return _MOCK_FACE_REPORTS.get(face_shape, _MOCK_FACE_REPORTS[FaceShape.OVAL])


def mock_detect_face_shape_error() -> FaceReport:
    return FaceReport(
        face_shape=FaceShape.OVAL, confidence=0.0,
        features=FaceFeatures(face_ratio=1.0, jaw_cheek_ratio=1.0, forehead_ratio=1.0),
        error="未检测到人脸,请上传清晰的正面照(需包含完整面部)。",
    )


# ============================================================================
#  B Mock: 10 条种子发型数据
# ============================================================================

_MOCK_HAIRSTYLES: List[Hairstyle] = [
    Hairstyle(id="hair_001", name="锁骨微卷发",
        image_url="images/hair_001_clavicle_wavy.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART, FaceShape.DIAMOND],
        style_vector=StyleVector(优雅=0.80, 甜美=0.60, 自然=0.50),
        length=HairLength.MEDIUM, curl=HairCurl.WAVY, popularity=0.85,
        warnings=["避免完全露出额头,侧分效果更佳"],
        care_tips="建议使用轻质定型喷雾,保持自然弧度。每4周修剪一次。"),
    Hairstyle(id="hair_002", name="层次中长发",
        image_url="images/hair_002_layered_medium.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.ROUND, FaceShape.SQUARE, FaceShape.LONG],
        style_vector=StyleVector(干练=0.85, 酷飒=0.40, 自然=0.60),
        length=HairLength.MEDIUM, curl=HairCurl.STRAIGHT, popularity=0.78,
        warnings=["方脸建议搭配斜刘海柔和下颌线条"],
        care_tips="适合忙碌的早晨:吹干即可成型。定期打薄保持层次感。"),
    Hairstyle(id="hair_003", name="日系空气刘海",
        image_url="images/hair_003_air_bangs.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART, FaceShape.LONG],
        style_vector=StyleVector(甜美=0.95, 自然=0.70, 优雅=0.30),
        length=HairLength.LONG, curl=HairCurl.WAVY, popularity=0.90,
        warnings=["圆脸不适合厚重齐刘海", "方脸可用侧分替代齐刘海"],
        care_tips="刘海需每天用卷梳吹整,保持蓬松感。"),
    Hairstyle(id="hair_004", name="黑长直",
        image_url="images/hair_004_black_straight.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.LONG, FaceShape.DIAMOND],
        style_vector=StyleVector(优雅=0.70, 复古=0.60, 自然=0.80),
        length=HairLength.LONG, curl=HairCurl.STRAIGHT, popularity=0.75,
        warnings=["圆脸和方脸可能显脸大"],
        care_tips="定期做发膜护理,避免分叉。"),
    Hairstyle(id="hair_005", name="干练齐肩短发",
        image_url="images/hair_005_bob.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART, FaceShape.DIAMOND],
        style_vector=StyleVector(干练=0.90, 酷飒=0.70, 优雅=0.50),
        length=HairLength.SHORT, curl=HairCurl.STRAIGHT, popularity=0.72,
        warnings=["圆脸和方脸需谨慎,可能显脸大"],
        care_tips="低维护发型。8周修剪一次。"),
    Hairstyle(id="hair_006", name="少年感碎短发",
        image_url="images/hair_006_pixie_cut.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART],
        style_vector=StyleVector(酷飒=0.85, 干练=0.75, 自然=0.50),
        length=HairLength.SHORT, curl=HairCurl.STRAIGHT, popularity=0.65,
        warnings=["圆脸/方脸/长脸不建议"],
        care_tips="需要频繁修剪(4-6周一次)。"),
    Hairstyle(id="hair_007", name="复古港风大波浪",
        image_url="images/hair_007_vintage_waves.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.LONG, FaceShape.DIAMOND],
        style_vector=StyleVector(复古=0.90, 优雅=0.75, 甜美=0.40),
        length=HairLength.LONG, curl=HairCurl.CURLY, popularity=0.80,
        warnings=["小个子不建议", "发量少的蓬松度不够"],
        care_tips="需卷发棒定型,搭配弹力素维持卷度。"),
    Hairstyle(id="hair_008", name="法式羊毛卷",
        image_url="images/hair_008_french_perm.png",
        suitable_shapes=[FaceShape.ROUND, FaceShape.OVAL, FaceShape.LONG],
        style_vector=StyleVector(复古=0.70, 自然=0.60, 甜美=0.50),
        length=HairLength.MEDIUM, curl=HairCurl.CURLY, popularity=0.82,
        warnings=["方脸可能让脸看起来更宽"],
        care_tips="不能用梳子!用手抓出纹理。"),
    Hairstyle(id="hair_009", name="韩式蛋卷头",
        image_url="images/hair_009_korean_curl.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.ROUND, FaceShape.HEART],
        style_vector=StyleVector(甜美=0.85, 自然=0.65, 优雅=0.40),
        length=HairLength.MEDIUM, curl=HairCurl.WAVY, popularity=0.88,
        warnings=["方脸需搭配侧分刘海"],
        care_tips="用大号卷发棒定型,搭配定型喷雾。"),
    Hairstyle(id="hair_010", name="一刀切短发",
        image_url="images/hair_010_blunt_bob.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART],
        style_vector=StyleVector(酷飒=0.80, 干练=0.70, 优雅=0.50),
        length=HairLength.SHORT, curl=HairCurl.STRAIGHT, popularity=0.76,
        warnings=["圆脸/方脸/长脸不推荐"],
        care_tips="每6周修剪保持线条。"),
]


def mock_search_hairstyles(
    face_shape:   Optional[FaceShape] = None,
    style_vector: Optional[StyleVector] = None,
    length:       Optional[HairLength] = None,
    curl:         Optional[HairCurl] = None,
    limit:        int = 10,
) -> List[Hairstyle]:
    result = list(_MOCK_HAIRSTYLES)
    if face_shape is not None:
        result = [h for h in result if face_shape in h.suitable_shapes]
    if length is not None:
        result = [h for h in result if h.length == length]
    if curl is not None:
        result = [h for h in result if h.curl == curl]
    if style_vector is not None:
        def _cos_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na > 0 and nb > 0 else 0.0
        sv_list = style_vector.to_list()
        result = sorted(result, key=lambda h: _cos_sim(sv_list, h.style_vector.to_list()), reverse=True)
    return result[:limit]


def mock_get_hairstyle_by_id(hairstyle_id: str) -> Optional[Hairstyle]:
    for h in _MOCK_HAIRSTYLES:
        if h.id == hairstyle_id:
            return h
    return None


# ============================================================================
#  C Mock: 推荐引擎
# ============================================================================

def mock_recommend(
    face_report: FaceReport,
    preferences: StylePreferences,
    top_n: int = 5,
) -> List[Recommendation]:
    sv = preferences.style_vector
    if sv is None and preferences.natural_language is not None:
        sv = mock_infer_style_vector(preferences.natural_language)
    candidates = mock_search_hairstyles(
        face_shape=face_report.face_shape, style_vector=sv,
        length=preferences.preferred_length, curl=preferences.preferred_curl, limit=50,
    )
    results = []
    for h in candidates:
        face_match = 1.0 if face_report.face_shape in h.suitable_shapes else 0.0
        style_sim = 0.5
        if sv is not None:
            a, b = sv.to_list(), h.style_vector.to_list()
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            style_sim = dot / (na * nb) if na > 0 and nb > 0 else 0.0
        score = face_match * 0.50 + style_sim * 0.30 + h.popularity * 0.20
        reasons = [f"你的{face_report.face_shape.value}与{h.name}高度匹配"]
        if style_sim > 0.6 and sv is not None:
            reasons.append(f"该发型的{sv.dominant_dim()}风格与你偏好一致(相似度{style_sim:.0%})")
        if h.popularity > 0.7:
            reasons.append(f"该发型当前很受欢迎(热度{h.popularity:.0%})")
        results.append(Recommendation(
            hairstyle=h, score=round(score, 4), reasons=reasons,
            details={"face_match": round(face_match, 4), "style_similarity": round(style_sim, 4), "popularity": h.popularity},
        ))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]


def mock_infer_style_vector(natural_language: str) -> StyleVector:
    s = natural_language.lower()
    return StyleVector(
        干练 = 0.80 if any(w in s for w in ["干练", "通勤", "职业", "利落", "简约", "职场"]) else 0.10,
        甜美 = 0.90 if any(w in s for w in ["甜美", "可爱", "日系", "少女", "甜妹", "甜"]) else 0.10,
        复古 = 0.80 if any(w in s for w in ["复古", "港风", "法式", "经典", "怀旧", "民国"]) else 0.10,
        酷飒 = 0.80 if any(w in s for w in ["酷", "帅气", "中性", "飒", "冷淡", "高冷", "御姐"]) else 0.10,
        自然 = 0.80 if any(w in s for w in ["自然", "慵懒", "随性", "日常", "普通", "简单"]) else 0.10,
        优雅 = 0.80 if any(w in s for w in ["优雅", "气质", "成熟", "高级", "名媛", "端庄"]) else 0.10,
    )


# ============================================================================
#  D Mock: 风格报告
# ============================================================================

def mock_generate_style_report(face_report: FaceReport, recommendations: List[Recommendation]) -> str:
    face_desc_map = {
        "鹅蛋脸": "鹅蛋脸被称为\"黄金比例\"脸型,面部线条流畅,几乎能驾驭所有发型。",
        "圆脸": "圆脸脸颊饱满,适合能拉长面部比例的发型,如中长发或层次感发型。",
        "方脸": "方脸下颌线条分明,需要柔和线条来平衡硬朗轮廓,侧分刘海效果最佳。",
        "长脸": "长脸纵向比例偏大,需要横向有层次感的发型来增加视觉宽度。",
        "心形脸": "心形脸上宽下窄,关键是平衡额头和下巴的比例,避免暴露额头。",
        "菱形脸": "菱形脸颧骨突出,需要柔化颧骨线条,空气刘海是不错选择。",
    }
    face_desc = face_desc_map.get(face_report.face_shape.value, f"你的脸型为{face_report.face_shape.value}。")
    ff = face_report.features

    feature_lines = []
    if ff.eye_distance == "偏宽":
        feature_lines.append("| 眼距 | 偏宽 | 适合留刘海或侧分来缩短视觉眼距 |")
    elif ff.eye_distance == "偏窄":
        feature_lines.append("| 眼距 | 偏窄 | 适合中分或露额发型来拉宽眼部视觉 |")
    else:
        feature_lines.append("| 眼距 | 适中 | 大多数发型都能驾驭 |")
    if ff.nose_type == "偏宽":
        feature_lines.append("| 鼻型 | 偏宽 | 可用蓬松头顶或侧分转移视觉焦点 |")
    elif ff.nose_type == "圆润":
        feature_lines.append("| 鼻型 | 圆润 | 适合利落线条发型来平衡柔和感 |")
    else:
        feature_lines.append("| 鼻型 | 直挺 | 五官立体,适合多种发型风格 |")
    if ff.chin_shape == "方":
        feature_lines.append("| 下巴 | 偏方 | 适合柔和卷发或层次发尾来柔化下颌线 |")
    elif ff.chin_shape == "尖":
        feature_lines.append("| 下巴 | 偏尖 | 适合齐肩发或蓬松发型来平衡下巴比例 |")
    else:
        feature_lines.append("| 下巴 | 偏圆 | 适合有层次感的发型来增加轮廓立体感 |")
    feature_table = "\n".join(feature_lines)

    sv = recommendations[0].hairstyle.style_vector if recommendations else StyleVector()
    style_stars = []
    for dim_name in ALL_STYLE_DIMS:
        val = getattr(sv, dim_name)
        stars = "★" * int(val * 5) + "☆" * (5 - int(val * 5))
        style_stars.append(f"| {dim_name} | {stars} | {val:.0%} |")
    style_table = "\n".join(style_stars) if style_stars else ""

    rec_lines = []
    for i, rec in enumerate(recommendations, 1):
        rec_lines.append(f"| {i} | **{rec.hairstyle.name}** | {rec.score:.0%} | {'；'.join(rec.reasons)} |")
    rec_table = "\n".join(rec_lines) if rec_lines else "| - | 暂无匹配发型 | - | - |"

    all_warnings = []
    for rec in recommendations:
        all_warnings.extend(rec.hairstyle.warnings)
    unique_warnings = list(dict.fromkeys(all_warnings))[:5]
    warnings_text = "\n".join(f"- {w}" for w in unique_warnings) if unique_warnings else "- 暂无特别避雷提示"

    all_tips = []
    for rec in recommendations:
        if rec.hairstyle.care_tips:
            all_tips.append(rec.hairstyle.care_tips)
    unique_tips = list(dict.fromkeys(all_tips))[:5]
    tips_text = "\n".join(f"{i}. {t}" for i, t in enumerate(unique_tips, 1)) if unique_tips else "- 请根据具体发型咨询专业发型师"

    return f"""## 你的个人风格分析报告

### 一、脸型分析
- **脸型**: {face_report.face_shape.value}
- **置信度**: {face_report.confidence:.0%}
- **长宽比**: {ff.face_ratio:.2f} | **颧颌比**: {ff.jaw_cheek_ratio:.2f} | **额颧比**: {ff.forehead_ratio:.2f}

> {face_desc}

### 二、五官特点与发型暗示
| 特征 | 描述 | 发型建议 |
|------|------|---------|
{feature_table}

### 三、风格象限
| 维度 | 星级 | 强度 |
|------|------|:---:|
{style_table}

### 四、推荐发型 Top {len(recommendations)}
| 排名 | 发型 | 匹配度 | 理由 |
|:---:|------|:---:|------|
{rec_table}

### 五、避雷提示
{warnings_text}

### 六、日常打理建议
{tips_text}

---
*报告由 AI 发型顾问 Agent 自动生成。*
"""
