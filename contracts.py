"""
================================================================================
  contracts.py — 发型推荐 AI Agent 系统 · API 合约（完整版）
================================================================================

  角色: 5 个人的"宪法"——所有模块的函数签名、数据类、Mock 实现在此定义。
  规则: 任何人修改合约 → 必须提 PR + 全员 Approve → 修改此文件的 docstring
        和自检脚本 → 自检通过后才能合并。

  项目: 基于 Agent 的脸型发型推荐系统
  团队: 5 人 / 3 周
  仓库: hair-advisor-agent

================================================================================
  数据流全景（以一次完整交互为例）:

  ┌─────────────────────────────────────────────────────────────────┐
  │ 用户: "帮我看看我适合什么发型"  + 上传 photo.jpg                   │
  │                                                                 │
  │ Agent 自动推理:                                                   │
  │   1. 用户上传了照片 → 调用 detect_face_shape("photo.jpg")         │
  │   2. 拿到 FaceReport{鹅蛋脸, 0.92} → 展示给用户                   │
  │   3. 用户: "我想要日系甜美风"                                     │
  │   4. Agent 调用 infer_style_vector("日系甜美风")                  │
  │      → StyleVector{甜美:0.9, 自然:0.7, ...}                      │
  │   5. Agent 调用 recommend(face_report, preferences)              │
  │      → [Recommendation(hair_003, 0.89, ...), ...]               │
  │   6. Agent 调用 generate_style_report(face_report, recs)         │
  │   7. Agent 把报告渲染到 Streamlit → 用户看到完整分析              │
  └─────────────────────────────────────────────────────────────────┘

================================================================================
  模块依赖链:

  A (face_analyzer) ─┐
                      ├──→ C (recommender) ──→ D (style_report)
  B (hairstyle_db)   ─┘         │                    │
                                 └────────────────────┘
                                          │
                                   E (agent_app)
                                 合约先行→各自开发→分阶段集成
================================================================================
"""

from __future__ import annotations  # 支持 Forward Reference (Python 3.10+)

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Union
from enum import Enum
import json
import inspect


# ============================================================================
#  第〇部分：枚举与常量
# ============================================================================

class FaceShape(str, Enum):
    """6 种脸型。"""
    OVAL    = "鹅蛋脸"
    ROUND   = "圆脸"
    SQUARE  = "方脸"
    LONG    = "长脸"
    HEART   = "心形脸"
    DIAMOND = "菱形脸"


class HairLength(str, Enum):
    """发型长度"""
    SHORT  = "短发"
    MEDIUM = "中发"
    LONG   = "长发"


class HairCurl(str, Enum):
    """卷度"""
    STRAIGHT = "直发"
    WAVY     = "微卷"
    CURLY    = "大卷"


class StyleDimension(str, Enum):
    """风格维度"""
    SMART   = "干练"
    SWEET   = "甜美"
    RETRO   = "复古"
    COOL    = "酷飒"
    NATURAL = "自然"
    ELEGANT = "优雅"


ALL_STYLE_DIMS: List[str] = [dim.value for dim in StyleDimension]


# ============================================================================
#  第一部分：共享数据类
# ============================================================================

@dataclass
class FaceFeatures:
    """面部五官特征（从 MediaPipe FaceMesh 468 关键点计算）"""
    face_ratio:      float
    jaw_cheek_ratio: float
    forehead_ratio:  float
    eye_distance:    str = "适中"
    nose_type:       str = "直挺"
    chin_shape:      str = "圆"

    def __post_init__(self):
        if not (0.5 <= self.face_ratio <= 2.5):
            raise ValueError(f"face_ratio {self.face_ratio} 超出合理范围 [0.5, 2.5]")
        if not (0.3 <= self.jaw_cheek_ratio <= 1.5):
            raise ValueError(f"jaw_cheek_ratio {self.jaw_cheek_ratio} 超出合理范围 [0.3, 1.5]")
        if not (0.5 <= self.forehead_ratio <= 1.5):
            raise ValueError(f"forehead_ratio {self.forehead_ratio} 超出合理范围 [0.5, 1.5]")


@dataclass
class FaceReport:
    """脸型识别结果——A 模块的唯一输出。"""
    face_shape:     FaceShape
    confidence:     float
    features:       FaceFeatures
    raw_landmarks:  Optional[List[dict]] = None
    error:          Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["face_shape"] = self.face_shape.value
        d.pop("raw_landmarks", None)
        if self.error is None:
            d.pop("error", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FaceReport":
        d = d.copy()
        d["face_shape"] = FaceShape(d["face_shape"])
        d["features"] = FaceFeatures(**d["features"])
        if "error" in d and d["error"] is not None:
            d["error"] = str(d["error"])
        return cls(**d)

    def is_reliable(self) -> bool:
        return self.confidence >= 0.85 and self.error is None


@dataclass
class StyleVector:
    """风格向量——6 维,每维 0.0~1.0。"""
    干练: float = 0.0
    甜美: float = 0.0
    复古: float = 0.0
    酷飒: float = 0.0
    自然: float = 0.0
    优雅: float = 0.0

    def __post_init__(self):
        for dim_name in ALL_STYLE_DIMS:
            val = getattr(self, dim_name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"StyleVector.{dim_name}={val} 超出 [0.0, 1.0] 范围")

    def to_list(self) -> List[float]:
        return [self.干练, self.甜美, self.复古, self.酷飒, self.自然, self.优雅]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StyleVector":
        return cls(**{k: float(v) for k, v in d.items() if k in ALL_STYLE_DIMS})

    def dominant_dim(self) -> str:
        best = max(ALL_STYLE_DIMS, key=lambda d: getattr(self, d))
        return best


@dataclass
class Hairstyle:
    """单条发型数据——B 模块的数据模型。"""
    id:              str
    name:            str
    image_url:       str
    suitable_shapes: List[FaceShape]
    style_vector:    StyleVector
    length:          HairLength
    curl:            HairCurl
    popularity:      float          = 0.5
    warnings:        List[str]      = field(default_factory=list)
    care_tips:       str            = ""

    def __post_init__(self):
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Hairstyle.id 不能为空")
        if not self.name:
            raise ValueError("Hairstyle.name 不能为空")
        if not (0.0 <= self.popularity <= 1.0):
            raise ValueError(f"popularity {self.popularity} 超出 [0.0, 1.0]")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["suitable_shapes"] = [s.value for s in self.suitable_shapes]
        d["style_vector"] = self.style_vector.to_dict()
        d["length"] = self.length.value
        d["curl"] = self.curl.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Hairstyle":
        d = d.copy()
        d["suitable_shapes"] = [FaceShape(s) for s in d["suitable_shapes"]]
        d["style_vector"] = StyleVector.from_dict(d["style_vector"])
        d["length"] = HairLength(d["length"])
        d["curl"] = HairCurl(d["curl"])
        return cls(**d)


@dataclass
class StylePreferences:
    """用户偏好——C 模块的输入之一。"""
    style_vector:      Optional[StyleVector] = None
    natural_language:  Optional[str]         = None
    preferred_length:  Optional[HairLength]  = None
    preferred_curl:    Optional[HairCurl]    = None

    def to_dict(self) -> dict:
        d: dict = {}
        if self.style_vector is not None:
            d["style_vector"] = self.style_vector.to_dict()
        if self.natural_language is not None:
            d["natural_language"] = self.natural_language
        if self.preferred_length is not None:
            d["preferred_length"] = self.preferred_length.value
        if self.preferred_curl is not None:
            d["preferred_curl"] = self.preferred_curl.value
        return d

    def is_empty(self) -> bool:
        return (self.style_vector is None
                and self.natural_language is None
                and self.preferred_length is None
                and self.preferred_curl is None)


@dataclass
class Recommendation:
    """单条推荐结果——C 模块的输出。"""
    hairstyle:  Hairstyle
    score:      float
    reasons:    List[str]
    details:    Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "hairstyle": self.hairstyle.to_dict(),
            "score": self.score,
            "reasons": self.reasons,
            "details": self.details,
        }


# ============================================================================
#  第二部分：模块函数签名（NotImplementedError 占位）
# ============================================================================

def detect_face_shape(image_path: str) -> FaceReport:
    raise NotImplementedError("A 模块尚未实现。")

def search_hairstyles(
    face_shape:     Optional[FaceShape] = None,
    style_vector:   Optional[StyleVector] = None,
    length:         Optional[HairLength] = None,
    curl:           Optional[HairCurl] = None,
    limit:          int = 10,
) -> List[Hairstyle]:
    if limit < 1 or limit > 50:
        raise ValueError(f"limit 必须在 1~50 之间,实际: {limit}")
    raise NotImplementedError("B 模块尚未实现。")

def get_hairstyle_by_id(hairstyle_id: str) -> Optional[Hairstyle]:
    if not hairstyle_id or not isinstance(hairstyle_id, str):
        raise ValueError("hairstyle_id 不能为空")
    raise NotImplementedError("B 模块尚未实现。")

def recommend(
    face_report:  FaceReport,
    preferences:  StylePreferences,
    top_n:        int = 5,
) -> List[Recommendation]:
    if top_n < 1 or top_n > 10:
        raise ValueError(f"top_n 必须在 1~10 之间,实际: {top_n}")
    raise NotImplementedError("C 模块尚未实现。")

def infer_style_vector(natural_language: str) -> StyleVector:
    if not natural_language or not natural_language.strip():
        raise ValueError("natural_language 不能为空")
    if len(natural_language) > 200:
        raise ValueError(f"natural_language 长度 {len(natural_language)} 超过 200 字限制")
    raise NotImplementedError("C 模块尚未实现(反向推理)。")

def generate_style_report(
    face_report:     FaceReport,
    recommendations: List[Recommendation],
) -> str:
    raise NotImplementedError("D 模块尚未实现。")


# ============================================================================
#  第三部分：Mock 实现
# ============================================================================

# ── A Mock ──
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


# ── B Mock: 10 条种子发型数据 ──
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
        import math
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


# ── C Mock ──
def mock_recommend(
    face_report: FaceReport,
    preferences: StylePreferences,
    top_n: int = 5,
) -> List[Recommendation]:
    import math
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


# ── D Mock ──
def mock_generate_style_report(face_report: FaceReport, recommendations: List[Recommendation]) -> str:
    face_desc_map = {
        "鹅蛋脸": "鹅蛋脸被称为\"黄金比例\"脸型,面部线条流畅。",
        "圆脸": "圆脸脸颊饱满,适合能拉长面部比例的发型。",
        "方脸": "方脸下颌线条分明,需要柔和线条来平衡硬朗轮廓。",
        "长脸": "长脸纵向比例偏大,需要横向有层次感的发型。",
        "心形脸": "心形脸上宽下窄,关键是平衡额头和下巴的比例。",
        "菱形脸": "菱形脸颧骨突出,需要柔化颧骨线条。",
    }
    face_desc = face_desc_map.get(face_report.face_shape.value, f"你的脸型为{face_report.face_shape.value}。")
    rec_lines = []
    for i, rec in enumerate(recommendations, 1):
        rec_lines.append(f"| {i} | **{rec.hairstyle.name}** | {rec.score:.0%} | {'；'.join(rec.reasons)} |")
    rec_table = "\n".join(rec_lines) if rec_lines else "| - | 暂无匹配发型 | - | - |"
    return f"""## 你的个人风格分析报告

### 一、脸型分析
- **脸型**: {face_report.face_shape.value}
- **置信度**: {face_report.confidence:.0%}

> {face_desc}

### 二、推荐发型 Top {len(recommendations)}

| 排名 | 发型 | 匹配度 | 理由 |
|:---:|------|:---:|------|
{rec_table}

---
*报告由 AI 发型顾问 Agent 自动生成。*
"""
