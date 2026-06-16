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
#  第三部分：从 contracts_mock 导入 Mock 实现（向后兼容）
# ============================================================================
#  所有 mock_* 函数和 _MOCK_* 数据已移至 contracts_mock.py
#  此处重导出确保所有已有代码无需修改即可继续使用
# ============================================================================

from contracts_mock import (  # noqa: E402, F401
    mock_detect_face_shape,
    mock_detect_face_shape_error,
    mock_search_hairstyles,
    mock_get_hairstyle_by_id,
    mock_recommend,
    mock_infer_style_vector,
    mock_generate_style_report,
    _MOCK_HAIRSTYLES,
    _MOCK_FACE_REPORTS,
)
