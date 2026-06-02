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

  并行策略:
  - 第1周: A 和 B 互不依赖,可完全并行开发(只需合约)
  - 第2周: C 和 D 互不依赖,可并行开发(依赖 A+B 但用 Mock 可提前开工)
  - 第3周: E 用 Mock 在第1周就能搭 Agent 骨架,第3周切换真模块联调
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
#  设计说明: str + Enum 双继承，序列化时自动取 .value，反序列化时 FaceShape("鹅蛋脸")。
#           这比纯 str 更安全（IDE 有补全），比纯 Enum 更方便（可以直接比较字符串）。
# ============================================================================

class FaceShape(str, Enum):
    """
    6 种脸型。

    分类依据（来自面部美学领域的共识标准）:
    - 鹅蛋脸: 长宽比≈1.5, 颧骨略宽, 额头/下颌均匀——"黄金比例"
    - 圆脸:   长宽比≈1.0, 脸颊饱满, 下颌圆润——"娃娃脸"
    - 方脸:   下颌角>140°, 额头宽≈下颌宽——"国字脸"/"方形"
    - 长脸:   长宽比>1.7, 脸型纵向拉伸——"马脸"
    - 心形脸: 额头宽>颧骨宽>下颌宽, 下巴尖——"倒三角"
    - 菱形脸: 颧骨最宽, 额头和下巴都窄——"钻石脸"

    注意: 这是一个主观美学分类,不是严格的生物学分类。
          不同文化可能有不同的脸型标准,本项目采用中国大陆通用标准。
    """
    OVAL    = "鹅蛋脸"
    ROUND   = "圆脸"
    SQUARE  = "方脸"
    LONG    = "长脸"
    HEART   = "心形脸"
    DIAMOND = "菱形脸"


class HairLength(str, Enum):
    """发型长度——影响推荐筛选的硬条件"""
    SHORT  = "短发"   # 耳朵以上
    MEDIUM = "中发"   # 耳朵～肩膀
    LONG   = "长发"   # 肩膀以下


class HairCurl(str, Enum):
    """卷度——影响发型质感和打理复杂度"""
    STRAIGHT = "直发"    # 自然垂顺
    WAVY     = "微卷"    # 自然弧度 / C 形卷
    CURLY    = "大卷"    # 明显卷曲 / S 形卷


class StyleDimension(str, Enum):
    """
    风格维度——用于"反向推理"的语义空间。

    设计说明: 为什么是 6 维而不是 3 维或 10 维？
    - 3 维(甜/飒/复古)太粗,无法区分"干练通勤"和"酷飒中性"
    - 10 维太细,标注成本高且维度间容易重叠
    - 6 维覆盖了主流中文发型搜索关键词,向量维度也适合余弦相似度计算

    使用方式:
    - 每个发型有一个 StyleVector(六维打分)
    - 用户说"日系甜美" → LLM 转为 StyleVector{甜美:0.9, 自然:0.6}
    - 推荐引擎用余弦相似度匹配
    """
    SMART   = "干练"    # 职场/通勤/利落/简约
    SWEET   = "甜美"    # 温柔/少女/可爱/日系甜
    RETRO   = "复古"    # 港风/法式/经典/怀旧
    COOL    = "酷飒"    # 中性/帅气/高冷/冷淡风
    NATURAL = "自然"    # 慵懒/随性/日常/裸感
    ELEGANT = "优雅"    # 气质/成熟/高级/名媛


# 所有风格维度列表(用于构建向量、断言检查等)
ALL_STYLE_DIMS: List[str] = [dim.value for dim in StyleDimension]


# ============================================================================
#  第一部分：共享数据类
#  设计原则:
#  1. 用 @dataclass 而非 dict——类型安全,IDE 有补全,字段修改时 CI 能抓到
#  2. 每个类自带 to_dict() / from_dict()——序列化边界只在函数调用入口/出口
#  3. Optional 字段标注清楚——哪些字段可空,什么情况下为空
# ============================================================================

@dataclass
class FaceFeatures:
    """
    面部五官特征（从 MediaPipe FaceMesh 468 关键点计算）。

    计算方式:
        face_ratio      = 脸宽(颧骨间距) / 脸长(发际线→下巴)
        jaw_cheek_ratio = 下颌角宽度 / 颧骨宽度
        forehead_ratio  = 额头宽度 / 颧骨宽度

    阈值参考(来自面部美学文献 + 实测调参):
        │ 指标          │ 鹅蛋脸 │ 圆脸  │ 方脸  │ 长脸  │ 心形脸 │ 菱形脸 │
        ├───────────────┼────────┼───────┼───────┼───────┼────────┼────────┤
        │ face_ratio     │ 1.4-1.6│ 0.9-1.2│ 1.1-1.4│ >1.7  │ 1.2-1.5│ 1.3-1.6│
        │ jaw_cheek_ratio│ 0.7-0.8│ 0.8-1.0│ >0.9  │ 0.7-0.8│ <0.7  │ <0.7  │
        │ forehead_ratio │ ≈1.0  │ ≈1.0  │ 0.9-1.1│ 0.8-1.0│ >1.05 │ 0.8-1.0│

    注意: 阈值会根据实际测试结果微调,不要硬编码在 contracts.py 里。
          A 模块应该把阈值放在 classifier.py 的配置中,方便调参。
    """
    face_ratio:      float                   # 脸宽/脸长,核心分类指标
    jaw_cheek_ratio: float                   # 下颌宽/颧骨宽,区分方脸 vs 心形脸
    forehead_ratio:  float                   # 额头宽/颧骨宽,辅助判断

    # 以下三个字段是语义化描述,由规则引擎从关键点坐标推断
    eye_distance:    str = "适中"             # "偏窄" | "适中" | "偏宽"
    nose_type:       str = "直挺"             # "直挺" | "圆润" | "偏宽"
    chin_shape:      str = "圆"               # "尖" | "圆" | "方"

    def __post_init__(self):
        """数据合法性校验"""
        if not (0.5 <= self.face_ratio <= 2.5):
            raise ValueError(f"face_ratio {self.face_ratio} 超出合理范围 [0.5, 2.5]")
        if not (0.3 <= self.jaw_cheek_ratio <= 1.5):
            raise ValueError(f"jaw_cheek_ratio {self.jaw_cheek_ratio} 超出合理范围 [0.3, 1.5]")
        if not (0.5 <= self.forehead_ratio <= 1.5):
            raise ValueError(f"forehead_ratio {self.forehead_ratio} 超出合理范围 [0.5, 1.5]")


@dataclass
class FaceReport:
    """
    脸型识别结果——A 模块的唯一输出。

    这是整个系统的"第一公里"数据,所有下游模块都依赖它。
    因此数据完整性和正确性至关重要。

    字段详解:
        face_shape:     分类标签(6 选 1)
        confidence:     置信度 0.0~1.0
                        - >0.85: 高置信度(直接采纳)
                        - 0.6~0.85: 中等置信度(Agent 应提示用户确认)
                        - <0.6: 低置信度(Agent 应提示用户无法判断,建议重新拍照)
        features:       五官特征详情(支撑风格报告的具体分析段落)
        raw_landmarks:  原始 468 个关键点坐标(调试用,生产环境不传输,太大)
        error:          错误信息(None=成功,非None=分析失败)

    使用示例:
        >>> report = detect_face_shape("photo.jpg")
        >>> if report.error:
        ...     print(f"分析失败: {report.error}")
        >>> elif report.confidence < 0.6:
        ...     print(f"置信度偏低,建议重新拍照")
        >>> else:
        ...     print(f"脸型: {report.face_shape.value}")
    """
    face_shape:     FaceShape
    confidence:     float
    features:       FaceFeatures
    raw_landmarks:  Optional[List[dict]] = None   # [{x: float, y: float, z: float}, ...]
    error:          Optional[str] = None           # 非 None 表示分析失败

    def to_dict(self) -> dict:
        """序列化——Agent 间传递用。不传 raw_landmarks(太大),不传 error=None"""
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
        """快速判断识别结果是否可靠(confidence >= 0.85)"""
        return self.confidence >= 0.85 and self.error is None


@dataclass
class StyleVector:
    """
    风格向量——6 维,每维 0.0~1.0。

    这是"反向推理"的核心数据结构:
    - 用户说"日系甜美风" → LLM 推理 → StyleVector{甜美:0.9, 自然:0.7}
    - 这个向量与每个发型的 style_vector 计算余弦相似度 = 风格匹配分

    为什么值域是 0.0~1.0 而不是 -1.0~1.0?
    → 风格维度之间是独立的:一个发型可以同时"甜美"又"干练",不存在"负甜美"。
    → 0.0~1.0 更直观:0=不相关,1=强烈相关。

    为什么向量不需要归一化?
    → 余弦相似度公式: cos(a,b) = dot(a,b) / (|a|×|b|)
    → 即使 a 和 b 没有归一化到单位向量,余弦值仍然在 [-1, 1] 范围内
    → 但为了方便调试和理解,建议标注时尽量让向量有区分度(不要全是 0.5)
    """
    干练: float = 0.0
    甜美: float = 0.0
    复古: float = 0.0
    酷飒: float = 0.0
    自然: float = 0.0
    优雅: float = 0.0

    def __post_init__(self):
        """校验每个维度在 0.0~1.0 范围内"""
        for dim_name in ALL_STYLE_DIMS:
            val = getattr(self, dim_name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"StyleVector.{dim_name}={val} 超出 [0.0, 1.0] 范围")

    def to_list(self) -> List[float]:
        """转为列表[干练,甜美,复古,酷飒,自然,优雅]——供 numpy/scipy 计算"""
        return [self.干练, self.甜美, self.复古, self.酷飒, self.自然, self.优雅]

    def to_dict(self) -> Dict[str, float]:
        """转为字典——供 JSON 序列化"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StyleVector":
        """从字典还原(只取 ALL_STYLE_DIMS 中的键)"""
        return cls(**{k: float(v) for k, v in d.items() if k in ALL_STYLE_DIMS})

    def dominant_dim(self) -> str:
        """返回最高分的维度名称(用于 Agent 简要描述)"""
        best = max(ALL_STYLE_DIMS, key=lambda d: getattr(self, d))
        return best


@dataclass
class Hairstyle:
    """
    单条发型数据——B 模块的数据模型。

    字段分组:
    ┌──────────────┬──────────────────────────────────┐
    │ 分组          │ 字段                               │
    ├──────────────┼──────────────────────────────────┤
    │ 标识          │ id, name                          │
    │ 展示          │ image_url                          │
    │ 匹配条件      │ suitable_shapes, style_vector      │
    │ 物理属性      │ length, curl                       │
    │ 元数据        │ popularity, warnings, care_tips    │
    └──────────────┴──────────────────────────────────┘

    suitable_shapes 说明:
    - 这是一个推荐列表,不是排他列表
    - [] 表示"所有脸型都可能适合"(但 confidence 会影响最终分数)
    - 标注原则:宁可多标(让推荐引擎打分筛选),不要漏标

    style_vector 说明:
    - 每维度 0.0~1.0,标注该发型在这个风格上的"强度"
    - 同一发型可以有多个高分维度(如锁骨微卷发: 优雅=0.8, 甜美=0.6)
    - 标注指南:
      · 0.0~0.2: 完全不具备这种风格
      · 0.3~0.5: 略微带有这种风格
      · 0.6~0.8: 明显具备这种风格(这是匹配的主力区间)
      · 0.9~1.0: 这种风格的"代表作"(用于精准匹配)
    """
    id:              str                     # 唯一标识,如 "hair_001"
    name:            str                     # 中文名称,如 "锁骨微卷发"
    image_url:       str                     # 展示图 URL(本地路径 or CDN),如 "images/hair_001.png"
    suitable_shapes: List[FaceShape]         # 适合的脸型列表
    style_vector:    StyleVector             # 风格向量
    length:          HairLength              # 长度
    curl:            HairCurl                # 卷度
    popularity:      float          = 0.5    # 热度 0.0~1.0(默认 0.5=中等)
    warnings:        List[str]      = field(default_factory=list)  # 避雷提示列表
    care_tips:       str            = ""     # 日常打理建议

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
    """
    用户偏好——C 模块的输入之一,来自 Agent 解析用户的自然语言。

    两种使用模式(互斥,但可以共存):
    1. 精确模式: style_vector 非空(用户选了具体风格标签)
    2. 模糊模式: natural_language 非空(用户说了段话,需要 LLM 转向量)
    3. 混合模式: 两者都非空 → 以 style_vector 为准,natural_language 作为补充

    为什么用 natural_language 而不是直接要求用户选标签?
    → 因为 Agent 的卖点就是"用户不需要知道怎么操作"。
    → 用户可以说"我想看起来更精神",Agent 负责把这句话转为可执行的查询。
    → 这就是"反向推理"的核心价值。

    示例:
        # 精确模式
        prefs = StylePreferences(style_vector=StyleVector(干练=0.9, 甜美=0.1))
        # 模糊模式
        prefs = StylePreferences(natural_language="日系甜美少女风")
        # 带附加条件
        prefs = StylePreferences(
            natural_language="日系甜美",
            preferred_length=HairLength.LONG,
            preferred_curl=HairCurl.WAVY
        )
    """
    style_vector:      Optional[StyleVector] = None   # 精确风格向量
    natural_language:  Optional[str]         = None   # 模糊自然语言描述
    preferred_length:  Optional[HairLength]  = None   # 长度偏好(可选)
    preferred_curl:    Optional[HairCurl]    = None   # 卷度偏好(可选)

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
        """是否完全没有偏好信息(没有风格也没有限制)"""
        return (self.style_vector is None
                and self.natural_language is None
                and self.preferred_length is None
                and self.preferred_curl is None)


@dataclass
class Recommendation:
    """
    单条推荐结果——C 模块的输出。

    字段说明:
        hairstyle: 发型数据(含名称、图片、风格向量等)
        score:     综合评分 0.0~1.0(用于排序)
        reasons:   推荐理由列表(自然语言,Agent 会直接使用)
        details:   各因子得分(调试/分析用,Agent 一般不需要展示)

    score 计算方式:
        score = face_match × 0.50 + style_similarity × 0.30 + popularity × 0.20

    其中:
        face_match:     发型.suitable_shapes 是否包含 脸型(1.0 or 0.0)
                        如果包含但 confidence<0.85,可以乘以 confidence 打折
        style_similarity:余弦相似度(用户偏好向量, 发型风格向量)
        popularity:      发型.popularity(0.0~1.0)

    注意:
        - 这是一个线性加权公式,不是推荐系统的最优算法
        - MVP 阶段够用,后续可升级为协同过滤 / 矩阵分解 / 学习排序
        - 权重(0.50, 0.30, 0.20)是可调的,放在 recommender/scoring.py 的配置中
    """
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
#  第二部分：模块函数签名
#  约定:
#  1. 每个函数包含完整的 docstring(做什么/输入/输出/异常/注意事项)
#  2. 所有 NotImplementedError 带说明(指向实现位置)
#  3. 返回类型精确标注,不用 Any
#  4. 输入参数都有类型标注 + 默认值(Optional 标注清楚)
# ============================================================================

# ── A 模块: 脸型识别 ────────────────────────────────────────────────

def detect_face_shape(image_path: str) -> FaceReport:
    """
    分析用户上传的人像照片,返回脸型 + 五官特征报告。

    ── 实现要点 ──
    1. 用 OpenCV 读取图片 → 转 RGB → 传给 MediaPipe FaceMesh
    2. FaceMesh 输出 468 个关键点的 (x, y, z) 归一化坐标(0~1)
    3. 从关键点中提取:
       - 脸宽:  关键点 234(左颧骨) 到 454(右颧骨) 的欧氏距离
       - 脸长:  关键点 10(发际线) 到 152(下巴) 的欧氏距离
       - 下颌宽: 关键点 58(左下颌) 到 288(右下颌) 的欧氏距离
       - 额头宽: 关键点 54(左额) 到 284(右额) 的欧氏距离
    4. 计算三个比例: face_ratio, jaw_cheek_ratio, forehead_ratio
    5. 根据阈值规则判断脸型(阈值在 classifier.py 中配置,可调)

    ── 异常处理 ──
    - FileNotFoundError: 图片路径不存在
    - ValueError: 图片中未检测到人脸
    - ValueError: 检测到多张人脸(取面积最大的那张,但 confidence 打折)
    - ValueError: 图片质量过低(分辨率<100×100 或 人脸占比<10%)
    - 上述异常都应被捕获,转为 FaceReport(error="具体错误信息")

    ── 性能要求 ──
    - 单张照片处理时间 < 2 秒(不含模型加载)
    - MediaPipe 模型可以预加载(懒加载模式,首次调用稍慢)

    ── 参数 ──
    image_path: 本地图片路径(支持 JPG / PNG / WebP)

    ── 返回 ──
    FaceReport: {face_shape, confidence, features, [error]}

    ── 测试建议 ──
    - 测试 6 种脸型各 2-3 张(正面、微侧)
    - 测试边界: 无人脸、多人脸、模糊照片、宠物照片
    - 期望准确率: 正面照 ≥ 85%, 微侧照 ≥ 70%
    """
    raise NotImplementedError(
        "A 模块尚未实现。请实现 face_analyzer/detector.py 中的 detect_face_shape()。\n"
        "提示: 用 contracts.mock_detect_face_shape() 作为临时替代。"
    )


# ── B 模块: 发型知识库 ──────────────────────────────────────────────

def search_hairstyles(
    face_shape:     Optional[FaceShape] = None,
    style_vector:   Optional[StyleVector] = None,
    length:         Optional[HairLength] = None,
    curl:           Optional[HairCurl] = None,
    limit:          int = 10,
) -> List[Hairstyle]:
    """
    根据条件搜索发型。

    ── 搜索逻辑 ──
    1. face_shape 非空 → 过滤适合该脸型的发型(suitable_shapes 包含该值)
    2. length 非空 → 过滤长度匹配的发型
    3. curl 非空 → 过滤卷度匹配的发型
    4. style_vector 非空 → 计算每个发型 style_vector 与查询向量的余弦相似度,按降序排序
    5. 取前 limit 条

    注意:
    - face_shape 匹配是精确匹配(1 或 0),不做模糊匹配
    - 如果所有过滤条件都为空,返回全部(按 popularity 降序,取前 limit)
    - 如果过滤后结果为空,返回空列表(不报错)

    ── 余弦相似度计算 ──
    cos(a, b) = dot(a.to_list(), b.to_list()) / (norm(a)×norm(b))
    其中 norm(v) = sqrt(sum(x² for x in v.to_list()))

    如果 norm(a)=0 或 norm(b)=0,返回 0.0(不报错)

    ── 参数 ──
    face_shape:   适合的脸型(精确过滤)
    style_vector: 风格偏好向量(余弦相似度排序),None 时不排序
    length:       长度过滤(可选)
    curl:         卷度过滤(可选)
    limit:        最大返回条数(默认 10,最大 50)

    ── 返回 ──
    匹配的发型列表(已排序,已截断)

    ── 实现位置 ──
    hairstyle_db/db.py
    """
    # 参数校验
    if limit < 1 or limit > 50:
        raise ValueError(f"limit 必须在 1~50 之间,实际: {limit}")

    raise NotImplementedError(
        "B 模块尚未实现。请实现 hairstyle_db/db.py 中的 search_hairstyles()。\n"
        "提示: 用 contracts.mock_search_hairstyles() 作为临时替代。"
    )


def get_hairstyle_by_id(hairstyle_id: str) -> Optional[Hairstyle]:
    """
    按 ID 获取单条发型数据。

    ── 参数 ──
    hairstyle_id: 发型唯一标识(如 "hair_001")

    ── 返回 ──
    Hairstyle 对象,未找到时返回 None(不抛异常)

    ── 实现位置 ──
    hairstyle_db/db.py
    """
    if not hairstyle_id or not isinstance(hairstyle_id, str):
        raise ValueError("hairstyle_id 不能为空")

    raise NotImplementedError(
        "B 模块尚未实现。请实现 hairstyle_db/db.py 中的 get_hairstyle_by_id()。\n"
        "提示: 用 contracts.mock_get_hairstyle_by_id() 作为临时替代。"
    )


# ── C 模块: 推荐引擎 ────────────────────────────────────────────────

def recommend(
    face_report:  FaceReport,
    preferences:  StylePreferences,
    top_n:        int = 5,
) -> List[Recommendation]:
    """
    综合脸型 + 风格偏好,返回 Top-N 发型推荐。

    ── 评分流程 ──
    1. 如果 preferences.natural_language 非空且 style_vector 为空
       → 先调用 infer_style_vector() 转为向量
    2. 调用 B 模块的 search_hairstyles(face_shape, style_vector, length, curl)
    3. 对返回的每条发型计算综合评分:
       score = face_match×0.50 + style_similarity×0.30 + popularity×0.20
    4. 按 score 降序 → 取前 top_n → 生成推荐理由(reasons)

    ── reasons 生成 ──
    每条推荐的 reasons 至少包含:
    - 脸型匹配理由: "你的{脸型}非常适合{发型名}"
    - 风格匹配理由: "该发型的{主导风格}风格与你偏好一致"
    - (可选)热度理由: "该发型当前很受欢迎"
    - (可选)注意事项: 如果发型有 warnings,可以作为 reason 的一部分

    ── 参数 ──
    face_report:  A 模块的输出
    preferences:  用户偏好(风格向量 or 自然语言)
    top_n:        返回 Top-N(默认 5,最大 10)

    ── 返回 ──
    推荐列表(按 score 降序),可能为空列表(无匹配)

    ── 实现位置 ──
    recommender/engine.py
    """
    if top_n < 1 or top_n > 10:
        raise ValueError(f"top_n 必须在 1~10 之间,实际: {top_n}")

    raise NotImplementedError(
        "C 模块尚未实现。请实现 recommender/engine.py 中的 recommend()。\n"
        "提示: 用 contracts.mock_recommend() 作为临时替代。"
    )


def infer_style_vector(natural_language: str) -> StyleVector:
    """
    "反向推理"核心: 把模糊的自然语言描述转为精确的风格向量。

    ── 这是本项目的创新点之一 ──
    市面上的发型推荐系统都需要用户先选择风格标签。
    我们让用户说一段话(如"日系甜美少女风,日常好打理"),
    Agent 自动调这个函数,用 LLM 推理出风格向量。

    ── 实现方式 ──
    1. MVP阶段: 关键词映射(如 mock 版本)
    2. 正式版: 调用 LLM API(GPT-4o-mini 即可,成本低)
       - Prompt 模板: "你是发型风格分析专家。用户描述了她想要的发型风格,
         请将这段描述映射到 6 个风格维度(干练/甜美/复古/酷飒/自然/优雅),
         每个维度 0.0~1.0。只输出 JSON。"
       - Fallback: 如果 LLM 调用失败,降级为关键词映射

    ── 参数 ──
    natural_language: 用户的自然语言风格描述(中文,1~200 字)

    ── 返回 ──
    StyleVector: 6 维风格向量,每维度 0.0~1.0

    ── 异常 ──
    ValueError: natural_language 为空或长度 > 200
    RuntimeError: LLM 调用失败且关键词映射也失败(极少发生)

    ── 实现位置 ──
    recommender/reverse_infer.py
    """
    if not natural_language or not natural_language.strip():
        raise ValueError("natural_language 不能为空")
    if len(natural_language) > 200:
        raise ValueError(f"natural_language 长度 {len(natural_language)} 超过 200 字限制")

    raise NotImplementedError(
        "C 模块尚未实现(反向推理)。请实现 recommender/reverse_infer.py。\n"
        "提示: 用 contracts.mock_infer_style_vector() 作为临时替代。"
    )


# ── D 模块: 风格报告生成 ────────────────────────────────────────────

def generate_style_report(
    face_report:     FaceReport,
    recommendations: List[Recommendation],
) -> str:
    """
    根据脸型分析 + 推荐结果,用 LLM 生成 Markdown 格式的个人风格报告。

    ── 报告结构(6 章) ──
    一、脸型分析
        - 你的脸型是什么、置信度、面部比例数据
        - 这种脸型的特点简介(1-2 句)
    二、五官特点与发型暗示
        - 眼距/鼻型/下巴 → 对发型选择的影响
        - 例:"你的眼距适中,适合中分或偏分;鼻型直挺,刘海不宜厚重"
    三、风格象限
        - 三线表: 风格 | 匹配度
        - 基于推荐结果的 style_vector,展示你最匹配的风格方向
    四、推荐发型 Top-N
        - 每个发型: 名称 + 图片 + 匹配度 + 推荐理由(来自 Recommendation.reasons)
    五、避雷提示
        - 基于脸型 + 推荐发型的 warnings,给出不适合的发型类型
    六、日常打理建议
        - 推荐造型产品、修剪周期、日常护理小贴士

    ── 注意事项 ──
    - 报告必须"千人千面":不同脸型/风格/推荐列表 → 输出内容不同
    - 不要硬编码示例文本,所有信息从 face_report 和 recommendations 提取
    - 边界情况:
      · recommendations 为空 → 报告提示"暂无匹配发型,请调整风格偏好"
      · face_report.confidence < 0.6 → 报告提示"识别置信度较低,仅供参考"
    - 表格必须用标准三线表格式(Markdown 表格)

    ── LLM Prompt 设计建议 ──
    - 用 system prompt 设定角色 + 报告模板
    - 用 user prompt 传入 face_report.to_dict() + recommendations 的 JSON
    - 输出要求: 纯 Markdown,不带 ```markdown``` 包裹
    - 温度: 0.3~0.5(有一定创造性但不过于随机)

    ── 参数 ──
    face_report:     A 模块的输出
    recommendations: C 模块的输出(Top-N)

    ── 返回 ──
    Markdown 字符串(可直接渲染)

    ── 实现位置 ──
    style_report/generator.py
    """
    if not recommendations:
        # 空推荐不应报错,应在生成函数内做优雅降级
        pass

    raise NotImplementedError(
        "D 模块尚未实现。请实现 style_report/generator.py 中的 generate_style_report()。\n"
        "提示: 用 contracts.mock_generate_style_report() 作为临时替代。"
    )


# ============================================================================
#  第三部分：Mock 实现
#  设计目标:
#  1. E 可以用这些 Mock 在第1天就搭好 Agent 骨架,不需要等任何人
#  2. C/D 可以用这些 Mock 写单元测试(因为他们依赖 A/B 的输出)
#  3. Mock 数据尽量覆盖所有脸型 × 所有风格 × 所有边界情况
# ============================================================================

# ── A Mock: 5 种不同脸型的 mock FaceReport ─────────────────────────

_MOCK_FACE_REPORTS: Dict[FaceShape, FaceReport] = {
    FaceShape.OVAL: FaceReport(
        face_shape=FaceShape.OVAL,
        confidence=0.92,
        features=FaceFeatures(
            face_ratio=1.48, jaw_cheek_ratio=0.78, forehead_ratio=1.05,
            eye_distance="适中", nose_type="直挺", chin_shape="圆",
        ),
    ),
    FaceShape.ROUND: FaceReport(
        face_shape=FaceShape.ROUND,
        confidence=0.88,
        features=FaceFeatures(
            face_ratio=1.05, jaw_cheek_ratio=0.92, forehead_ratio=0.98,
            eye_distance="偏宽", nose_type="圆润", chin_shape="圆",
        ),
    ),
    FaceShape.SQUARE: FaceReport(
        face_shape=FaceShape.SQUARE,
        confidence=0.85,
        features=FaceFeatures(
            face_ratio=1.22, jaw_cheek_ratio=0.95, forehead_ratio=1.02,
            eye_distance="适中", nose_type="偏宽", chin_shape="方",
        ),
    ),
    FaceShape.LONG: FaceReport(
        face_shape=FaceShape.LONG,
        confidence=0.86,
        features=FaceFeatures(
            face_ratio=1.78, jaw_cheek_ratio=0.72, forehead_ratio=0.90,
            eye_distance="偏窄", nose_type="直挺", chin_shape="尖",
        ),
    ),
    FaceShape.HEART: FaceReport(
        face_shape=FaceShape.HEART,
        confidence=0.80,
        features=FaceFeatures(
            face_ratio=1.35, jaw_cheek_ratio=0.62, forehead_ratio=1.12,
            eye_distance="适中", nose_type="直挺", chin_shape="尖",
        ),
    ),
    # 菱形脸单独定义(confidence 设置低一点,测试边界情况)
    FaceShape.DIAMOND: FaceReport(
        face_shape=FaceShape.DIAMOND,
        confidence=0.65,
        features=FaceFeatures(
            face_ratio=1.42, jaw_cheek_ratio=0.55, forehead_ratio=0.82,
            eye_distance="偏窄", nose_type="直挺", chin_shape="尖",
        ),
    ),
}


def mock_detect_face_shape(image_path: str = "", face_shape: FaceShape = FaceShape.OVAL) -> FaceReport:
    """
    Mock: 返回指定脸型的 FaceReport(默认鹅蛋脸)。

    用法:
        >>> report = mock_detect_face_shape()        # 鹅蛋脸
        >>> report = mock_detect_face_shape(face_shape=FaceShape.ROUND)  # 圆脸
    """
    return _MOCK_FACE_REPORTS.get(face_shape, _MOCK_FACE_REPORTS[FaceShape.OVAL])


def mock_detect_face_shape_error() -> FaceReport:
    """Mock: 返回一个"无人脸"错误(测试错误处理流程)"""
    return FaceReport(
        face_shape=FaceShape.OVAL,   # 占位值
        confidence=0.0,
        features=FaceFeatures(face_ratio=1.0, jaw_cheek_ratio=1.0, forehead_ratio=1.0),
        error="未检测到人脸,请上传清晰的正面照(需包含完整面部)。",
    )


# ── B Mock: 10 条种子发型数据(覆盖全部 6 种脸型 + 全部 6 种风格) ──

_MOCK_HAIRSTYLES: List[Hairstyle] = [
    # ── 中发 × 微卷 / 直发(百搭款) ──
    Hairstyle(
        id="hair_001", name="锁骨微卷发",
        image_url="images/hair_001_clavicle_wavy.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART, FaceShape.DIAMOND],
        style_vector=StyleVector(优雅=0.80, 甜美=0.60, 自然=0.50),
        length=HairLength.MEDIUM, curl=HairCurl.WAVY, popularity=0.85,
        warnings=["避免完全露出额头,侧分效果更佳"],
        care_tips="建议使用轻质定型喷雾,保持自然弧度。每4周修剪一次。",
    ),
    Hairstyle(
        id="hair_002", name="层次中长发",
        image_url="images/hair_002_layered_medium.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.ROUND, FaceShape.SQUARE, FaceShape.LONG],
        style_vector=StyleVector(干练=0.85, 酷飒=0.40, 自然=0.60),
        length=HairLength.MEDIUM, curl=HairCurl.STRAIGHT, popularity=0.78,
        warnings=["方脸建议搭配斜刘海柔和下颌线条"],
        care_tips="适合忙碌的早晨:吹干即可成型。定期打薄保持层次感。",
    ),

    # ── 长发 × 直发 / 微卷(气质款) ──
    Hairstyle(
        id="hair_003", name="日系空气刘海",
        image_url="images/hair_003_air_bangs.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART, FaceShape.LONG],
        style_vector=StyleVector(甜美=0.95, 自然=0.70, 优雅=0.30),
        length=HairLength.LONG, curl=HairCurl.WAVY, popularity=0.90,
        warnings=["圆脸不适合厚重齐刘海", "方脸可用侧分替代齐刘海"],
        care_tips="刘海需每天用卷梳吹整,保持蓬松感。建议备用干发喷雾。",
    ),
    Hairstyle(
        id="hair_004", name="黑长直",
        image_url="images/hair_004_black_straight.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.LONG, FaceShape.DIAMOND],
        style_vector=StyleVector(优雅=0.70, 复古=0.60, 自然=0.80),
        length=HairLength.LONG, curl=HairCurl.STRAIGHT, popularity=0.75,
        warnings=["圆脸和方脸可能显脸大", "需要发质好才出效果,受损发质慎选"],
        care_tips="定期做发膜护理,避免分叉。每周2-3次精油护理。",
    ),

    # ── 短发 × 直发(干练款) ──
    Hairstyle(
        id="hair_005", name="干练齐肩短发",
        image_url="images/hair_005_bob.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART, FaceShape.DIAMOND],
        style_vector=StyleVector(干练=0.90, 酷飒=0.70, 优雅=0.50),
        length=HairLength.SHORT, curl=HairCurl.STRAIGHT, popularity=0.72,
        warnings=["圆脸和方脸需谨慎,可能显脸大"],
        care_tips="低维护发型。8周修剪一次,日常免打理。",
    ),
    Hairstyle(
        id="hair_006", name="少年感碎短发",
        image_url="images/hair_006_pixie_cut.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART],
        style_vector=StyleVector(酷飒=0.85, 干练=0.75, 自然=0.50),
        length=HairLength.SHORT, curl=HairCurl.STRAIGHT, popularity=0.65,
        warnings=["圆脸/方脸/长脸不建议", "额头过宽或过窄不适合"],
        care_tips="需要频繁修剪(4-6周一次)。造型用发蜡抓出纹理。",
    ),

    # ── 长发 × 大卷(复古/优雅款) ──
    Hairstyle(
        id="hair_007", name="复古港风大波浪",
        image_url="images/hair_007_vintage_waves.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.LONG, FaceShape.DIAMOND],
        style_vector=StyleVector(复古=0.90, 优雅=0.75, 甜美=0.40),
        length=HairLength.LONG, curl=HairCurl.CURLY, popularity=0.80,
        warnings=["小个子不建议(长度压身高)", "发量少的蓬松度不够"],
        care_tips="需卷发棒定型,搭配弹力素维持卷度。每周2次深层护理。",
    ),
    Hairstyle(
        id="hair_008", name="法式羊毛卷",
        image_url="images/hair_008_french_perm.png",
        suitable_shapes=[FaceShape.ROUND, FaceShape.OVAL, FaceShape.LONG],
        style_vector=StyleVector(复古=0.70, 自然=0.60, 甜美=0.50),
        length=HairLength.MEDIUM, curl=HairCurl.CURLY, popularity=0.82,
        warnings=["方脸可能让脸看起来更宽", "发量极多者慎选(爆炸头风险)"],
        care_tips="不能用梳子!用手抓出纹理。搭配泡沫发蜡定型。",
    ),

    # ── 中发 × 微卷(甜美/自然款) ──
    Hairstyle(
        id="hair_009", name="韩式蛋卷头",
        image_url="images/hair_009_korean_curl.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.ROUND, FaceShape.HEART],
        style_vector=StyleVector(甜美=0.85, 自然=0.65, 优雅=0.40),
        length=HairLength.MEDIUM, curl=HairCurl.WAVY, popularity=0.88,
        warnings=["方脸需搭配侧分刘海", "长脸需注意刘海长度不能太短"],
        care_tips="用大号卷发棒定型,搭配定型喷雾。避免频繁梳头。",
    ),

    # ── 直发 × 短发(酷飒款) ──
    Hairstyle(
        id="hair_010", name="一刀切短发",
        image_url="images/hair_010_blunt_bob.png",
        suitable_shapes=[FaceShape.OVAL, FaceShape.HEART],
        style_vector=StyleVector(酷飒=0.80, 干练=0.70, 优雅=0.50),
        length=HairLength.SHORT, curl=HairCurl.STRAIGHT, popularity=0.76,
        warnings=["圆脸/方脸/长脸不推荐", "需要经常打理维持整齐感"],
        care_tips="每6周修剪保持线条。用直发夹定型。",
    ),
]


def mock_search_hairstyles(
    face_shape:   Optional[FaceShape] = None,
    style_vector: Optional[StyleVector] = None,
    length:       Optional[HairLength] = None,
    curl:         Optional[HairCurl] = None,
    limit:        int = 10,
) -> List[Hairstyle]:
    """
    Mock: 对种子数据做简单过滤 + 排序(真实版用 SQLite + 余弦相似度)。

    这个 mock 实现了与真实函数相同的过滤逻辑,方便 C/E 写测试。
    """
    result = list(_MOCK_HAIRSTYLES)

    # 1. 脸型过滤
    if face_shape is not None:
        result = [h for h in result if face_shape in h.suitable_shapes]

    # 2. 长度过滤
    if length is not None:
        result = [h for h in result if h.length == length]

    # 3. 卷度过滤
    if curl is not None:
        result = [h for h in result if h.curl == curl]

    # 4. 风格相似度排序(简易版余弦)
    if style_vector is not None:
        import math
        def _cos_sim(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na > 0 and nb > 0 else 0.0

        sv_list = style_vector.to_list()
        result = sorted(
            result,
            key=lambda h: _cos_sim(sv_list, h.style_vector.to_list()),
            reverse=True,
        )

    return result[:limit]


def mock_get_hairstyle_by_id(hairstyle_id: str) -> Optional[Hairstyle]:
    """Mock: 按 ID 查找(遍历种子数据)"""
    for h in _MOCK_HAIRSTYLES:
        if h.id == hairstyle_id:
            return h
    return None


# ── C Mock: 推荐引擎 ────────────────────────────────────────────────

def mock_recommend(
    face_report: FaceReport,
    preferences: StylePreferences,
    top_n: int = 5,
) -> List[Recommendation]:
    """
    Mock: 用 mock_search_hairstyles + 简单打分模拟推荐流程。

    比之前版本更好:真正计算 face_match + style_similarity + popularity,
    而不是只按人气排序。这样 C 写单元测试时有正确的参考值。
    """
    import math

    # 1. 如果有自然语言描述 → 先转风格向量
    sv = preferences.style_vector
    if sv is None and preferences.natural_language is not None:
        sv = mock_infer_style_vector(preferences.natural_language)

    # 2. 搜索候选集
    candidates = mock_search_hairstyles(
        face_shape=face_report.face_shape,
        style_vector=sv,
        length=preferences.preferred_length,
        curl=preferences.preferred_curl,
        limit=50,
    )

    # 3. 打分
    results: List[Recommendation] = []
    for h in candidates:
        # 脸型匹配: 1.0 or 0.0(因为 search 已过滤,全为1.0; 若未过滤则检查)
        face_match = 1.0 if face_report.face_shape in h.suitable_shapes else 0.0

        # 风格相似度
        style_sim = 0.5  # 默认
        if sv is not None:
            a = sv.to_list()
            b = h.style_vector.to_list()
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            style_sim = dot / (na * nb) if na > 0 and nb > 0 else 0.0

        # 综合分
        score = face_match * 0.50 + style_sim * 0.30 + h.popularity * 0.20

        # 理由
        reasons = [f"你的{face_report.face_shape.value}与{h.name}高度匹配"]

        if style_sim > 0.6 and sv is not None:
            best_dim = sv.dominant_dim()
            reasons.append(f"该发型的{best_dim}风格与你偏好一致(相似度{style_sim:.0%})")

        if h.popularity > 0.7:
            reasons.append(f"该发型当前很受欢迎(热度{h.popularity:.0%})")

        # 避雷提醒(只在必要时加)
        if h.warnings and face_report.face_shape == FaceShape.ROUND:
            reasons.append(f"注意: {h.warnings[0]}")

        results.append(Recommendation(
            hairstyle=h,
            score=round(score, 4),
            reasons=reasons,
            details={
                "face_match": round(face_match, 4),
                "style_similarity": round(style_sim, 4),
                "popularity": h.popularity,
            },
        ))

    # 4. 排序 + 截断
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]


def mock_infer_style_vector(natural_language: str) -> StyleVector:
    """
    Mock: 关键词映射(真实版用 LLM)。

    支持的关键词覆盖所有 6 个风格维度。
    如果同一句话包含多个风格关键词,向量会混合(如"日系甜美干练风")。
    """
    s = natural_language.lower()
    return StyleVector(
        干练 = 0.80 if any(w in s for w in ["干练", "通勤", "职业", "利落", "简约", "职场"]) else 0.10,
        甜美 = 0.90 if any(w in s for w in ["甜美", "可爱", "日系", "少女", "甜妹", "甜"]) else 0.10,
        复古 = 0.80 if any(w in s for w in ["复古", "港风", "法式", "经典", "怀旧", "民国"]) else 0.10,
        酷飒 = 0.80 if any(w in s for w in ["酷", "帅气", "中性", "飒", "冷淡", "高冷", "御姐"]) else 0.10,
        自然 = 0.80 if any(w in s for w in ["自然", "慵懒", "随性", "日常", "普通", "简单"]) else 0.10,
        优雅 = 0.80 if any(w in s for w in ["优雅", "气质", "成熟", "高级", "名媛", "端庄"]) else 0.10,
    )


# ── D Mock: 风格报告 ────────────────────────────────────────────────

def mock_generate_style_report(
    face_report: FaceReport,
    recommendations: List[Recommendation],
) -> str:
    """
    Mock: 返回完整的 Markdown 风格报告。

    真实版用 LLM 生成,这个 mock 版用字符串模板生成。
    包含了完整的 6 章结构 + 三线表 + 千人千面的内容(从参数提取)。
    """

    # ── 第一章: 脸型分析 ──
    face_desc_map = {
        "鹅蛋脸": "鹅蛋脸被称为\"黄金比例\"脸型,面部线条流畅,长宽比例近乎完美。几乎任何发型都能轻松驾驭。",
        "圆脸":   "圆脸的特点是脸颊饱满、线条柔和,视觉上减龄效果显著。适合能拉长面部比例的发型。",
        "方脸":   "方脸下颌线条分明,给人干练率性的感觉。需要用柔和线条来平衡硬朗轮廓。",
        "长脸":   "长脸纵向比例偏大,需要在视觉上增加宽度来平衡。横向有层次感的发型是最佳选择。",
        "心形脸": "心形脸上宽下窄,额头饱满、下巴精致。关键是平衡额头和下巴的比例。",
        "菱形脸": "菱形脸颧骨突出、额头和下巴偏窄,是非常有辨识度的脸型。需要柔化颧骨线条。",
    }

    face_desc = face_desc_map.get(face_report.face_shape.value,
        f"你的脸型经系统分析为{face_report.face_shape.value}。")

    # ── 第二章: 五官特点 ──
    eye_hint_map = {
        "偏窄": "建议选择中分或露出额头的发型,避免厚重刘海加重视觉压迫",
        "适中": "适合大部分刘海类型(空气刘海、侧分、八字刘海),发型选择自由度很高",
        "偏宽": "偏分或侧分效果更好,可以适当缩短眼距的视觉感受",
    }
    nose_hint_map = {
        "直挺": "几乎任何发型都能驾驭,刘海厚度不影响整体协调感",
        "圆润": "建议选择有层次的发型,侧分刘海比齐刘海更能突出鼻部优势",
        "偏宽": "避免过于贴脸的直发,微卷发可以分散视觉焦点",
    }
    chin_hint_map = {
        "尖":   "锁骨发或大波浪可以平衡尖下巴,增加下半脸的量感",
        "圆":   "层次感发型可以打破圆润感,锁骨长度最适合",
        "方":   "带弧度的卷发可以柔和下颌线条,避免齐耳短发",
    }

    eye_hint   = eye_hint_map.get(face_report.features.eye_distance, "适中")
    nose_hint  = nose_hint_map.get(face_report.features.nose_type, "直挺")
    chin_hint  = chin_hint_map.get(face_report.features.chin_shape, "圆")

    # ── 第三章: 风格象限 ──
    # 从推荐结果中聚合最匹配的风格向量
    agg_vector = StyleVector()
    if recommendations:
        for rec in recommendations:
            sv = rec.hairstyle.style_vector
            agg_vector.干练 += sv.干练
            agg_vector.甜美 += sv.甜美
            agg_vector.复古 += sv.复古
            agg_vector.酷飒 += sv.酷飒
            agg_vector.自然 += sv.自然
            agg_vector.优雅 += sv.优雅
        n = len(recommendations)
        for dim in ALL_STYLE_DIMS:
            setattr(agg_vector, dim, getattr(agg_vector, dim) / n)

    # 转为星级(0~1 → 0~5 星)
    def to_stars(val: float) -> str:
        full = int(round(val * 5))
        return "★" * full + "☆" * (5 - full)

    # ── 第四章: 推荐列表 ──
    rec_lines: List[str] = []
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            reasons_text = "；".join(rec.reasons)
            rec_lines.append(
                f"| {i} | **{rec.hairstyle.name}** | {rec.score:.0%} | "
                f"{rec.hairstyle.length.value} | {rec.hairstyle.curl.value} | "
                f"{reasons_text} |"
            )
    else:
        rec_lines.append("| - | 暂无匹配发型,请调整风格偏好后重试 | - | - | - | - |")

    rec_table = "\n".join(rec_lines)

    # ── 第五章: 避雷提示 ──
    warnings_set: set = set()
    for rec in recommendations:
        for w in rec.hairstyle.warnings:
            warnings_set.add(w)
    if not warnings_set:
        warnings_set.add("暂无特殊避雷提示,你可以大胆尝试各种风格!")

    warning_lines = "\n".join(f"- {w}" for w in sorted(warnings_set))

    # ── 第六章: 打理建议 ──
    all_tips = [rec.hairstyle.care_tips for rec in recommendations if rec.hairstyle.care_tips]
    if not all_tips:
        all_tips = ["根据发型类型选择合适的造型产品,保持定期修剪习惯。"]

    tip_lines = "\n".join(f"- {t}" for t in all_tips[:3])  # 最多3条

    # ── 组装 ──
    md = f"""## 🎯 你的个人风格分析报告

### 一、脸型分析

- **脸型**: {face_report.face_shape.value}
- **置信度**: {face_report.confidence:.0%}
- **长宽比**: {face_report.features.face_ratio:.2f}
- **颧颌比**: {face_report.features.jaw_cheek_ratio:.2f}

> {face_desc}

### 二、五官特点与发型暗示

| 特征 | 描述 | 发型建议 |
|------|------|---------|
| 眼距 | {face_report.features.eye_distance} | {eye_hint} |
| 鼻型 | {face_report.features.nose_type} | {nose_hint} |
| 下巴 | {face_report.features.chin_shape} | {chin_hint} |

### 三、风格象限

| 干练 | 甜美 | 复古 | 酷飒 | 自然 | 优雅 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| {to_stars(agg_vector.干练)} | {to_stars(agg_vector.甜美)} | {to_stars(agg_vector.复古)} | {to_stars(agg_vector.酷飒)} | {to_stars(agg_vector.自然)} | {to_stars(agg_vector.优雅)} |

### 四、推荐发型 Top {len(recommendations) if recommendations else 0}

| 排名 | 发型 | 匹配度 | 长度 | 卷度 | 理由 |
|:---:|------|:---:|:---:|:---:|------|
{rec_table}

### 五、避雷提示

{warning_lines}

### 六、日常打理建议

{tip_lines}

---
*报告由 AI 发型顾问 Agent 自动生成,仅供参考。最终选择请结合个人喜好和实际试戴效果。*
"""
    return md


# ============================================================================
#  第四部分：合约自检
#  用法: python contracts.py
#  如果全部通过 → ✅ 合约无问题,可以开工
#  如果某条失败 → ❌ 检查对应模块的数据类和 Mock
# ============================================================================

if __name__ == "__main__":
    import sys
    import traceback

    passed = 0
    failed = 0
    total  = 0

    def check(name: str, condition: bool, detail: str = ""):
        """简单的测试断言,带计数和彩色输出"""
        global passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ [{name}] {detail}")
        else:
            failed += 1
            print(f"  ❌ [{name}] FAILED: {detail}")
            return False
        return True

    print("=" * 65)
    print("  contracts.py — 合约自检(完整版)")
    print("=" * 65)

    # ── 第一节: 枚举检查 ──
    print("\n── 枚举检查 ──")
    check("FaceShape 共6类", len(FaceShape) == 6,
          f"实际: {len(FaceShape)} → {[s.value for s in FaceShape]}")
    check("HairLength 共3类", len(HairLength) == 3)
    check("HairCurl 共3类", len(HairCurl) == 3)
    check("StyleDimension 共6类", len(StyleDimension) == 6)
    check("Enum→str 可用", FaceShape.OVAL == "鹅蛋脸" or FaceShape.OVAL.value == "鹅蛋脸")
    check("str→Enum 可重建", FaceShape("鹅蛋脸") == FaceShape.OVAL)

    # ── 第二节: 数据类检查 ──
    print("\n── 数据类检查 ──")

    # FaceFeatures
    ff = FaceFeatures(face_ratio=1.48, jaw_cheek_ratio=0.78, forehead_ratio=1.05)
    check("FaceFeatures 创建", ff.face_ratio == 1.48 and ff.eye_distance == "适中")
    try:
        FaceFeatures(face_ratio=5.0, jaw_cheek_ratio=0.5, forehead_ratio=1.0)
        check("FaceFeatures 边界校验", False, "应该抛出 ValueError")
    except ValueError:
        check("FaceFeatures 边界校验", True, "face_ratio=5.0 正确抛出 ValueError")

    # FaceReport
    report = _MOCK_FACE_REPORTS[FaceShape.OVAL]
    check("FaceReport 创建", report.face_shape == FaceShape.OVAL and report.confidence == 0.92)
    d = report.to_dict()
    check("FaceReport.to_dict()", d["face_shape"] == "鹅蛋脸" and "confidence" in d)
    check("FaceReport.to_dict() 不含 raw_landmarks", "raw_landmarks" not in d)
    restored = FaceReport.from_dict(d)
    check("FaceReport 反序列化", restored.face_shape == FaceShape.OVAL and restored.confidence == 0.92)
    check("FaceReport.is_reliable()", report.is_reliable() is True)
    check("FaceReport.is_reliable() 低置信度",
          mock_detect_face_shape(face_shape=FaceShape.DIAMOND).is_reliable() is False)

    # StyleVector
    sv = StyleVector(干练=0.9, 甜美=0.1, 复古=0.0, 酷飒=0.0, 自然=0.5, 优雅=0.3)
    check("StyleVector 创建", sv.干练 == 0.9)
    check("StyleVector.to_list()", len(sv.to_list()) == 6 and sv.to_list()[0] == 0.9)
    check("StyleVector.dominant_dim()", sv.dominant_dim() == "干练")
    d_sv = sv.to_dict()
    restored_sv = StyleVector.from_dict(d_sv)
    check("StyleVector 反序列化", restored_sv.干练 == 0.9)

    try:
        StyleVector(干练=1.5, 甜美=0.1, 复古=0.0, 酷飒=0.0, 自然=0.5, 优雅=0.3)
        check("StyleVector 边界校验", False, "应该抛出 ValueError")
    except ValueError:
        check("StyleVector 边界校验", True, "干练=1.5 正确抛出 ValueError")

    # Hairstyle
    h = _MOCK_HAIRSTYLES[0]
    check("Hairstyle 创建", h.id == "hair_001" and h.name == "锁骨微卷发")
    d_h = h.to_dict()
    restored_h = Hairstyle.from_dict(d_h)
    check("Hairstyle 反序列化", restored_h.id == "hair_001" and restored_h.name == "锁骨微卷发")
    check("Hairstyle 风格向量保留",
          restored_h.style_vector.优雅 == 0.80 and restored_h.style_vector.甜美 == 0.60)

    # StylePreferences
    prefs_empty = StylePreferences()
    check("StylePreferences.is_empty()", prefs_empty.is_empty())
    prefs_full = StylePreferences(natural_language="日系甜美", preferred_length=HairLength.LONG)
    check("StylePreferences.is_empty() 非空", not prefs_full.is_empty())

    # Recommendation
    rec = Recommendation(
        hairstyle=h, score=0.85,
        reasons=["适合鹅蛋脸", "风格匹配"],
        details={"face_match": 0.9, "style_similarity": 0.8, "popularity": 0.7}
    )
    check("Recommendation.to_dict()", rec.to_dict()["score"] == 0.85
          and len(rec.to_dict()["reasons"]) == 2)

    # ── 第三节: Mock 数据完整性 ──
    print("\n── Mock 数据完整性 ──")

    # A: 6 种脸型都有 mock FaceReport
    check("6种脸型 Mock 齐全", len(_MOCK_FACE_REPORTS) == 6,
          f"实际: {len(_MOCK_FACE_REPORTS)} → {list(_MOCK_FACE_REPORTS.keys())}")
    check("Mock 误差报告可用",
          mock_detect_face_shape_error().error is not None)

    # B: 10 条种子数据
    check("种子数据 10 条", len(_MOCK_HAIRSTYLES) == 10)

    # B: 覆盖所有脸型
    all_shapes_in_seed: set = set()
    for hs in _MOCK_HAIRSTYLES:
        all_shapes_in_seed.update(hs.suitable_shapes)
    check("种子数据覆盖6种脸型", len(all_shapes_in_seed) == 6,
          f"实际覆盖: {[s.value for s in all_shapes_in_seed]}")

    # B: 覆盖所有风格维度
    max_style_per_dim: Dict[str, float] = {d: 0.0 for d in ALL_STYLE_DIMS}
    for hs in _MOCK_HAIRSTYLES:
        for d in ALL_STYLE_DIMS:
            val = getattr(hs.style_vector, d)
            if val > max_style_per_dim[d]:
                max_style_per_dim[d] = val
    all_dims_covered = all(v > 0.5 for v in max_style_per_dim.values())
    check("种子数据覆盖6种风格维度", all_dims_covered,
          f"各维度最高分: {max_style_per_dim}")

    # B: mock_search_hairstyles 过滤测试
    short_styles = mock_search_hairstyles(length=HairLength.SHORT)
    check("过滤: 短发", len(short_styles) == 3,
          f"实际: {len(short_styles)} → {[h.name for h in short_styles]}")

    oval_styles = mock_search_hairstyles(face_shape=FaceShape.OVAL)
    check("过滤: 鹅蛋脸", len(oval_styles) >= 8,
          f"实际: {len(oval_styles)}")  # 鹅蛋脸百搭,应该很多

    curly_styles = mock_search_hairstyles(curl=HairCurl.CURLY)
    check("过滤: 大卷", len(curly_styles) == 2,
          f"实际: {len(curly_styles)} → {[h.name for h in curly_styles]}")

    # 组合过滤
    comb = mock_search_hairstyles(
        face_shape=FaceShape.ROUND, curl=HairCurl.STRAIGHT, limit=5
    )
    check("组合过滤: 圆脸+直发", all(
        FaceShape.ROUND in h.suitable_shapes and h.curl == HairCurl.STRAIGHT
        for h in comb
    ), f"实际: {len(comb)} 条")

    # B: get_hairstyle_by_id
    check("get_hairstyle_by_id 找到", mock_get_hairstyle_by_id("hair_001") is not None)
    check("get_hairstyle_by_id 未找到", mock_get_hairstyle_by_id("nonexistent") is None)

    # C: mock_recommend 测试
    print("\n── Mock 推荐引擎 ──")

    prefs = StylePreferences(natural_language="干练通勤风")
    recs = mock_recommend(report, prefs, top_n=3)
    check("推荐: 返回 Top-3", len(recs) == 3,
          f"实际: {len(recs)} → {[r.hairstyle.name for r in recs]}")
    check("推荐: 按 score 降序", recs[0].score >= recs[1].score >= recs[2].score,
          f"分数: {[r.score for r in recs]}")
    check("推荐: 包含 reasons", all(len(r.reasons) >= 1 for r in recs))
    check("推荐: 包含 details", all(len(r.details) == 3 for r in recs),
          f"实际: {[r.details for r in recs]}")

    # 空偏好
    empty_recs = mock_recommend(report, StylePreferences(), top_n=5)
    check("推荐: 空偏好仍返回结果", len(empty_recs) == 5)

    # 反向推理: 6 种风格各测一种
    infer_tests = [
        ("干练通勤风", "干练", 0.80),
        ("日系甜美少女风", "甜美", 0.90),
        ("复古港风", "复古", 0.80),
        ("酷飒高冷", "酷飒", 0.80),
        ("自然慵懒日常", "自然", 0.80),
        ("优雅气质风", "优雅", 0.80),
    ]
    for text, dim, expected_min in infer_tests:
        svi = mock_infer_style_vector(text)
        actual = getattr(svi, dim)
        check(f"反向推理: '{text}' → {dim}={actual}",
              actual >= expected_min,
              f"预期≥{expected_min},实际={actual}")

    # D: mock_generate_style_report
    print("\n── Mock 风格报告 ──")
    md = mock_generate_style_report(report, recs)
    check("报告: 包含 6 个章节标题",
          all(s in md for s in ["脸型分析", "五官特点", "风格象限", "推荐发型", "避雷提示", "打理建议"]))
    check("报告: 包含三线表", "|" in md and "---" in md)
    check("报告: 包含脸型名", report.face_shape.value in md)
    check("报告: 包含推荐发型名", recs[0].hairstyle.name in md)
    check("报告: 包含星级评分", "★" in md)
    check("报告: 长度 > 500 字符", len(md) > 500,
          f"实际: {len(md)} 字符")

    # 空推荐报告
    empty_md = mock_generate_style_report(report, [])
    check("报告: 空推荐优雅降级",
          "暂无匹配发型" in empty_md or "0" in empty_md.split("推荐发型")[1][:30])

    # ── 第四节: 函数签名验证 ──
    print("\n── 函数签名验证 ──")
    funcs = {
        "detect_face_shape":     detect_face_shape,
        "search_hairstyles":     search_hairstyles,
        "get_hairstyle_by_id":   get_hairstyle_by_id,
        "recommend":             recommend,
        "infer_style_vector":    infer_style_vector,
        "generate_style_report": generate_style_report,
    }
    for name, func in funcs.items():
        sig = inspect.signature(func)
        check(f"函数签名: {name}", True,
              f"参数: {list(sig.parameters.keys())}")
        src = inspect.getsource(func)
        check(f"未实现: {name}", "NotImplementedError" in src or "raise NotImplementedError" in src)

    # ── 总结 ──
    print("\n" + "=" * 65)
    if failed == 0:
        print(f"  🎉 全部 {passed}/{total} 项检查通过!")
        print(f"  合约无问题,5 个人可以各自开工了。")
    else:
        print(f"  ⚠️  {passed}/{total} 通过, {failed} 项失败")
        print(f"  请检查上述 ❌ 标记的测试项。")
    print("=" * 65)
    sys.exit(0 if failed == 0 else 1)
