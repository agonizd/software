"""
reverse_infer.py — C 模块创新功能：反向推理

"反向推理"核心: 把模糊的自然语言描述转为精确的风格向量。
市面上的发型推荐系统都需要用户先选择风格标签。
我们让用户说一段话，Agent 自动调这个函数，用 LLM 推理出风格向量。

实现方式:
  1. MVP阶段: 关键词映射
  2. 正式版: 调用 LLM API (GPT-4o-mini, 成本低)
  3. Fallback: LLM 失败 → 降级为关键词映射
"""

from __future__ import annotations
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contracts import StyleVector


# ─────────────────────────────────────────────────────────────
# LLM Prompt 模板
# ─────────────────────────────────────────────────────────────

INFER_PROMPT = """\
你是发型风格分析专家。用户描述了她想要的发型风格，
请将这段描述映射到 6 个风格维度，每个维度 0.0~1.0。

风格维度说明（每维取值 0.0~1.0，多个维度可同时高分）：
- 干练：职场通勤、利落、精神、简约
- 甜美：温柔、可爱、少女感、日系甜
- 复古：港风、法式、经典、怀旧
- 酷飒：中性、帅气、高冷、冷淡风
- 自然：慵懒、随性、日常、裸感
- 优雅：气质、成熟、高级、名媛

用户描述：{user_input}

只输出 JSON，不包含任何其他文字或markdown标记：
{{"干练": 0.0, "甜美": 0.0, "复古": 0.0, "酷飒": 0.0, "自然": 0.0, "优雅": 0.0}}"""


# ─────────────────────────────────────────────────────────────
# 关键词映射表（降级方案）
# ─────────────────────────────────────────────────────────────

KEYWORD_MAP: Dict[str, list] = {
    "干练": ["干练", "通勤", "职场", "利落", "OL", "商务", "简约", "职业"],
    "甜美": ["甜美", "可爱", "少女", "软妹", "甜系", "日系", "温柔", "萌"],
    "复古": ["复古", "港风", "法式", "经典", "复古风", "怀旧", "vintage"],
    "酷飒": ["酷", "帅气", "中性", "高冷", "朋克", "酷飒", "冷淡", "御姐"],
    "自然": ["自然", "慵懒", "随性", "清爽", "日常", "简约", "低调"],
    "优雅": ["优雅", "气质", "成熟", "高级感", "知性", "女人味", "名媛", "端庄"],
}


# ═════════════════════════════════════════════════════════════
# 核心函数
# ═════════════════════════════════════════════════════════════

def infer_style_vector(natural_language: str) -> StyleVector:
    """
    把用户自然语言描述转为风格向量。
    优先调用 LLM（精确），失败自动降级到关键词匹配（快速）。

    参数:
        natural_language: 用户对风格的自由文字描述，如"我想要干练通勤风"

    返回:
        StyleVector: 6 维风格向量

    异常:
        ValueError: natural_language 为空或长度 > 200
    """
    if not natural_language or not natural_language.strip():
        raise ValueError("natural_language 不能为空")
    if len(natural_language) > 200:
        raise ValueError(f"natural_language 长度 {len(natural_language)} 超过 200 字限制")

    try:
        return _infer_by_llm(natural_language)
    except Exception:
        return _infer_by_keywords(natural_language)


# ═════════════════════════════════════════════════════════════
# 方案 1：LLM 推理（需要 OPENAI_API_KEY）
# ═════════════════════════════════════════════════════════════

def _infer_by_llm(text: str) -> StyleVector:
    """调用 OpenAI 接口提取风格向量。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai 未安装，请执行: pip install openai")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("未找到 OPENAI_API_KEY 环境变量")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": INFER_PROMPT.format(user_input=text)}],
        temperature=0.1,
        max_tokens=100,
    )
    raw = resp.choices[0].message.content.strip()

    # 清理可能的 markdown 包裹
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    data = json.loads(raw)

    # 数值合法性检查
    for k, v in data.items():
        if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"LLM 返回值不合法: {k}={v}")

    return StyleVector(**{k: float(v) for k, v in data.items()})


# ═════════════════════════════════════════════════════════════
# 方案 2：关键词匹配（降级备用）
# ═════════════════════════════════════════════════════════════

def _infer_by_keywords(text: str) -> StyleVector:
    """
    基于预定义关键词表的简单匹配。
    每命中一个关键词 +0.5，上限 1.0。
    没有任何命中时，默认返回「自然 0.5」。
    """
    scores = {dim: 0.0 for dim in KEYWORD_MAP}

    for dim, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text:
                scores[dim] += 0.5
        scores[dim] = min(scores[dim], 1.0)

    # 全零时默认自然风
    if all(v == 0.0 for v in scores.values()):
        scores["自然"] = 0.5

    return StyleVector(**scores)
