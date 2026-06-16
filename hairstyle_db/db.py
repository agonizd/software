"""
hairstyle_db/db.py — B 模块：发型 JSON 数据库

提供：
- load_hairstyles()        从 JSON 加载全部发型
- save_hairstyles()        保存到 JSON（支持增删改后回写）
- search_hairstyles()      按脸型/风格/长度/卷度多维筛选
- get_hairstyle_by_id()    按 ID 精确查询
- count_hairstyles()       统计总数
- add_hairstyle()          新增发型
- delete_hairstyle()       删除发型

数据源：hairstyle_db/hairstyles.json（10 条种子数据）
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import List, Optional

# 确保能导入同级的 contracts 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts import (
    FaceShape, HairLength, HairCurl, StyleVector, Hairstyle,
)

# ── 数据库路径 ──────────────────────────────────────────────────
_DB_DIR = os.path.dirname(os.path.abspath(__file__))
_JSON_PATH = os.path.join(_DB_DIR, "hairstyles.json")


# ======================================================================
#  核心读写
# ======================================================================

def load_hairstyles() -> List[Hairstyle]:
    """从 JSON 加载全部发型数据，返回 Hairstyle 对象列表。"""
    if not os.path.exists(_JSON_PATH):
        return []
    with open(_JSON_PATH, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    return [_dict_to_hairstyle(d) for d in raw_list]


def save_hairstyles(hairstyles: List[Hairstyle]) -> None:
    """将 Hairstyle 列表序列化并写回 JSON 文件。"""
    raw_list = [h.to_dict() for h in hairstyles]
    with open(_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_list, f, ensure_ascii=False, indent=2)


def count_hairstyles() -> int:
    """返回当前发型总数。"""
    return len(load_hairstyles())


# ======================================================================
#  查询
# ======================================================================

def search_hairstyles(
    face_shape:   Optional[FaceShape] = None,
    style_vector: Optional[StyleVector] = None,
    length:       Optional[HairLength] = None,
    curl:         Optional[HairCurl] = None,
    limit:        int = 10,
) -> List[Hairstyle]:
    """
    多维度筛选发型。
    - face_shape:  按适合脸型筛选
    - style_vector: 按风格向量排序（余弦相似度）
    - length/curl: 按长度/卷度精确筛选
    - limit:       最多返回条数
    """
    if limit < 1 or limit > 50:
        raise ValueError(f"limit 必须在 1~50 之间,实际: {limit}")

    results = load_hairstyles()

    if face_shape is not None:
        results = [h for h in results if face_shape in h.suitable_shapes]
    if length is not None:
        results = [h for h in results if h.length == length]
    if curl is not None:
        results = [h for h in results if h.curl == curl]

    if style_vector is not None:
        sv_list = style_vector.to_list()
        def _cos_sim(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na > 0 and nb > 0 else 0.0
        results.sort(key=lambda h: _cos_sim(sv_list, h.style_vector.to_list()), reverse=True)

    return results[:limit]


def get_hairstyle_by_id(hairstyle_id: str) -> Optional[Hairstyle]:
    """按 ID 精确查询，不存在返回 None。"""
    if not hairstyle_id:
        raise ValueError("hairstyle_id 不能为空")
    for h in load_hairstyles():
        if h.id == hairstyle_id:
            return h
    return None


# ======================================================================
#  增删
# ======================================================================

def add_hairstyle(hairstyle: Hairstyle) -> bool:
    """
    新增发型（ID 不能重复），返回 True 表示成功。
    """
    all_hs = load_hairstyles()
    if any(h.id == hairstyle.id for h in all_hs):
        return False  # ID 重复
    all_hs.append(hairstyle)
    save_hairstyles(all_hs)
    return True


def delete_hairstyle(hairstyle_id: str) -> bool:
    """按 ID 删除发型，返回 True 表示成功。"""
    all_hs = load_hairstyles()
    new_list = [h for h in all_hs if h.id != hairstyle_id]
    if len(new_list) == len(all_hs):
        return False  # 未找到
    save_hairstyles(new_list)
    return True


# ======================================================================
#  内部辅助
# ======================================================================

def _dict_to_hairstyle(d: dict) -> Hairstyle:
    """JSON dict → Hairstyle 对象。"""
    return Hairstyle(
        id=d["id"],
        name=d["name"],
        image_url=d["image_url"],
        suitable_shapes=[FaceShape(s) for s in d["suitable_shapes"]],
        style_vector=StyleVector(**{k: v for k, v in d["style_vector"].items()}),
        length=HairLength(d["length"]),
        curl=HairCurl(d["curl"]),
        popularity=float(d.get("popularity", 0.5)),
        warnings=d.get("warnings", []),
        care_tips=d.get("care_tips", ""),
    )


# ======================================================================
#  自检
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  hairstyle_db/db.py 自检")
    print("=" * 60)

    total = count_hairstyles()
    print(f"  发型总数: {total}")
    assert total == 10, f"期望 10 条，实际 {total} 条"

    # 按脸型搜
    oval = search_hairstyles(face_shape=FaceShape.OVAL)
    print(f"  鹅蛋脸适合: {[h.name for h in oval]}")
    assert len(oval) == 10, f"鹅蛋脸应匹配全部 10 款"

    # 按 ID 查
    h = get_hairstyle_by_id("hair_003")
    print(f"  ID 查询 hair_003: {h.name if h else '未找到'}")
    assert h is not None and h.name == "日系空气刘海"

    # 按风格向量排序
    sv = StyleVector(干练=0.9)
    ranked = search_hairstyles(style_vector=sv, limit=3)
    ranked_names = [(h.name, round(h.style_vector.干练, 2)) for h in ranked]
    print(f"  干练风 Top3: {ranked_names}")
    assert ranked[0].id in ["hair_002", "hair_005", "hair_010"]

    print("  ✅ 全部自检通过！")
