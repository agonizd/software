"""
recommend_engine 包 — C 模块：推荐引擎
对外暴露: recommend(), infer_style_vector()
"""
from recommend_engine.engine import recommend
from recommend_engine.reverse_infer import infer_style_vector

__all__ = ["recommend", "infer_style_vector"]
