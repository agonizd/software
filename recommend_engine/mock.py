"""
mock.py — C 模块对外暴露的 Mock 实现
供 E（组长）在 Agent 开发阶段使用，无需等待 C 真实交付。
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts import (
    Recommendation, Hairstyle, StyleVector,
    FaceReport, StylePreferences, FaceShape,
    HairLength, HairCurl, FaceFeatures,
)


def mock_recommend(
    face_report: FaceReport,
    preferences: StylePreferences,
    top_n: int = 5,
) -> list:
    """
    返回固定的 5 条测试推荐数据。
    E 接入 Agent 时可直接用此函数替代真实 recommend()。
    """
    from contracts import mock_recommend as _real_mock
    return _real_mock(face_report, preferences, top_n)
