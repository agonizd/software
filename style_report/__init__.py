"""
D 模块 — 风格报告生成器

根据脸型分析结果 + 发型推荐列表生成 Markdown 格式的个人风格分析报告。
"""

from .generator import generate_style_report, mock_generate_style_report_d

__all__ = ["generate_style_report", "mock_generate_style_report_d"]
