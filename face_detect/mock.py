# face_detect/mock.py
"""
A 模块 Mock 实现 — 复用 contracts.py 中已有的 mock 函数。

其他模块（C、D、E）在 A 模块未完成时，可以使用这些 mock 来开发测试。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contracts import mock_detect_face_shape, mock_detect_face_shape_error, FaceShape

__all__ = ["mock_detect_face_shape", "mock_detect_face_shape_error", "FaceShape"]
