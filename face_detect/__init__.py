# A模块：脸型识别（MediaPipe FaceMesh）
"""
face_detect — 脸型识别模块

对外唯一入口：detect_face_shape(image_path) -> FaceReport
"""

from face_detect.detector import detect_face_shape

__all__ = ["detect_face_shape"]
