# face_detect/detector.py
"""
脸型检测器 — 模块 A 的核心文件。

实现 contracts.py 中定义的函数签名:
    detect_face_shape(image_path: str) -> FaceReport

流程:
    1. OpenCV 读取图片 → 转 RGB
    2. MediaPipe FaceLandmarker (Tasks API) 检测 → 输出 468 个关键点
    3. landmarks.py 提取面部比例
    4. classifier.py 判断脸型
    5. 组装 FaceReport 返回

注意:
    本文件兼容 MediaPipe 0.10.35+ (Tasks API)。
    首次运行会自动下载 face_landmarker.task 模型文件（约 3.8MB）。
"""

import os
import sys
import urllib.request

# 确保能导入项目根目录的 contracts.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

from contracts import FaceReport, FaceFeatures, FaceShape
from face_detect.landmarks  import extract_face_features, infer_eye_distance, infer_nose_type, infer_chin_shape
from face_detect.classifier import classify_face_shape


# ── 模型文件路径 ────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "face_landmarker.task",
)

# ── 懒加载：MediaPipe FaceLandmarker 只初始化一次 ────────────
_face_landmarker = None


def _ensure_model() -> str:
    """确保模型文件存在，不存在则自动下载。返回本地路径。"""
    if not os.path.exists(_MODEL_PATH):
        print(f"[face_detect] 正在下载模型文件... ({_MODEL_URL})")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print(f"[face_detect] 模型下载完成: {_MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(
                f"无法下载 FaceLandmarker 模型: {e}\n"
                f"请手动下载: {_MODEL_URL}\n"
                f"并保存到: {_MODEL_PATH}"
            )
    return _MODEL_PATH


def _get_face_landmarker():
    """获取或初始化 FaceLandmarker 实例（懒加载模式）"""
    global _face_landmarker
    if _face_landmarker is None:
        model_path = _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _face_landmarker = FaceLandmarker.create_from_options(options)
    return _face_landmarker


def _make_error_report(error_msg: str) -> FaceReport:
    """生成一个标准的错误报告"""
    return FaceReport(
        face_shape=FaceShape.OVAL,  # 占位值
        confidence=0.0,
        features=FaceFeatures(face_ratio=1.0, jaw_cheek_ratio=1.0, forehead_ratio=1.0),
        error=error_msg,
    )


def detect_face_shape(image_path: str) -> FaceReport:
    """
    分析用户上传的人像照片，返回脸型 + 五官特征报告。

    参数:
        image_path: 本地图片路径（支持 JPG / PNG / WebP）

    返回:
        FaceReport: {face_shape, confidence, features, [error]}

    异常处理:
        所有异常都被捕获，转为 FaceReport(error="具体错误信息")，
        不会抛出异常到调用方。
    """

    # ── 1. 检查文件是否存在 ──
    if not os.path.exists(image_path):
        return _make_error_report(f"图片路径不存在: {image_path}")

    # ── 2. 读取图片 ──
    image = cv2.imread(image_path)
    if image is None:
        return _make_error_report("图片读取失败，请检查格式（支持 JPG/PNG/WebP）")

    # ── 3. 检查分辨率 ──
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return _make_error_report(f"图片分辨率过低({w}×{h})，请上传至少 100×100 的照片")

    # ── 4. 预处理：BGR → RGB ──
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ── 5. MediaPipe FaceLandmarker 检测 ──
    try:
        face_landmarker = _get_face_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = face_landmarker.detect(mp_image)
    except Exception as e:
        return _make_error_report(f"MediaPipe 处理失败: {str(e)}")

    # ── 6. 检查是否检测到人脸 ──
    if not result.face_landmarks:
        return _make_error_report("未检测到人脸，请上传清晰的正面照（需包含完整面部）。")

    # ── 7. 提取关键点（只取第一张/最大人脸）──
    #    result.face_landmarks[0] 是 List[NormalizedLandmark]
    landmarks = result.face_landmarks[0]

    # ── 8. 计算面部比例 ──
    raw_features = extract_face_features(landmarks)

    # ── 9. 推断五官语义描述 ──
    eye_distance = infer_eye_distance(landmarks)
    nose_type    = infer_nose_type(landmarks)
    chin_shape   = infer_chin_shape(landmarks)

    # ── 10. 组装 FaceFeatures ──
    features = FaceFeatures(
        face_ratio      = raw_features["face_ratio"],
        jaw_cheek_ratio = raw_features["jaw_cheek_ratio"],
        forehead_ratio  = raw_features["forehead_ratio"],
        eye_distance    = eye_distance,
        nose_type       = nose_type,
        chin_shape      = chin_shape,
    )

    # ── 11. 脸型分类 ──
    face_shape, confidence = classify_face_shape(raw_features)

    # ── 12. 组装 FaceReport ──
    return FaceReport(
        face_shape = face_shape,
        confidence = confidence,
        features   = features,
    )


# ── 命令行快速测试 ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="脸型识别测试工具")
    parser.add_argument("image", help="图片路径")
    args = parser.parse_args()

    report = detect_face_shape(args.image)

    if report.error:
        print(f"❌ 错误: {report.error}")
    else:
        print(f"✅ 脸型: {report.face_shape.value}")
        print(f"   置信度: {report.confidence:.0%}")
        print(f"   脸长宽比: {report.features.face_ratio}")
        print(f"   下颌颧骨比: {report.features.jaw_cheek_ratio}")
        print(f"   额头颧骨比: {report.features.forehead_ratio}")
        print(f"   眼距: {report.features.eye_distance}")
        print(f"   鼻型: {report.features.nose_type}")
        print(f"   下巴: {report.features.chin_shape}")
