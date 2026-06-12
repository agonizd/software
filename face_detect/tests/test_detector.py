# face_detect/tests/test_detector.py
"""
A 模块单元测试 — 脸型识别

测试分为两类:
    1. TestMockDetector: 用 Mock 数据测试（不依赖真实图片，随时可跑）
    2. TestRealDetector: 用真实图片测试（需要 test_images/ 目录下有照片）

运行方式:
    pytest face_detect/tests/ -v
"""

import sys
import os

# 确保能导入项目根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from contracts import FaceShape, FaceReport, FaceFeatures, mock_detect_face_shape, mock_detect_face_shape_error


class TestMockDetector:
    """用 Mock 测试数据结构（不依赖真实图片）"""

    def test_mock_returns_face_report(self):
        """Mock 应该返回 FaceReport 类型"""
        report = mock_detect_face_shape()
        assert isinstance(report, FaceReport)

    def test_mock_oval_face(self):
        """默认返回鹅蛋脸"""
        report = mock_detect_face_shape(face_shape=FaceShape.OVAL)
        assert report.face_shape == FaceShape.OVAL
        assert report.confidence >= 0.85
        assert report.error is None

    def test_mock_all_6_shapes(self):
        """6 种脸型都有 Mock 数据"""
        for shape in FaceShape:
            report = mock_detect_face_shape(face_shape=shape)
            assert report.face_shape == shape

    def test_mock_error_report(self):
        """错误报告应包含 error 字段"""
        report = mock_detect_face_shape_error()
        assert report.error is not None
        assert report.confidence == 0.0

    def test_face_report_serialization(self):
        """FaceReport 序列化/反序列化一致性"""
        report = mock_detect_face_shape()
        d = report.to_dict()
        assert "face_shape" in d
        assert "confidence" in d
        assert d["face_shape"] == "鹅蛋脸"
        # 还原
        restored = FaceReport.from_dict(d)
        assert restored.face_shape == FaceShape.OVAL

    def test_face_report_is_reliable(self):
        """高置信度报告应可靠"""
        report = mock_detect_face_shape(face_shape=FaceShape.OVAL)
        assert report.is_reliable() is True

    def test_face_report_low_confidence_not_reliable(self):
        """低置信度报告不可靠"""
        report = mock_detect_face_shape(face_shape=FaceShape.DIAMOND)
        assert report.is_reliable() is False

    def test_error_report_not_reliable(self):
        """错误报告不可靠"""
        report = mock_detect_face_shape_error()
        assert report.is_reliable() is False


class TestLandmarks:
    """测试 landmarks.py 的特征提取"""

    def test_extract_features_returns_dict(self):
        """extract_face_features 应返回包含三个比例的字典"""
        from face_detect.landmarks import extract_face_features

        # 创建模拟的 landmark 对象
        class FakeLandmark:
            def __init__(self, x, y, z=0):
                self.x = x
                self.y = y
                self.z = z

        # 构造一个简单的"鹅蛋脸"关键点集
        # 脸宽 0.5, 脸长 0.7, 颧骨宽 0.45, 下颌宽 0.35, 额头宽 0.45
        fake_landmarks = [FakeLandmark(0, 0)] * 468
        fake_landmarks[234] = FakeLandmark(0.25, 0.5)   # 左颧骨
        fake_landmarks[454] = FakeLandmark(0.75, 0.5)   # 右颧骨
        fake_landmarks[10]  = FakeLandmark(0.5, 0.1)    # 发际线
        fake_landmarks[152] = FakeLandmark(0.5, 0.9)    # 下巴
        fake_landmarks[123] = FakeLandmark(0.22, 0.5)   # 左颧骨最外点
        fake_landmarks[352] = FakeLandmark(0.78, 0.5)   # 右颧骨最外点
        fake_landmarks[58]  = FakeLandmark(0.32, 0.75)  # 左下颌角
        fake_landmarks[288] = FakeLandmark(0.68, 0.75)  # 右下颌角
        fake_landmarks[54]  = FakeLandmark(0.27, 0.3)   # 左额
        fake_landmarks[284] = FakeLandmark(0.73, 0.3)   # 右额

        result = extract_face_features(fake_landmarks)

        assert "face_ratio" in result
        assert "jaw_cheek_ratio" in result
        assert "forehead_ratio" in result
        assert 0.5 <= result["face_ratio"] <= 2.5
        assert 0.3 <= result["jaw_cheek_ratio"] <= 1.5
        assert 0.5 <= result["forehead_ratio"] <= 1.5


class TestClassifier:
    """测试 classifier.py 的脸型分类"""

    def test_classify_long_face(self):
        """长脸：face_ratio > 1.7"""
        from face_detect.classifier import classify_face_shape
        shape, conf = classify_face_shape({
            "face_ratio": 1.85, "jaw_cheek_ratio": 0.75, "forehead_ratio": 0.90
        })
        assert shape == FaceShape.LONG

    def test_classify_square_face(self):
        """方脸：jaw_cheek_ratio > 0.9"""
        from face_detect.classifier import classify_face_shape
        shape, conf = classify_face_shape({
            "face_ratio": 1.25, "jaw_cheek_ratio": 0.95, "forehead_ratio": 1.02
        })
        assert shape == FaceShape.SQUARE

    def test_classify_round_face(self):
        """圆脸：face_ratio < 1.2 且 jaw_cheek_ratio > 0.8"""
        from face_detect.classifier import classify_face_shape
        shape, conf = classify_face_shape({
            "face_ratio": 1.05, "jaw_cheek_ratio": 0.92, "forehead_ratio": 0.98
        })
        assert shape == FaceShape.ROUND

    def test_classify_heart_face(self):
        """心形脸：forehead_ratio > 1.05 且 jaw_cheek_ratio < 0.7"""
        from face_detect.classifier import classify_face_shape
        shape, conf = classify_face_shape({
            "face_ratio": 1.35, "jaw_cheek_ratio": 0.62, "forehead_ratio": 1.12
        })
        assert shape == FaceShape.HEART

    def test_classify_diamond_face(self):
        """菱形脸：jaw_cheek_ratio < 0.7 且 forehead_ratio < 0.95"""
        from face_detect.classifier import classify_face_shape
        shape, conf = classify_face_shape({
            "face_ratio": 1.42, "jaw_cheek_ratio": 0.55, "forehead_ratio": 0.82
        })
        assert shape == FaceShape.DIAMOND

    def test_classify_oval_face(self):
        """鹅蛋脸：默认"""
        from face_detect.classifier import classify_face_shape
        shape, conf = classify_face_shape({
            "face_ratio": 1.48, "jaw_cheek_ratio": 0.78, "forehead_ratio": 1.00
        })
        assert shape == FaceShape.OVAL

    def test_confidence_range(self):
        """置信度在合理范围内"""
        from face_detect.classifier import classify_face_shape
        for fr, jcr, fhr in [
            (1.85, 0.75, 0.90),  # 长脸
            (1.25, 0.95, 1.02),  # 方脸
            (1.05, 0.92, 0.98),  # 圆脸
            (1.48, 0.78, 1.00),  # 鹅蛋脸
        ]:
            shape, conf = classify_face_shape({
                "face_ratio": fr, "jaw_cheek_ratio": jcr, "forehead_ratio": fhr
            })
            assert 0.0 <= conf <= 1.0, f"置信度 {conf} 超出范围"


class TestRealDetector:
    """真实图片测试（需要有 test_images/ 目录）"""

    def test_nonexistent_image_returns_error(self):
        """不存在的图片应返回错误报告"""
        from face_detect.detector import detect_face_shape
        report = detect_face_shape("nonexistent_image_12345.jpg")
        assert report.error is not None
        assert report.confidence == 0.0

    def test_real_image_returns_face_report(self):
        """真实图片应返回 FaceReport"""
        test_img = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "test_images", "face_01.jpg"
        )
        if not os.path.exists(test_img):
            pytest.skip("没有测试图片，跳过真实图片测试")

        from face_detect.detector import detect_face_shape
        report = detect_face_shape(test_img)
        assert isinstance(report, FaceReport)
        assert report.face_shape in FaceShape
        assert report.error is None
