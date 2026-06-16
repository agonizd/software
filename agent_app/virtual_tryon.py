"""
virtual_tryon.py — 虚拟试发型模块

将发型预览图叠加到用户上传的照片上，实现「发型虚拟试戴」效果。

算法:
  1. MediaPipe 检测用户照片中的人脸 → 获取 468 个关键点
  2. 同样检测发型预览图中的人脸 → 获取关键点
  3. 基于关键点确定两张脸的「面部映射」：缩放 + 平移
  4. 从发型预览图中裁剪出头发区域（眉毛以上 + 两侧延伸）
  5. 将头发区域缩放后贴合到用户照片的对应位置
  6. 底部边缘应用羽化渐变，自然融合

依赖: opencv-python, mediapipe, Pillow, numpy
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
from datetime import datetime

# 确保项目根目录可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MediaPipe landmarks key point indices (copied from face_detect/landmarks.py)
FACE_TOP = 10       # 发际线
FACE_BOTTOM = 152   # 下巴
FACE_WIDTH_L = 234  # 左颧骨
FACE_WIDTH_R = 454  # 右颧骨
CHEEK_L = 123       # 左颧骨最外
CHEEK_R = 352       # 右颧骨最外
JAW_L = 58          # 左下颌
JAW_R = 288         # 右下颌
FOREHEAD_L = 54     # 左额骨
FOREHEAD_R = 284    # 右额骨
EYEBROW_L_INNER = 55   # 左眉内侧
EYEBROW_R_INNER = 285  # 右眉内侧
LEFT_EYE_TOP = 159     # 左眼上缘
RIGHT_EYE_TOP = 386    # 右眼上缘

# 模型路径
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "face_landmarker.task",
)

_face_landmarker = None


def _get_landmarker():
    """懒加载 MediaPipe FaceLandmarker"""
    global _face_landmarker
    if _face_landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        _face_landmarker = FaceLandmarker.create_from_options(options)
    return _face_landmarker


def _read_image_rgb(image_path: str):
    """读取图片，返回 (RGB numpy array, (h,w))"""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, rgb.shape[:2]


def get_face_landmarks(image_path: str):
    """
    获取图片的人脸关键点（normalized coordinates 0~1）。

    Returns:
        list of NormalizedLandmark 或者 None
    """
    try:
        rgb, (h, w) = _read_image_rgb(image_path)
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _get_landmarker().detect(mp_image)
        if result.face_landmarks:
            return result.face_landmarks[0], (h, w)
        return None, (h, w)
    except Exception as e:
        print(f"[virtual_tryon] 人脸检测失败: {e}")
        return None, None


def _landmark_to_pixel(lm, w, h):
    """归一化坐标 → 像素坐标"""
    return int(lm.x * w), int(lm.y * h)


def _get_hair_region_from_preview(hair_image_path: str, face_landmarks, img_size):
    """
    从发型预览图中提取头发区域。

    头发区域 = 眉毛以上 + 额头 + 全部头顶 + 两侧延伸至颧骨外

    Returns:
        cropped hair region as PIL Image, or None
    """
    if face_landmarks is None:
        return None

    h, w = img_size
    lms = face_landmarks

    # ── 关键点像素坐标 ──
    top_y = _landmark_to_pixel(lms[FACE_TOP], w, h)[1]          # 发际线
    left_eye_top_y = _landmark_to_pixel(lms[LEFT_EYE_TOP], w, h)[1]   # 左眼上缘
    right_eye_top_y = _landmark_to_pixel(lms[RIGHT_EYE_TOP], w, h)[1] # 右眼上缘
    eyebrow_y = int((left_eye_top_y + right_eye_top_y) / 2) - int((eyebrow_y if 'eyebrow_y' in dir() else 0) or 0)

    # 眉毛位置：用眼睛上缘上方约 10% 脸高
    face_height = _landmark_to_pixel(lms[FACE_BOTTOM], w, h)[1] - _landmark_to_pixel(lms[FACE_TOP], w, h)[1]
    brow_y = max(0, top_y - int(face_height * 0.15))  # 眉毛大约在发际线下方 15%

    # 实际上，眉毛用 landmarks 更准：
    brow_left_y = _landmark_to_pixel(lms[EYEBROW_L_INNER], w, h)[1]
    brow_right_y = _landmark_to_pixel(lms[EYEBROW_R_INNER], w, h)[1]
    brow_avg_y = int((brow_left_y + brow_right_y) / 2)

    # 头发底部边界：眉毛上方一点点（给额头留空间）
    hair_bottom_y = max(0, brow_avg_y - int(face_height * 0.05))

    # 头发顶部边界：从图片顶部开始
    hair_top_y = 0

    # 头发水平边界：颧骨两侧再外扩 25%
    cheek_left_x = _landmark_to_pixel(lms[CHEEK_L], w, h)[0]
    cheek_right_x = _landmark_to_pixel(lms[CHEEK_R], w, h)[0]
    cheek_width = cheek_right_x - cheek_left_x
    padding = int(cheek_width * 0.30)

    hair_left_x = max(0, cheek_left_x - padding)
    hair_right_x = min(w, cheek_right_x + padding)

    # ── 裁剪 ──
    pil_img = Image.open(hair_image_path).convert("RGBA")
    hair_crop = pil_img.crop((hair_left_x, hair_top_y, hair_right_x, hair_bottom_y))

    return hair_crop, {
        "top": hair_top_y, "bottom": hair_bottom_y,
        "left": hair_left_x, "right": hair_right_x,
        "face_height": face_height,
        "brow_y": brow_avg_y,
        "cheek_left": cheek_left_x,
        "cheek_right": cheek_right_x,
    }


def _calculate_target_region(user_face_landmarks, user_img_size):
    """
    根据用户照片中的面部关键点，计算头发应该放置的目标区域。
    """
    h, w = user_img_size
    lms = user_face_landmarks

    top_y = _landmark_to_pixel(lms[FACE_TOP], w, h)[1]
    brow_left_y = _landmark_to_pixel(lms[EYEBROW_L_INNER], w, h)[1]
    brow_right_y = _landmark_to_pixel(lms[EYEBROW_R_INNER], w, h)[1]
    brow_avg_y = int((brow_left_y + brow_right_y) / 2)
    face_height = _landmark_to_pixel(lms[FACE_BOTTOM], w, h)[1] - top_y

    hair_bottom_y = max(0, brow_avg_y - int(face_height * 0.05))
    hair_top_y = 0  # 从图片顶部

    cheek_left_x = _landmark_to_pixel(lms[CHEEK_L], w, h)[0]
    cheek_right_x = _landmark_to_pixel(lms[CHEEK_R], w, h)[0]
    cheek_width = cheek_right_x - cheek_left_x
    padding = int(cheek_width * 0.30)

    hair_left_x = max(0, cheek_left_x - padding)
    hair_right_x = min(w, cheek_right_x + padding)

    return {
        "left": hair_left_x, "right": hair_right_x,
        "top": hair_top_y, "bottom": hair_bottom_y,
        "face_height": face_height,
    }


def _create_feathered_alpha_mask(width, height, feather_px=40):
    """
    创建羽化 alpha 遮罩：
    - 顶部 100% 不透明
    - 底部逐渐过渡到 0% 透明
    - 左右边缘也有轻微羽化
    """
    mask = np.ones((height, width), dtype=np.float32)

    # 底部羽化
    if height > feather_px:
        for y in range(height - feather_px, height):
            alpha = 1.0 - (y - (height - feather_px)) / feather_px
            alpha = max(0.0, alpha)
            mask[y, :] = alpha * mask[y, :]

    # 左右边缘轻微羽化（10% 宽度）
    side_feather = int(width * 0.08)
    if side_feather > 5:
        for x in range(side_feather):
            alpha = x / side_feather
            mask[:, x] = alpha * mask[:, x]
        for x in range(width - side_feather, width):
            alpha = (width - x) / side_feather
            mask[:, x] = alpha * mask[:, x]

    # 顶部微微羽化（5px）
    top_feather = min(5, height // 10)
    for y in range(top_feather):
        alpha = y / top_feather
        mask[y, :] = alpha * mask[y, :]

    return (mask * 255).astype(np.uint8)


def virtual_tryon(
    user_photo_path: str,
    hairstyle_image_path: str,
    output_dir: str = None,
) -> str:
    """
    将发型叠加到用户照片上。

    Args:
        user_photo_path: 用户上传的照片路径
        hairstyle_image_path: 发型预览图路径
        output_dir: 输出目录（默认在 hairstyle_db/tryon_results/）

    Returns:
        合成后的图片路径
    """
    # ── 1. 检测双方面部关键点 ──
    user_lms, user_size = get_face_landmarks(user_photo_path)
    hair_lms, hair_size = get_face_landmarks(hairstyle_image_path)

    if user_lms is None:
        raise ValueError("用户照片中未检测到人脸，请上传清晰的正面照片")
    if hair_lms is None:
        # 发型预览图可能检测不到人脸（如果发型遮挡太多），用默认比例裁剪
        print("[virtual_tryon] 发型预览图未检测到人脸，使用默认头部比例裁剪")
        hair_lms = None

    user_h, user_w = user_size
    hair_h, hair_w = hair_size if hair_size else (1024, 1024)

    # ── 2. 计算目标区域（用户照片中头发应放置的位置）──
    target = _calculate_target_region(user_lms, user_size)
    target_w = target["right"] - target["left"]
    target_h = target["bottom"] - target["top"]

    if target_w <= 0 or target_h <= 0:
        raise ValueError("无法确定用户面部区域")

    # ── 3. 提取发型预览图中的头发区域 ──
    hair_img = Image.open(hairstyle_image_path).convert("RGBA")

    if hair_lms is not None:
        # 用 landmarks 精确裁剪
        hair_info = _get_hair_region_from_preview(hairstyle_image_path, hair_lms, hair_size)
        if hair_info is not None:
            hair_crop, hinfo = hair_info
        else:
            hair_crop = None
    else:
        hair_crop = None

    if hair_crop is None:
        # 回退：按比例裁剪（取图片上部分作为头发）
        crop_ratio = 0.50  # 取上头 50%
        crop_bottom = int(hair_h * crop_ratio)
        # 头发区域略宽于面部（取中间 80% 宽度）
        margin = int(hair_w * 0.10)
        hair_crop = hair_img.crop((margin, 0, hair_w - margin, crop_bottom))

    # ── 4. 缩放头发区域到目标大小 ──
    hair_crop = hair_crop.resize((target_w, target_h), Image.LANCZOS)

    # ── 5. 创建羽化 alpha 遮罩 ──
    feather_px = max(20, int(target_h * 0.35))  # 底部 35% 羽化
    alpha_mask = _create_feathered_alpha_mask(target_w, target_h, feather_px)

    # 应用遮罩到头发图层的 alpha 通道
    hair_array = np.array(hair_crop)
    hair_alpha = hair_array[:, :, 3].astype(np.float32)
    hair_alpha = hair_alpha * (alpha_mask.astype(np.float32) / 255.0)
    hair_array[:, :, 3] = hair_alpha.astype(np.uint8)
    hair_layer = Image.fromarray(hair_array, "RGBA")

    # ── 6. 合成到用户照片 ──
    user_img = Image.open(user_photo_path).convert("RGBA")
    # 如果用户照片比目标区域窄，先略微放大
    composite = user_img.copy()

    # 粘贴头发图层
    composite.paste(hair_layer, (target["left"], target["top"]), hair_layer)

    # 转回 RGB
    result = Image.new("RGB", composite.size, (255, 255, 255))
    result.paste(composite, mask=composite.split()[3])

    # ── 7. 保存结果 ──
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "hairstyle_db", "tryon_results"
        )
    os.makedirs(output_dir, exist_ok=True)

    hairstyle_name = os.path.splitext(os.path.basename(hairstyle_image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"tryon_{hairstyle_name}_{timestamp}.jpg")
    result.save(output_path, "JPEG", quality=92)

    return output_path


def batch_tryon(
    user_photo_path: str,
    hairstyle_image_paths: list,
    output_dir: str = None,
) -> list:
    """
    批量为多个发型生成试戴效果。

    Returns:
        list of output paths
    """
    results = []
    for hair_path in hairstyle_image_paths:
        try:
            result_path = virtual_tryon(user_photo_path, hair_path, output_dir)
            results.append({"path": result_path, "error": None})
        except Exception as e:
            results.append({"path": None, "error": str(e)})
    return results


# ═══════════════════════════════════════════════════════════════
# 快速测试
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="虚拟试发型工具")
    parser.add_argument("user_photo", help="用户照片路径")
    parser.add_argument("hairstyle", help="发型预览图路径")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    args = parser.parse_args()

    try:
        result = virtual_tryon(args.user_photo, args.hairstyle, args.output)
        print(f"✅ 试戴效果已生成: {result}")
    except Exception as e:
        print(f"❌ 失败: {e}")
