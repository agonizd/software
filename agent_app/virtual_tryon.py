"""
virtual_tryon.py — 虚拟试发型模块（v8: Laplacian Pyramid 融合）

将发型预览图叠加到用户上传的照片上，实现自然的「发型虚拟试戴」效果。

算法（v8 改进）:
  1. MediaPipe 检测双方面部关键点（468点）
  2. 去除用户原有发型：HSV 肤色采样 + 发区检测 + 纹理填充
  3. 仿射变换：基于眼心 + 下巴的相似变换矩阵，对齐面部
  4. 提取参考发型：基于 landmarks 裁剪 + 软 alpha 遮罩
  5. 颜色匹配：Reinhard 色彩迁移（均值/标准差对齐）
  6. Laplacian Pyramid 多频融合：6层金字塔，频率分离混合
     避免"幽灵边缘"，不同频率不同融合宽度

依赖: opencv-python, mediapipe, numpy
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

# 确保项目根目录可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════
# MediaPipe Face Landmarks 关键点索引
# ═══════════════════════════════════════════════════════════════
FACE_TOP = 10        # 发际线中点
FACE_BOTTOM = 152    # 下巴中点
FACE_WIDTH_L = 234   # 左颧骨
FACE_WIDTH_R = 454   # 右颧骨
CHEEK_L = 123        # 左颧骨最外侧
CHEEK_R = 352        # 右颧骨最外侧
JAW_L = 58           # 左下颌角
JAW_R = 288          # 右下颌角
FOREHEAD_L = 54      # 左额角
FOREHEAD_R = 284     # 右额角
EYEBROW_L_INNER = 55    # 左眉内侧
EYEBROW_R_INNER = 285   # 右眉内侧
LEFT_EYE_TOP = 159      # 左眼上缘
RIGHT_EYE_TOP = 386     # 右眼上缘
EYE_LEFT = 33           # 左眼外角
EYE_RIGHT = 263         # 右眼外角
EYE_LEFT_INNER = 133    # 左眼内角
EYE_RIGHT_INNER = 362   # 右眼内角
NOSE_TIP = 4            # 鼻尖

# 面部轮廓点 (0-16)
FACE_OVAL = list(range(0, 17))

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
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions, RunningMode
        )
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
        (list of NormalizedLandmark, (h,w)) 或者 (None, None)
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


# ═══════════════════════════════════════════════════════════════
# 1. 高斯金字塔 & 拉普拉斯金字塔
# ═══════════════════════════════════════════════════════════════

def _build_gaussian_pyramid(img, levels):
    """
    构建高斯金字塔。

    Args:
        img: numpy array (H, W) 或 (H, W, C)
        levels: 金字塔层数

    Returns:
        list of numpy arrays, [G0, G1, ..., Gn]
    """
    pyramid = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def _build_laplacian_pyramid(img, levels):
    """
    构建拉普拉斯金字塔。

    Args:
        img: numpy array (H, W) 或 (H, W, C)
        levels: 金字塔层数

    Returns:
        list of numpy arrays, [L0, L1, ..., Ln]
        其中 L0..L(n-1) 是拉普拉斯层, Ln 是最后一层高斯层
    """
    gaussian = _build_gaussian_pyramid(img, levels)
    laplacian = []
    for i in range(levels):
        # G_i 上采样到 G_{i-1} 的大小，然后相减
        up = cv2.pyrUp(gaussian[i + 1])
        # 确保尺寸匹配
        h, w = gaussian[i].shape[:2]
        up = up[:h, :w]
        laplacian.append(cv2.subtract(gaussian[i], up))
    laplacian.append(gaussian[-1])
    return laplacian


def _reconstruct_from_laplacian(pyramid):
    """
    从拉普拉斯金字塔重建图像。

    Args:
        pyramid: 拉普拉斯金字塔 (L0..Ln), Ln 是最低分辨率的高斯层

    Returns:
        重建的全分辨率图像
    """
    result = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        result = cv2.pyrUp(result)
        h, w = pyramid[i].shape[:2]
        result = result[:h, :w]
        result = cv2.add(result, pyramid[i])
    return result


def _laplacian_blend(dst, src, mask, levels=6):
    """
    Laplacian Pyramid 多频率融合 —— 核心算法。

    原理:
      - 在不同频率层使用不同的融合宽度
      - 高频层（纹理细节）：窄过渡，保留清晰纹理
      - 低频层（颜色光照）：宽过渡，平滑颜色变化
      - 避免了"一刀切"羽化造成的鬼影或明显接缝

    Args:
        dst: 目标图像 (用户照片，已去除原发型), BGR, uint8
        src: 源图像 (参考发型), BGR, uint8
        mask: alpha 遮罩, float32 [0,1]
        levels: 金字塔层数

    Returns:
        融合后的 BGR 图像
    """
    # 构建掩膜高斯金字塔（控制每层的融合宽度）
    mask_gaussian = _build_gaussian_pyramid(mask, levels)

    # 构建源/目标拉普拉斯金字塔
    lp_dst = _build_laplacian_pyramid(dst.astype(np.float32), levels)
    lp_src = _build_laplacian_pyramid(src.astype(np.float32), levels)

    # 逐层融合
    lp_result = []
    for i in range(levels + 1):
        # 确保掩膜通道数匹配
        m = mask_gaussian[i]
        if len(m.shape) == 2 and len(lp_dst[i].shape) == 3:
            m = np.expand_dims(m, axis=2)

        # 每层按掩膜混合: L_result = L_src * mask + L_dst * (1 - mask)
        blended = lp_src[i] * m + lp_dst[i] * (1.0 - m)
        lp_result.append(blended)

    # 重建
    result = _reconstruct_from_laplacian(lp_result)
    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# 2. 用户原有发型去除
# ═══════════════════════════════════════════════════════════════

def _remove_user_hair(rgb, landmarks, img_size):
    """
    检测并去除用户原有发型。
    
    方法:
      1. 基于 landmarks 确定发区范围（眉毛以上 + 颞部）
      2. 从额头/脸颊采样肤色
      3. 用 HSV 颜色距离检测头发像素
      4. 用肤色均值 + Perlin-like 纹理填充发区

    Args:
        rgb: BGR uint8 numpy array
        landmarks: MediaPipe face landmarks
        img_size: (h, w)

    Returns:
        (cleaned_rgb, hair_mask) — BGR uint8, 和 bool mask
    """
    h, w = img_size
    lms = landmarks

    # ── 确定发区 ──
    brow_left_y = _landmark_to_pixel(lms[EYEBROW_L_INNER], w, h)[1]
    brow_right_y = _landmark_to_pixel(lms[EYEBROW_R_INNER], w, h)[1]
    brow_avg_y = int((brow_left_y + brow_right_y) / 2)
    face_height = _landmark_to_pixel(lms[FACE_BOTTOM], w, h)[1] - \
                  _landmark_to_pixel(lms[FACE_TOP], w, h)[1]
    # 发区底部：眉毛上方 5% 脸高
    hair_bottom = max(0, brow_avg_y - int(face_height * 0.06))

    # 发区水平范围：颧骨外扩 20%
    cheek_left = _landmark_to_pixel(lms[CHEEK_L], w, h)[0]
    cheek_right = _landmark_to_pixel(lms[CHEEK_R], w, h)[0]
    cheek_w = cheek_right - cheek_left
    hair_left = max(0, cheek_left - int(cheek_w * 0.25))
    hair_right = min(w, cheek_right + int(cheek_w * 0.25))

    # ── 采样肤色 ──
    # 额头中心区域（在眉毛上方，避开发际线）
    forehead_top = _landmark_to_pixel(lms[FACE_TOP], w, h)[1]
    sample_region = rgb[
        max(0, forehead_top):hair_bottom,
        cheek_left + cheek_w // 4:cheek_left + 3 * cheek_w // 4
    ]
    
    if sample_region.size == 0:
        # fallback: 用脸颊
        sample_region = rgb

    # 转换为 HSV 采样肤色
    sample_hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    skin_H_mean = np.median(sample_hsv[:, 0])
    skin_S_mean = np.median(sample_hsv[:, 1])
    skin_V_mean = np.median(sample_hsv[:, 2])

    # 肤色范围
    H_range = 20   # 色相范围
    S_range = 50   # 饱和度范围
    V_range = 60   # 明度范围

    # ── 检测头发像素 ──
    full_hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    skin_mask = (
        (np.abs(full_hsv[:, :, 0].astype(int) - int(skin_H_mean)) < H_range) &
        (np.abs(full_hsv[:, :, 1].astype(int) - int(skin_S_mean)) < S_range) &
        (np.abs(full_hsv[:, :, 2].astype(int) - int(skin_V_mean)) < V_range)
    )

    # 发区内的非肤色 = 头发
    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[:hair_bottom, hair_left:hair_right] = True
    hair_mask = region_mask & ~skin_mask

    # 形态学处理：保守操作，只填充最明显的头发像素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair_mask = cv2.morphologyEx(hair_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
    # 轻度膨胀（3px），刚好覆盖发丝边缘
    hair_mask = cv2.dilate(hair_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    hair_mask = hair_mask.astype(bool)

    # 额外约束：只在发际线上方填充，额头以下保持原样
    forehead_top = _landmark_to_pixel(lms[FACE_TOP], w, h)[1]
    safe_margin = max(5, int(face_height * 0.04))
    above_hairline = np.zeros((h, w), dtype=bool)
    above_hairline[:forehead_top + safe_margin, :] = True
    hair_mask = hair_mask & above_hairline

    # ── 肤色采样（额头+脸颊）──
    forehead_sample = rgb[
        forehead_top:brow_avg_y,
        cheek_left + cheek_w // 4:cheek_left + 3 * cheek_w // 4
    ]
    cheek_mid_h = brow_avg_y + int(face_height * 0.3)
    cheek_sample = rgb[
        cheek_mid_h:cheek_mid_h + int(face_height * 0.15),
        cheek_left + cheek_w // 8:cheek_left + 7 * cheek_w // 8
    ]

    all_skin = []
    for sample_region in [forehead_sample, cheek_sample]:
        if sample_region.size > 0:
            sf = sample_region.reshape(-1, 3)
            sh = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
            skin_in_sample = (
                (sh[:, :, 0] < 25) & (sh[:, :, 1] > 15) & (sh[:, :, 2] > 40)
            ).reshape(-1)
            if np.sum(skin_in_sample) > 10:
                all_skin.append(sf[skin_in_sample])
            else:
                all_skin.append(sf)
    if all_skin:
        skin_pixels = np.concatenate(all_skin, axis=0)
        skin_mean = np.mean(skin_pixels, axis=0).astype(np.float32)
        skin_std = np.std(skin_pixels, axis=0).astype(np.float32)
    else:
        skin_mean = np.array([200, 170, 150], dtype=np.float32)
        skin_std = np.array([20, 20, 20], dtype=np.float32)

    # ── 多层纹理填充发区 ──
    cleaned = rgb.copy()
    if np.any(hair_mask):
        # 第一层：肤色基底 + 较大随机噪声
        noise_level = skin_std * 0.6
        noise = np.random.normal(0, noise_level,
                                 (np.sum(hair_mask), 3)).astype(np.float32)
        fill_color = skin_mean.reshape(1, 3) + noise
        brightness_jitter = np.random.uniform(-8, 8, (np.sum(hair_mask), 1)).astype(np.float32)
        fill_color = fill_color + brightness_jitter
        fill_color = np.clip(fill_color, 0, 255).astype(np.uint8)
        cleaned[hair_mask] = fill_color

        # 第二层：边缘过渡（半透明混合）
        dilated = cv2.dilate(hair_mask.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        edge_region = dilated.astype(bool) & ~hair_mask
        if np.any(edge_region):
            edge_orig = cleaned.copy()[edge_region].astype(np.float32)
            edge_fill = np.full_like(edge_orig, skin_mean)
            cleaned[edge_region] = np.clip(
                edge_orig * 0.5 + edge_fill * 0.5, 0, 255
            ).astype(np.uint8)

        # 第三层：高斯模糊整合
        fill_region = dilated.astype(bool)
        blur_cleaned = cv2.GaussianBlur(cleaned, (7, 7), 0)
        cleaned[fill_region] = blur_cleaned[fill_region]

        # 第四层：微纹理（毛孔质感）
        micro = np.random.normal(0, 2.0, cleaned.shape).astype(np.float32)
        for c in range(3):
            cleaned[:, :, c] = np.clip(
                cleaned[:, :, c].astype(np.float32) + micro[:, :, c], 0, 255
            ).astype(np.uint8)

    return cleaned, hair_mask


# ═══════════════════════════════════════════════════════════════
# 3. 面部仿射变换
# ═══════════════════════════════════════════════════════════════

def _get_eye_center(lms, w, h):
    """计算眼睛中心坐标（左右眼各6个关键点的均值）"""
    # 左眼关键点: 33, 133, 157, 158, 159, 160 (? actually the indices vary)
    # 左眼轮廓: indices around eye
    left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 173, 246]
    right_eye_indices = [263, 362, 384, 385, 386, 387, 388, 398, 466]

    left_pts = np.array([_landmark_to_pixel(lms[i], w, h) for i in left_eye_indices])
    right_pts = np.array([_landmark_to_pixel(lms[i], w, h) for i in right_eye_indices])

    left_center = left_pts.mean(axis=0)
    right_center = right_pts.mean(axis=0)
    return left_center, right_center


def _compute_similarity_matrix(src_lms, dst_lms, src_size, dst_size):
    """
    计算从源脸到目标脸的相似变换矩阵（旋转+缩放+平移）。

    使用三个参考点：左眼中心、右眼中心、下巴中点。
    """
    sh, sw = src_size
    dh, dw = dst_size

    # 源面关键点
    src_left_eye, src_right_eye = _get_eye_center(src_lms, sw, sh)
    src_chin = _landmark_to_pixel(src_lms[FACE_BOTTOM], sw, sh)

    # 目标面关键点
    dst_left_eye, dst_right_eye = _get_eye_center(dst_lms, dw, dh)
    dst_chin = _landmark_to_pixel(dst_lms[FACE_BOTTOM], dw, dh)

    src_pts = np.float32([src_left_eye, src_right_eye, src_chin])
    dst_pts = np.float32([dst_left_eye, dst_right_eye, dst_chin])

    # 计算相似变换矩阵
    M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    if M is None:
        # fallback: 只用平移 + 缩放
        src_eye_center = (src_left_eye + src_right_eye) / 2
        dst_eye_center = (dst_left_eye + dst_right_eye) / 2
        src_eye_dist = np.linalg.norm(src_right_eye - src_left_eye)
        dst_eye_dist = np.linalg.norm(dst_right_eye - dst_left_eye)
        scale = dst_eye_dist / max(src_eye_dist, 1e-6)
        M = np.array([
            [scale, 0, dst_eye_center[0] - scale * src_eye_center[0]],
            [0, scale, dst_eye_center[1] - scale * src_eye_center[1]],
        ], dtype=np.float32)

    return M


# ═══════════════════════════════════════════════════════════════
# 4. 提取参考发型 + 软遮罩
# ═══════════════════════════════════════════════════════════════

def _extract_hair_from_reference(rgb, landmarks, img_size):
    """
    从参考图中提取头发区域，带软 alpha 遮罩。
    
    Returns:
        (hair_bgr, hair_alpha) — 都是 uint8, alpha 范围 [0,255]
    """
    h, w = img_size
    lms = landmarks

    if lms is None:
        # 无 landmarks → 取上半部分
        crop_h = h // 2
        hair = rgb[:crop_h, :].copy()
        alpha = np.ones((crop_h, w), dtype=np.float32)
        # 底部渐变
        feather = crop_h // 4
        for y in range(crop_h - feather, crop_h):
            alpha[y, :] = (crop_h - y) / feather
        return hair, (alpha * 255).astype(np.uint8)

    # 有 landmarks → 精确定义发区
    brow_left_y = _landmark_to_pixel(lms[EYEBROW_L_INNER], w, h)[1]
    brow_right_y = _landmark_to_pixel(lms[EYEBROW_R_INNER], w, h)[1]
    brow_avg_y = int((brow_left_y + brow_right_y) / 2)
    face_height = _landmark_to_pixel(lms[FACE_BOTTOM], w, h)[1] - \
                  _landmark_to_pixel(lms[FACE_TOP], w, h)[1]
    hair_bottom = max(0, brow_avg_y - int(face_height * 0.05))

    cheek_left = _landmark_to_pixel(lms[CHEEK_L], w, h)[0]
    cheek_right = _landmark_to_pixel(lms[CHEEK_R], w, h)[0]
    cheek_w = cheek_right - cheek_left
    hair_left = max(0, cheek_left - int(cheek_w * 0.4))
    hair_right = min(w, cheek_right + int(cheek_w * 0.4))
    hair_top = 0

    # 裁剪发区
    hair_crop = rgb[hair_top:hair_bottom, hair_left:hair_right].copy()
    crop_h, crop_w = hair_crop.shape[:2]

    if crop_h == 0 or crop_w == 0:
        return None, None

    # ── 创建软 alpha 遮罩 ──
    # 核心区域（头顶中部）：100% 不透明
    # 边缘区域（底边、侧边）：渐变到 0%
    alpha = np.ones((crop_h, crop_w), dtype=np.float32)

    # 底边渐变 — 在发区底部（眉毛上方）自然过渡
    feather_bottom = max(5, int(crop_h * 0.30))
    for y in range(crop_h - feather_bottom, crop_h):
        alpha[y, :] = float(crop_h - y) / feather_bottom

    # 左右边渐变
    feather_side = max(5, int(crop_w * 0.12))
    for x in range(feather_side):
        alpha[:, x] = np.minimum(alpha[:, x], float(x) / feather_side)
    for x in range(crop_w - feather_side, crop_w):
        alpha[:, x] = np.minimum(alpha[:, x], float(crop_w - x) / feather_side)

    # 顶部微渐变
    feather_top = min(3, crop_h // 20)
    if feather_top > 0:
        for y in range(feather_top):
            alpha[y, :] = np.minimum(alpha[y, :], float(y) / feather_top)

    # 使用 HSV 检测非头发内容（脸、背景），降低那些区域的 alpha
    hair_hsv = cv2.cvtColor(hair_crop, cv2.COLOR_BGR2HSV)
    # 脸部范围（典型亚洲肤色 HSV）: H [0, 25], S [20, 180], V [50, 255]
    face_mask = (
        (hair_hsv[:, :, 0] < 25) &
        (hair_hsv[:, :, 1] > 20) &
        (hair_hsv[:, :, 2] > 50)
    ).astype(np.float32)
    # 脸区域 alpha 降低
    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
    alpha = alpha * (1.0 - face_mask * 0.5)
    alpha = np.clip(alpha, 0, 1)

    alpha_u8 = (alpha * 255).astype(np.uint8)
    return hair_crop, alpha_u8


# ═══════════════════════════════════════════════════════════════
# 5. 颜色匹配 (Reinhard 色彩迁移)
# ═══════════════════════════════════════════════════════════════

def _match_colors(src, target_skin_pixels):
    """
    将源图像的颜色统计匹配到目标肤色。

    使用 Reinhard 色彩迁移：调整 src 的每个通道使其均值/标准差与目标肤色一致。

    Args:
        src: 源 BGR 图像 (头发), shape (H, W, 3)
        target_skin_pixels: 目标肤色像素数组, shape (N, 3)

    Returns:
        颜色调整后的 src
    """
    src_f = src.astype(np.float32)

    if target_skin_pixels is None or len(target_skin_pixels) == 0:
        return src

    src_pixels = src_f.reshape(-1, 3)

    src_mean = np.mean(src_pixels, axis=0)
    src_std = np.std(src_pixels, axis=0) + 1e-6
    dst_mean = np.mean(target_skin_pixels, axis=0)
    dst_std = np.std(target_skin_pixels, axis=0) + 1e-6

    result = (src_f - src_mean) * (dst_std / src_std) + dst_mean
    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# 6. 遮罩生成
# ═══════════════════════════════════════════════════════════════

def _create_blend_mask(hair_rgb, hair_alpha, reg_h, reg_w, user_hair_mask):
    """
    创建用于 Laplacian 融合的遮罩。

    Args:
        hair_rgb: 对齐后的头发 (BGR), shape (reg_h, reg_w, 3)
        hair_alpha: 头发软 alpha, float32 [0,1], shape (reg_h, reg_w)
        reg_h, reg_w: 区域尺寸
        user_hair_mask: 用户原发区遮罩 (bool, shape (reg_h, reg_w))

    Returns:
        blend_mask: float32, 范围 [0,1]
    """
    # 基础：参考头发 alpha
    blend_mask = hair_alpha.copy()

    # 检测参考头发中有实际内容的区域（非黑色/非背景）
    hair_gray = cv2.cvtColor(hair_rgb, cv2.COLOR_BGR2GRAY)
    hair_content = hair_gray > 15  # 有内容的像素

    # 在用户原发区 且 参考有内容的区域，提升遮罩
    if user_hair_mask is not None and np.any(user_hair_mask):
        # 只膨胀一点
        hair_region = cv2.dilate(
            user_hair_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ).astype(bool)
        # 与参考内容重叠的区域才设高 alpha
        combined = hair_region & hair_content
        blend_mask = np.maximum(blend_mask, combined.astype(np.float32) * 0.9)

    # 底部自然过渡（非线性，先慢后快）
    feather_px = max(15, int(reg_h * 0.35))
    for y in range(reg_h - feather_px, reg_h):
        t = float(reg_h - y) / feather_px
        # easeInOut: 开始时保持高，最后快速衰减
        fade = t * t * (3 - 2 * t)  # smoothstep
        blend_mask[y, :] = blend_mask[y, :] * fade

    # 左右边温和过渡
    feather_side = max(10, int(reg_w * 0.10))
    for x in range(feather_side):
        fade = (float(x) / feather_side) ** 0.6
        blend_mask[:, x] = np.minimum(blend_mask[:, x], fade)
    for x in range(reg_w - feather_side, reg_w):
        fade = (float(reg_w - x) / feather_side) ** 0.6
        blend_mask[:, x] = np.minimum(blend_mask[:, x], fade)

    return np.clip(blend_mask, 0, 1)


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def virtual_tryon(
    user_photo_path: str,
    hairstyle_image_path: str,
    output_dir: str = None,
) -> str:
    """
    将发型自然地叠加到用户照片上（v8: Laplacian Pyramid 融合）。

    算法流程:
      1. 检测双方面部关键点
      2. 去除用户原有发型（HSV 肤色采样 + 发区检测 + 纹理填充）
      3. 提取参考发型 + 软 alpha 遮罩
      4. 仿射变换对齐面部
      5. 颜色匹配（Reinhard 色彩迁移）
      6. Laplacian Pyramid 6层融合
      7. 保存结果

    Args:
        user_photo_path: 用户上传的照片路径
        hairstyle_image_path: 发型预览图路径
        output_dir: 输出目录

    Returns:
        合成后的图片路径
    """
    # ── 1. 检测关键点 ──
    user_lms, user_size = get_face_landmarks(user_photo_path)
    hair_lms, hair_size = get_face_landmarks(hairstyle_image_path)

    if user_lms is None:
        raise ValueError("用户照片中未检测到人脸，请上传清晰的正面照片")

    user_h, user_w = user_size
    hair_h, hair_w = hair_size if hair_size else (1024, 1024)
    hair_lms_available = hair_lms is not None

    # 加载图像为 BGR (OpenCV 格式)
    user_bgr = cv2.imread(user_photo_path)
    hair_bgr = cv2.imread(hairstyle_image_path)

    if user_bgr is None:
        raise FileNotFoundError(f"无法读取用户照片: {user_photo_path}")
    if hair_bgr is None:
        raise FileNotFoundError(f"无法读取发型图: {hairstyle_image_path}")

    # ── 2. 去除用户原有发型 ──
    cleaned_user, user_hair_mask = _remove_user_hair(user_bgr, user_lms, user_size)

    # ── 3. 提取参考发型 ──
    hair_extract, hair_alpha = _extract_hair_from_reference(
        hair_bgr, hair_lms if hair_lms_available else None, hair_size
    )

    if hair_extract is None or hair_alpha is None:
        # fallback: 取参考图上半部分
        crop_h = hair_h // 2
        hair_extract = hair_bgr[:crop_h, :].copy()
        hair_alpha = np.ones((crop_h, hair_w), dtype=np.uint8) * 255
        # 底部渐变
        feather = crop_h // 3
        for y in range(crop_h - feather, crop_h):
            hair_alpha[y, :] = int(255 * (crop_h - y) / feather)

    # ── 4. 计算目标区域 ──
    brow_left_y = _landmark_to_pixel(user_lms[EYEBROW_L_INNER], user_w, user_h)[1]
    brow_right_y = _landmark_to_pixel(user_lms[EYEBROW_R_INNER], user_w, user_h)[1]
    brow_avg_y = int((brow_left_y + brow_right_y) / 2)
    face_top_y = _landmark_to_pixel(user_lms[FACE_TOP], user_w, user_h)[1]
    face_height = _landmark_to_pixel(user_lms[FACE_BOTTOM], user_w, user_h)[1] - face_top_y
    hair_bottom_y = max(0, brow_avg_y - int(face_height * 0.06))

    cheek_left_x = _landmark_to_pixel(user_lms[CHEEK_L], user_w, user_h)[0]
    cheek_right_x = _landmark_to_pixel(user_lms[CHEEK_R], user_w, user_h)[0]
    cheek_w = cheek_right_x - cheek_left_x
    hair_left_x = max(0, cheek_left_x - int(cheek_w * 0.35))
    hair_right_x = min(user_w, cheek_right_x + int(cheek_w * 0.35))
    hair_top_y = 0

    target = {
        "left": hair_left_x, "right": hair_right_x,
        "top": hair_top_y, "bottom": hair_bottom_y,
    }
    target_w = hair_right_x - hair_left_x
    target_h = hair_bottom_y - hair_top_y

    if target_w <= 10 or target_h <= 10:
        raise ValueError("无法确定面部区域，请使用正面照片")

    # ── 5. 仿射变换对齐 ──
    if hair_lms_available:
        M = _compute_similarity_matrix(hair_lms, user_lms, hair_size, user_size)
    else:
        # fallback: 简单平移+缩放
        src_eye_x = hair_w // 2
        src_eye_y = hair_h // 3
        dst_eye_x = (cheek_left_x + cheek_right_x) // 2
        dst_eye_y = brow_avg_y
        scale = target_w / hair_w
        M = np.array([
            [scale, 0, dst_eye_x - scale * src_eye_x],
            [0, scale, dst_eye_y - scale * src_eye_y],
        ], dtype=np.float32)

    # 将头发变形到目标区域
    hair_warped = cv2.warpAffine(
        hair_extract, M, (user_w, user_h),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
    )
    alpha_warped = cv2.warpAffine(
        hair_alpha, M, (user_w, user_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    alpha_warped = alpha_warped.astype(np.float32) / 255.0

    # 裁剪到目标区域
    hair_cropped = hair_warped[target["top"]:target["bottom"],
                               target["left"]:target["right"]]
    alpha_cropped = alpha_warped[target["top"]:target["bottom"],
                                  target["left"]:target["right"]]
    user_cropped = cleaned_user[target["top"]:target["bottom"],
                                 target["left"]:target["right"]]

    # 修正尺寸（warpAffine 可能产生微小偏差）
    actual_h, actual_w = min(hair_cropped.shape[0], user_cropped.shape[0], alpha_cropped.shape[0]), \
                          min(hair_cropped.shape[1], user_cropped.shape[1], alpha_cropped.shape[1])
    hair_cropped = hair_cropped[:actual_h, :actual_w]
    alpha_cropped = alpha_cropped[:actual_h, :actual_w]
    user_cropped = user_cropped[:actual_h, :actual_w]

    if actual_h < 5 or actual_w < 5:
        raise ValueError("对齐后的区域太小")

    # ── 6. 颜色匹配 ──
    # 从 user_cropped 中采样肤色像素（底部接近眉毛的皮肤区域）
    patch_h, patch_w = user_cropped.shape[:2]
    skin_sample_h = max(5, patch_h // 5)
    skin_region = user_cropped[-skin_sample_h:, patch_w // 4:3 * patch_w // 4]

    skin_pixels = []
    if skin_region.size > 0:
        skin_hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)
        skin_pixel_mask = (
            (skin_hsv[:, :, 0] < 25) &
            (skin_hsv[:, :, 1] > 20) &
            (skin_hsv[:, :, 2] > 50)
        )
        if np.any(skin_pixel_mask):
            skin_pixels = skin_region[skin_pixel_mask].astype(np.float32)
        else:
            skin_pixels = skin_region.reshape(-1, 3).astype(np.float32)
    else:
        skin_pixels = user_cropped.reshape(-1, 3).astype(np.float32)

    # 对头发做颜色匹配
    hair_cropped = _match_colors(hair_cropped, skin_pixels)

    # ── 7. 创建融合遮罩 ──
    # 提取用户发区遮罩的对应区域
    user_hair_cropped = None
    if user_hair_mask is not None and np.any(user_hair_mask):
        user_hair_cropped = user_hair_mask[target["top"]:target["top"] + actual_h,
                                            target["left"]:target["left"] + actual_w]

    region_mask = _create_blend_mask(
        hair_cropped, alpha_cropped, actual_h, actual_w, user_hair_cropped
    )

    # ── 8. Laplacian Pyramid 融合 ──
    # 确保尺寸是偶数（pyrDown/pyrUp 要求）
    def _pad_to_even(img_h, img_w):
        return img_h - (img_h % 2), img_w - (img_w % 2)

    even_h, even_w = _pad_to_even(actual_h, actual_w)

    dst_patch = user_cropped[:even_h, :even_w]
    src_patch = hair_cropped[:even_h, :even_w]
    mask_patch = region_mask[:even_h, :even_w]

    pyramid_levels = min(6, int(np.log2(min(even_h, even_w))) - 1)
    pyramid_levels = max(3, pyramid_levels)

    blended_patch = _laplacian_blend(dst_patch, src_patch, mask_patch, pyramid_levels)

    # 替换回全图
    result = cleaned_user.copy()
    result[target["top"]:target["top"] + even_h,
           target["left"]:target["left"] + even_w] = blended_patch

    # ── 9. 后处理：微调 ──
    # 过渡区轻微模糊
    transition = np.abs(region_mask[:even_h, :even_w] - 0.5) < 0.3
    if np.any(transition):
        blur_result = cv2.GaussianBlur(result, (5, 5), 0)
        result[target["top"]:target["top"] + even_h,
               target["left"]:target["left"] + even_w][transition] = \
            blur_result[target["top"]:target["top"] + even_h,
                        target["left"]:target["left"] + even_w][transition]

    # ── 10. 保存 ──
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "hairstyle_db", "tryon_results"
        )
    os.makedirs(output_dir, exist_ok=True)

    hairstyle_name = os.path.splitext(os.path.basename(hairstyle_image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"tryon_{hairstyle_name}_{timestamp}.jpg")
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return output_path


def batch_tryon(
    user_photo_path: str,
    hairstyle_image_paths: list,
    output_dir: str = None,
) -> list:
    """批量为多个发型生成试戴效果。"""
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
    parser = argparse.ArgumentParser(description="虚拟试发型工具 v8")
    parser.add_argument("user_photo", help="用户照片路径")
    parser.add_argument("hairstyle", help="发型预览图路径")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    args = parser.parse_args()

    try:
        result = virtual_tryon(args.user_photo, args.hairstyle, args.output)
        print(f"✅ 试戴效果已生成: {result}")
    except Exception as e:
        print(f"❌ 失败: {e}")
