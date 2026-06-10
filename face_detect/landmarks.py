# face_detect/landmarks.py
"""
从 MediaPipe FaceMesh 468 关键点中提取面部比例。

关键点索引（来自 contracts.py 文档 + MediaPipe FaceMesh 标准定义）:
    脸宽(颧骨间距):  234(左颧骨) ↔ 454(右颧骨)
    脸长(发际线→下巴): 10(发际线) ↔ 152(下巴)
    颧骨宽:          123(左颧骨最外点) ↔ 352(右颧骨最外点)
    下颌宽:          58(左下颌角) ↔ 288(右下颌角)
    额头宽:          54(左额骨) ↔ 284(右额骨)

输出三个核心比例:
    face_ratio       = 脸长 / 脸宽  （分类主指标）
    jaw_cheek_ratio  = 下颌宽 / 颧骨宽
    forehead_ratio   = 额头宽 / 颧骨宽
"""

import math


# ── 关键点索引常量 ──────────────────────────────────────────────

# 脸宽（左右颧骨）
FACE_WIDTH_L = 234
FACE_WIDTH_R = 454

# 脸长（发际线 → 下巴）
FACE_TOP     = 10
FACE_BOTTOM  = 152

# 颧骨宽（左右颧骨最外点）
CHEEK_L = 123
CHEEK_R = 352

# 下颌宽（左右下颌角）
JAW_L = 58
JAW_R = 288

# 额头宽（左右额骨）
FOREHEAD_L = 54
FOREHEAD_R = 284

# 眼距相关
LEFT_EYE_INNER  = 133
RIGHT_EYE_INNER = 362
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263

# 鼻子相关
NOSE_TIP   = 1
NOSE_LEFT  = 48
NOSE_RIGHT = 278

# 下巴相关
CHIN_CENTER = 152
CHIN_LEFT   = 58
CHIN_RIGHT  = 288


def euclidean_distance(p1, p2) -> float:
    """计算两个关键点的欧氏距离（归一化坐标）"""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def extract_face_features(landmarks) -> dict:
    """
    从 MediaPipe landmarks 提取面部比例数据。

    参数:
        landmarks: face_mesh.process() 返回的 landmark 列表
                   每个元素有 .x, .y, .z 属性（归一化 0~1）

    返回:
        dict: {
            "face_ratio":      float,  # 脸长/脸宽（核心分类指标）
            "jaw_cheek_ratio": float,  # 下颌宽/颧骨宽
            "forehead_ratio":  float,  # 额头宽/颧骨宽
        }
    """
    # ── 提取各部位距离 ──
    face_width   = euclidean_distance(landmarks[FACE_WIDTH_L], landmarks[FACE_WIDTH_R])
    face_height  = euclidean_distance(landmarks[FACE_TOP],    landmarks[FACE_BOTTOM])
    cheek_width  = euclidean_distance(landmarks[CHEEK_L],     landmarks[CHEEK_R])
    jaw_width    = euclidean_distance(landmarks[JAW_L],       landmarks[JAW_R])
    forehead_w   = euclidean_distance(landmarks[FOREHEAD_L],  landmarks[FOREHEAD_R])

    # ── 计算三个核心比例 ──
    # 注意：face_ratio = 脸长/脸宽（参考 contracts.py 中的阈值表）
    face_ratio      = face_height / face_width if face_width > 0 else 1.0
    jaw_cheek_ratio = jaw_width   / cheek_width if cheek_width > 0 else 1.0
    forehead_ratio  = forehead_w  / cheek_width if cheek_width > 0 else 1.0

    return {
        "face_ratio":      round(face_ratio,      3),
        "jaw_cheek_ratio": round(jaw_cheek_ratio, 3),
        "forehead_ratio":  round(forehead_ratio,  3),
    }


def infer_eye_distance(landmarks) -> str:
    """
    推断眼距类型。

    逻辑：两眼内角距离 / 脸宽
    - < 0.25 → "偏窄"
    - 0.25~0.35 → "适中"
    - > 0.35 → "偏宽"
    """
    eye_inner_dist = euclidean_distance(landmarks[LEFT_EYE_INNER], landmarks[RIGHT_EYE_INNER])
    face_width = euclidean_distance(landmarks[FACE_WIDTH_L], landmarks[FACE_WIDTH_R])

    if face_width == 0:
        return "适中"

    ratio = eye_inner_dist / face_width
    if ratio < 0.25:
        return "偏窄"
    elif ratio > 0.35:
        return "偏宽"
    else:
        return "适中"


def infer_nose_type(landmarks) -> str:
    """
    推断鼻型。

    逻辑：鼻翼宽 / 脸宽
    - < 0.20 → "直挺"
    - 0.20~0.28 → "圆润"
    - > 0.28 → "偏宽"
    """
    nose_width = euclidean_distance(landmarks[NOSE_LEFT], landmarks[NOSE_RIGHT])
    face_width = euclidean_distance(landmarks[FACE_WIDTH_L], landmarks[FACE_WIDTH_R])

    if face_width == 0:
        return "直挺"

    ratio = nose_width / face_width
    if ratio > 0.28:
        return "偏宽"
    elif ratio > 0.20:
        return "圆润"
    else:
        return "直挺"


def infer_chin_shape(landmarks) -> str:
    """
    推断下巴形状。

    逻辑：比较下巴中心与下颌角的相对位置
    - 下巴到下颌角连线形成的角度较大 → "方"
    - 中等 → "圆"
    - 较小 → "尖"
    """
    chin_center_y = landmarks[CHIN_CENTER].y
    jaw_left_y    = landmarks[JAW_L].y
    jaw_right_y   = landmarks[JAW_R].y

    jaw_avg_y = (jaw_left_y + jaw_right_y) / 2

    if jaw_avg_y == chin_center_y:
        return "圆"

    # 下巴相对下颌角的纵向落差
    chin_drop = chin_center_y - jaw_avg_y

    # 下颌宽度
    jaw_width = euclidean_distance(landmarks[JAW_L], landmarks[JAW_R])

    if jaw_width == 0:
        return "圆"

    # 下巴尖锐度 = 纵向落差 / 横向宽度
    chin_sharpness = abs(chin_drop) / jaw_width

    if chin_sharpness > 0.15:
        return "尖"
    elif chin_sharpness < 0.05:
        return "方"
    else:
        return "圆"
