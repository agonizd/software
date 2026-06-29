"""
app.py — Streamlit 前端入口（E 模块）

自动检测运行模式：
- 检测到 DASHSCOPE_API_KEY 环境变量 + LangChain 已安装 → 阿里云百炼 Agent 模式
- 否则 → Mock 模式（离线规则引擎，无需任何配置）

启动方式：
    streamlit run agent_app/app.py
"""

import streamlit as st
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent_app.tools import (
    detect_face_shape_tool,
    search_hairstyles_tool,
    recommend_tool,
    generate_report_tool,
    TOOL_DEFINITIONS,
)
from agent_app.agent import MockAgent
from agent_app.prompts import SYSTEM_PROMPT, SIMPLE_PROMPT
from agent_app.virtual_tryon import virtual_tryon, batch_tryon

# ── 尝试导入 B 的 UI 组件 ──
B_COMPONENTS_AVAILABLE = False
try:
    pass
except ImportError:
    pass

# ── 尝试导入 LangChain（Agent 模式）──
LANGCHAIN_AVAILABLE = False
DASHSCOPE_AVAILABLE = False
try:
    from langchain_community.chat_models import ChatTongyi
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.tools import Tool
    LANGCHAIN_AVAILABLE = True
    DASHSCOPE_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="发型顾问 AI",
    page_icon="✿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# 全局样式 — 暖色调 · 柔和自然主题
# ============================================================
st.markdown("""
<style>
/* ── 1. 根背景：强制覆盖 Streamlit 主题 ── */
html, body, .stApp, #root {
    background-color: #FAF7F4 !important;
}
footer, #MainMenu {
    visibility: hidden !important;
}

/* ── 2. 主内容区：所有中间容器全部收归暖色 ── */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
section[data-testid="stSidebar"] + section,
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
.block-container,
div[data-testid="stBlock"] {
    background-color: #FAF7F4 !important;
}

/* ── 3. 侧边栏：奶油底色 × 独立配色 ── */
[data-testid="stSidebar"] {
    background-color: #F5F0EB !important;
    border-right: 1px solid #EDE3DA !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #F5F0EB !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #3D2C1E !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #8C7268 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #EDE3DA !important;
}

/* ── 4. 聊天消息：白色气泡 + 软边 ── */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #EDE3DA !important;
    border-radius: 18px !important;
    box-shadow: none !important;
}

/* ── 5. 按钮柔和化 ── */
.stButton > button {
    border-radius: 14px !important;
    border: 1px solid #EDE3DA !important;
    background: #FFFFFF !important;
    color: #3D2C1E !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #FDF0E8 !important;
    border-color: #C17B4A !important;
    color: #C17B4A !important;
}

/* ── 6. 文件上传：虚线框 + 居中 ── */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1.5px dashed #D4C8BC !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #C17B4A !important;
}
[data-testid="stFileUploader"] button {
    background: #FFFFFF !important;
    border: 1px solid #EDE3DA !important;
    color: #3D2C1E !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #FDF0E8 !important;
    border-color: #C17B4A !important;
}

/* ── 7. 输入框 ── */
input[type="text"],
textarea,
[data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    border: 1px solid #EDE3DA !important;
    background: #FFFFFF !important;
    color: #3D2C1E !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: 16px !important;
}

/* ── 8. 分割线 / 提示框 / 展开 ── */
hr, .stDivider {
    border-color: #EDE3DA !important;
}
.stAlert,
[data-testid="stExpander"] details {
    border-radius: 14px !important;
}
[data-testid="stExpander"] details {
    border: 1px solid #EDE3DA !important;
}

/* ── 9. 滚动条 ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #FAF7F4; }
::-webkit-scrollbar-thumb { background: #D1C4B6; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #B0A098; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State 初始化
# ============================================================
def init_session():
    # 自动检测运行模式：有 API Key 且 LangChain 可用 → Agent，否则 → Mock
    if "mode" not in st.session_state:
        st.session_state.mode = _detect_mode()

    defaults = {
        "messages": [],
        "face_report": None,
        "last_recommendations": None,
        "uploaded_image_path": None,
        "agent_executor": None,
        "tryon_results": {},
        "pending_tryon": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _detect_mode() -> str:
    """自动检测：有 DASHSCOPE_API_KEY 环境变量 且 LangChain 已安装 → agent，否则 mock"""
    if DASHSCOPE_AVAILABLE and os.getenv("DASHSCOPE_API_KEY", "").strip():
        return "agent"
    return "mock"


init_session()


# ============================================================
# 工具函数
# ============================================================
FACE_EMOJI = {
    "鹅蛋脸": "🥚", "圆脸": "🌕", "方脸": "⬜",
    "长脸": "📏", "心形脸": "🌸", "菱形脸": "💎",
}

STYLE_TAGS = [
    ("干练通勤", "👔"), ("甜美约会", "🌸"), ("复古港风", "📼"),
    ("酷飒少年", "✦"), ("自然清新", "🌿"), ("优雅知性", "🎀"),
]


def _resolve_image_path(image_url: str) -> str:
    if not image_url:
        return ""
    candidate = os.path.join(PROJECT_ROOT, "hairstyle_db", image_url)
    if os.path.isfile(candidate):
        return candidate
    candidate2 = os.path.join(PROJECT_ROOT, image_url)
    if os.path.isfile(candidate2):
        return candidate2
    return ""


def _on_photo_upload():
    f = st.session_state.get("sidebar_uploader")
    if not f:
        return
    save_dir = Path(PROJECT_ROOT) / "uploaded_photos"
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = str(save_dir / f"upload_{timestamp}.jpg")
    with open(img_path, "wb") as fh:
        fh.write(f.getbuffer())
    st.session_state.uploaded_image_path = img_path

    try:
        from face_detect.detector import detect_face_shape
        report = detect_face_shape(img_path)
        if report.error:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"照片上传了，不过人脸识别遇到了一点问题：{report.error}\n\n建议换一张光线更好的正面照试试 ☀️",
            })
        elif report.face_shape is not None:
            shape = report.face_shape.value
            conf = report.confidence
            st.session_state.face_report = {
                "face_shape": shape,
                "confidence": conf,
                "features": {
                    "face_ratio": report.features.face_ratio,
                    "jaw_cheek_ratio": report.features.jaw_cheek_ratio,
                    "forehead_ratio": report.features.forehead_ratio,
                    "eye_distance": report.features.eye_distance,
                    "nose_type": report.features.nose_type,
                    "chin_shape": report.features.chin_shape,
                },
            }
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"照片收到啦 ✿\n\n"
                    f"分析结果显示你是 **{shape}** {FACE_EMOJI.get(shape, '')}，"
                    f"置信度 **{conf:.0%}**\n\n"
                    f"现在告诉我你想要什么风格，我来为你推荐最适合的发型 🌿"
                ),
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "照片上传成功，但没有识别到清晰的正面人脸。建议拍一张正面、免冠、光线充足的照片重新上传 ☀️",
            })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"照片已保存，脸型分析暂时遇到了些问题（{e}）。你可以直接告诉我你的脸型，或者稍后重试 🌿",
        })


# ============================================================
# 侧边栏 — 暖奶油风格
# ============================================================
def render_sidebar():
    with st.sidebar:
        # 顶部品牌区
        st.markdown("""
        <div style="padding: 20px 4px 16px 4px;">
            <div style="font-size: 28px; margin-bottom: 6px;">✿</div>
            <div style="font-size: 17px; font-weight: 500; color: #3D2C1E; letter-spacing: 0.3px;">
                发型顾问
            </div>
            <div style="font-size: 12px; color: #8C7268; margin-top: 2px; line-height: 1.5;">
                你的专属智能造型师
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # 上传照片
        st.markdown('<div style="font-size:12px; color:#8C7268; margin-bottom:6px;">上传正面照</div>', unsafe_allow_html=True)
        st.file_uploader(
            "选择照片",
            type=["jpg", "jpeg", "png"],
            key="sidebar_uploader",
            on_change=_on_photo_upload,
            help="正面、免冠、光线充足效果最佳",
            label_visibility="collapsed",
        )

        if st.session_state.uploaded_image_path and os.path.isfile(st.session_state.uploaded_image_path):
            st.image(st.session_state.uploaded_image_path, width=160)

        st.divider()

        # 脸型状态
        if st.session_state.face_report:
            fr = st.session_state.face_report
            shape = fr["face_shape"]
            conf = fr["confidence"]
            emoji = FACE_EMOJI.get(shape, "")
            st.markdown(f"""
            <div style="background:#FDF0E8; border-radius:14px; padding:12px 14px;
            border:1px solid #E8C9A8;">
                <div style="font-size:13px; font-weight:500; color:#3D2C1E;">{emoji} {shape}</div>
                <div style="font-size:11px; color:#8C7268; margin-top:2px;">置信度 {conf:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#F5F0EB; border-radius:14px; padding:12px 14px;
            border:1px solid #EDE3DA;">
                <div style="font-size:12px; color:#8C7268;">暂未分析脸型</div>
                <div style="font-size:11px; color:#B0A098; margin-top:2px;">上传照片后自动识别</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # API Key（可选，自动降级 Mock）
        with st.expander("⚙ 高级设置", expanded=False):
            api_key = st.text_input(
                "阿里云百炼 API Key",
                type="password",
                value=os.getenv("DASHSCOPE_API_KEY", ""),
                placeholder="sk-...",
                label_visibility="collapsed",
            )
            if api_key:
                os.environ["DASHSCOPE_API_KEY"] = api_key
                if DASHSCOPE_AVAILABLE:
                    st.session_state.mode = "agent"
                    st.caption("已启用 通义千问 Agent")
                else:
                    st.caption("未安装 dashscope，使用 Mock 模式")
            else:
                if st.session_state.mode == "agent":
                    st.session_state.mode = "mock"
                if DASHSCOPE_AVAILABLE:
                    st.caption("未设置 Key，使用 Mock 模式")
                else:
                    st.caption("未安装 dashscope，使用 Mock 模式")

        st.divider()

        # 底部统计 & 重置
        msg_count = len(st.session_state.messages)
        rec_count = len(st.session_state.last_recommendations) if st.session_state.last_recommendations else 0
        st.caption(f"{msg_count} 条消息  ·  {rec_count} 条推荐")

        if st.button("重新开始", use_container_width=True):
            for key in ["messages", "face_report", "last_recommendations",
                        "uploaded_image_path", "agent_executor", "tryon_results"]:
                st.session_state[key] = [] if key == "messages" else ({} if key == "tryon_results" else None)
            st.rerun()

        st.caption("v1.0")


# ============================================================
# Landing Hero — 柔和欢迎页
# ============================================================
def render_landing_hero():
    st.markdown("""
    <div style="text-align:center; padding: 56px 16px 40px 16px;">
        <div style="font-size: 52px; margin-bottom: 14px; color: #C17B4A;">✿</div>
        <div style="font-size: 26px; font-weight: 500; color: #3D2C1E;
        letter-spacing: 0.5px; margin-bottom: 10px;">
            找到属于你的发型
        </div>
        <div style="font-size: 14px; color: #8C7268; max-width: 380px;
        margin: 0 auto 36px auto; line-height: 1.8;">
            上传一张照片，AI 分析脸型，为你精准推荐<br>
            最适合的发型风格
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 风格标签
    st.markdown('<div style="text-align:center; margin-bottom:8px; font-size:12px; color:#8C7268;">选择你喜欢的风格开始</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    clicked_tag = None
    for i, (tag, icon) in enumerate(STYLE_TAGS):
        with cols[i]:
            if st.button(f"{icon} {tag}", key=f"tag_{tag}", use_container_width=True):
                clicked_tag = tag
    return clicked_tag


# ============================================================
# 推荐卡片 — 柔和暖色风
# ============================================================
def fallback_recommendation_card(rec: dict, index: int):
    hs = rec.get("hairstyle", {})
    score = rec.get("score", 0)
    reasons = rec.get("reasons", [])
    details = rec.get("details", {})
    hairstyle_id = hs.get("id", "")
    hairstyle_name = hs.get("name", "")

    # 匹配度颜色（暖色系）
    if score >= 0.85:
        bar_color = "#7A9E7E"; badge_bg = "#EEF5EE"; badge_text = "完美匹配"
    elif score >= 0.7:
        bar_color = "#C17B4A"; badge_bg = "#FDF0E8"; badge_text = "高度推荐"
    elif score >= 0.55:
        bar_color = "#B08B6E"; badge_bg = "#F5EDE5"; badge_text = "可以尝试"
    else:
        bar_color = "#A09080"; badge_bg = "#F0EBEA"; badge_text = "仅供参考"

    img_path = _resolve_image_path(hs.get("image_url", ""))

    with st.container():
        # 顶部徽章行
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <div style="width:28px; height:28px; border-radius:50%;
            background:{badge_bg}; border:1px solid {bar_color}30;
            display:flex; align-items:center; justify-content:center;
            font-size:12px; font-weight:500; color:{bar_color};">
            {index}</div>
            <div style="font-size:16px; font-weight:500; color:#3D2C1E;">{hairstyle_name}</div>
            <div style="margin-left:auto; background:{badge_bg}; color:{bar_color};
            font-size:11px; padding:3px 10px; border-radius:20px;
            border:1px solid {bar_color}30;">
            {badge_text}</div>
        </div>
        """, unsafe_allow_html=True)

        col_img, col_info = st.columns([1, 2])
        with col_img:
            if img_path and os.path.isfile(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.markdown(f"""
                <div style="aspect-ratio:3/4; background:#F5F0EB;
                border-radius:16px; display:flex; align-items:center;
                justify-content:center; font-size:32px; color:#C17B4A;">✿</div>
                """, unsafe_allow_html=True)

        with col_info:
            # 匹配度进度条
            st.markdown(f"""
            <div style="margin:4px 0 14px 0;">
                <div style="font-size:11px; color:#8C7268; margin-bottom:5px;">匹配度</div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="flex:1; height:6px; background:#EDE3DA;
                    border-radius:3px; overflow:hidden;">
                        <div style="width:{score:.0%}; height:100%;
                        background:{bar_color}; border-radius:3px;"></div>
                    </div>
                    <span style="font-size:13px; font-weight:500;
                    color:{bar_color};">{score:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 推荐理由
            for reason in reasons[:3]:
                st.markdown(f"""
                <div style="font-size:13px; color:#5A4035; margin:3px 0;
                padding-left:10px; border-left:2px solid {bar_color}60;">
                {reason}</div>
                """, unsafe_allow_html=True)

            # 属性标签
            tags = []
            if hs.get("length"):
                tags.append(hs["length"])
            if hs.get("curl"):
                tags.append(hs["curl"])
            if hs.get("popularity"):
                tags.append(f"热度 {hs['popularity']:.0%}")
            if tags:
                tag_html = "".join([
                    f'<span style="display:inline-block; background:#F5F0EB; '
                    f'border-radius:20px; padding:2px 10px; font-size:11px; '
                    f'color:#8C7268; margin:3px 3px 0 0;">{t}</span>'
                    for t in tags
                ])
                st.markdown(f'<div style="margin-top:10px;">{tag_html}</div>', unsafe_allow_html=True)

        # 操作按钮
        user_photo = st.session_state.get("uploaded_image_path")
        can_tryon = user_photo and os.path.isfile(user_photo) and img_path and os.path.isfile(img_path)

        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            if st.button(
                "试戴预览" if can_tryon else "试戴（请先上传照片）",
                key=f"tryon_btn_{hairstyle_id}_{index}",
                disabled=not can_tryon,
                use_container_width=True,
            ):
                if can_tryon:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"帮我试戴「{hairstyle_name}」看看效果",
                        "is_tryon": True,
                        "tryon_info": {
                            "hairstyle_id": hairstyle_id,
                            "hairstyle_name": hairstyle_name,
                            "img_path": img_path,
                            "user_photo": user_photo,
                        },
                    })
                    st.rerun()
        with col_btn2:
            if hairstyle_id in st.session_state.tryon_results:
                if st.button("查看效果", key=f"view_{hairstyle_id}_{index}", use_container_width=True):
                    st.session_state.show_tryon = hairstyle_id
                    st.rerun()
        with col_btn3:
            with st.expander("打理建议与注意事项"):
                if hs.get("warnings"):
                    st.markdown(f'<div style="font-size:13px; color:#A07060;">注意：{"；".join(hs["warnings"])}</div>', unsafe_allow_html=True)
                if hs.get("care_tips"):
                    st.markdown(f'<div style="font-size:13px; color:#6A8A6E; margin-top:6px;">{hs["care_tips"]}</div>', unsafe_allow_html=True)
                if details:
                    st.caption(
                        f"脸型匹配 {details.get('face_match', 0):.0%}  ·  "
                        f"风格相似 {details.get('style_similarity', 0):.0%}  ·  "
                        f"热度 {details.get('popularity', 0):.0%}"
                    )

        # 试戴结果
        if st.session_state.get("show_tryon") == hairstyle_id:
            result_path = st.session_state.tryon_results.get(hairstyle_id)
            if result_path and os.path.isfile(result_path):
                st.image(result_path, caption=f"{hairstyle_name} · 试戴效果（仅供参考）", width=400)

        st.markdown('<div style="height:4px; background:#EDE3DA; border-radius:2px; margin:16px 0 20px 0;"></div>', unsafe_allow_html=True)


def fallback_report_display(report_md: str):
    with st.container():
        st.markdown(f"""
        <div style="background:#FDF8F4; border-radius:18px; padding:22px 24px;
        border:1px solid #EDE3DA; margin:10px 0;">
        {report_md}
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 聊天渲染
# ============================================================
def render_chat():
    if not st.session_state.messages:
        tag = render_landing_hero()
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "你好，很高兴见到你 🌿\n\n"
                "我可以帮你分析脸型、推荐最适合的发型，还能做虚拟试戴预览。\n\n"
                "**可以这样开始：**\n"
                "- 在左侧上传一张正面照，AI 会自动识别你的脸型\n"
                "- 或者直接点击上面的风格标签，告诉我你的偏好\n"
                "- 也可以直接和我聊，描述你想要的效果 ☁️"
            ),
        })
        if tag:
            st.session_state.messages.append({"role": "user", "content": f"我想要{tag}风格"})
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_tryon") or msg.get("is_tryon_result"):
                st.markdown(msg["content"])
            elif msg.get("is_report"):
                fallback_report_display(msg["content"])
            elif msg.get("recommendations"):
                st.markdown(msg["content"])
                for i, rec in enumerate(msg["recommendations"], 1):
                    fallback_recommendation_card(rec, i)
            else:
                st.markdown(msg["content"])


# ============================================================
# 主函数
# ============================================================
def main():
    render_sidebar()
    render_chat()

    # 处理试戴请求
    pending = [m for m in st.session_state.messages[-3:] if m.get("is_tryon")]
    if pending:
        tryon_info = pending[-1]["tryon_info"]
        with st.chat_message("assistant"):
            response = run_tryon(tryon_info)
            st.session_state.messages.append({
                "role": "assistant", "content": response, "is_tryon_result": True,
            })
        st.rerun()

    # 聊天输入
    if prompt := st.chat_input("告诉我你想要的风格，或描述你的需求…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        image_path = st.session_state.get("uploaded_image_path")

        with st.chat_message("assistant"):
            if st.session_state.mode == "agent" and LANGCHAIN_AVAILABLE:
                response = run_langchain_agent(prompt, image_path)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = run_mock_agent(prompt, image_path)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        st.rerun()


def run_mock_agent(prompt: str, image_path: str = None) -> str:
    agent = MockAgent()

    if st.session_state.face_report:
        agent.session["face_report"] = st.session_state.face_report
        agent.session["stage"] = "face_analyzed"
    if st.session_state.last_recommendations:
        agent.session["recommendations"] = st.session_state.last_recommendations
        agent.session["stage"] = "recommended"

    response = agent.process(prompt, image_path)

    st.session_state.face_report = agent.session["face_report"]
    st.session_state.last_recommendations = agent.session["recommendations"]

    return response


def run_tryon(tryon_info: dict) -> str:
    hairstyle_name = tryon_info["hairstyle_name"]
    hairstyle_id = tryon_info["hairstyle_id"]
    img_path = tryon_info["img_path"]
    user_photo = tryon_info["user_photo"]

    with st.spinner(f"正在为你合成「{hairstyle_name}」的试戴效果，稍等一下…"):
        try:
            output_dir = os.path.join(PROJECT_ROOT, "hairstyle_db", "tryon_results")
            result_path = virtual_tryon(user_photo, img_path, output_dir)

            st.session_state.tryon_results[hairstyle_id] = result_path
            st.session_state.show_tryon = hairstyle_id

            st.image(result_path, caption=f"{hairstyle_name} · 试戴效果", width=400)

            return (
                f"「{hairstyle_name}」的试戴效果来了 ✿\n\n"
                f"上面是 AI 合成的预览，可以感受一下整体风格。\n\n"
                f"实际效果会因发质和发量有所不同，可以把这张图给发型师参考，沟通会更顺畅 🌿\n\n"
                f"想看其他发型？点击推荐卡片里的「试戴预览」就可以~"
            )

        except ValueError as e:
            err_msg = str(e)
            if "未检测到人脸" in err_msg:
                return (
                    f"试戴暂时无法完成：{err_msg}\n\n"
                    f"建议换一张正面、免冠的清晰照片再试试 ☀️"
                )
            return f"试戴遇到了一点问题：{err_msg}\n\n可以稍后重试或换张照片~"
        except Exception as e:
            return f"合成过程出了些小差错：{str(e)}\n\n请检查照片格式或稍后重试。"


def run_langchain_agent(prompt: str, image_path: str = None) -> str:
    try:
        if st.session_state.agent_executor is None:
            tools = [
                Tool(
                    name="detect_face_shape",
                    func=lambda p: detect_face_shape_tool(p),
                    description="分析用户上传的人像照片，识别脸型。参数 image_path 是照片路径。",
                ),
                Tool(
                    name="search_hairstyles",
                    func=lambda p: search_hairstyles_tool(p),
                    description="搜索发型数据库。参数 query_json 是 JSON 字符串，可包含 face_shape/style_vector/length/curl/limit。",
                ),
                Tool(
                    name="recommend",
                    func=lambda p: recommend_tool(p),
                    description="根据脸型和偏好推荐 Top-N 发型。参数 params_json 包含 face_report + preferences + top_n。",
                ),
                Tool(
                    name="generate_style_report",
                    func=lambda p: generate_report_tool(p),
                    description="生成完整风格分析报告。参数 params_json 包含 face_report + recommendations。",
                ),
            ]

            from langchain_community.chat_models import ChatTongyi
            from langchain.agents import create_react_agent, AgentExecutor
            from langchain import hub

            llm = ChatTongyi(model="qwen-plus", temperature=0.7)
            agent = create_react_agent(llm, tools, SYSTEM_PROMPT)
            st.session_state.agent_executor = AgentExecutor(
                agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,
            )

        input_text = prompt
        if image_path:
            input_text += f"\n\n用户上传了照片：{image_path}"

        result = st.session_state.agent_executor.invoke({"input": input_text})
        return result["output"]

    except Exception as e:
        return f"Agent 模式出错（已回退到 Mock 模式）：{str(e)}"


if __name__ == "__main__":
    main()
