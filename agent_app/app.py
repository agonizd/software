"""
app.py — Streamlit 前端入口（E 模块）

两种运行模式：
1. Mock 模式（默认）：不需要 OpenAI API Key，
   Agent 用简单的规则引擎自动调度工具，适合 Day 1-3 开发调试
2. Agent 模式：需要 OPENAI_API_KEY，用 LangChain ReAct Agent，
   真正的自然语言理解 + 自主工具调度

启动方式：
    streamlit run agent_app/app.py

B 的 UI 组件集成说明：
    B 负责 delivery 推荐卡片 + 报告展示的 Streamlit 组件。
    E 在 app.py 中 import 这些组件并嵌入聊天消息流。
    如果 B 还未交付，使用内置的 fallback 展示（纯文本 + Markdown）。
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
    # B 的组件路径（等 B 交付后确认）
    # from hairstyle_db.components import render_recommendation_card, render_report
    pass
except ImportError:
    pass

# ── 尝试导入 LangChain（Agent 模式）──
LANGCHAIN_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.tools import Tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="发型顾问 AI Agent",
    page_icon="💇",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# 全局样式 — 现代发型顾问 AI 主题
# ============================================================
st.markdown("""
<style>
/* ── 基础变量与主题 ── */
:root {
    --primary: #8B5CF6;
    --primary-light: #A78BFA;
    --primary-dark: #7C3AED;
    --accent: #F59E0B;
    --accent-light: #FCD34D;
    --bg: #FAFAFA;
    --card-bg: #FFFFFF;
    --text: #1F2937;
    --text-secondary: #6B7280;
    --border: #E5E7EB;
    --success: #10B981;
    --radius: 16px;
    --shadow: 0 2px 12px rgba(0,0,0,0.06);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.10);
}

/* ── 全局重置 ── */
.stApp { background: var(--bg) !important; }
footer, #MainMenu { visibility: hidden !important; }

/* ── 侧边栏美化 ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
    color: #F1F5F9 !important;
}
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stCheckbox label {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}

/* ── 聊天消息美化 ── */
.stChatMessage { border-radius: 18px !important; padding: 16px 20px !important; }
.stChatMessage[data-testid="stChatMessage"] {
    background: var(--card-bg) !important;
    box-shadow: var(--shadow) !important;
    margin-bottom: 12px !important;
}

/* ── 按钮统一风格 ── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    border: none !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-lg) !important;
}

/* ── 成功/信息框 ── */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State 初始化
# ============================================================
def init_session():
    """初始化 Streamlit 会话状态"""
    defaults = {
        "messages": [],                # 对话历史: [{"role": "user"/"assistant", "content": ...}]
        "face_report": None,           # 当前脸型分析结果 (dict)
        "last_recommendations": None,  # 最近一次推荐结果 (list of dict)
        "uploaded_image_path": None,   # 上传照片的本地路径
        "mode": "mock",                # "mock" | "agent"
        "agent_executor": None,        # LangChain AgentExecutor 实例
        "tryon_results": {},          # 试戴结果 {hairstyle_id: result_image_path}
        "pending_tryon": None,        # 待处理的试戴请求
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ============================================================
# 侧边栏
# ============================================================
def _on_photo_upload():
    """照片上传回调：保存 + 即时脸型分析"""
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

    # 即时脸型分析
    try:
        from agent_app.face_detect import detect_face_shape
        shape, conf, _landmarks = detect_face_shape(img_path)
        st.session_state.face_report = {"face_shape": shape, "confidence": conf}
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"📸 脸型分析完成！你的脸型是 **{shape}**（置信度 {conf:.0%}）",
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ 人脸检测失败：{e}\n\n请确保照片光线充足、正面免冠。",
        })


def render_sidebar():
    """暗色侧边栏：上传 + 状态 + 设置"""
    with st.sidebar:
        # Logo & 标题
        st.markdown("""
        <div style="text-align:center; padding:12px 0 8px 0;">
            <div style="font-size:36px; margin-bottom:4px;">💇</div>
            <div style="font-size:18px; font-weight:700; color:#F1F5F9;">
            发型顾问 AI</div>
            <div style="font-size:11px; color:#94A3B8; margin-top:2px;">
            你的专属智能造型师</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── 照片上传（紧凑卡片式）──
        st.markdown('<p style="color:#94A3B8; font-size:12px; font-weight:600; margin-bottom:6px;">📷 上传正面照</p>', unsafe_allow_html=True)
        st.file_uploader(
            "选择 JPG/PNG 照片",
            type=["jpg", "jpeg", "png"],
            key="sidebar_uploader",
            on_change=_on_photo_upload,
            help="光线充足、正面免冠效果最佳",
        )

        # 已上传缩略图
        if st.session_state.uploaded_image_path and os.path.isfile(st.session_state.uploaded_image_path):
            st.image(st.session_state.uploaded_image_path, width=140)

        st.divider()

        # ── 脸型状态 ──
        if st.session_state.face_report:
            fr = st.session_state.face_report
            shape_emoji = FACE_EMOJI.get(fr["face_shape"], "😊")
            st.success(f"{shape_emoji} {fr['face_shape']} — 置信度 {fr['confidence']:.0%}")
        else:
            st.info("📋 尚未分析脸型")

        st.divider()

        # ── 运行模式 ──
        st.markdown('<p style="color:#94A3B8; font-size:12px; font-weight:600; margin-bottom:6px;">⚙️ 运行模式</p>', unsafe_allow_html=True)
        mode = st.radio(
            "模式",
            ["🤖 Mock（离线）", "🧠 Agent（需 Key）"],
            index=0 if st.session_state.mode == "mock" else 1,
            label_visibility="collapsed",
        )
        st.session_state.mode = "mock" if "Mock" in mode else "agent"

        if st.session_state.mode == "agent":
            api_key = st.text_input("OpenAI API Key", type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                placeholder="sk-...")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        st.divider()

        # ── 快速统计 ──
        msg_count = len(st.session_state.messages)
        rec_count = len(st.session_state.last_recommendations) if st.session_state.last_recommendations else 0
        st.caption(f"💬 {msg_count} 条消息 · 🎯 {rec_count} 条推荐")

        # 重置
        if st.button("🔄 重新开始", use_container_width=True):
            for key in ["messages", "face_report", "last_recommendations",
                        "uploaded_image_path", "agent_executor", "tryon_results"]:
                st.session_state[key] = [] if key == "messages" else ({} if key == "tryon_results" else None)
            st.rerun()

        st.caption("v1.0 · contracts.py")


# ============================================================
# 常量
# ============================================================
FACE_EMOJI = {
    "鹅蛋脸": "🥚", "圆脸": "😊", "方脸": "🔲",
    "长脸": "📏", "心形脸": "❤️", "菱形脸": "💎",
}

STYLE_TAGS = [
    ("干练通勤", "💼"), ("甜美约会", "🌸"), ("复古港风", "📼"),
    ("酷飒少年", "⚡"), ("自然清新", "🌿"), ("优雅知性", "💜"),
]
def _resolve_image_path(image_url: str) -> str:
    """将相对路径解析为绝对路径"""
    if not image_url:
        return ""
    # 先尝试 hairstyle_db 目录
    candidate = os.path.join(PROJECT_ROOT, "hairstyle_db", image_url)
    if os.path.isfile(candidate):
        return candidate
    # 再尝试项目根目录
    candidate2 = os.path.join(PROJECT_ROOT, image_url)
    if os.path.isfile(candidate2):
        return candidate2
    return ""


# ============================================================
# 推荐卡片（现代卡片式设计）
# ============================================================
def fallback_recommendation_card(rec: dict, index: int):
    """现代卡片：左侧预览图 + 右侧信息 + 底部操作"""
    hs = rec.get("hairstyle", {})
    score = rec.get("score", 0)
    reasons = rec.get("reasons", [])
    details = rec.get("details", {})
    hairstyle_id = hs.get("id", "")
    hairstyle_name = hs.get("name", "")

    # 匹配度颜色
    if score >= 0.85:
        bar_color = "#10B981"; badge = "🏆 完美匹配"
    elif score >= 0.7:
        bar_color = "#8B5CF6"; badge = "⭐ 高度推荐"
    elif score >= 0.55:
        bar_color = "#F59E0B"; badge = "👍 可以尝试"
    else:
        bar_color = "#6B7280"; badge = "💡 仅供参考"

    # 解析图片路径
    img_path = _resolve_image_path(hs.get("image_url", ""))

    # 卡片容器
    with st.container():
        st.markdown(f"""
        <div style="background: white; border-radius: 20px; padding: 20px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06); margin-bottom: 16px;
        border: 1px solid #F3F4F6; position: relative; overflow: hidden;">
        <!-- 顶部色条 -->
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px;
        background: {bar_color}; border-radius: 20px 20px 0 0;"></div>
        <!-- 排名徽章 -->
        <div style="position: absolute; top: 14px; right: 16px;
        background: {bar_color}15; color: {bar_color}; font-size: 12px;
        font-weight: 700; padding: 4px 12px; border-radius: 20px;">
        #{index} {badge}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── 图片 + 基本信息 ──
        col_img, col_info = st.columns([1, 2])
        with col_img:
            if img_path and os.path.isfile(img_path):
                st.image(img_path, width=180)
            else:
                st.markdown(f"""
                <div style="width:180px; height:220px; background: linear-gradient(135deg, #F3F4F6, #E5E7EB);
                border-radius: 14px; display: flex; align-items: center; justify-content: center;
                font-size: 48px;">💇</div>
                """, unsafe_allow_html=True)

        with col_info:
            # 发型名称 + 匹配度条
            st.markdown(f"### {hairstyle_name}")
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin: 8px 0 12px 0;">
                <div style="flex: 1; height: 8px; background: #F3F4F6; border-radius: 4px; overflow: hidden;">
                    <div style="width: {score:.0%}; height: 100%; background: {bar_color};
                    border-radius: 4px; transition: width 0.6s ease;"></div>
                </div>
                <span style="font-size: 14px; font-weight: 700; color: {bar_color};">{score:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

            # 推荐理由
            for reason in reasons:
                st.markdown(f"<p style='color:#4B5563; font-size:14px; margin:2px 0;'>✦ {reason}</p>", unsafe_allow_html=True)

            # 标签行
            tags = []
            if hs.get("length"):
                tags.append(f"📏 {hs['length']}")
            if hs.get("curl"):
                tags.append(f"🌀 {hs['curl']}")
            if hs.get("popularity"):
                tags.append(f"🔥 {hs['popularity']:.0%}")
            if tags:
                st.caption("  |  ".join(tags))

        # ── 操作按钮 ──
        user_photo = st.session_state.get("uploaded_image_path")
        can_tryon = user_photo and os.path.isfile(user_photo) and img_path and os.path.isfile(img_path)

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button(
                f"👗 试戴预览" if can_tryon else "👗 试戴预览（请先上传照片）",
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
                if st.button(
                    "📸 查看效果",
                    key=f"view_{hairstyle_id}_{index}",
                    use_container_width=True,
                ):
                    st.session_state.show_tryon = hairstyle_id
                    st.rerun()
        with col_btn3:
            with st.expander("📋 详情"):
                if hs.get("warnings"):
                    st.warning("⚠️ " + "；".join(hs["warnings"]))
                if hs.get("care_tips"):
                    st.info("💡 " + hs["care_tips"])
                if details:
                    st.caption(
                        f"脸型: {details.get('face_match', 0):.0%} | "
                        f"风格: {details.get('style_similarity', 0):.0%} | "
                        f"热度: {details.get('popularity', 0):.0%}"
                    )

        # 试戴结果展示
        if st.session_state.get("show_tryon") == hairstyle_id:
            result_path = st.session_state.tryon_results.get(hairstyle_id)
            if result_path and os.path.isfile(result_path):
                st.image(result_path, caption=f"✨ {hairstyle_name} 试戴效果", width=480)
                st.caption("💡 合成效果仅供参考，实际效果因发质发量略有差异")

        st.markdown("</div>", unsafe_allow_html=True)


def _handle_tryon_click(hairstyle_id, hairstyle_name, img_path, user_photo):
    """试戴按钮回调"""
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


def fallback_report_display(report_md: str):
    """B 交付前的报告展示 fallback"""
    with st.container():
        st.markdown(f"""
        <div style="background: white; border-radius: 20px; padding: 24px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06); margin: 12px 0;
        border-left: 4px solid #8B5CF6;">
        {report_md}
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 聊天渲染（含 Landing Hero）
# ============================================================
def render_landing_hero():
    """首次访问的欢迎页面"""
    st.markdown("""
    <div style="text-align: center; padding: 48px 20px 32px 20px;">
        <div style="font-size: 64px; margin-bottom: 12px; animation: float 3s ease-in-out infinite;">
            💇
        </div>
        <h1 style="font-size: 36px; font-weight: 800; color: #1F2937; margin: 0 0 8px 0;">
            你的专属 <span style="color: #8B5CF6;">AI 发型顾问</span>
        </h1>
        <p style="font-size: 16px; color: #6B7280; max-width: 480px; margin: 0 auto 28px auto; line-height: 1.6;">
            上传照片 → AI 分析脸型 → 精准推荐最适合你的发型
        </p>
        <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 8px;">
    """, unsafe_allow_html=True)

    # 快速风格标签
    cols = st.columns(6)
    for i, (tag, emoji) in enumerate(STYLE_TAGS):
        with cols[i]:
            if st.button(f"{emoji} {tag}", key=f"tag_{tag}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": f"我想要{tag}风格"})
                return tag

    st.markdown("</div></div>", unsafe_allow_html=True)
    return None


def render_chat():
    """渲染聊天消息（含欢迎 Hero）"""
    if not st.session_state.messages:
        # 首次访问 → Landing Hero
        tag = render_landing_hero()
        # 引导消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "你好！我是你的专属发型顾问 💇\n\n"
                "**三步搞定完美发型：**\n"
                "1️⃣ 在左侧上传一张正面照\n"
                "2️⃣ 点击上方风格标签，或直接告诉我你想要的风格\n"
                "3️⃣ 我会为你推荐最适合的发型，还能虚拟试戴！\n\n"
                "开始试试吧~ 😊"
            ),
        })
        # 风格标签被点击
        if tag:
            st.session_state.messages.append({"role": "user", "content": f"我想要{tag}风格"})
        st.rerun()

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_tryon"):
                st.markdown(msg["content"])
            elif msg.get("is_tryon_result"):
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

    # 处理试戴请求（来自按钮回调，无需 chat_input）
    pending = [m for m in st.session_state.messages[-3:] if m.get("is_tryon")]
    if pending:
        tryon_info = pending[-1]["tryon_info"]
        with st.chat_message("assistant"):
            response = run_tryon(tryon_info)
            st.session_state.messages.append({
                "role": "assistant", "content": response, "is_tryon_result": True,
            })
        st.rerun()

    # chat_input 收消息
    if prompt := st.chat_input("描述你想要的风格，例如「干练通勤」「甜美约会」..."):
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
    """运行 Mock Agent"""
    agent = MockAgent()

    # 从 session state 恢复状态
    if st.session_state.face_report:
        agent.session["face_report"] = st.session_state.face_report
        agent.session["stage"] = "face_analyzed"
    if st.session_state.last_recommendations:
        agent.session["recommendations"] = st.session_state.last_recommendations
        agent.session["stage"] = "recommended"

    response = agent.process(prompt, image_path)

    # 同步状态回 session state
    st.session_state.face_report = agent.session["face_report"]
    st.session_state.last_recommendations = agent.session["recommendations"]

    return response


def run_tryon(tryon_info: dict) -> str:
    """
    执行虚拟试戴：将发型预览图合成到用户照片上。

    Args:
        tryon_info: {
            "hairstyle_id": str,
            "hairstyle_name": str,
            "img_path": str,       # 发型预览图路径
            "user_photo": str,     # 用户照片路径
        }

    Returns:
        结果描述文本
    """
    hairstyle_name = tryon_info["hairstyle_name"]
    hairstyle_id = tryon_info["hairstyle_id"]
    img_path = tryon_info["img_path"]
    user_photo = tryon_info["user_photo"]

    with st.spinner(f"✨ 正在为你合成「{hairstyle_name}」的试戴效果..."):
        try:
            output_dir = os.path.join(PROJECT_ROOT, "hairstyle_db", "tryon_results")
            result_path = virtual_tryon(user_photo, img_path, output_dir)

            # 保存结果到 session state
            st.session_state.tryon_results[hairstyle_id] = result_path
            st.session_state.show_tryon = hairstyle_id

            # 同时显示结果图片
            st.image(result_path, caption=f"✨ {hairstyle_name} 试戴效果", width=400)

            return (
                f"👗 **{hairstyle_name}** 试戴效果来啦！\n\n"
                f"上面是 AI 合成的预览效果图，可以看到你换上这款发型的整体感觉。\n\n"
                f"💡 小提示：\n"
                f"- 合成效果主要展示发型轮廓和风格，实际效果会因发质发量略有差异\n"
                f"- 建议拿这张图给发型师参考，沟通更高效！\n"
                f"- 想看其他发型的试戴效果？点击推荐卡片里的「试戴预览」按钮~"
            )

        except ValueError as e:
            err_msg = str(e)
            if "未检测到人脸" in err_msg:
                return (
                    f"⚠️ 试戴失败：{err_msg}\n\n"
                    f"请先上传一张清晰正面照（光线充足、面部无遮挡），"
                    f"再进行试戴预览。"
                )
            return f"⚠️ 试戴遇到问题：{err_msg}\n\n请稍后重试或换一张照片试试~"
        except Exception as e:
            return f"❌ 试戴合成出错：{str(e)}\n\n请检查照片格式或稍后重试。"


def run_langchain_agent(prompt: str, image_path: str = None) -> str:
    """
    运行 LangChain ReAct Agent。

    需要：
    - OPENAI_API_KEY 环境变量
    - langchain + langchain-openai 已安装
    """
    try:
        if st.session_state.agent_executor is None:
            # 构建 LangChain Tools
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

            from langchain_openai import ChatOpenAI
            from langchain.agents import create_react_agent, AgentExecutor
            from langchain import hub

            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            agent = create_react_agent(llm, tools, SYSTEM_PROMPT)
            st.session_state.agent_executor = AgentExecutor(
                agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,
            )

        # 构建输入
        input_text = prompt
        if image_path:
            input_text += f"\n\n用户上传了照片：{image_path}"

        result = st.session_state.agent_executor.invoke({"input": input_text})
        return result["output"]

    except Exception as e:
        return f"Agent 模式出错（回退到 Mock 模式）：{str(e)}"


if __name__ == "__main__":
    main()
