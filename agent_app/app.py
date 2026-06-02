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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ============================================================
# 侧边栏
# ============================================================
def render_sidebar():
    """渲染侧边栏：模式切换 + 状态面板"""
    with st.sidebar:
        st.title("💇 发型顾问 AI")

        # 模式选择
        st.subheader("⚙️ 运行模式")
        mode = st.radio(
            "选择运行模式",
            options=["🤖 Mock（离线调试）", "🧠 Agent（需要 API Key）"],
            index=0 if st.session_state.mode == "mock" else 1,
            help="Mock 模式用规则引擎模拟 Agent，无需 API Key。Agent 模式需要 OpenAI API Key。",
        )
        st.session_state.mode = "mock" if "Mock" in mode else "agent"

        # Agent 模式：API Key 输入
        if st.session_state.mode == "agent":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                help="不会存储到文件，仅本次会话有效",
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        st.divider()

        # 当前状态
        st.subheader("📊 会话状态")
        if st.session_state.face_report:
            fr = st.session_state.face_report
            st.success(f"脸型: {fr['face_shape']} ({fr['confidence']:.0%})")
        else:
            st.info("尚未分析脸型")

        if st.session_state.last_recommendations:
            st.caption(f"最近推荐: {len(st.session_state.last_recommendations)} 条")

        st.caption(f"对话轮次: {len(st.session_state.messages) // 2}")

        st.divider()

        # 重置按钮
        if st.button("🔄 重新开始", use_container_width=True):
            for key in ["messages", "face_report", "last_recommendations",
                        "uploaded_image_path", "agent_executor"]:
                st.session_state[key] = [] if key == "messages" else None
            st.rerun()

        st.caption(f"---\n*合约版本: contracts.py v1.0*")


# ============================================================
# Fallback UI 组件（B 未交付前的替代方案）
# ============================================================
def fallback_recommendation_card(rec: dict, index: int):
    """B 交付前的推荐卡片 fallback"""
    hs = rec.get("hairstyle", {})
    score = rec.get("score", 0)
    reasons = rec.get("reasons", [])
    details = rec.get("details", {})

    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            # 颜色条表示匹配度
            color = "#4CAF50" if score > 0.8 else "#FF9800" if score > 0.6 else "#F44336"
            st.markdown(
                f"<div style='width:60px;height:60px;border-radius:50%;"
                f"background:{color};display:flex;align-items:center;justify-content:center;"
                f"color:white;font-weight:bold;font-size:18px;'>{score:.0%}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(f"### {index}. {hs.get('name', '')}")
            for reason in reasons:
                st.caption(f"• {reason}")

        # 详情展开
        with st.expander("📋 详细信息"):
            st.write(f"**长度**: {hs.get('length', '')} | **卷度**: {hs.get('curl', '')}")
            st.write(f"**热度**: {hs.get('popularity', 0):.0%}")
            if hs.get("warnings"):
                st.warning("⚠️ " + "；".join(hs["warnings"]))
            if hs.get("care_tips"):
                st.info("💡 " + hs["care_tips"])

            # 因子得分
            if details:
                st.caption(
                    f"脸型匹配: {details.get('face_match', 0):.0%} | "
                    f"风格相似: {details.get('style_similarity', 0):.0%} | "
                    f"热度: {details.get('popularity', 0):.0%}"
                )


def fallback_report_display(report_md: str):
    """B 交付前的报告展示 fallback"""
    st.markdown(report_md)


# ── Mock Agent 已移至 agent_app/agent.py ──
# ============================================================
# 聊天消息展示
# ============================================================
def render_chat():
    """渲染聊天界面"""
    # 欢迎消息
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "你好！我是你的专属发型顾问 💇\n\n"
                "上传一张正面照，我就能帮你分析脸型、推荐最适合你的发型。\n\n"
                "或者直接告诉我你想要什么风格，比如「干练通勤风」「日系甜美风」~"
            ),
        })

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_report"):
                # 渲染 Markdown 报告
                fallback_report_display(msg["content"])
            elif msg.get("recommendations"):
                # 如果有推荐结果，渲染推荐卡片
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

    # ── 图片上传区域 ──
    uploaded_file = st.file_uploader(
        "📎 上传正面照（JPG/PNG）",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
        help="建议光线充足、正面免冠照片",
    )

    # ── 聊天输入 ──
    if prompt := st.chat_input("描述你想要的风格，或问任何发型相关问题..."):
        # 保存用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 处理上传的照片
        image_path = None
        if uploaded_file:
            # 保存到临时目录
            save_dir = Path(PROJECT_ROOT) / "uploaded_photos"
            save_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = str(save_dir / f"upload_{timestamp}.jpg")
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_image_path = image_path

            # 显示上传的照片
            with st.chat_message("user"):
                st.image(uploaded_file, caption="上传的照片", width=200)

        # ── 处理消息 ──
        with st.chat_message("assistant"):
            if st.session_state.mode == "agent" and LANGCHAIN_AVAILABLE:
                # Agent 模式（需要 AI）
                response = run_langchain_agent(prompt, image_path)
            else:
                # Mock 模式：规则引擎
                response = run_mock_agent(prompt, image_path)

            st.markdown(response)

        # 保存助手消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
        })

        # 清空上传组件（避免重复处理）
        if uploaded_file:
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
