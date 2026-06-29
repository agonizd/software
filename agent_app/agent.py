"""
agent.py — Mock Agent（规则引擎模拟 Agent 调度）

这是 E 模块的核心逻辑——不需要 LLM 就能跑通完整对话流程。
用户上传照片 → 分析 → 推荐 → 生成报告，全部用 if-else 规则驱动。
"""

import json
from agent_app.tools import (
    detect_face_shape_tool,
    recommend_tool,
    generate_report_tool,
)


class MockAgent:
    """
    模拟 Agent 的决策逻辑——用规则引擎替代 LLM 推理。

    这不是真正的 Agent，但它让你在 Day 1 就能跑通完整流程：
    上传照片 → 分析脸型 → 推荐发型 → 生成报告

    决策树（按优先级）：
    1. 有照片 → 分析脸型
    2. 有脸型 + 有风格描述 → 推荐
    3. 有推荐 + 用户想看报告 → 生成报告
    4. 其他 → 引导用户
    """

    def __init__(self):
        self.session = {
            "face_report": None,
            "recommendations": None,
            "stage": "start",  # start → face_analyzed → recommended → reported
        }

    def process(self, user_message: str, image_path: str = None) -> str:
        """
        根据用户输入和当前阶段，决定下一步。

        Args:
            user_message: 用户输入文本
            image_path: 照片路径（None = 未上传）

        Returns:
            助手的回复文本（可能是纯文本或 Markdown）
        """
        msg = user_message.strip()

        # ── Step 1: 用户上传照片 → 分析脸型 ──
        # 只在首次上传（stage=start）且无文本输入时触发
        # 修复：之前 stage in ("start","face_analyzed") 导致已分析脸型时
        # 用户输入文字仍被误判为"上传了新照片"而重复分析
        if image_path and self.session["stage"] == "start":
            return self._handle_face_analysis(image_path)

        # ── Step 2: 有脸型 + 有风格关键词 → 推荐 ──
        if self.session["stage"] in ("face_analyzed", "recommended"):
            style_keywords = [
                "干练", "甜美", "复古", "酷飒", "自然", "优雅",
                "通勤", "日系", "港风", "法式", "飒", "气质", "可爱",
                "帅气", "慵懒", "成熟", "少女",
            ]
            has_style = any(kw in msg.lower() for kw in style_keywords)
            wants_recommend = any(
                w in msg.lower()
                for w in ["推荐", "适合", "发型", "换", "试试", "建议", "帮我"]
            )

            if has_style or wants_recommend:
                return self._handle_recommend(msg)

        # ── Step 3: 用户想看报告 ──
        if any(w in msg.lower() for w in ["报告", "详细", "为什么", "分析", "生成"]):
            # "生成照片"/"生成图片"/"生成效果" 不能触发报告
            preview_kw = ["照片", "图片", "效果图", "预览", "效果"]
            wants_preview = any(w in msg.lower() for w in preview_kw)
            if self.session["recommendations"] and wants_preview:
                return self._handle_preview(msg)
            if self.session["recommendations"]:
                return self._handle_report()

        # ── Step 3.5: 用户想看效果预览 ──
        if any(w in msg.lower() for w in ["照片", "图片", "效果图", "预览", "看看效果", "长什么样"]):
            if self.session["recommendations"]:
                return self._handle_preview(msg)

        # ── Step 4: 用户想试戴特定发型 ──
        tryon_kw = ["试戴", "换上", "试试效果", "帮我换", "换上看看"]
        if any(w in msg.lower() for w in tryon_kw):
            if self.session["recommendations"]:
                return self._handle_tryon(msg)

        # ── Default: 引导用户 ──
        return self._handle_default()

    def _handle_face_analysis(self, image_path: str) -> str:
        """处理脸型分析请求"""
        result_json = detect_face_shape_tool(image_path)
        result = json.loads(result_json)

        if result.get("status") == "error":
            return (
                f"抱歉，照片分析遇到了一点问题😅\n\n"
                f"{result['message']}\n\n"
                f"可以换一张清晰的正面照试试吗？"
            )

        # 去掉工具层的 status 字段，只保留 FaceReport 需要的字段
        result.pop("status", None)
        self.session["face_report"] = result
        self.session["stage"] = "face_analyzed"

        face_shape = result["face_shape"]
        confidence = result["confidence"]
        features = result["features"]

        response = (
            f"分析完成！你的脸型是 **{face_shape}**"
            f"（置信度 {confidence:.0%}）\n\n"
        )

        face_descriptions = {
            "鹅蛋脸": '鹅蛋脸是面部美学的"黄金比例"，线条流畅，几乎能驾驭所有发型！你有很大的选择空间。',
            "圆脸": "圆脸的特点是脸颊饱满、线条柔和，视觉上非常减龄。适合能拉长面部比例的发型。",
            "方脸": "方脸下颌线条分明，给人干练率性的感觉。可以用柔和线条来平衡硬朗轮廓。",
            "长脸": "长脸纵向比例偏大，需要在视觉上增加宽度来平衡。有层次感的横向发型是最佳选择。",
            "心形脸": "心形脸上宽下窄，额头饱满、下巴精致。关键是平衡额头和下巴的比例。",
            "菱形脸": "菱形脸颧骨突出，辨识度很高。需要柔化颧骨线条的发型。",
        }
        response += face_descriptions.get(face_shape, "")

        response += (
            f"\n\n📐 面部比例：长宽比 {features['face_ratio']:.2f}，"
            f"颧颌比 {features['jaw_cheek_ratio']:.2f}"
        )

        response += (
            "\n\n---\n你想尝试什么风格呢？比如「干练通勤风」"
            "「日系甜美风」「复古港风」……"
        )
        return response

    def _handle_recommend(self, msg: str) -> str:
        """处理推荐请求"""
        params = {
            "face_report": self.session["face_report"],
            "preferences": {"natural_language": msg},
            "top_n": 3,
        }

        result_json = recommend_tool(json.dumps(params, ensure_ascii=False))
        result = json.loads(result_json)

        if isinstance(result, dict) and result.get("status") == "error":
            return f"推荐遇到了问题：{result['message']}"

        self.session["recommendations"] = result
        self.session["stage"] = "recommended"

        face_shape = self.session["face_report"]["face_shape"]
        response = (
            f"根据你的 **{face_shape}** 和风格偏好，"
            f"为你推荐以下 {len(result)} 款发型：\n\n"
        )

        for i, rec in enumerate(result, 1):
            hs = rec["hairstyle"]
            response += f"**{i}. {hs['name']}** — 匹配度 {rec['score']:.0%}\n"
            first_reason = rec["reasons"][0] if rec["reasons"] else ""
            response += f"> {first_reason}\n\n"

        response += (
            "---\n"
            "想看详细分析报告吗？告诉我你对哪个发型感兴趣，"
            "或者回复「生成报告」~"
        )
        return response

    def _handle_report(self) -> str:
        """生成风格报告"""
        params = {
            "face_report": self.session["face_report"],
            "recommendations": self.session["recommendations"],
        }
        report_md = generate_report_tool(json.dumps(params, ensure_ascii=False))

        if report_md.startswith("{"):
            error = json.loads(report_md)
            return f"报告生成失败：{error.get('message', '')}"

        self.session["stage"] = "reported"
        return report_md

    def _handle_preview(self, msg: str) -> str:
        """展示推荐发型的预览图"""
        if not self.session["recommendations"]:
            return "还没有推荐结果哦～先说你的风格偏好，我帮你推荐后再生成预览！"

        recs = self.session["recommendations"]
        response = "📸 **发型效果预览**\n\n以下是为您推荐发型的预览效果图，每张都是根据发型特征 AI 生成的参考图：\n\n"

        for i, rec in enumerate(recs[:3], 1):
            hs = rec["hairstyle"]
            response += f"**{i}. {hs['name']}** (匹配度 {rec['score']:.0%})\n"
            response += f"> {hs.get('length', '')} · {hs.get('curl', '')} · 热度 {hs.get('popularity', 0):.0%}\n"
            if hs.get("care_tips"):
                response += f"> 💡 {hs['care_tips']}\n"
            response += "\n"

        response += (
            "---\n"
            "💡 提示：点击上方推荐卡片的「👗 试戴预览」按钮，可以把这个发型合成到你的照片上哦！\n"
            "需要我帮你对比这几款在不同场合的适配度吗？"
        )
        return response

    def _handle_tryon(self, msg: str) -> str:
        """引导用户通过试戴按钮使用试戴功能"""
        if not self.session["recommendations"]:
            return "还没有推荐结果哦～先说你的风格偏好，我帮你推荐后再试戴！"

        # 尝试匹配用户提到的发型
        matched = None
        for rec in self.session["recommendations"]:
            hs_name = rec["hairstyle"]["name"]
            if hs_name in msg:
                matched = rec
                break

        if not matched:
            # 没匹配到具体发型，列出可试戴的选项
            rec_names = [rec["hairstyle"]["name"] for rec in self.session["recommendations"][:3]]
            names_text = "、".join(rec_names)
            return (
                f"👗 想试戴哪一款呢？当前可试戴的发型有：\n\n"
                + "\n".join(f"- {name}" for name in rec_names)
                + f"\n\n💡 点击上方推荐卡片里的 **「👗 试戴预览」** 按钮就能看到效果！\n"
                f"也可以直接说「试戴{rec_names[0]}」来快速体验~"
            )

        return (
            f"👗 好的！想试戴 **{matched['hairstyle']['name']}** 对吧？\n\n"
            f"💡 请点击上方推荐卡片里的 **「👗 试戴预览」** 按钮，"
            f"我会把这个发型合成到你的照片上，让你看到真实效果！"
        )

    def _handle_default(self) -> str:
        """当用户输入无法匹配任何阶段时，给出引导"""
        if self.session["stage"] == "start":
            return (
                "你好！我是你的专属发型顾问 💇\n\n"
                "上传一张正面照，我就能帮你分析脸型、推荐最适合你的发型。\n\n"
                "或者直接告诉我你想要什么风格，比如「干练通勤风」「日系甜美风」~"
            )

        if self.session["stage"] == "face_analyzed":
            return (
                f"你的脸型是 **{self.session['face_report']['face_shape']}**。\n\n"
                "告诉我你的风格偏好吧！比如：\n"
                "- 🏢 干练通勤风\n"
                "- 🌸 日系甜美风\n"
                "- 🎞️ 复古港风\n"
                "- 😎 酷飒中性风\n"
                "- 🌿 自然慵懒风\n"
                "- 💎 优雅气质风"
            )

        if self.session["stage"] == "recommended":
            return (
                f"你已经有了 {len(self.session['recommendations'])} 条推荐。\n\n"
                "你可以：\n"
                "- 回复「生成报告」查看详细分析\n"
                "- 告诉我新的风格偏好，重新推荐\n"
                "- 上传新的照片重新分析"
            )

        return (
            "我还不太确定你想要什么。试试告诉我风格偏好，"
            "或者上传一张照片？"
        )
