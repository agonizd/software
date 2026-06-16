# 💇 AI 发型顾问 (AI Hair Advisor)

基于 AI 的智能发型推荐系统。上传一张自拍照片，即可获得：
- **脸型分析** — MediaPipe 468 点人脸关键点检测，6 种脸型分类
- **个性化推荐** — 多因子打分（脸型匹配 + 风格相似度 + 流行度）
- **虚拟试戴** — 将推荐发型合成到你的照片上预览效果
- **风格报告** — 生成详细的风格分析报告

## 功能特性

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| A · 脸型检测 | 468 关键点提取 → 面部比例分析 → 6 种脸型分类 | MediaPipe + OpenCV |
| B · 发型数据库 | JSON 数据库 + 风格向量检索 + CRUD | Python 标准库 |
| C · 推荐引擎 | face_match×0.5 + style_similarity×0.3 + popularity×0.2 | NumPy |
| D · 风格报告 | Markdown 风格分析报告生成 | OpenAI API（可选） |
| E · 前端交互 | Streamlit 聊天式 UI + LangChain ReAct Agent | Streamlit + LangChain |
| 虚拟试戴 | 发型预览图关键点对齐 → alpha 羽化合成 | MediaPipe + OpenCV + Pillow |

## 项目结构

```
software/
├── agent_app/             # E 模块 — Streamlit 前端 + Agent 调度
│   ├── app.py             # 主入口、UI 渲染
│   ├── agent.py           # MockAgent 规则引擎
│   ├── tools.py           # LangChain Tool 封装（动态导入）
│   ├── virtual_tryon.py   # 虚拟试戴合成
│   └── tests/
├── face_detect/           # A 模块 — 脸型检测
│   ├── detector.py        # MediaPipe 检测器
│   ├── landmarks.py       # 关键点提取与计算
│   └── classifier.py      # 脸型分类器
├── hairstyle_db/          # B 模块 — 发型数据库
│   ├── hairstyles.json    # 发型数据
│   ├── db.py              # CRUD 检索
│   └── images/            # 发型预览图（10 款）
├── recommend_engine/      # C 模块 — 推荐引擎
│   ├── engine.py          # 推荐主逻辑
│   ├── scoring.py         # 多因子打分
│   └── reverse_infer.py   # 风格向量反推
├── style_report/          # D 模块 — 风格报告
│   └── generator.py       # 报告生成器
├── contracts.py           # 数据合约（数据类定义）
├── contracts_mock.py      # Mock 实现（无 API 模式）
├── tests/                 # 集成测试
│   └── test_integration.py
├── requirements.txt       # Python 依赖
├── .streamlit/            # Streamlit 部署配置
│   └── config.toml
└── .gitignore
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/agonizd/software.git
cd software
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行

**Mock 模式（默认，无需 API Key）：**

```bash
streamlit run agent_app/app.py
```

浏览器打开 http://localhost:8501 即可使用。

**Agent 模式（需要 OpenAI API Key）：**

```bash
# 先配置 API Key
cp recommend_engine/.env.example .env
# 编辑 .env，填入 OPENAI_API_KEY=sk-xxx

# 启动（自动检测 API Key 后启用 Agent 模式）
streamlit run agent_app/app.py
```

### 4. 运行测试

```bash
pytest tests/ -v
```

## 两种运行模式

| | Mock 模式 | Agent 模式 |
|---|---|---|
| **需要 API Key** | 不需要 | 需要 OpenAI |
| **费用** | 免费 | 按 API 用量计费 |
| **推理方式** | 规则引擎自动调度 | LLM 自然语言推理 |
| **功能** | 完整（检测+推荐+试戴+报告） | 完整 + 更智能的对话 |
| **推荐** | 日常使用、演示 | 需要自然语言交互时 |

## 使用方法

1. 启动后，在左侧边栏上传一张正面自拍照片
2. 系统自动分析脸型，显示分析结果
3. 点击推荐标签（如"适合圆脸""韩式卷发"）或直接在聊天框输入需求
4. 查看推荐发型卡片，点击"虚拟试戴"预览效果
5. 获取详细的风格分析报告

## 技术细节

### 脸型分类

基于 468 个面部关键点，计算以下比例特征进行分类：

- 脸型长宽比（Face Aspect Ratio）
- 颧颌宽度比（Cheekbone-Jaw Ratio）  
- 额头比（Forehead Ratio）

支持的 6 种脸型：鹅蛋脸、圆脸、方脸、长脸、心形脸、菱形脸。

### 推荐算法

```
综合得分 = 脸型匹配度 × 0.50 + 风格相似度 × 0.30 + 流行度 × 0.20
```

### 模型下载

首次运行时，MediaPipe 会自动从 Google 下载 `face_landmarker.task`（约 3.8 MB）。如网络受限，可手动下载后放入项目根目录。

## 系统要求

- Python 3.8+
- 内存 1GB+
- 首次运行需要网络（下载模型）

## License

MIT
