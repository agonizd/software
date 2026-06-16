"""
generate_previews.py — 发型预览图生成脚本
批量调用 ImageGen 并整理命名
"""

import os
import shutil

IMAGES_DIR = os.path.dirname(os.path.abspath(__file__)) + "/images"

# 10 款发型的生成提示词（英文，用于 AI 绘图）
HAIRSTYLE_PROMPTS = [
    {
        "id": "hair_001",
        "name": "锁骨微卷发",
        "filename": "hair_001_clavicle_wavy.png",
        "prompt": (
            "Clavicle wavy hair on East Asian woman: shoulder-length soft waves, "
            "side-parted chestnut brown hair, elegant feminine look, "
            "studio lighting, front-facing portrait, beauty magazine style, 8k"
        ),
    },
    {
        "id": "hair_002",
        "name": "层次中长发",
        "filename": "hair_002_layered_medium.png",
        "prompt": (
            "Layered medium haircut on East Asian woman: sleek straight layers "
            "at collarbone length, modern clean cut, professional look, "
            "white studio background, front-facing, beauty salon photo, 8k"
        ),
    },
    {
        "id": "hair_003",
        "name": "日系空气刘海",
        "filename": "hair_003_air_bangs.png",
        "prompt": (
            "Air bangs hairstyle on East Asian woman: wispy see-through bangs, "
            "long soft wavy hair, Japanese kawaii sweet style, pink-toned backdrop, "
            "front-facing portrait, beauty salon photo, 8k"
        ),
    },
    {
        "id": "hair_004",
        "name": "黑长直",
        "filename": "hair_004_black_straight.png",
        "prompt": (
            "Black long straight hair on East Asian woman: sleek glossy straight hair "
            "past shoulders, classic elegant beauty, clean white background, "
            "front-facing portrait, beauty salon photo, 8k"
        ),
    },
    {
        "id": "hair_005",
        "name": "干练齐肩短发",
        "filename": "hair_005_bob.png",
        "prompt": (
            "Bob haircut on East Asian woman: sleek chin-length blunt bob, "
            "sharp clean lines, professional capable look, gray background, "
            "front-facing portrait, beauty salon photo, 8k"
        ),
    },
    {
        "id": "hair_006",
        "name": "少年感碎短发",
        "filename": "hair_006_pixie_cut.png",
        "prompt": (
            "Pixie cut on East Asian woman: short textured choppy crop, "
            "youthful fresh boyish look, messy but styled, white studio, "
            "front-facing portrait, beauty salon photo, 8k"
        ),
    },
    {
        "id": "hair_007",
        "name": "复古港风大波浪",
        "filename": "hair_007_vintage_waves.png",
        "prompt": (
            "Vintage waves on East Asian woman: retro Hong Kong style big curls, "
            "voluminous glamorous waves, side-parted 90s HK cinema look, "
            "warm-toned studio, front-facing portrait, beauty magazine, 8k"
        ),
    },
    {
        "id": "hair_008",
        "name": "法式羊毛卷",
        "filename": "hair_008_french_perm.png",
        "prompt": (
            "French perm on East Asian woman: tight curly voluminous perm, "
            "textured fluffy curls, Parisian chic effortless look, "
            "natural light studio, front-facing portrait, beauty salon photo, 8k"
        ),
    },
    {
        "id": "hair_009",
        "name": "韩式蛋卷头",
        "filename": "hair_009_korean_curl.png",
        "prompt": (
            "Korean curl on East Asian woman: soft bouncy C-curl waves, "
            "K-beauty egg roll perm, sweet romantic look, pastel studio, "
            "front-facing portrait, beauty salon photo, 8k"
        ),
    },
    {
        "id": "hair_010",
        "name": "一刀切短发",
        "filename": "hair_010_blunt_bob.png",
        "prompt": (
            "Blunt bob on East Asian woman: sharp straight-across cut at jawline, "
            "geometric clean edge, bold edgy cool look, dark backdrop, "
            "front-facing portrait, beauty salon photo, 8k"
        ),
    },
]


def print_generation_instructions():
    """打印需要逐个生成的提示词列表"""
    print("=" * 60)
    print("  发型预览图生成 — 请逐个复制以下提示词到 ImageGen 工具")
    print("=" * 60)
    print()
    for h in HAIRSTYLE_PROMPTS:
        print(f"  [{h['id']}] {h['name']} → {h['filename']}")
        print(f"  Prompt: {h['prompt']}")
        print()


if __name__ == "__main__":
    print_generation_instructions()
