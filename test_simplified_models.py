#!/usr/bin/env python3
"""
测试公司内部 API 的简化模型名称
"""

import os
import anthropic

# 配置环境变量
os.environ['ANTHROPIC_BASE_URL'] = "https://aidev.deyecloud.com/api"
os.environ['ANTHROPIC_AUTH_TOKEN'] = "cr_fd8489bac5fac5a8cc9d234e8a93baf15c65a0fa96e64731c3f36201fe0417b1"

# 基于用户提供的信息，测试这些模型名称
MODELS_TO_TEST = [
    # 完全匹配用户提供的名称
    "Default",
    "Opus",
    "Haiku",

    # 小写版本
    "default",
    "opus",
    "haiku",

    # 带版本号
    "Sonnet 4.5",
    "Opus 4.5",
    "Haiku 4.5",
    "sonnet-4.5",
    "opus-4.5",
    "haiku-4.5",
    "sonnet_4.5",
    "opus_4.5",
    "haiku_4.5",

    # 只有模型系列名
    "Sonnet",
    "sonnet",

    # 其他可能的格式
    "claude-sonnet-4.5",
    "claude-opus-4.5",
    "claude-haiku-4.5",
]

def test_model(model_name: str) -> tuple:
    """测试模型是否可用"""
    try:
        client = anthropic.Anthropic(
            api_key=os.environ['ANTHROPIC_AUTH_TOKEN'],
            base_url=os.environ['ANTHROPIC_BASE_URL']
        )

        # 发送简单的测试请求
        response = client.messages.create(
            model=model_name,
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": "请用中文回复：你好"
            }]
        )

        response_text = response.content[0].text
        return True, response_text[:100]

    except Exception as e:
        error_msg = str(e)
        if "No available Claude accounts" in error_msg:
            return False, "模型不可用"
        elif "model" in error_msg.lower():
            return False, "模型名称错误"
        else:
            return False, f"错误: {error_msg[:80]}"

def main():
    print("=" * 80)
    print("测试公司内部 API 的简化模型名称")
    print("=" * 80)
    print("基于用户提供的信息:")
    print("  1. Default (Sonnet 4.5)")
    print("  2. Opus (Opus 4.5)")
    print("  3. Haiku (Haiku 4.5)")
    print("=" * 80)
    print()

    available_models = []

    for model in MODELS_TO_TEST:
        print(f"测试: {model:<30} ", end="", flush=True)

        success, result = test_model(model)

        if success:
            print(f"✅ 可用! 响应: {result}")
            available_models.append(model)
        else:
            print(f"❌ {result}")

    print()
    print("=" * 80)
    print("测试结果")
    print("=" * 80)

    if available_models:
        print(f"✅ 找到 {len(available_models)} 个可用模型:")
        print()
        for idx, model in enumerate(available_models, 1):
            print(f"{idx}. {model}")

        print()
        print("=" * 80)
        print("推荐配置")
        print("=" * 80)

        # 根据用户提供的价格推荐
        print("根据任务选择模型:")
        print()

        for model in available_models:
            model_lower = model.lower()
            if 'haiku' in model_lower:
                print(f"🚀 快速任务 (Haiku 4.5): {model}")
                print("   - 最快速度")
                print("   - 成本最低 ($1/$5 per Mtok)")
                print("   - 适合: 简单分类、快速标注")
                haiku_model = model
                break

        for model in available_models:
            model_lower = model.lower()
            if 'default' in model_lower or ('sonnet' in model_lower and '4.5' in model_lower):
                print(f"⭐ 推荐 (Sonnet 4.5): {model}")
                print("   - 性能平衡")
                print("   - 成本适中 ($3/$15 per Mtok)")
                print("   - 适合: 教学风格标注 (首选)")
                sonnet_model = model
                break

        for model in available_models:
            model_lower = model.lower()
            if 'opus' in model_lower:
                print(f"💎 高精度 (Opus 4.5): {model}")
                print("   - 最强能力")
                print("   - 成本最高 ($5/$25 per Mtok)")
                print("   - 适合: 复杂分析、低置信度样本")
                opus_model = model
                break

        print()
        print("=" * 80)
        print("立即使用")
        print("=" * 80)

        # 使用找到的第一个模型
        recommended = available_models[0]

        print(f"修改配置使用模型: {recommended}")
        print()
        print("# 方法1: 测试标注器")
        print(f"python -c \"")
        print(f"from src.annotation.vlm_annotator import VLMStyleAnnotator")
        print(f"import os")
        print(f"annotator = VLMStyleAnnotator(")
        print(f"    api_key=os.environ['ANTHROPIC_AUTH_TOKEN'],")
        print(f"    base_url=os.environ['ANTHROPIC_BASE_URL'],")
        print(f"    model='{recommended}'  # 使用这个！")
        print(f")")
        print(f"result = annotator.annotate_single_sample(")
        print(f"    behavior_sequence=['Writing', 'Explaining'],")
        print(f"    lecture_text='今天学习数学...'")
        print(f")")
        print(f"print(f'风格: {{result[\\\"style\\\"]}} (置信度: {{result[\\\"confidence\\\"]:.2f}})')")
        print(f"\"")
        print()
        print("# 方法2: 修改批量标注脚本")
        print(f"python annotate_tbu.py annotate \\")
        print(f"    --input data/tbu/tbu_for_annotation.json \\")
        print(f"    --output data/tbu/tbu_annotated.json \\")
        print(f"    --model {recommended} \\")
        print(f"    --max_samples 10  # 先测试10个")

    else:
        print("❌ 没有找到可用的模型")
        print()
        print("建议:")
        print("  1. 检查 API 密钥是否有效")
        print("  2. 确认网络连接正常")
        print("  3. 联系 IT 确认正确的模型名称")

    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
