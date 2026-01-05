#!/usr/bin/env python3
"""
测试公司内部 Claude API 支持哪些模型
"""

import os
import anthropic

# 配置环境变量
os.environ['ANTHROPIC_BASE_URL'] = "https://aidev.deyecloud.com/api"
os.environ['ANTHROPIC_AUTH_TOKEN'] = "cr_fd8489bac5fac5a8cc9d234e8a93baf15c65a0fa96e64731c3f36201fe0417b1"

# 常见的 Claude 模型列表
MODELS_TO_TEST = [
    "claude-3-5-sonnet-20241022",  # 最新的 Sonnet 3.5
    "claude-3-5-sonnet-20240620",  # 旧版 Sonnet 3.5
    "claude-3-sonnet-20240229",    # Claude 3 Sonnet
    "claude-3-opus-20240229",      # Claude 3 Opus
    "claude-3-haiku-20240307",     # Claude 3 Haiku
    "claude-2.1",                  # Claude 2.1
    "claude-2.0",                  # Claude 2.0
    "claude-instant-1.2",          # Claude Instant
]

def test_model(model_name: str) -> bool:
    """测试模型是否可用"""
    try:
        client = anthropic.Anthropic(
            api_key=os.environ['ANTHROPIC_AUTH_TOKEN'],
            base_url=os.environ['ANTHROPIC_BASE_URL']
        )

        # 发送简单的测试请求
        response = client.messages.create(
            model=model_name,
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": "Hi"
            }]
        )

        return True, response.content[0].text

    except Exception as e:
        error_msg = str(e)
        if "No available Claude accounts" in error_msg:
            return False, "模型不可用"
        else:
            return False, f"错误: {error_msg[:100]}"

def main():
    print("=" * 80)
    print("测试公司内部 Claude API 支持的模型")
    print("=" * 80)
    print(f"API地址: {os.environ['ANTHROPIC_BASE_URL']}")
    print(f"API密钥: {os.environ['ANTHROPIC_AUTH_TOKEN'][:20]}...")
    print("=" * 80)
    print()

    available_models = []

    for model in MODELS_TO_TEST:
        print(f"测试模型: {model:<40} ", end="", flush=True)

        success, result = test_model(model)

        if success:
            print(f"✅ 可用")
            available_models.append(model)
        else:
            print(f"❌ {result}")

    print()
    print("=" * 80)
    print("总结")
    print("=" * 80)

    if available_models:
        print(f"✅ 找到 {len(available_models)} 个可用模型:")
        for model in available_models:
            print(f"  - {model}")
        print()
        print("推荐使用:")
        print(f"  {available_models[0]}")
    else:
        print("❌ 没有找到可用的模型")
        print()
        print("可能的原因:")
        print("  1. API 密钥无效")
        print("  2. API 地址错误")
        print("  3. 网络连接问题")
        print("  4. 公司内部 API 使用不同的模型命名")

    print("=" * 80)

if __name__ == '__main__':
    main()
