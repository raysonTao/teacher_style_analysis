#!/bin/bash

# 直接测试 API 的原始 HTTP 请求

echo "========================================"
echo "直接测试 Claude API HTTP 请求"
echo "========================================"
echo ""

API_URL="https://aidev.deyecloud.com/api/v1/messages"
API_KEY="cr_fd8489bac5fac5a8cc9d234e8a93baf15c65a0fa96e64731c3f36201fe0417b1"

# 测试不同的模型名称
MODELS=(
    "claude-sonnet-4.5"
    "sonnet"
    "default"
    "Default"
    "claude-3-5-sonnet-20241022"
)

for MODEL in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "测试模型: $MODEL"
    echo "----------------------------------------"

    # 构建请求体
    REQUEST_BODY=$(cat <<EOF
{
    "model": "$MODEL",
    "max_tokens": 10,
    "messages": [
        {
            "role": "user",
            "content": "Hi"
        }
    ]
}
EOF
)

    # 发送请求
    RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
        -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -H "x-api-key: $API_KEY" \
        -H "anthropic-version: 2023-06-01" \
        -d "$REQUEST_BODY")

    # 提取 HTTP 状态码
    HTTP_CODE=$(echo "$RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
    RESPONSE_BODY=$(echo "$RESPONSE" | sed '/HTTP_CODE:/d')

    echo "HTTP 状态码: $HTTP_CODE"
    echo "响应:"
    echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_BODY"
    echo ""
done

echo "========================================"
echo "测试完成"
echo "========================================"
