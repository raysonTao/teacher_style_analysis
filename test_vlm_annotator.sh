#!/bin/bash

# 快速测试 VLM 标注器
# 测试公司内部 Claude API 是否正常工作

set -e

echo "=================================="
echo "VLM 标注器快速测试"
echo "=================================="

# 配置环境变量
export ANTHROPIC_BASE_URL="https://aidev.deyecloud.com/api"
export ANTHROPIC_AUTH_TOKEN="cr_fd8489bac5fac5a8cc9d234e8a93baf15c65a0fa96e64731c3f36201fe0417b1"

echo ""
echo "配置信息:"
echo "  API地址: $ANTHROPIC_BASE_URL"
echo "  API密钥: ${ANTHROPIC_AUTH_TOKEN:0:20}..."
echo ""

echo "运行测试..."
echo ""

python -m src.annotation.vlm_annotator

echo ""
echo "=================================="
echo "测试完成！"
echo "=================================="
echo ""
echo "如果看到标注结果，说明 API 工作正常"
echo "现在可以运行完整流程:"
echo "  ./annotate_and_train_tbu.sh /path/to/TBU"
echo ""
