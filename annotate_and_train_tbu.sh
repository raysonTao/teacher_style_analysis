#!/bin/bash

# TBU 数据集标注和训练流程
# 使用公司内部 Claude API 进行全自动标注

set -e  # 遇到错误立即退出

echo "=================================="
echo "TBU 数据集标注和训练流程"
echo "=================================="

# 配置环境变量
export ANTHROPIC_BASE_URL="https://aidev.deyecloud.com/api"
export ANTHROPIC_AUTH_TOKEN="cr_fd8489bac5fac5a8cc9d234e8a93baf15c65a0fa96e64731c3f36201fe0417b1"

# CUDA 配置
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7

# 配置参数
TBU_PATH=${1:-"data/TBU"}  # TBU 数据集路径（第一个参数）
MAX_SAMPLES=${2:-""}       # 最大样本数（可选，用于测试）
MODEL="claude-3-5-sonnet-20241022"

echo ""
echo "配置信息:"
echo "  TBU 数据路径: $TBU_PATH"
echo "  API地址: $ANTHROPIC_BASE_URL"
echo "  使用模型: $MODEL"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  最大样本数: $MAX_SAMPLES (测试模式)"
fi
echo ""

# 检查 TBU 数据是否存在
if [ ! -d "$TBU_PATH" ]; then
    echo "错误: TBU 数据路径不存在: $TBU_PATH"
    echo ""
    echo "请先下载 TBU 数据集:"
    echo "  git clone https://github.com/cai-KU/TBU $TBU_PATH"
    exit 1
fi

echo "步骤 1/4: 转换 TBU 数据集为标注格式"
echo "------------------------------------"
CONVERT_ARGS="--tbu_path $TBU_PATH --output data/tbu/tbu_for_annotation.json"
if [ -n "$MAX_SAMPLES" ]; then
    CONVERT_ARGS="$CONVERT_ARGS --max_samples $MAX_SAMPLES"
fi

python -m src.annotation.convert_tbu $CONVERT_ARGS

if [ $? -ne 0 ]; then
    echo "错误: 数据转换失败"
    exit 1
fi

echo ""
echo "步骤 2/4: 使用 VLM 批量标注教学风格"
echo "------------------------------------"
echo "注意: 这将调用大量 API，完整数据集可能需要数小时"
echo ""

ANNOTATE_ARGS="--input data/tbu/tbu_for_annotation.json --output data/tbu/tbu_annotated.json --model $MODEL --save_interval 10"
if [ -n "$MAX_SAMPLES" ]; then
    ANNOTATE_ARGS="$ANNOTATE_ARGS --max_samples $MAX_SAMPLES"
fi

python annotate_tbu.py annotate $ANNOTATE_ARGS

if [ $? -ne 0 ]; then
    echo "错误: 标注失败"
    exit 1
fi

echo ""
echo "步骤 3/4: 转换标注结果为训练格式"
echo "------------------------------------"

python annotate_tbu.py convert \
    --input data/tbu/tbu_annotated.json \
    --output data/tbu/tbu_training.json \
    --train_ratio 0.7 \
    --val_ratio 0.15

if [ $? -ne 0 ]; then
    echo "错误: 格式转换失败"
    exit 1
fi

echo ""
echo "步骤 4/4: 训练深度学习模型"
echo "------------------------------------"

python -m src.models.deep_learning.train \
    --data_path data/tbu/tbu_training.json \
    --model_config default \
    --batch_size 64 \
    --num_epochs 150 \
    --lr 5e-5 \
    --device cuda \
    --num_workers 4 \
    --checkpoint_dir ./checkpoints/tbu \
    --log_dir ./logs/tbu

if [ $? -ne 0 ]; then
    echo "错误: 训练失败"
    exit 1
fi

echo ""
echo "=================================="
echo "完成！"
echo "=================================="
echo ""
echo "训练结果:"
echo "  模型文件: ./checkpoints/tbu/best_model.pth"
echo "  训练日志: ./logs/tbu/"
echo "  标注文件: data/tbu/tbu_annotated.json"
echo "  训练数据: data/tbu/tbu_training.json"
echo ""
echo "使用模型:"
echo "  python -m src.main analyze --video your_video.mp4 --mode deep_learning --device cuda"
echo ""
