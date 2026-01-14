#!/bin/bash
# 批量处理课堂录播视频 - 特征提取 + VLM标注
# 创建时间: 2026-01-13
# 用法: nohup bash process_new_videos.sh > process_log.txt 2>&1 &

set -e  # 遇到错误立即退出

echo "==================================="
echo "开始处理课堂录播视频"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================="

# 定义变量
VIDEOS_DIR="data/custom/videos"
FEATURES_OUTPUT="data/custom/extracted_features/features_classroom_$(date +%Y%m%d_%H%M%S).json"
ANNOTATIONS_OUTPUT="data/custom/vlm_annotations/annotations_classroom_$(date +%Y%m%d_%H%M%S).json"
VLM_MODEL="opus"  # 使用opus获得高质量标注
NUM_KEYFRAMES=10
TEACHER_ID="classroom_recordings"
DISCIPLINE="计算机科学"
GRADE="大学"

# 检查目录是否存在
if [ ! -d "$VIDEOS_DIR" ]; then
    echo "错误: 视频目录不存在: $VIDEOS_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p data/custom/extracted_features
mkdir -p data/custom/vlm_annotations

echo ""
echo "步骤 1/2: 提取视频特征"
echo "-----------------------------------"
echo "视频目录: $VIDEOS_DIR"
echo "输出文件: $FEATURES_OUTPUT"
echo "教师ID: $TEACHER_ID"
echo "学科: $DISCIPLINE"
echo "年级: $GRADE"
echo ""

# 步骤1: 提取特征
python data/custom/tools/extract_features.py \
    --videos_dir "$VIDEOS_DIR" \
    --output "$FEATURES_OUTPUT" \
    --teacher_id "$TEACHER_ID" \
    --discipline "$DISCIPLINE" \
    --grade "$GRADE"

if [ $? -ne 0 ]; then
    echo "错误: 特征提取失败"
    exit 1
fi

echo ""
echo "✓ 特征提取完成"
echo "输出文件: $FEATURES_OUTPUT"
echo "文件大小: $(du -h "$FEATURES_OUTPUT" | cut -f1)"
echo ""

echo "步骤 2/2: VLM标注"
echo "-----------------------------------"
echo "特征文件: $FEATURES_OUTPUT"
echo "输出文件: $ANNOTATIONS_OUTPUT"
echo "VLM模型: $VLM_MODEL"
echo "关键帧数: $NUM_KEYFRAMES"
echo ""

# 步骤2: VLM标注
python data/custom/tools/annotate_videos.py \
    --features_path "$FEATURES_OUTPUT" \
    --output "$ANNOTATIONS_OUTPUT" \
    --model "$VLM_MODEL" \
    --max_samples 0 \
    --num_keyframes $NUM_KEYFRAMES

if [ $? -ne 0 ]; then
    echo "错误: VLM标注失败"
    exit 1
fi

echo ""
echo "✓ VLM标注完成"
echo "输出文件: $ANNOTATIONS_OUTPUT"
echo "文件大小: $(du -h "$ANNOTATIONS_OUTPUT" | cut -f1)"
echo ""

# 统计信息
echo "==================================="
echo "处理完成汇总"
echo "==================================="
echo "特征文件: $FEATURES_OUTPUT"
echo "标注文件: $ANNOTATIONS_OUTPUT"

# 统计视频数量
VIDEO_COUNT=$(find "$VIDEOS_DIR" -type f -name "*.mp4" | wc -l)
echo "处理视频数: $VIDEO_COUNT"

# 统计标注样本数
if [ -f "$ANNOTATIONS_OUTPUT" ]; then
    SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('$ANNOTATIONS_OUTPUT'))))" 2>/dev/null || echo "未知")
    echo "标注样本数: $SAMPLE_COUNT"
fi

echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================="
echo ""
echo "下一步操作："
echo "1. 查看标注结果: cat $ANNOTATIONS_OUTPUT | jq ."
echo "2. 转换为训练格式: python data/custom/tools/convert_to_training.py \\"
echo "      --annotations $ANNOTATIONS_OUTPUT \\"
echo "      --features $FEATURES_OUTPUT \\"
echo "      --output data/custom/training_data/training_classroom.json"
echo ""
