#!/bin/bash
# GPU训练脚本 - 使用合成数据

# 设置CUDA路径（避免cuDNN版本问题）
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7

# 切换到项目目录
cd /home/rayson/code/teacher_style_analysis

# 训练参数
MODEL_CONFIG="default"  # 可选: default, lightweight, high_accuracy
BATCH_SIZE=64           # GPU可以使用更大的batch size
NUM_EPOCHS=200          # GPU训练可以训练更多轮
LEARNING_RATE=1e-4
NUM_SYNTHETIC=5000      # 合成数据样本数

echo "========================================="
echo "开始GPU训练 - MMAN教师风格分类模型"
echo "========================================="
echo "模型配置: $MODEL_CONFIG"
echo "Batch Size: $BATCH_SIZE"
echo "训练轮数: $NUM_EPOCHS"
echo "合成样本数: $NUM_SYNTHETIC"
echo "========================================="

# 使用合成数据训练
python -m src.models.deep_learning.train \
    --use_synthetic \
    --num_synthetic $NUM_SYNTHETIC \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --weight_decay 1e-5 \
    --optimizer adamw \
    --scheduler cosine \
    --early_stopping 20 \
    --device cuda \
    --num_workers 4 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs \
    --seed 42

echo "========================================="
echo "训练完成！"
echo "模型保存在: ./checkpoints/best_model.pth"
echo "训练日志保存在: ./logs/"
echo "========================================="

