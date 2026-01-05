#!/bin/bash
# GPU训练脚本 - 使用MM-TBA转换数据

# 设置CUDA路径（避免cuDNN版本问题）
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.7

# 切换到项目目录
cd /home/rayson/code/teacher_style_analysis

echo "========================================="
echo "步骤 1: 转换 MM-TBA 数据集"
echo "========================================="

# 转换数据
python convert_mmtba.py

echo ""
echo "========================================="
echo "步骤 2: 使用 MM-TBA 数据训练模型"
echo "========================================="

# 训练参数
MODEL_CONFIG="default"  # 可选: default, lightweight, high_accuracy
BATCH_SIZE=32           # MM-TBA数据较少，使用较小的batch size
NUM_EPOCHS=150
LEARNING_RATE=5e-5      # 真实数据使用较小学习率

echo "模型配置: $MODEL_CONFIG"
echo "Batch Size: $BATCH_SIZE"
echo "训练轮数: $NUM_EPOCHS"
echo "========================================="

# 训练模型
python -m src.models.deep_learning.train \
    --data_path data/mm-tba/mmtba_converted.json \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --weight_decay 1e-5 \
    --optimizer adamw \
    --scheduler cosine \
    --early_stopping 15 \
    --device cuda \
    --num_workers 4 \
    --checkpoint_dir ./checkpoints/mmtba \
    --log_dir ./logs/mmtba \
    --seed 42

echo ""
echo "========================================="
echo "训练完成！"
echo "模型保存在: ./checkpoints/mmtba/best_model.pth"
echo "训练日志保存在: ./logs/mmtba/"
echo "========================================="
echo ""
echo "使用训练好的模型："
echo "python -m src.main analyze \\"
echo "  --video your_video.mp4 \\"
echo "  --teacher teacher001 \\"
echo "  --discipline 数学 \\"
echo "  --grade 高中 \\"
echo "  --mode deep_learning \\"
echo "  --device cuda"
