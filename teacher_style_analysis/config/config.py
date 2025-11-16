"""系统配置文件"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 数据路径配置
DATA_DIR = BASE_DIR / 'data'
VIDEO_DIR = DATA_DIR / 'videos'
AUDIO_DIR = DATA_DIR / 'audio'
TEXT_DIR = DATA_DIR / 'text'
FEATURES_DIR = DATA_DIR / 'extracted_features'
RESULTS_DIR = DATA_DIR / 'results'

# 创建必要的目录
for dir_path in [VIDEO_DIR, AUDIO_DIR, TEXT_DIR, FEATURES_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# 模型配置
MODEL_CONFIG = {
    'yolo_model_path': 'yolov8n.pt',  # 动作检测模型
    'openpose_model_dir': str(BASE_DIR / 'models/openpose'),  # 姿态估计模型
    'whisper_model': 'base',  # 语音识别模型
    'bert_model': 'bert-base-chinese',  # 文本分析模型
    'cmat_model_path': str(BASE_DIR / 'models/cmat_model.pkl'),  # 风格分类模型
}

# 风格标签配置
STYLE_LABELS = [
    '理论讲授型',
    '启发引导型',
    '互动导向型',
    '逻辑推导型',
    '题目驱动型',
    '情感表达型',
    '耐心细致型'
]

# 系统参数配置
SYSTEM_CONFIG = {
    'frame_rate': 30,  # 视频处理帧率
    'audio_sample_rate': 16000,  # 音频采样率
    'feature_extraction_batch_size': 4,  # 特征提取批处理大小
    'style_match_threshold': 0.6,  # 风格匹配阈值
    'lambda_weight': 0.4,  # 规则与学习结果融合权重
}

# API配置
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'max_upload_size': 100 * 1024 * 1024,  # 100MB
}

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',
    'database': 'teacher_style_analysis',
    'port': 3306
}

# Redis配置
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}