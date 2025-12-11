"""系统配置文件"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR
DATA_DIR = BASE_DIR / 'data'

# 数据路径配置
VIDEO_DIR = DATA_DIR / 'videos'
AUDIO_DIR = DATA_DIR / 'audio'
TEXT_DIR = DATA_DIR / 'text'
FEATURES_DIR = DATA_DIR / 'extracted_features'
RESULTS_DIR = DATA_DIR / 'results'
TEMP_DIR = BASE_DIR / 'temp'
FEEDBACK_DIR = BASE_DIR / 'feedback'

# 初始化目录函数
def init_directories():
    """创建所有必要的目录"""
    directories = [
        VIDEO_DIR, AUDIO_DIR, TEXT_DIR, FEATURES_DIR, RESULTS_DIR,
        BASE_DIR / 'models', BASE_DIR / 'experiments' / 'results',
        BASE_DIR / 'experiments' / 'visualizations'
    ]
    for dir_path in directories:
        dir_path.mkdir(exist_ok=True, parents=True)

# 创建必要的目录
for dir_path in [VIDEO_DIR, AUDIO_DIR, TEXT_DIR, FEATURES_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# 模型配置
MODEL_CONFIG = {
    'yolo_model_path': 'models/weights/yolov8n.pt',  # 动作检测模型
    'openpose_model_dir': str(BASE_DIR / 'models/openpose'),  # 姿态估计模型（已替换为MediaPipe）
    'whisper_model': 'medium',  # 语音识别模型
    'bert_model': 'bert-base-chinese',  # 文本特征提取模型
    'style_classifier_path': 'models/style_classifier.pth',  # 风格分类器模型
    'cmat_model_path': 'models/cmat_model.pkl',  # CMAT模型路径
    # MediaPipe配置
    'mediapipe_model_complexity': 1,  # 模型复杂度 0, 1, 2
    'mediapipe_smooth_landmarks': True,  # 平滑关键点
    'mediapipe_min_detection_confidence': 0.5,  # 最小检测置信度
    'mediapipe_min_tracking_confidence': 0.5  # 最小跟踪置信度
}

# 视频配置
VIDEO_CONFIG = {
    'frame_width': 640,
    'frame_height': 480,
    'fps': 30,
    'max_frames': 1000,
    'motion_energy_frame_interval': 10,
    'detection_frame_interval': 30,
    'detection_confidence_threshold': 0.5,
    'test_mode': False,
    'test_frame_limit': 100
}

# 动作配置
ACTION_CONFIG = {
    'num_classes': 10,
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45
}

# 音频配置
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 128,
    'whisper_model_path': 'models/weights/medium.pt',  # Whisper模型路径
    'whisper_model_size': 'medium'  # Whisper模型大小
}

# 文本配置
TEXT_CONFIG = {
    'bert_model_name': 'bert-base-chinese'
}

# 多模态融合配置
FUSION_CONFIG = {
    'fusion_method': 'concat',
    'feature_weights': {
        'video': 0.4,
        'audio': 0.3,
        'text': 0.3
    },
    'normalization': True
}

# API配置
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'version': '1.0.0',
    'max_video_size': 50 * 1024 * 1024,  # 50MB
    'cors_origins': ['*']
}

# 系统配置
SYSTEM_CONFIG = {
    'app_name': '教师风格画像分析系统',
    'version': '1.0.0',
    'description': '基于多模态数据的教师教学风格分析系统',
    'lambda_weight': 0.5  # 融合规则驱动和机器学习的权重
}

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'teacher_style_db',
    'user': 'admin',
    'password': 'admin123'
}

# Redis配置
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

# 风格标签
STYLE_LABELS = {
    'enthusiastic': '热情',
    'explanatory': '讲解',
    'interactive': '互动',
    'analytical': '分析',
    'structured': '结构化',
    'casual': '随意'
}