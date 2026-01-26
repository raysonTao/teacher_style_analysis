"""系统配置文件"""
import os
import logging
import sys
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
LOG_DIR = BASE_DIR.parent / 'log'  # 日志目录配置在/src同级
VISUALIZATION_DIR = BASE_DIR.parent / 'result'  # 可视化结果目录

# 初始化目录函数
def init_directories():
    """创建所有必要的目录"""
    directories = [
        VIDEO_DIR, AUDIO_DIR, TEXT_DIR, FEATURES_DIR, RESULTS_DIR,
        BASE_DIR / 'models', BASE_DIR / 'experiments' / 'results',
        BASE_DIR / 'experiments' / 'visualizations', LOG_DIR
    ]
    for dir_path in directories:
        dir_path.mkdir(exist_ok=True, parents=True)

# 创建必要的目录
for dir_path in [VIDEO_DIR, AUDIO_DIR, TEXT_DIR, FEATURES_DIR, RESULTS_DIR, LOG_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# 自定义类将stdout和stderr重定向到logger
class StdoutToLogging:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ''
    
    def write(self, message):
        self.buffer += message
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line:
                self.logger.log(self.level, line)
    
    def flush(self):
        if self.buffer:
            self.logger.log(self.level, self.buffer)
            self.buffer = ''

# 配置全局logger
logger = logging.getLogger('teacher_style_analysis')
logger.setLevel(logging.INFO)
logger.propagate = False  # 防止日志传播到父日志器

# 确保logger只有一个处理器
if not logger.handlers:
    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建文件处理器
    from datetime import datetime
    log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    log_filepath = LOG_DIR / log_filename
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 重定向stdout和stderr到logger
sys.stdout = StdoutToLogging(logger, logging.INFO)
sys.stderr = StdoutToLogging(logger, logging.ERROR)


# 模型配置
MODEL_CONFIG = {
    'yolo_model_path': 'models/weights/yolov8n.pt',  # 动作检测模型
    'openpose_model_dir': str(BASE_DIR / 'models/openpose'),  # 姿态估计模型（已替换为MediaPipe）
    'whisper_model': 'medium',  # 语音识别模型
    'bert_model': 'bert-base-chinese',  # 文本特征提取模型
    'style_classifier_path': 'models/style_classifier.pth',  # 风格分类器模型
    'cmat_model_path': 'models/cmat_model.pkl',  # CMAT模型路径
    'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',  # 自监督声学表征模型
    'wav2vec2_emotion_model_name': 'superb/wav2vec2-base-superb-er',  # 情感识别模型
    'dialogue_act_model_name': 'bert-base-chinese',  # 对话行为识别模型
    'stgcn_model_path': 'models/weights/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth',  # ST-GCN模型路径
    'deepsort_model_path': 'models/weights/osnet_x0_25_msmt17.pth',  # DeepSORT外观模型
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
    'test_frame_limit': 100,
    # 可视化配置
    'enable_visualization': True,  # 启用可视化
    'visualization_frame_interval': 30,  # 可视化采样间隔（与检测间隔一致）
    'save_visualization_video': True,  # 保存可视化视频
    'save_visualization_frames': True,  # 保存可视化帧图片
    'bbox_color': (0, 0, 255),  # 红色边界框 (BGR: 蓝,绿,红)
    'bbox_thickness': 2,  # 边界框线条粗细
    'pose_text_color': (255, 0, 0),  # 蓝色文本 (BGR: 蓝,绿,红)
    'text_font': 0,  # 字体 (cv2.FONT_HERSHEY_SIMPLEX = 0)
    'text_font_scale': 0.6,  # 字体大小
    'text_thickness': 2,  # 文本粗细
    'keypoint_color': (0, 255, 0),  # 绿色关键点 (BGR)
    'keypoint_radius': 3,  # 关键点半径
    'skeleton_color': (0, 255, 255),  # 黄色骨架线 (BGR)
    'skeleton_thickness': 2,  # 骨架线粗细
    # 跟踪与时序动作识别
    'tracker_type': 'deepsort',
    'tracker_max_age': 30,
    'tracker_iou_threshold': 0.3,
    'teacher_track_patience': 45,
    'stgcn_sequence_length': 32,
    'stgcn_stride': 8,
    'stgcn_min_confidence': 0.4,
    'stgcn_action_labels': [
        'wave', 'raise_hand_hold', 'pointing', 'writing', 'standing', 'walking'
    ]
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
    'whisper_model_size': 'medium',  # Whisper模型大小
    'wav2vec2_model_name': MODEL_CONFIG['wav2vec2_model_name'],
    'wav2vec2_emotion_model_name': MODEL_CONFIG['wav2vec2_emotion_model_name'],
    'wav2vec2_cache_dir': str(BASE_DIR.parent / 'models' / 'weights' / 'huggingface'),
    'emotion_label_map': {},
    'local_files_only': True
}

# 文本配置
TEXT_CONFIG = {
    'bert_model_name': str(
        BASE_DIR / 'models' / 'weights' / 'huggingface' / 'hub'
        / 'models--bert-base-chinese' / 'snapshots'
        / '8f23c25b06e129b6c986331a13d8d025a92cf0ea'
    ),
    'dialogue_act_model_name': str(
        BASE_DIR / 'models' / 'weights' / 'huggingface' / 'hub'
        / 'models--bert-base-chinese' / 'snapshots'
        / '8f23c25b06e129b6c986331a13d8d025a92cf0ea'
    ),
    'dialogue_act_labels': ['question', 'instruction', 'explanation', 'feedback'],
    'dialogue_act_label_map': {
        'question': '提问',
        'instruction': '指令',
        'explanation': '讲解',
        'feedback': '反馈'
    },
    'max_length': 256,
    'local_files_only': True
}

# 离线模式：避免运行时访问Hugging Face
if AUDIO_CONFIG.get('local_files_only') or TEXT_CONFIG.get('local_files_only'):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# 多模态融合配置
FUSION_CONFIG = {
    'fusion_method': 'concat',
    'feature_weights': {
        'video': 0.4,
        'audio': 0.3,
        'text': 0.3
    },
    'normalization': True,
    'use_mman': True,
    'mman_config': 'default',
    'mman_checkpoint': 'checkpoints/best_model.pth'
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
