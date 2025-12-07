"""教学风格分析特征提取模块"""

# 导出主特征提取器
from .feature_extractor import FeatureExtractor

# 导出视频相关特征提取模块
from .object_detection import YOLOObjectDetector
from .pose_estimation import MediaPipePoseEstimator
from .action_recognition import PoseActionRecognizer
from .video_feature_extractor import VideoFeatureExtractor

# 导出音频特征提取模块
from .audio_feature_extractor import AudioFeatureExtractor

# 导出文本特征提取模块
from .text_feature_extractor import TextFeatureExtractor

# 导出多模态融合模块
from .multimodal_fusion import MultimodalFeatureFusion

# 导出版本信息
__version__ = "1.0.0"
__author__ = "Teacher Style Analysis Team"

# 提供便捷的导入方式
__all__ = [
    # 主特征提取器
    "FeatureExtractor",
    
    # 视频相关模块
    "YOLOObjectDetector",
    "MediaPipePoseEstimator",
    "PoseActionRecognizer",
    "VideoFeatureExtractor",
    
    # 音频模块
    "AudioFeatureExtractor",
    
    # 文本模块
    "TextFeatureExtractor",
    
    # 多模态融合
    "MultimodalFeatureFusion",
]

# 模块描述
__doc__ = """教学风格分析特征提取模块

该模块提供了多模态特征提取的完整功能，包括：
1. 视频特征提取（对象检测、姿态估计、动作识别）
2. 音频特征提取（语音识别、情感分析、语速语调）
3. 文本特征提取（语义分析、情感分析、教学术语识别）
4. 多模态特征融合（教学风格评估）

使用示例：
from teacher_style_analysis.features import FeatureExtractor

# 初始化特征提取器
extractor = FeatureExtractor()

# 处理视频文件
features = extractor.process_video("path/to/video.mp4")

# 保存特征到文件
extractor.extract_and_save_features("path/to/video.mp4", "output/path/features.json")
"""

