"""主特征提取模块，协调所有子模块进行多模态特征提取"""
import os
import sys
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import BASE_DIR, FEATURES_DIR, logger
from .video_feature_extractor import VideoFeatureExtractor
from .audio_feature_extractor import AudioFeatureExtractor
from .text_feature_extractor import TextFeatureExtractor
from .multimodal_fusion import MultimodalFeatureFusion

class FeatureExtractor:
    """主特征提取类，协调所有子模块进行多模态特征提取"""
    
    # 单例模式实现
    _instance = None
    
    def __new__(cls):
        """控制实例创建，确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super(FeatureExtractor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化主特征提取器"""
        # 确保只初始化一次
        if not hasattr(self, 'video_extractor'):
            logger.info("初始化主特征提取器...")
            
            # 初始化各个子模块
            self.video_extractor = VideoFeatureExtractor()
            self.audio_extractor = AudioFeatureExtractor()
            self.text_extractor = TextFeatureExtractor()
            self.multimodal_fusion = MultimodalFeatureFusion()
            
            # 初始化特征变量
            self.features = None
            
            logger.info("主特征提取器初始化完成")
    
    def process_video(self, video_path: str) -> Dict:
        """
        处理视频文件，提取所有模态的特征并融合
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            融合后的多模态特征字典
        """
        logger.info(f"开始处理视频: {video_path}")
        
        # 确保输出目录存在
        output_dir = str(FEATURES_DIR)
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 提取视频特征
        logger.info("开始提取视频特征...")
        video_features = self.video_extractor.extract_features(video_path)
        logger.info("视频特征提取完成")
        
        # 2. 提取音频特征
        logger.info("开始提取音频特征...")
        try:
            # 从视频中提取音频
            temp_audio_path = self.audio_extractor.extract_audio_from_video(video_path)
            
            if temp_audio_path and os.path.exists(temp_audio_path):
                logger.info(f"临时音频文件已创建: {temp_audio_path}")
                
                # 提取音频特征
                audio_features = self.audio_extractor.extract_features(temp_audio_path)
                
                # 删除临时音频文件
                try:
                    os.remove(temp_audio_path)
                    logger.info("临时音频文件已删除")
                except Exception as e:
                    logger.warning(f"删除临时音频文件失败: {e}")
            else:
                logger.warning("无法提取音频，使用默认音频特征")
                audio_features = self.audio_extractor.extract_features(" ")
                
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            import traceback
            traceback.print_exc()
            audio_features = self.audio_extractor.extract_features(" ")
        
        logger.info("音频特征提取完成")
        
        # 3. 提取文本特征
        logger.info("开始提取文本特征...")
        transcription = audio_features.get("transcription", "")
        text_features = self.text_extractor.extract_features(transcription)
        logger.info("文本特征提取完成")
        
        # 4. 融合多模态特征
        logger.info("开始融合多模态特征...")
        self.features = self.multimodal_fusion.fuse_features(
            video_features, 
            audio_features, 
            text_features
        )
        logger.info("多模态特征融合完成")
        
        # 添加视频元信息
        self.features["video_path"] = video_path
        self.features["video_name"] = os.path.basename(video_path)
        
        return self.features
    
    def extract_video_features(self, video_path: str) -> Dict:
        """
        仅提取视频特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频特征字典
        """
        return self.video_extractor.extract_features(video_path)
    
    def extract_audio_features(self, audio_path: str) -> Dict:
        """
        仅提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频特征字典
        """
        return self.audio_extractor.extract_features(audio_path)
    
    def extract_text_features(self, text: str) -> Dict:
        """
        仅提取文本特征
        
        Args:
            text: 输入文本
            
        Returns:
            文本特征字典
        """
        return self.text_extractor.extract_features(text)
    
    def extract_and_save_features(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        提取特征并保存到JSON文件
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径，默认为None
            
        Returns:
            融合后的多模态特征字典
        """
        # 处理视频
        features = self.process_video(video_path)
        
        # 如果没有指定输出路径，使用默认路径
        if output_path is None:
            video_name = os.path.basename(video_path)
            video_name_without_ext = os.path.splitext(video_name)[0]
            output_path = os.path.join(
                str(FEATURES_DIR),
                f"{video_name_without_ext}_features.json"
            )
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存特征到JSON文件
        try:
            # 将numpy数组转换为列表以便JSON序列化
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            # 转换特征字典
            serializable_features = convert_numpy(features)
            
            # 保存到JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_features, f, ensure_ascii=False, indent=4)
            
            logger.info(f"特征已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存特征失败: {e}")
            import traceback
            traceback.print_exc()
        
        return features
    
    def get_features(self) -> Dict:
        """
        获取最近一次处理的特征
        
        Returns:
            融合后的多模态特征字典
        """
        return self.features


