"""多模态特征融合模块，负责融合不同模态的特征"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import FUSION_CONFIG

class MultimodalFeatureFusion:
    """多模态特征融合类，用于融合视频、音频和文本特征"""
    
    def __init__(self):
        """初始化多模态特征融合器"""
        self.fusion_config = FUSION_CONFIG
    
    def fuse_features(self, video_features: Dict, audio_features: Dict, text_features: Dict) -> Dict:
        """
        融合多模态特征
        
        Args:
            video_features: 视频特征字典
            audio_features: 音频特征字典
            text_features: 文本特征字典
            
        Returns:
            融合后的特征字典
        """
        fused_features = {
            "video_features": video_features,
            "audio_features": audio_features,
            "text_features": text_features,
            "teaching_style": {
                "lecturing": 0.0,
                "guiding": 0.0,
                "interactive": 0.0,
                "demonstrative": 0.0
            },
            "engagement_level": 0.0,
            "interaction_score": 0.0,
            "clarity_score": 0.0,
            "fusion_vector": None
        }
        
        try:
            # 计算教学风格指标
            self._calculate_teaching_style(fused_features)
            
            # 计算互动水平
            self._calculate_interaction_score(fused_features)
            
            # 计算讲解清晰度
            self._calculate_clarity_score(fused_features)
            
            # 计算参与度
            self._calculate_engagement_level(fused_features)
            
            # 生成融合向量
            self._generate_fusion_vector(fused_features)
            
        except Exception as e:
            print(f"多模态特征融合失败: {e}")
            import traceback
            traceback.print_exc()
        
        return fused_features
    
    def _calculate_teaching_style(self, fused_features: Dict):
        """
        计算教学风格指标
        
        Args:
            fused_features: 融合特征字典
        """
        video_features = fused_features["video_features"]
        audio_features = fused_features["audio_features"]
        text_features = fused_features["text_features"]
        
        # 初始化风格分数
        lecturing = 0.0
        guiding = 0.0
        interactive = 0.0
        demonstrative = 0.0
        
        # 基于视频特征的风格分析
        action_counts = video_features.get("action_counts", {})
        action_frequency = video_features.get("action_frequency", {})
        
        # lecturing: 站立、讲解
        if action_frequency.get("standing", 0) > 0.5:
            lecturing += 0.5
        
        # guiding: 指向、引导
        if action_frequency.get("pointing", 0) > 0.2:
            guiding += 0.4
        
        # interactive: 手势、互动
        if action_frequency.get("gesturing", 0) > 0.3:
            interactive += 0.4
        
        # demonstrative: 书写、演示
        if action_frequency.get("writing", 0) > 0.2:
            demonstrative += 0.4
        
        # 基于音频特征的风格分析
        audio_duration = audio_features.get("audio_duration", 0)
        voice_activity = audio_features.get("voice_activity", [])
        
        if voice_activity:
            speaking_ratio = sum(voice_activity) / len(voice_activity)
            if speaking_ratio > 0.8:
                lecturing += 0.3
            elif speaking_ratio > 0.5:
                interactive += 0.2
        
        # 基于文本特征的风格分析
        word_count = text_features.get("word_count", 0)
        if word_count > 500:
            lecturing += 0.2
        elif word_count > 200:
            guiding += 0.2
        
        # 归一化分数
        total = lecturing + guiding + interactive + demonstrative
        if total > 0:
            lecturing /= total
            guiding /= total
            interactive /= total
            demonstrative /= total
        
        # 更新教学风格
        fused_features["teaching_style"] = {
            "lecturing": float(lecturing),
            "guiding": float(guiding),
            "interactive": float(interactive),
            "demonstrative": float(demonstrative)
        }
    
    def _calculate_interaction_score(self, fused_features: Dict):
        """
        计算互动分数
        
        Args:
            fused_features: 融合特征字典
        """
        video_features = fused_features["video_features"]
        audio_features = fused_features["audio_features"]
        
        interaction_score = 0.0
        
        # 基于动作的互动分析
        action_frequency = video_features.get("action_frequency", {})
        
        # 手势和指向动作增加互动分数
        interaction_score += action_frequency.get("gesturing", 0) * 0.4
        interaction_score += action_frequency.get("pointing", 0) * 0.3
        
        # 基于音频的互动分析
        voice_activity = audio_features.get("voice_activity", [])
        if voice_activity:
            speaking_ratio = sum(voice_activity) / len(voice_activity)
            # 适中的说话比例（不是一直说）表示可能有互动
            if 0.5 < speaking_ratio < 0.8:
                interaction_score += 0.3
        
        # 归一化分数
        interaction_score = min(interaction_score, 1.0)
        fused_features["interaction_score"] = float(interaction_score)
    
    def _calculate_clarity_score(self, fused_features: Dict):
        """
        计算讲解清晰度分数
        
        Args:
            fused_features: 融合特征字典
        """
        audio_features = fused_features["audio_features"]
        text_features = fused_features["text_features"]
        
        clarity_score = 0.5  # 默认清晰度
        
        # 基于音频的清晰度分析
        volume = audio_features.get("volume", [])
        if volume:
            avg_volume = np.mean(volume)
            # 适中的音量表示清晰度较高
            if 0.05 < avg_volume < 0.2:
                clarity_score += 0.3
        
        # 基于文本的清晰度分析
        word_count = text_features.get("word_count", 0)
        sentence_count = text_features.get("sentence_count", 0)
        
        if sentence_count > 0:
            avg_words_per_sentence = word_count / sentence_count
            # 适中的句子长度表示清晰度较高
            if 10 < avg_words_per_sentence < 20:
                clarity_score += 0.2
        
        # 归一化分数
        clarity_score = min(clarity_score, 1.0)
        fused_features["clarity_score"] = float(clarity_score)
    
    def _calculate_engagement_level(self, fused_features: Dict):
        """
        计算参与度
        
        Args:
            fused_features: 融合特征字典
        """
        video_features = fused_features["video_features"]
        audio_features = fused_features["audio_features"]
        
        engagement_level = 0.0
        
        # 基于视频特征的参与度分析
        motion_energy = video_features.get("motion_energy", [])
        if motion_energy:
            avg_motion = np.mean(motion_energy)
            # 适当的运动表示较高的参与度
            if avg_motion > 0.01:
                engagement_level += 0.4
        
        action_frequency = video_features.get("action_frequency", {})
        # 手势和书写动作增加参与度
        engagement_level += action_frequency.get("gesturing", 0) * 0.3
        engagement_level += action_frequency.get("writing", 0) * 0.3
        
        # 基于音频特征的参与度分析
        pitch = audio_features.get("pitch", [])
        if pitch:
            avg_pitch = np.mean(pitch)
            # 适当的语调变化表示较高的参与度
            if 100 < avg_pitch < 300:
                engagement_level += 0.2
        
        # 归一化分数
        engagement_level = min(engagement_level, 1.0)
        fused_features["engagement_level"] = float(engagement_level)
    
    def _generate_fusion_vector(self, fused_features: Dict):
        """
        生成融合向量
        
        Args:
            fused_features: 融合特征字典
        """
        fusion_vector = []
        
        # 添加视频特征
        video_features = fused_features["video_features"]
        fusion_vector.append(video_features.get("avg_motion_energy", 0.0))
        
        # 添加动作频率特征
        action_frequency = video_features.get("action_frequency", {})
        for action in ["standing", "walking", "gesturing", "writing", "pointing"]:
            fusion_vector.append(action_frequency.get(action, 0.0))
        
        # 添加音频特征
        audio_features = fused_features["audio_features"]
        if audio_features.get("volume", []):
            fusion_vector.append(np.mean(audio_features["volume"]))
        else:
            fusion_vector.append(0.0)
        
        if audio_features.get("pitch", []):
            fusion_vector.append(np.mean(audio_features["pitch"]))
        else:
            fusion_vector.append(0.0)
        
        # 添加文本特征
        text_features = fused_features["text_features"]
        fusion_vector.append(text_features.get("word_count", 0) / 100.0)  # 归一化
        fusion_vector.append(text_features.get("sentence_count", 0) / 10.0)  # 归一化
        
        # 添加教学风格特征
        teaching_style = fused_features["teaching_style"]
        fusion_vector.append(teaching_style["lecturing"])
        fusion_vector.append(teaching_style["guiding"])
        fusion_vector.append(teaching_style["interactive"])
        fusion_vector.append(teaching_style["demonstrative"])
        
        # 添加互动和清晰度分数
        fusion_vector.append(fused_features["interaction_score"])
        fusion_vector.append(fused_features["clarity_score"])
        fusion_vector.append(fused_features["engagement_level"])
        
        fused_features["fusion_vector"] = fusion_vector
