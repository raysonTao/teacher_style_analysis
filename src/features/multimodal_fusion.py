"""多模态特征融合模块，负责融合不同模态的特征"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import FUSION_CONFIG, logger

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
            "video": video_features,
            "audio": audio_features,
            "text": text_features,
            "fusion": {
                "teaching_style_metrics": {},
                "interaction_level": 0.0,
                "explanation_clarity": 0.0,
                "emotional_engagement": 0.0,
                "logical_structure": 0.0,
                "interaction_score": 0.0,
                "clarity_score": 0.0,
                "engagement_level": 0.0
            },
            "teaching_style": {},
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
            logger.error(f"多模态特征融合失败: {e}")
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
        
        action_frequency = video_features.get("action_frequency", {})
        gesture = action_frequency.get("gesturing", 0.0)
        pointing = action_frequency.get("pointing", 0.0)
        writing = action_frequency.get("writing", 0.0)
        standing = action_frequency.get("standing", 0.0)

        speaking_ratio = self._extract_speaking_ratio(audio_features)
        speech_rate = audio_features.get("speech_rate", 0.0)
        silence_ratio = audio_features.get("silence_ratio", 1.0)
        pitch_variation = audio_features.get("pitch_variation", 0.0)
        emotion_scores = audio_features.get("emotion_scores", {})

        question_frequency = text_features.get("question_frequency", 0.0)
        vocabulary_richness = text_features.get("vocabulary_richness", 0.0)
        sentence_complexity = text_features.get("sentence_complexity", 0.0)
        keyword_density = text_features.get("keyword_density", {})
        logical_indicators = text_features.get("logical_indicators", {})

        interaction_level = self._clamp01(
            0.4 * self._clamp01(question_frequency) +
            0.3 * gesture +
            0.2 * pointing +
            0.1 * (1.0 - abs(speaking_ratio - 0.6))
        )

        explanation_clarity = self._clamp01(
            0.4 * self._bell_score(speech_rate, 150.0, 60.0) +
            0.3 * self._volume_stability(audio_features) +
            0.3 * self._sentence_clarity(text_features)
        )

        logical_structure = self._clamp01(
            0.6 * self._clamp01(sum(logical_indicators.values())) +
            0.4 * self._clamp01(sentence_complexity)
        )

        emotional_engagement = self._clamp01(
            0.6 * (emotion_scores.get("happy", 0.0) + emotion_scores.get("surprise", 0.0)) +
            0.2 * gesture +
            0.2 * self._clamp01(pitch_variation)
        )

        teaching_style_metrics = {
            "lecturing": self._clamp01(
                0.4 * standing +
                0.3 * self._bell_score(speech_rate, 150.0, 60.0) +
                0.3 * self._clamp01(speaking_ratio)
            ),
            "guiding": self._clamp01(
                0.4 * self._clamp01(question_frequency) +
                0.3 * interaction_level +
                0.3 * pointing
            ),
            "interactive": self._clamp01(
                0.5 * interaction_level +
                0.3 * gesture +
                0.2 * self._clamp01(1.0 - silence_ratio)
            ),
            "logical": self._clamp01(
                0.5 * logical_structure +
                0.3 * vocabulary_richness +
                0.2 * sentence_complexity
            ),
            "problem_driven": self._clamp01(
                0.4 * self._clamp01(question_frequency) +
                0.4 * keyword_density.get("problem", 0.0) +
                0.2 * logical_structure
            ),
            "emotional": self._clamp01(
                0.6 * emotional_engagement +
                0.2 * gesture +
                0.2 * self._clamp01(pitch_variation)
            ),
            "patient": self._clamp01(
                0.5 * silence_ratio +
                0.3 * self._bell_score(speech_rate, 110.0, 50.0) +
                0.2 * self._clamp01(1.0 - pitch_variation)
            )
        }

        fused_features["teaching_style"] = teaching_style_metrics
        fused_features["fusion"]["teaching_style_metrics"] = teaching_style_metrics
        fused_features["fusion"]["interaction_level"] = interaction_level
        fused_features["fusion"]["explanation_clarity"] = explanation_clarity
        fused_features["fusion"]["logical_structure"] = logical_structure
        fused_features["fusion"]["emotional_engagement"] = emotional_engagement
    
    def _calculate_interaction_score(self, fused_features: Dict):
        """
        计算互动分数
        
        Args:
            fused_features: 融合特征字典
        """
        video_features = fused_features["video_features"]
        audio_features = fused_features["audio_features"]
        
        action_frequency = video_features.get("action_frequency", {})
        speaking_ratio = self._extract_speaking_ratio(audio_features)

        interaction_score = self._clamp01(
            0.4 * action_frequency.get("gesturing", 0.0) +
            0.3 * action_frequency.get("pointing", 0.0) +
            0.3 * (1.0 - abs(speaking_ratio - 0.6))
        )

        fused_features["interaction_score"] = float(interaction_score)
        fused_features["fusion"]["interaction_score"] = float(interaction_score)
    
    def _calculate_clarity_score(self, fused_features: Dict):
        """
        计算讲解清晰度分数
        
        Args:
            fused_features: 融合特征字典
        """
        audio_features = fused_features["audio_features"]
        text_features = fused_features["text_features"]
        
        speech_rate = audio_features.get("speech_rate", 0.0)
        clarity_score = self._clamp01(
            0.4 * self._bell_score(speech_rate, 150.0, 60.0) +
            0.3 * self._volume_stability(audio_features) +
            0.3 * self._sentence_clarity(text_features)
        )
        fused_features["clarity_score"] = float(clarity_score)
        fused_features["fusion"]["clarity_score"] = float(clarity_score)
    
    def _calculate_engagement_level(self, fused_features: Dict):
        """
        计算参与度
        
        Args:
            fused_features: 融合特征字典
        """
        video_features = fused_features["video_features"]
        audio_features = fused_features["audio_features"]
        
        motion_energy = video_features.get("motion_energy", [])
        avg_motion = np.mean(motion_energy) if motion_energy else 0.0
        motion_score = float(np.tanh(avg_motion * 50.0))

        action_frequency = video_features.get("action_frequency", {})
        pitch_variation = audio_features.get("pitch_variation", 0.0)

        engagement_level = self._clamp01(
            0.4 * motion_score +
            0.3 * action_frequency.get("gesturing", 0.0) +
            0.2 * action_frequency.get("writing", 0.0) +
            0.1 * self._clamp01(pitch_variation)
        )

        fused_features["engagement_level"] = float(engagement_level)
        fused_features["fusion"]["engagement_level"] = float(engagement_level)
    
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
        for key in ["lecturing", "guiding", "interactive", "logical", "problem_driven", "emotional", "patient"]:
            fusion_vector.append(teaching_style.get(key, 0.0))

        # 添加互动和清晰度分数
        fusion_vector.append(fused_features["fusion"]["interaction_level"])
        fusion_vector.append(fused_features["fusion"]["explanation_clarity"])
        fusion_vector.append(fused_features["fusion"]["logical_structure"])
        fusion_vector.append(fused_features["fusion"]["emotional_engagement"])
        fusion_vector.append(fused_features["fusion"]["engagement_level"])
        
        fused_features["fusion_vector"] = fusion_vector

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _bell_score(value: float, center: float, width: float) -> float:
        if width <= 0:
            return 0.0
        return float(np.exp(-((value - center) / width) ** 2))

    @staticmethod
    def _extract_speaking_ratio(audio_features: Dict) -> float:
        if audio_features.get("voice_activity"):
            return float(np.mean(audio_features["voice_activity"]))
        silence_ratio = audio_features.get("silence_ratio", None)
        if silence_ratio is not None:
            return float(1.0 - silence_ratio)
        return 0.0

    @staticmethod
    def _volume_stability(audio_features: Dict) -> float:
        stats = audio_features.get("volume_statistics", {})
        mean_val = stats.get("mean", None)
        std_val = stats.get("std", None)
        if mean_val is None or std_val is None:
            volume = audio_features.get("volume", [])
            if not volume:
                return 0.0
            mean_val = float(np.mean(volume))
            std_val = float(np.std(volume))
        variation = std_val / (mean_val + 1e-6)
        return max(0.0, min(1.0, 1.0 - variation))

    @staticmethod
    def _sentence_clarity(text_features: Dict) -> float:
        word_count = text_features.get("word_count", 0)
        sentence_count = text_features.get("sentence_count", 0)
        if sentence_count <= 0:
            return 0.0
        avg_words = word_count / sentence_count
        return float(np.exp(-((avg_words - 15.0) / 7.0) ** 2))
