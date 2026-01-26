"""多模态特征融合模块，负责融合不同模态的特征"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import FUSION_CONFIG, logger
from .feature_encoder import FeatureEncoder

try:
    from models.deep_learning.mman_model import create_model
except Exception:
    create_model = None


class MMANFusionEngine:
    """多模态注意力融合引擎"""

    def __init__(self):
        self.encoder = FeatureEncoder()
        self.model = None
        self.device = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if not FUSION_CONFIG.get('use_mman', False) or create_model is None:
            return

        try:
            import torch

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = create_model(FUSION_CONFIG.get('mman_config', 'default')).to(self.device).eval()

            checkpoint_path = FUSION_CONFIG.get('mman_checkpoint')
            if checkpoint_path and os.path.exists(checkpoint_path):
                state = torch.load(checkpoint_path, map_location=self.device)
                state_dict = state.get('model_state_dict', state)
                self.model.load_state_dict(state_dict, strict=False)
                self.is_loaded = True
                logger.info(f"MMAN模型加载成功: {checkpoint_path}")
            else:
                logger.error("未找到MMAN检查点，无法启用多模态注意力融合")
        except Exception as e:
            logger.warning(f"MMAN模型加载失败: {e}")
            self.model = None
            self.is_loaded = False

    def compute_outputs(self, video_features: Dict, audio_features: Dict, text_features: Dict):
        """
        计算模态权重与融合向量

        Returns:
            weights: 模态权重字典
            embedding: 融合向量（numpy数组或None）
        """
        encoded = {
            'video': self.encoder.encode_video_features(video_features),
            'audio': self.encoder.encode_audio_features(audio_features),
            'text': self.encoder.encode_text_features(text_features)
        }

        if not self.is_loaded or self.model is None:
            logger.error("MMAN模型未加载，无法计算融合输出")
            return None, None

        try:
            import torch

            inputs = {
                'video': torch.tensor(encoded['video'], dtype=torch.float32).unsqueeze(0).to(self.device),
                'audio': torch.tensor(encoded['audio'], dtype=torch.float32).unsqueeze(0).to(self.device),
                'text': torch.tensor(encoded['text'], dtype=torch.float32).unsqueeze(0).to(self.device)
            }
            with torch.no_grad():
                outputs = self.model(inputs, return_attention=True)
                embedding = self.model.get_embedding(inputs).squeeze(0).cpu().numpy()

            attn = outputs.get('transformer_attention', [])
            if attn:
                last_layer = attn[-1]  # [batch, heads, seq, seq]
                weights = last_layer.mean(dim=1).mean(dim=1).squeeze(0)
                weights = weights / (weights.sum() + 1e-6)
                weight_dict = {
                    'video': float(weights[0]),
                    'audio': float(weights[1]),
                    'text': float(weights[2])
                }
            else:
                logger.error("MMAN未返回注意力权重")
                weight_dict = None

            return weight_dict, embedding
        except Exception as e:
            logger.error(f"MMAN输出计算失败: {e}")
            return None, None

    def compute_weights(self, video_features: Dict, audio_features: Dict, text_features: Dict) -> Dict:
        """计算模态权重（兼容旧接口）"""
        weights, _ = self.compute_outputs(video_features, audio_features, text_features)
        return weights

class MultimodalFeatureFusion:
    """多模态特征融合类，用于融合视频、音频和文本特征"""
    
    def __init__(self):
        """初始化多模态特征融合器"""
        self.fusion_config = FUSION_CONFIG
        self.mman_engine = MMANFusionEngine()
    
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
                "engagement_level": 0.0,
                "modality_weights": {}
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

            # 计算多模态注意力权重与融合向量
            weights, embedding = self.mman_engine.compute_outputs(
                video_features, audio_features, text_features
            )
            fused_features["fusion"]["modality_weights"] = weights
            if weights is None:
                fused_features["fusion"]["error"] = "MMAN权重计算失败"

            if embedding is not None:
                fused_features["fusion_vector"] = embedding.tolist()
            else:
                # 生成融合向量（回退方案）
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
        dialogue_act_scores = text_features.get("dialogue_act_scores", {})
        if dialogue_act_scores:
            question_frequency = max(question_frequency, dialogue_act_scores.get("question", 0.0))
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
        weights = fused_features.get("fusion", {}).get("modality_weights")
        if not weights:
            fused_features["fusion_vector"] = None
            return

        video_weight = weights.get("video", 0.0)
        audio_weight = weights.get("audio", 0.0)
        text_weight = weights.get("text", 0.0)
        
        # 添加视频特征
        video_features = fused_features["video_features"]
        fusion_vector.append(video_weight * video_features.get("avg_motion_energy", 0.0))
        
        # 添加动作频率特征
        action_frequency = video_features.get("action_frequency", {})
        for action in ["standing", "walking", "gesturing", "writing", "pointing"]:
            fusion_vector.append(video_weight * action_frequency.get(action, 0.0))
        
        # 添加音频特征
        audio_features = fused_features["audio_features"]
        if audio_features.get("volume", []):
            fusion_vector.append(audio_weight * np.mean(audio_features["volume"]))
        else:
            fusion_vector.append(0.0)
        
        if audio_features.get("pitch", []):
            fusion_vector.append(audio_weight * np.mean(audio_features["pitch"]))
        else:
            fusion_vector.append(0.0)
        
        # 添加文本特征
        text_features = fused_features["text_features"]
        fusion_vector.append(text_weight * (text_features.get("word_count", 0) / 100.0))
        fusion_vector.append(text_weight * (text_features.get("sentence_count", 0) / 10.0))
        
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
        fusion_vector.append(video_weight)
        fusion_vector.append(audio_weight)
        fusion_vector.append(text_weight)
        
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
