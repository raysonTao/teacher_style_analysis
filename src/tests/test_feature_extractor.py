import unittest
import numpy as np
from unittest.mock import Mock, patch
from features.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        # 创建特征提取器实例
        self.feature_extractor = FeatureExtractor()
        # 模拟视频文件路径
        self.mock_video_path = "tests/data/sample_video.mp4"
    
    def test_extract_video_features(self):
        # 测试提取视频特征
        features = self.feature_extractor.extract_video_features(self.mock_video_path)
        
        # 验证特征包含正确的键
        self.assertIn('action_sequence', features)
        self.assertIn('pose_estimation', features)
        self.assertIn('motion_energy', features)
        self.assertIn('spatial_distribution', features)
        self.assertIn('behavior_frequency', features)
    
    def test_extract_audio_features(self):
        # 测试提取音频特征
        features = self.feature_extractor.extract_audio_features(self.mock_video_path)
        
        # 验证特征包含正确的键
        self.assertIn('speech_rate', features)
        self.assertIn('pitch_variation', features)
        self.assertIn('volume_level', features)
        self.assertIn('silence_ratio', features)
        self.assertIn('emotion_scores', features)
    
    def test_extract_text_features(self):
        # 测试文本特征提取
        transcript = "这是一个教学视频示例"
        features = self.feature_extractor.extract_text_features(transcript)
        
        # 验证特征包含正确的键
        self.assertIn('semantic_features', features)
        self.assertIn('sentiment_score', features)
        self.assertIn('teaching_terms', features)
        
        # 验证语义特征是numpy数组
        self.assertIsInstance(features['semantic_features'], np.ndarray)
    
    def test_fuse_multimodal_features(self):
        # 模拟各模态特征
        video_features = {
            'action_sequence': [],
            'pose_estimation': [],
            'motion_energy': [],
            'spatial_distribution': {'front': 0.6, 'middle': 0.3, 'side': 0.1},
            'behavior_frequency': {'standing': 0.85, 'walking': 0.10, 'gesturing': 0.65}
        }
        
        audio_features = {
            'speech_rate': 120.0,
            'pitch_variation': 0.5,
            'volume_level': 0.8,
            'silence_ratio': 0.1,
            'emotion_scores': {'happy': 0.6, 'neutral': 0.4}
        }
        
        text_features = {
            'vocabulary_richness': 0.65,
            'sentence_complexity': 0.50,
            'question_frequency': 0.12,
            'keyword_density': {'概念': 0.05, '原理': 0.04},
            'logical_indicators': {'因为': 0.02, '所以': 0.03},
            'semantic_features': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'sentiment_score': 0.75,
            'teaching_terms': ['教学', '讲解', '提问']
        }
        
        # 测试特征融合
        fused_features = self.feature_extractor.fuse_multimodal_features(
            video_features, audio_features, text_features
        )
        
        # 验证融合特征结构正确
        self.assertIn('video', fused_features)
        self.assertIn('audio', fused_features)
        self.assertIn('text', fused_features)
        self.assertIn('fusion', fused_features)
        
        # 验证融合特征包含教学风格指标
        fusion_data = fused_features['fusion']
        self.assertIn('interaction_level', fusion_data)
        self.assertIn('explanation_clarity', fusion_data)
        self.assertIn('teaching_style_metrics', fusion_data)
    
    def test_process_video(self):
        # 测试处理视频
        result = self.feature_extractor.process_video(self.mock_video_path)
        
        # 验证结果包含所有必要的键
        self.assertIn('video_features', result)
        self.assertIn('audio_features', result)
        self.assertIn('text_features', result)
        self.assertIn('fused_features', result)
        
        # 验证融合特征包含教学风格指标
        self.assertIn('fusion', result['fused_features'])
        self.assertIn('teaching_style_metrics', result['fused_features']['fusion'])

if __name__ == '__main__':
    unittest.main()