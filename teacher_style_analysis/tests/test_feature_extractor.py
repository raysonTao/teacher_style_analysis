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
    
    @patch('features.feature_extractor.cv2.VideoCapture')
    def test_extract_video_features(self, mock_video_capture):
        # 模拟视频捕获对象
        mock_capture = Mock()
        mock_video_capture.return_value = mock_capture
        mock_capture.isOpened.return_value = True
        
        # 模拟视频帧
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture.read.side_effect = [(True, mock_frame), (False, None)]
        
        # 模拟YOLOv8检测结果
        self.feature_extractor.yolo_model = Mock()
        self.feature_extractor.yolo_model.track.return_value = Mock(
            boxes=[Mock(xyxy=[0, 0, 100, 100], conf=0.9, cls=0)],
            masks=None
        )
        
        # 模拟OpenPose姿态估计
        self.feature_extractor.pose_estimator = Mock()
        self.feature_extractor.pose_estimator.process.return_value = {
            'pose_keypoints': np.zeros((1, 25, 3))
        }
        
        # 测试提取视频特征
        features = self.feature_extractor.extract_video_features(self.mock_video_path)
        
        # 验证特征包含正确的键
        self.assertIn('body_language_features', features)
        self.assertIn('facial_expression_features', features)
        self.assertIn('gesture_features', features)
        
        # 验证特征是numpy数组
        self.assertIsInstance(features['body_language_features'], np.ndarray)
    
    @patch('features.feature_extractor.whisper.load_model')
    def test_extract_audio_features(self, mock_load_model):
        # 模拟Whisper模型
        mock_whisper_model = Mock()
        mock_load_model.return_value = mock_whisper_model
        mock_whisper_model.transcribe.return_value = {
            'text': "这是一个教学视频",
            'segments': [{'start': 0, 'end': 5, 'text': "这是一个"}, 
                         {'start': 5, 'end': 10, 'text': "教学视频"}]
        }
        
        # 测试提取音频特征
        features = self.feature_extractor.extract_audio_features(self.mock_video_path)
        
        # 验证特征包含正确的键
        self.assertIn('transcript', features)
        self.assertIn('speaking_rate', features)
        self.assertIn('pitch_features', features)
        
        # 验证转录文本存在
        self.assertTrue(len(features['transcript']) > 0)
    
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
            'body_language_features': np.array([0.1, 0.2, 0.3]),
            'facial_expression_features': np.array([0.4, 0.5]),
            'gesture_features': np.array([0.6])
        }
        
        audio_features = {
            'speaking_rate': 120.0,
            'pitch_features': np.array([0.7, 0.8]),
            'energy_features': np.array([0.9])
        }
        
        text_features = {
            'semantic_features': np.array([1.0, 1.1, 1.2]),
            'sentiment_score': 0.85,
            'teaching_terms': ['教学', '示例']
        }
        
        # 测试特征融合
        fused_features = self.feature_extractor.fuse_multimodal_features(
            video_features, audio_features, text_features
        )
        
        # 验证融合特征是numpy数组
        self.assertIsInstance(fused_features, np.ndarray)
        # 验证融合特征维度合理
        self.assertTrue(fused_features.ndim == 1)
    
    @patch.object(FeatureExtractor, 'extract_video_features')
    @patch.object(FeatureExtractor, 'extract_audio_features')
    @patch.object(FeatureExtractor, 'extract_text_features')
    @patch.object(FeatureExtractor, 'fuse_multimodal_features')
    def test_process_video(self, mock_fuse, mock_text, mock_audio, mock_video):
        # 设置模拟返回值
        mock_video.return_value = {'body_language_features': np.array([0.1, 0.2])}
        mock_audio.return_value = {'transcript': '测试文本', 'speaking_rate': 100.0}
        mock_text.return_value = {'semantic_features': np.array([0.3, 0.4])}
        mock_fuse.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        
        # 测试处理视频
        result = self.feature_extractor.process_video(self.mock_video_path)
        
        # 验证所有方法都被调用
        mock_video.assert_called_once_with(self.mock_video_path)
        mock_audio.assert_called_once_with(self.mock_video_path)
        mock_text.assert_called_once()
        mock_fuse.assert_called_once()
        
        # 验证结果包含所有必要的键
        self.assertIn('video_features', result)
        self.assertIn('audio_features', result)
        self.assertIn('text_features', result)
        self.assertIn('fused_features', result)

if __name__ == '__main__':
    unittest.main()