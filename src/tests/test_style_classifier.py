import unittest
import numpy as np
from unittest.mock import Mock, patch
from models.core.style_classifier import StyleClassifier

class TestStyleClassifier(unittest.TestCase):
    def setUp(self):
        # 创建风格分类器实例
        self.style_classifier = StyleClassifier()
        # 模拟多模态特征
        self.mock_features = np.random.rand(1, 256)  # 假设256维特征向量
    
    def test_apply_rule_driven_layer(self):
        # 模拟多模态特征和原始数据
        features = np.array([[0.8, 0.2, 0.6, 0.9]])
        raw_data = {
            'video_features': {'gesture_frequency': 0.7},
            'audio_features': {'speaking_rate': 150.0},
            'text_features': {'sentiment_score': 0.9, 'teaching_terms': ['互动', '参与']}
        }
        
        # 测试规则驱动层
        rule_results = self.style_classifier.apply_rule_driven_layer(features, raw_data)
        
        # 验证规则结果包含预期的键
        self.assertIn('analytical_score', rule_results)
        self.assertIn('interactive_score', rule_results)
        self.assertIn('authoritative_score', rule_results)
        self.assertIn('supportive_score', rule_results)
        
        # 验证分数在0-1范围内
        for score in rule_results.values():
            self.assertTrue(0 <= score <= 1)
    
    @patch('models.style_classifier.torch')
    def test_apply_ml_layer(self, mock_torch):
        # 模拟PyTorch模型
        mock_model = Mock()
        mock_model.forward.return_value = mock_torch.tensor([[0.8, 0.6, 0.3, 0.9]])
        mock_model.predict.return_value = [np.array([0.8, 0.6, 0.3, 0.9])]
        self.style_classifier.cmat_model = mock_model
        
        # 测试机器学习层
        ml_results = self.style_classifier.apply_ml_layer(self.mock_features)
        
        # 验证机器学习结果是numpy数组
        self.assertIsInstance(ml_results, np.ndarray)
        # 验证结果维度正确
        self.assertEqual(ml_results.shape, (1, 4))  # 假设4种风格
    
    def test_fuse_outputs(self):
        # 模拟规则驱动和机器学习结果
        rule_results = {
            'analytical_score': 0.7,
            'interactive_score': 0.8,
            'authoritative_score': 0.5,
            'supportive_score': 0.6
        }
        
        ml_results = np.array([[0.8, 0.7, 0.6, 0.5]])
        
        # 测试输出融合
        fused_results = self.style_classifier.fuse_outputs(rule_results, ml_results)
        
        # 验证融合结果包含预期的键
        self.assertIn('analytical', fused_results)
        self.assertIn('interactive', fused_results)
        self.assertIn('authoritative', fused_results)
        self.assertIn('supportive', fused_results)
        self.assertIn('dominant_style', fused_results)
        
        # 验证分数在0-1范围内
        for style, score in fused_results.items():
            if style != 'dominant_style':
                self.assertTrue(0 <= score <= 1)
    
    @patch.object(StyleClassifier, 'apply_rule_driven_layer')
    @patch.object(StyleClassifier, 'apply_ml_layer')
    @patch.object(StyleClassifier, 'fuse_outputs')
    def test_classify_style(self, mock_fuse, mock_ml, mock_rule):
        # 设置模拟返回值
        mock_rule.return_value = {
            'analytical_score': 0.7,
            'interactive_score': 0.8,
            'authoritative_score': 0.5,
            'supportive_score': 0.6
        }
        
        mock_ml.return_value = np.array([[0.8, 0.7, 0.6, 0.5]])
        
        mock_fuse.return_value = {
            'analytical': 0.75,
            'interactive': 0.75,
            'authoritative': 0.55,
            'supportive': 0.55,
            'dominant_style': 'analytical_interactive'
        }
        
        # 模拟原始数据
        raw_data = {
            'video_features': {},
            'audio_features': {},
            'text_features': {}
        }
        
        # 测试风格分类
        result = self.style_classifier.classify_style(self.mock_features, raw_data)
        
        # 验证所有方法都被调用
        mock_rule.assert_called_once_with(self.mock_features, raw_data)
        mock_ml.assert_called_once_with(self.mock_features)
        mock_fuse.assert_called_once()
        
        # 验证结果包含必要的键
        self.assertIn('style_scores', result)
        self.assertIn('dominant_style', result)
        self.assertIn('confidence', result)
    
    def test_calculate_feature_contributions(self):
        # 模拟特征和模型
        features = np.random.rand(1, 10)  # 10维特征
        self.style_classifier.cmat_model = Mock()
        
        # 测试特征贡献度计算
        contributions = self.style_classifier.calculate_feature_contributions(features)
        
        # 验证贡献度是字典类型
        self.assertIsInstance(contributions, dict)
        # 验证包含预期的键
        for style in ['analytical', 'interactive', 'authoritative', 'supportive']:
            self.assertIn(style, contributions)
    
    def test_explain_prediction(self):
        # 模拟分类结果和特征贡献度
        classification_result = {
            'style_scores': {
                'analytical': 0.8,
                'interactive': 0.6,
                'authoritative': 0.4,
                'supportive': 0.7
            },
            'dominant_style': 'analytical',
            'confidence': 0.9
        }
        
        feature_contributions = {
            'analytical': {0: 0.3, 1: 0.2, 2: 0.1},
            'interactive': {3: 0.4, 4: 0.2},
            'authoritative': {5: 0.1, 6: 0.1},
            'supportive': {7: 0.3, 8: 0.2, 9: 0.1}
        }
        
        # 测试预测解释
        explanation = self.style_classifier.explain_prediction(
            classification_result, feature_contributions
        )
        
        # 验证解释包含必要的键
        self.assertIn('dominant_style_justification', explanation)
        self.assertIn('top_contributing_features', explanation)
        self.assertIn('style_breakdown', explanation)

if __name__ == '__main__':
    unittest.main()