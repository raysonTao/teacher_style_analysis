import unittest
import numpy as np
from unittest.mock import Mock, patch
from feedback.feedback_generator import FeedbackGenerator

class TestFeedbackGenerator(unittest.TestCase):
    def setUp(self):
        # 创建反馈生成器实例
        self.feedback_generator = FeedbackGenerator()
        
        # 模拟分类结果
        self.mock_classification = {
            'style_scores': {
                'analytical': 0.8,
                'interactive': 0.6,
                'authoritative': 0.4,
                'supportive': 0.7
            },
            'dominant_style': 'analytical',
            'confidence': 0.9
        }
        
        # 模拟学科类型
        self.mock_subject = '数学'
        
        # 模拟历史记录
        self.mock_history = [
            {'date': '2023-01-01', 'style_scores': {'analytical': 0.6, 'interactive': 0.5, 'authoritative': 0.4, 'supportive': 0.5}},
            {'date': '2023-02-01', 'style_scores': {'analytical': 0.7, 'interactive': 0.55, 'authoritative': 0.4, 'supportive': 0.6}}
        ]
    
    def test_calculate_smi(self):
        # 测试SMI计算
        smi = self.feedback_generator.calculate_smi(self.mock_classification, self.mock_subject)
        
        # 验证SMI是字典类型
        self.assertIsInstance(smi, dict)
        # 验证包含预期的键
        self.assertIn('overall_smi', smi)
        self.assertIn('style_match_scores', smi)
        self.assertIn('recommended_adjustments', smi)
        
        # 验证整体SMI在0-100范围内
        self.assertTrue(0 <= smi['overall_smi'] <= 100)
        
        # 验证风格匹配分数在0-1范围内
        for score in smi['style_match_scores'].values():
            self.assertTrue(0 <= score <= 1)
    
    def test_generate_improvement_suggestions(self):
        # 模拟SMI结果
        smi_result = {
            'overall_smi': 75.5,
            'style_match_scores': {
                'analytical': 0.9,
                'interactive': 0.6,
                'authoritative': 0.7,
                'supportive': 0.8
            },
            'recommended_adjustments': ['增加互动性']
        }
        
        # 模拟详细特征
        detailed_features = {
            'gesture_frequency': 0.4,
            'question_asking_rate': 0.3,
            'feedback_positivity': 0.8
        }
        
        # 测试生成改进建议
        suggestions = self.feedback_generator.generate_improvement_suggestions(
            smi_result, detailed_features, self.mock_subject
        )
        
        # 验证建议是列表类型
        self.assertIsInstance(suggestions, list)
        # 验证列表不为空
        self.assertTrue(len(suggestions) > 0)
        # 验证每个建议是字典类型
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, dict)
            self.assertIn('area', suggestion)
            self.assertIn('suggestion', suggestion)
            self.assertIn('priority', suggestion)
    
    def test_analyze_teaching_growth(self):
        # 测试教学成长分析
        growth_analysis = self.feedback_generator.analyze_teaching_growth(
            self.mock_classification, self.mock_history
        )
        
        # 验证分析结果是字典类型
        self.assertIsInstance(growth_analysis, dict)
        # 验证包含预期的键
        self.assertIn('growth_trends', growth_analysis)
        self.assertIn('improved_areas', growth_analysis)
        self.assertIn('areas_needing_attention', growth_analysis)
        self.assertIn('consistency_score', growth_analysis)
        
        # 验证一致性分数在0-1范围内
        self.assertTrue(0 <= growth_analysis['consistency_score'] <= 1)
    
    @patch.object(FeedbackGenerator, 'calculate_smi')
    @patch.object(FeedbackGenerator, 'generate_improvement_suggestions')
    @patch.object(FeedbackGenerator, 'analyze_teaching_growth')
    def test_generate_feedback_report(self, mock_growth, mock_suggestions, mock_smi):
        # 设置模拟返回值
        mock_smi.return_value = {
            'overall_smi': 80.0,
            'style_match_scores': {
                'analytical': 0.9,
                'interactive': 0.7,
                'authoritative': 0.8,
                'supportive': 0.8
            },
            'recommended_adjustments': ['略微增加互动性']
        }
        
        mock_suggestions.return_value = [
            {'area': '互动教学', 'suggestion': '增加提问频率', 'priority': 'medium'},
            {'area': '反馈机制', 'suggestion': '提供更具体的学生反馈', 'priority': 'high'}
        ]
        
        mock_growth.return_value = {
            'growth_trends': {'analytical': 'increasing', 'interactive': 'stable'},
            'improved_areas': ['分析能力'],
            'areas_needing_attention': ['互动性'],
            'consistency_score': 0.85
        }
        
        # 模拟详细特征
        detailed_features = {
            'gesture_frequency': 0.5,
            'question_asking_rate': 0.4,
            'feedback_positivity': 0.9
        }
        
        # 测试生成反馈报告
        report = self.feedback_generator.generate_feedback_report(
            self.mock_classification, detailed_features, 
            self.mock_subject, self.mock_history
        )
        
        # 验证所有方法都被调用
        mock_smi.assert_called_once_with(self.mock_classification, self.mock_subject)
        mock_suggestions.assert_called_once()
        mock_growth.assert_called_once_with(self.mock_classification, self.mock_history)
        
        # 验证报告包含必要的部分
        self.assertIn('smi_analysis', report)
        self.assertIn('improvement_suggestions', report)
        self.assertIn('growth_analysis', report)
        self.assertIn('summary', report)
        self.assertIn('actionable_next_steps', report)
    
    def test_get_recommended_resources(self):
        # 测试获取推荐资源
        resources = self.feedback_generator.get_recommended_resources(
            self.mock_classification, self.mock_subject, self.mock_history
        )
        
        # 验证资源是列表类型
        self.assertIsInstance(resources, list)
        # 验证每个资源项是字典类型
        for resource in resources:
            self.assertIsInstance(resource, dict)
            self.assertIn('type', resource)
            self.assertIn('title', resource)
            self.assertIn('description', resource)
            self.assertIn('relevance', resource)

if __name__ == '__main__':
    unittest.main()