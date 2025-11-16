import unittest
import numpy as np
import os
from unittest.mock import Mock, patch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # 创建测试所需的路径和配置
        self.test_video_path = "tests/data/sample_video.mp4"
        self.test_output_dir = "tests/integration_output"
        self.teacher_id = "test_teacher_001"
        self.subject = "数学"
        
        # 创建输出目录
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)
        
        # 创建测试数据目录
        if not os.path.exists("tests/data"):
            os.makedirs("tests/data")
    
    def tearDown(self):
        # 清理测试生成的文件
        if os.path.exists(self.test_output_dir):
            for file in os.listdir(self.test_output_dir):
                os.remove(os.path.join(self.test_output_dir, file))
            os.rmdir(self.test_output_dir)
    
    @patch('features.feature_extractor.FeatureExtractor')
    @patch('models.style_classifier.StyleClassifier')
    @patch('feedback.feedback_generator.FeedbackGenerator')
    def test_complete_analysis_pipeline(self, mock_feedback_class, mock_classifier_class, mock_extractor_class):
        # 模拟特征提取器
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # 设置特征提取器返回值
        mock_extractor.process_video.return_value = {
            'video_features': {
                'body_language_features': np.array([0.1, 0.2, 0.3]),
                'facial_expression_features': np.array([0.4, 0.5]),
                'gesture_features': np.array([0.6])
            },
            'audio_features': {
                'transcript': "这是一个数学教学视频，讲解了微积分的基本概念。",
                'speaking_rate': 120.0,
                'pitch_features': np.array([0.7, 0.8]),
                'energy_features': np.array([0.9])
            },
            'text_features': {
                'semantic_features': np.array([1.0, 1.1, 1.2]),
                'sentiment_score': 0.85,
                'teaching_terms': ['微积分', '基本概念']
            },
            'fused_features': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        }
        
        # 模拟风格分类器
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier
        
        # 设置分类器返回值
        mock_classifier.classify_style.return_value = {
            'style_scores': {
                'analytical': 0.85,
                'interactive': 0.60,
                'authoritative': 0.50,
                'supportive': 0.70
            },
            'dominant_style': 'analytical',
            'confidence': 0.92,
            'feature_contributions': {
                'analytical': {0: 0.3, 2: 0.2, 5: 0.15},
                'interactive': {3: 0.2, 7: 0.15},
                'authoritative': {1: 0.1, 4: 0.1},
                'supportive': {6: 0.25, 8: 0.15}
            },
            'explanation': "教学风格倾向于分析型，主要表现为逻辑清晰的讲解和系统化的知识传授。"
        }
        
        # 模拟反馈生成器
        mock_feedback = Mock()
        mock_feedback_class.return_value = mock_feedback
        
        # 设置反馈生成器返回值
        mock_feedback.generate_feedback_report.return_value = {
            'smi_analysis': {
                'overall_smi': 87.5,
                'style_match_scores': {
                    'analytical': 0.95,
                    'interactive': 0.75,
                    'authoritative': 0.65,
                    'supportive': 0.85
                },
                'recommended_adjustments': ['适度增加互动环节', '加强情感支持表达']
            },
            'improvement_suggestions': [
                {
                    'area': '互动教学',
                    'suggestion': '在讲解概念后增加简短的学生参与环节，如提问或小组讨论',
                    'priority': 'medium',
                    'expected_impact': '提高互动性评分约10-15%'
                },
                {
                    'area': '反馈机制',
                    'suggestion': '增加正面反馈频率，特别是在学生回答问题后',
                    'priority': 'high',
                    'expected_impact': '提升支持性评分约15-20%'
                }
            ],
            'growth_analysis': {
                'growth_trends': {
                    'analytical': 'stable',
                    'interactive': 'improving',
                    'authoritative': 'stable',
                    'supportive': 'improving'
                },
                'improved_areas': ['互动性', '支持性'],
                'areas_needing_attention': [],
                'consistency_score': 0.88
            },
            'summary': '总体而言，教学风格与数学学科高度匹配（SMI：87.5）。分析型教学风格能够有效传达复杂的数学概念。建议适度增加互动环节以进一步提升教学效果。',
            'actionable_next_steps': [
                '设计2-3个简短的课堂互动问题',
                '准备正面反馈的具体语言模板',
                '在讲解新概念后预留2-3分钟的讨论时间'
            ],
            'recommended_resources': [
                {
                    'type': 'article',
                    'title': '数学教学中的互动策略',
                    'description': '探讨如何在保持分析型教学优势的同时增加互动性',
                    'relevance': 'high'
                },
                {
                    'type': 'workshop',
                    'title': '有效的教学反馈技巧',
                    'description': '提升教师反馈能力的实践工作坊',
                    'relevance': 'medium'
                }
            ]
        }
        
        # 导入并运行完整的分析管道
        from main import run_analysis_pipeline
        
        # 执行分析
        results = run_analysis_pipeline(
            video_path=self.test_video_path,
            teacher_id=self.teacher_id,
            subject=self.subject,
            output_dir=self.test_output_dir
        )
        
        # 验证所有组件都被正确调用
        mock_extractor.process_video.assert_called_once_with(self.test_video_path)
        mock_classifier.classify_style.assert_called_once()
        mock_feedback.generate_feedback_report.assert_called_once()
        
        # 验证结果结构完整性
        self.assertIn('features', results)
        self.assertIn('classification', results)
        self.assertIn('feedback', results)
        self.assertEqual(results['teacher_id'], self.teacher_id)
        self.assertEqual(results['subject'], self.subject)
        
        # 验证特征提取结果
        features = results['features']
        self.assertIn('video_features', features)
        self.assertIn('audio_features', features)
        self.assertIn('text_features', features)
        self.assertIn('fused_features', features)
        
        # 验证分类结果
        classification = results['classification']
        self.assertIn('style_scores', classification)
        self.assertIn('dominant_style', classification)
        self.assertIn('confidence', classification)
        self.assertEqual(classification['dominant_style'], 'analytical')
        self.assertTrue(classification['confidence'] > 0.9)
        
        # 验证反馈结果
        feedback = results['feedback']
        self.assertIn('smi_analysis', feedback)
        self.assertIn('improvement_suggestions', feedback)
        self.assertIn('growth_analysis', feedback)
        self.assertIn('summary', feedback)
        
        # 验证SMI分析
        smi_analysis = feedback['smi_analysis']
        self.assertIn('overall_smi', smi_analysis)
        self.assertTrue(smi_analysis['overall_smi'] > 85)
        
        # 验证改进建议
        suggestions = feedback['improvement_suggestions']
        self.assertTrue(len(suggestions) > 0)
        for suggestion in suggestions:
            self.assertIn('area', suggestion)
            self.assertIn('suggestion', suggestion)
            self.assertIn('priority', suggestion)
    
    @patch('main.run_analysis_pipeline')
    def test_batch_processing_integration(self, mock_run_pipeline):
        # 设置模拟返回值
        mock_result = {
            'video_path': "video.mp4",
            'teacher_id': self.teacher_id,
            'subject': self.subject,
            'classification': {
                'style_scores': {'analytical': 0.85, 'interactive': 0.60},
                'dominant_style': 'analytical',
                'confidence': 0.92
            },
            'feedback': {
                'smi_analysis': {'overall_smi': 87.5},
                'summary': '教学风格分析摘要'
            }
        }
        mock_run_pipeline.return_value = mock_result
        
        # 导入并运行批量分析
        from main import batch_analysis
        
        # 模拟视频路径列表（通过patch避免实际文件系统操作）
        with patch('main.get_video_paths_from_directory', return_value=['video1.mp4', 'video2.mp4']):
            results = batch_analysis(
                directory="tests/data",
                teacher_id=self.teacher_id,
                subject=self.subject,
                output_dir=self.test_output_dir,
                max_workers=1  # 单线程以确保mock按顺序调用
            )
        
        # 验证批量分析运行正常
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_run_pipeline.call_count, 2)
        
        # 验证每个结果的正确性
        for result in results:
            self.assertEqual(result['teacher_id'], self.teacher_id)
            self.assertEqual(result['subject'], self.subject)
            self.assertIn('classification', result)
            self.assertIn('feedback', result)

if __name__ == '__main__':
    unittest.main()