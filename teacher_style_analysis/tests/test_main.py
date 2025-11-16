import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import run_analysis_pipeline, batch_analysis, export_results, setup_database, check_system_status

class TestMain(unittest.TestCase):
    def setUp(self):
        # 模拟配置和环境
        self.mock_video_path = "tests/data/sample_video.mp4"
        self.mock_output_dir = "tests/output"
        self.mock_teacher_id = "teacher1"
        self.mock_subject = "数学"
        
        # 创建必要的目录
        if not os.path.exists(self.mock_output_dir):
            os.makedirs(self.mock_output_dir)
    
    def tearDown(self):
        # 清理测试生成的文件
        if os.path.exists(self.mock_output_dir):
            for file in os.listdir(self.mock_output_dir):
                os.remove(os.path.join(self.mock_output_dir, file))
            os.rmdir(self.mock_output_dir)
    
    @patch('main.feature_extractor')
    @patch('main.style_classifier')
    @patch('main.feedback_generator')
    @patch('main.save_results')
    def test_run_analysis_pipeline(self, mock_save, mock_feedback, mock_classifier, mock_extractor):
        # 设置模拟返回值
        mock_extractor.process_video.return_value = {
            'video_features': {'gesture_features': np.array([0.1, 0.2])},
            'audio_features': {'transcript': '测试文本', 'speaking_rate': 100.0},
            'text_features': {'semantic_features': np.array([0.3, 0.4])},
            'fused_features': np.array([0.1, 0.2, 0.3, 0.4])
        }
        
        mock_classifier.classify_style.return_value = {
            'style_scores': {
                'analytical': 0.8,
                'interactive': 0.6,
                'authoritative': 0.4,
                'supportive': 0.7
            },
            'dominant_style': 'analytical',
            'confidence': 0.9
        }
        
        mock_feedback.generate_feedback_report.return_value = {
            'smi_analysis': {'overall_smi': 85.0},
            'improvement_suggestions': [{'area': '互动性', 'suggestion': '增加提问'}],
            'growth_analysis': {'improved_areas': ['分析能力']},
            'summary': '教学风格分析摘要'
        }
        
        # 测试分析管道
        results = run_analysis_pipeline(
            self.mock_video_path,
            self.mock_teacher_id,
            self.mock_subject,
            output_dir=self.mock_output_dir
        )
        
        # 验证所有方法都被调用
        mock_extractor.process_video.assert_called_once_with(self.mock_video_path)
        mock_classifier.classify_style.assert_called_once()
        mock_feedback.generate_feedback_report.assert_called_once()
        mock_save.assert_called_once()
        
        # 验证结果包含必要的键
        self.assertIn('features', results)
        self.assertIn('classification', results)
        self.assertIn('feedback', results)
        self.assertIn('video_path', results)
        self.assertIn('teacher_id', results)
        self.assertIn('subject', results)
    
    @patch('main.run_analysis_pipeline')
    @patch('main.get_video_paths_from_directory')
    def test_batch_analysis(self, mock_get_paths, mock_run_pipeline):
        # 模拟视频路径列表
        mock_video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
        mock_get_paths.return_value = mock_video_paths
        
        # 模拟单个分析结果
        mock_result = {
            'video_path': "video.mp4",
            'classification': {'dominant_style': 'analytical'},
            'feedback': {'smi_analysis': {'overall_smi': 85.0}}
        }
        mock_run_pipeline.return_value = mock_result
        
        # 测试批量分析
        results = batch_analysis(
            "tests/data",
            self.mock_teacher_id,
            self.mock_subject,
            output_dir=self.mock_output_dir,
            max_workers=2
        )
        
        # 验证方法调用
        mock_get_paths.assert_called_once_with("tests/data")
        self.assertEqual(mock_run_pipeline.call_count, 3)
        
        # 验证结果
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result, mock_result)
    
    def test_export_results(self):
        # 模拟分析结果
        analysis_results = [
            {
                'video_path': 'video1.mp4',
                'teacher_id': 'teacher1',
                'subject': '数学',
                'classification': {
                    'style_scores': {'analytical': 0.8, 'interactive': 0.6},
                    'dominant_style': 'analytical',
                    'confidence': 0.9
                },
                'feedback': {
                    'smi_analysis': {'overall_smi': 85.0},
                    'summary': '分析摘要1'
                }
            },
            {
                'video_path': 'video2.mp4',
                'teacher_id': 'teacher1',
                'subject': '物理',
                'classification': {
                    'style_scores': {'authoritative': 0.7, 'supportive': 0.5},
                    'dominant_style': 'authoritative',
                    'confidence': 0.8
                },
                'feedback': {
                    'smi_analysis': {'overall_smi': 78.0},
                    'summary': '分析摘要2'
                }
            }
        ]
        
        # 测试导出为CSV
        csv_path = os.path.join(self.mock_output_dir, "results.csv")
        export_results(analysis_results, csv_path, format="csv")
        
        # 验证CSV文件存在
        self.assertTrue(os.path.exists(csv_path))
        
        # 测试导出为JSON
        json_path = os.path.join(self.mock_output_dir, "results.json")
        export_results(analysis_results, json_path, format="json")
        
        # 验证JSON文件存在
        self.assertTrue(os.path.exists(json_path))
    
    @patch('main.sqlalchemy.create_engine')
    @patch('main.sqlalchemy.orm.sessionmaker')
    def test_setup_database(self, mock_sessionmaker, mock_create_engine):
        # 模拟数据库连接
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_session = Mock()
        mock_sessionmaker.return_value = mock_session
        
        # 测试数据库设置
        db = setup_database("sqlite:///:memory:")
        
        # 验证数据库连接创建
        mock_create_engine.assert_called_once_with("sqlite:///:memory:")
        mock_sessionmaker.assert_called_once_with(bind=mock_engine)
    
    @patch('main.feature_extractor')
    @patch('main.style_classifier')
    @patch('main.feedback_generator')
    def test_check_system_status(self, mock_feedback, mock_classifier, mock_extractor):
        # 设置模拟状态
        mock_extractor.get_status.return_value = {"status": "healthy", "models_loaded": 3}
        mock_classifier.get_status.return_value = {"status": "healthy", "model_version": "1.0"}
        mock_feedback.get_status.return_value = {"status": "healthy"}
        
        # 测试系统状态检查
        status = check_system_status()
        
        # 验证状态检查
        mock_extractor.get_status.assert_called_once()
        mock_classifier.get_status.assert_called_once()
        mock_feedback.get_status.assert_called_once()
        
        # 验证状态结果
        self.assertIn('status', status)
        self.assertIn('components', status)
        self.assertEqual(status['status'], 'healthy')
        self.assertEqual(len(status['components']), 3)

if __name__ == '__main__':
    unittest.main()