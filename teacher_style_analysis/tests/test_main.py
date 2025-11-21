import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
import tempfile
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
    @patch('main.data_manager')
    def test_run_analysis_pipeline(self, mock_data_manager, mock_feedback, mock_classifier, mock_extractor):
        # 设置模拟返回值
        mock_extractor.extract_and_save.return_value = "features/test_features.json"
        
        mock_classifier.classify_and_save.return_value = "results/test_results.json"
        
        mock_feedback.generate_feedback_report.return_value = {
            'smi': {'score': 85.0},
            'teaching_style': {
                'main_styles': [['analytical', 0.8]],
                'detailed_scores': {
                    'analytical': 0.8,
                    'interactive': 0.6,
                    'authoritative': 0.4,
                    'supportive': 0.7
                }
            },
            'improvement_suggestions': [{'area': '互动性', 'suggestion': '增加提问'}],
            'growth_analysis': {'improved_areas': ['分析能力']},
            'summary': '教学风格分析摘要'
        }
        
        mock_data_manager.save_video_info.return_value = None
        mock_data_manager.update_video_status.return_value = None
        
        # 测试分析管道
        results = run_analysis_pipeline(
            self.mock_video_path,
            self.mock_teacher_id,
            self.mock_subject,
            "高中"
        )
        
        # 验证所有方法都被调用
        mock_data_manager.save_video_info.assert_called_once()
        mock_extractor.extract_and_save.assert_called_once_with(self.mock_video_path, unittest.mock.ANY)
        mock_classifier.classify_and_save.assert_called_once_with(unittest.mock.ANY)
        mock_feedback.generate_feedback_report.assert_called_once_with(unittest.mock.ANY, self.mock_subject, "高中")
        mock_data_manager.update_video_status.assert_called_once_with(unittest.mock.ANY, "completed")
        
        # 验证结果包含必要的键
        self.assertIn('video_id', results)
        self.assertIn('status', results)
        self.assertIn('feedback', results)
        self.assertIn('features_path', results)
        self.assertIn('result_path', results)
    
    @patch('main.run_analysis_pipeline')
    def test_batch_analysis(self, mock_run_pipeline):
        # 创建临时测试目录和文件
        test_dir = tempfile.mkdtemp()
        test_files = []
        for i in range(3):
            test_file = os.path.join(test_dir, f"video{i+1}.mp4")
            with open(test_file, 'w') as f:
                f.write("dummy video content")
            test_files.append(test_file)
        
        # 模拟单个分析结果
        mock_result = {
            'video_id': 'test123',
            'status': 'completed',
            'feedback': {'smi': {'score': 85.0}},
            'features_path': 'features/test.json',
            'result_path': 'results/test.json'
        }
        mock_run_pipeline.return_value = mock_result
        
        try:
            # 测试批量分析
            results = batch_analysis(
                test_dir,
                self.mock_teacher_id,
                self.mock_subject,
                "高中"
            )
            
            # 验证方法调用
            self.assertEqual(mock_run_pipeline.call_count, 3)
            
            # 验证结果
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertEqual(result['success'], True)
                self.assertEqual(result['data'], mock_result)
        finally:
            # 清理临时文件
            for test_file in test_files:
                if os.path.exists(test_file):
                    os.remove(test_file)
            os.rmdir(test_dir)
    
    def test_export_results(self):
        # 创建临时反馈文件用于测试
        import json
        from teacher_style_analysis.config.config import FEEDBACK_DIR
        
        # 确保反馈目录存在
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        
        # 创建测试反馈文件
        test_video_id = "test_video_123"
        test_feedback = {
            'teacher_id': 'teacher1',
            'discipline': '数学',
            'grade': '高中',
            'smi': {'score': 85.0},
            'teaching_style': {
                'main_styles': [['analytical', 0.8]],
                'detailed_scores': {
                    'analytical': 0.8,
                    'interactive': 0.6,
                    'authoritative': 0.4,
                    'supportive': 0.7
                }
            },
            'improvement_suggestions': [{'area': '互动性', 'suggestion': '增加提问'}],
            'growth_analysis': {'improved_areas': ['分析能力']},
            'summary': '教学风格分析摘要'
        }
        
        feedback_file = FEEDBACK_DIR / f"{test_video_id}_feedback.json"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(test_feedback, f, ensure_ascii=False, indent=2)
        
        # 初始化路径变量
        json_path = None
        csv_path = None
        
        try:
            # 测试导出为JSON
            json_path = export_results(test_video_id, 'json')
            self.assertTrue(os.path.exists(json_path))
            
            # 测试导出为CSV
            csv_path = export_results(test_video_id, 'csv')
            self.assertTrue(os.path.exists(csv_path))
            
            # 测试导出为Excel（如果openpyxl可用）
            try:
                xlsx_path = export_results(test_video_id, 'excel')
                self.assertTrue(os.path.exists(xlsx_path))
                # 清理Excel文件
                if os.path.exists(xlsx_path):
                    os.remove(xlsx_path)
            except ImportError:
                # 如果缺少openpyxl，跳过Excel测试
                pass
            
        finally:
            # 清理测试文件
            if feedback_file.exists():
                feedback_file.unlink()
            # 清理导出的文件
            for path in [json_path, csv_path]:
                if path and os.path.exists(path):
                    os.remove(path)
    
    def test_setup_database(self):
        # 测试数据库设置 - 新版本不需要参数
        try:
            setup_database()
            # 如果成功执行，测试通过
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"数据库设置失败: {e}")
    
    def test_check_system_status(self):
        # 测试系统状态检查 - 不需要模拟，直接调用
        status = check_system_status()
        
        # 验证状态结果
        self.assertIn('status', status)
        self.assertIn('missing_packages', status)
        self.assertIn('directories', status)
        # 状态应该是healthy、warning或error之一
        self.assertIn(status['status'], ['healthy', 'warning', 'error'])

if __name__ == '__main__':
    unittest.main()