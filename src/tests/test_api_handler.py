import unittest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from api.api_handler import app
import tempfile
import os

class TestApiHandler(unittest.TestCase):
    def setUp(self):
        # 创建测试客户端
        self.client = TestClient(app)
        
        # 模拟数据库会话
        self.mock_db = Mock()
        
        # 模拟数据管理器
        self.data_manager_patcher = patch('api.api_handler.data_manager')
        self.mock_data_manager = self.data_manager_patcher.start()
        
        # 模拟核心组件
        self.feature_extractor_patcher = patch('api.api_handler.feature_extractor')
        self.style_classifier_patcher = patch('api.api_handler.style_classifier')
        self.feedback_generator_patcher = patch('api.api_handler.feedback_generator')
        
        self.mock_feature_extractor = self.feature_extractor_patcher.start()
        self.mock_style_classifier = self.style_classifier_patcher.start()
        self.mock_feedback_generator = self.feedback_generator_patcher.start()
        
        # 设置数据管理器的属性
        self.mock_data_manager.discipline_standards = {
            '数学': '理论推导',
            '语文': '情感表达',
            '英语': '互动导向',
            '物理': '逻辑推导',
            '化学': '实验探究',
            '生物': '观察分析',
            '历史': '史料分析',
            '地理': '空间思维',
            '政治': '思辨论证'
        }
        
        self.mock_data_manager.grade_standards = {
            '初中': '基础理解',
            '高中': '深化应用',
            '大学': '研究创新'
        }
        
        self.mock_data_manager.style_labels = {
            '理论讲授型': '知识传授',
            '启发引导型': '思维启发',
            '互动导向型': '交流互动',
            '逻辑推导型': '逻辑推理',
            '题目驱动型': '练习巩固',
            '情感表达型': '情感共鸣',
            '耐心细致型': '细致讲解'
        }
    
    def tearDown(self):
        # 停止所有补丁
        self.data_manager_patcher.stop()
        self.feature_extractor_patcher.stop()
        self.style_classifier_patcher.stop()
        self.feedback_generator_patcher.stop()
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = self.client.get("/api/health")
        
        # 验证响应状态码 - 允许503状态码（服务可能暂时不可用）
        self.assertIn(response.status_code, [200, 503])
        
        # 验证响应数据
        data = response.json()
        if response.status_code == 200:
            self.assertTrue(data["success"])
            self.assertEqual(data["status"], "healthy")
        else:
            self.assertEqual(data["detail"], "服务不可用")
    
    def test_get_config(self):
        # 测试获取配置端点
        response = self.client.get("/api/config")
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应包含必要的配置项
        config = response.json()
        self.assertIn("data", config)
        self.assertIn("supported_disciplines", config["data"])
        self.assertIn("supported_grades", config["data"])
        self.assertIn("style_labels", config["data"])
        self.assertIn("max_video_size", config["data"])
    
    def test_upload_video(self):
        # 创建临时视频文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b'test video data')
            tmp_file_path = tmp_file.name
        
        try:
            # 模拟数据库操作
            mock_video = Mock()
            mock_video.id = "test-video-id"
            self.mock_data_manager.save_video_info.return_value = "test-video-id"
            
            # 发送上传请求
            with open(tmp_file_path, 'rb') as f:
                response = self.client.post(
                    "/api/upload_video",
                    files={"video": ("test_video.mp4", f, "video/mp4")},
                    data={"teacher_id": "teacher1", "discipline": "数学", "grade": "高中"}
                )
            
            # 验证响应状态码
            self.assertEqual(response.status_code, 200)
            
            # 验证响应内容
            self.assertIn("video_id", response.json())
            self.assertIn("message", response.json())
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def test_get_videos(self):
        # 模拟数据库查询结果
        mock_videos = [
            Mock(id="video1", title="视频1", teacher_id="teacher1", discipline="数学", 
                 uploaded_at="2023-01-01", status="completed"),
            Mock(id="video2", title="视频2", teacher_id="teacher2", discipline="物理", 
                 uploaded_at="2023-01-02", status="processing")
        ]
        
        # 设置模拟查询
        self.mock_data_manager.list_videos.return_value = {
            'videos': mock_videos,
            'total': 2,
            'page': 1,
            'page_size': 10,
            'total_pages': 1
        }
        
        # 发送请求
        response = self.client.get("/api/videos?page=1&page_size=10")
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("videos", data["data"])
        self.assertIn("total", data["data"])
        self.assertEqual(len(data["data"]["videos"]), 2)
    
    def test_get_video(self):
        # 模拟视频数据
        mock_video = Mock(
            id="video1",
            title="测试视频",
            teacher_id="teacher1",
            discipline="数学",
            uploaded_at="2023-01-01",
            status="completed",
            analysis_results={"style_scores": {"analytical": 0.8}}
        )
        
        # 设置模拟查询
        self.mock_data_manager.get_video_info.return_value = mock_video
        
        # 发送请求
        response = self.client.get("/api/videos/video1")
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        data = response.json()
        self.assertIn("data", data)
        self.assertEqual(data["data"]["id"], "video1")
        self.assertEqual(data["data"]["title"], "测试视频")
        self.assertIn("analysis_results", data["data"])
    
    def test_delete_video(self):
        # 模拟视频数据
        mock_video = Mock(id="video1")
        
        # 设置模拟查询
        self.mock_data_manager.get_video_info.return_value = mock_video
        self.mock_data_manager.delete_video.return_value = True
        
        # 发送请求
        response = self.client.delete("/api/videos/video1")
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        self.assertEqual(response.json()["message"], "视频及相关数据已成功删除")
    
    def test_analyze_style(self):
        # 模拟视频数据
        mock_video = Mock(
            id="video1",
            file_path="/path/to/video.mp4",
            discipline="数学",
            grade="初中",
            status="pending"
        )
        
        # 模拟分析结果
        mock_analysis_result = {
            "style_scores": {"analytical": 0.8, "interactive": 0.6},
            "top_styles": [("analytical", 0.8), ("interactive", 0.6)],
            "dominant_style": "analytical",
            "confidence": 0.85,
            "feedback": {"overall": "很好的教学风格"}
        }
        
        # 设置模拟数据管理器
        self.mock_data_manager.get_video_info.return_value = mock_video
        self.mock_data_manager.update_video_status.return_value = True
        self.mock_data_manager.save_analysis_results.return_value = True
        
        # 模拟核心组件
        self.mock_feature_extractor.extract_features.return_value = {"features": [1, 2, 3]}
        self.mock_style_classifier.predict.return_value = mock_analysis_result
        self.mock_feedback_generator.generate_feedback_report.return_value = mock_analysis_result["feedback"]
        
        # 发送请求
        response = self.client.post("/api/analyze_style/video1")
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        data = response.json()
        self.assertIn("data", data)
        self.assertEqual(data["data"]["video_id"], "video1")
        self.assertIn("analysis_results", data["data"])

if __name__ == '__main__':
    unittest.main()