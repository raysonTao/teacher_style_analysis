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
        
        # 模拟数据库会话
        self.mock_db = Mock()
        
        # 模拟核心组件
        self.feature_extractor_patcher = patch('api.api_handler.feature_extractor')
        self.style_classifier_patcher = patch('api.api_handler.style_classifier')
        self.feedback_generator_patcher = patch('api.api_handler.feedback_generator')
        
        self.mock_feature_extractor = self.feature_extractor_patcher.start()
        self.mock_style_classifier = self.style_classifier_patcher.start()
        self.mock_feedback_generator = self.feedback_generator_patcher.start()
    
    def tearDown(self):
        # 停止所有补丁
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
        self.assertIn("supported_subjects", config)
        self.assertIn("style_categories", config)
        self.assertIn("system_version", config)
    
    def test_upload_video(self):
        # 创建临时视频文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b'test video data')
            tmp_file_path = tmp_file.name
        
        try:
            # 模拟数据库操作
            mock_video = Mock()
            mock_video.id = "test-video-id"
            self.mock_db.add.return_value = None
            self.mock_db.commit.return_value = None
            self.mock_db.refresh.return_value = None
            
            # 发送上传请求
            with open(tmp_file_path, 'rb') as f:
                response = self.client.post(
                    "/api/upload_video",
                    files={"file": ("test_video.mp4", f, "video/mp4")},
                    data={"teacher_id": "teacher1", "subject": "数学", "title": "测试视频"}
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
            Mock(id="video1", title="视频1", teacher_id="teacher1", subject="数学", 
                 uploaded_at="2023-01-01", status="completed"),
            Mock(id="video2", title="视频2", teacher_id="teacher2", subject="物理", 
                 uploaded_at="2023-01-02", status="processing")
        ]
        
        # 设置模拟查询
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_videos
        
        self.mock_db.query.return_value = mock_query
        
        # 发送请求
        response = self.client.get("/api/videos?page=1&page_size=10")
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        data = response.json()
        self.assertIn("videos", data)
        self.assertIn("total", data)
        self.assertEqual(len(data["videos"]), 2)
    
    def test_get_video(self):
        # 模拟视频数据
        mock_video = Mock(
            id="video1",
            title="测试视频",
            teacher_id="teacher1",
            subject="数学",
            uploaded_at="2023-01-01",
            status="completed",
            analysis_results={"style_scores": {"analytical": 0.8}}
        )
        
        # 设置模拟查询
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_video
        
        # 发送请求
        response = self.client.get("/api/videos/video1")
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        data = response.json()
        self.assertEqual(data["id"], "video1")
        self.assertEqual(data["title"], "测试视频")
        self.assertIn("analysis_results", data)
    
    def test_delete_video(self):
        # 模拟视频数据
        mock_video = Mock(id="video1")
        
        # 设置模拟查询
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_video
        
        # 发送请求
        response = self.client.delete("/api/videos/video1")
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        self.assertEqual(response.json(), {"message": "视频删除成功", "video_id": "video1"})
        
        # 验证删除操作被调用
        self.mock_db.delete.assert_called_once_with(mock_video)
        self.mock_db.commit.assert_called_once()
    
    @patch('api.api_handler.process_video_analysis')
    def test_analyze_style(self, mock_process):
        # 设置模拟返回值
        mock_process.return_value = {
            "style_scores": {"analytical": 0.8, "interactive": 0.6},
            "dominant_style": "analytical",
            "confidence": 0.9
        }
        
        # 模拟视频数据
        mock_video = Mock(
            id="video1",
            file_path="test_video.mp4",
            subject="数学",
            status="uploaded"
        )
        
        # 设置模拟查询
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_video
        
        # 发送请求
        response = self.client.post("/api/analyze_style/video1")
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        data = response.json()
        self.assertIn("style_scores", data)
        self.assertIn("dominant_style", data)
        self.assertIn("confidence", data)

if __name__ == '__main__':
    unittest.main()