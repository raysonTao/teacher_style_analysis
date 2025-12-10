"""姿态估计模块，负责使用MediaPipe进行人体姿态估计"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import BASE_DIR, MODEL_CONFIG

class MediaPipePoseEstimator:
    """MediaPipe姿态估计类，用于检测人体姿态关键点"""
    
    def __init__(self):
        """初始化MediaPipe Pose模型"""
        self.pose = None
        self._load_model()
    
    def _load_model(self):
        """加载MediaPipe Pose模型"""
        try:
            import mediapipe as mp
            print("初始化MediaPipe Pose模型...")
            
            mp_pose = mp.solutions.pose
            
            # 配置姿态估计模型参数
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=MODEL_CONFIG['mediapipe_model_complexity'],
                smooth_landmarks=MODEL_CONFIG['mediapipe_smooth_landmarks'],
                enable_segmentation=False,
                min_detection_confidence=MODEL_CONFIG['mediapipe_min_detection_confidence'],
                min_tracking_confidence=MODEL_CONFIG['mediapipe_min_tracking_confidence']
            )
            
            print("MediaPipe Pose模型初始化成功")
            
        except Exception as e:
            print(f"MediaPipe Pose模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.pose = None
    
    def estimate_pose(self, frame: np.ndarray) -> Dict:
        """
        对输入帧进行姿态估计
        
        Args:
            frame: 输入的视频帧
            
        Returns:
            姿态估计结果，包含关键点和置信度
        """
        if self.pose is None:
            return {
                'success': False,
                'keypoints': None,
                'confidence': 0.0
            }
        
        # 转换颜色空间从BGR到RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 进行姿态估计
        results = self.pose.process(rgb_frame)
        
        # 如果没有检测到姿态关键点
        if not results.pose_landmarks:
            return {
                'success': False,
                'keypoints': None,
                'confidence': 0.0
            }
        
        # 提取姿态关键点
        pose_keypoints = []
        total_confidence = 0
        landmark_count = 0
        
        for landmark in results.pose_landmarks.landmark:
            # 获取关键点坐标和置信度
            x = landmark.x
            y = landmark.y
            z = landmark.z
            visibility = landmark.visibility
            
            pose_keypoints.append([x, y, z, visibility])
            total_confidence += visibility
            landmark_count += 1
        
        # 计算平均置信度
        average_confidence = total_confidence / landmark_count if landmark_count > 0 else 0
        
        return {
            'success': True,
            'keypoints': np.array(pose_keypoints),
            'confidence': average_confidence
        }
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.pose is not None
