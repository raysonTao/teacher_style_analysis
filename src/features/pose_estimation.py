"""姿态估计模块，负责使用MediaPipe进行人体姿态估计"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import BASE_DIR, MODEL_CONFIG, logger

class MediaPipePoseEstimator:
    """MediaPipe姿态估计类，用于检测人体姿态关键点"""
    
    def __init__(self):
        """初始化MediaPipe Pose模型"""
        self.pose = None
        self._load_model()
    
    def _load_model(self):
        """加载MediaPipe Pose模型"""
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_tasks
            import urllib.request
            logger.info("初始化MediaPipe Pose模型...")

            # 定义模型路径
            model_dir = os.path.join(BASE_DIR, "models", "weights")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "pose_landmarker_lite.task")

            # 如果模型不存在，下载它
            if not os.path.exists(model_path):
                logger.info("下载MediaPipe Pose模型...")
                model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    logger.info(f"模型下载成功: {model_path}")
                except Exception as download_error:
                    logger.error(f"模型下载失败: {download_error}")
                    logger.info("尝试使用镜像源...")
                    # 如果官方源失败，可以添加镜像源
                    raise

            # 配置姿态估计模型参数（使用新API）
            base_options = mp_tasks.BaseOptions(
                model_asset_path=model_path
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=MODEL_CONFIG['mediapipe_min_detection_confidence'],
                min_tracking_confidence=MODEL_CONFIG['mediapipe_min_tracking_confidence']
            )

            self.pose = vision.PoseLandmarker.create_from_options(options)
            self._frame_counter = 0  # 用于视频模式的帧计数器

            logger.info("MediaPipe Pose模型初始化成功")

        except Exception as e:
            logger.error(f"MediaPipe Pose模型初始化失败: {e}")
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

        try:
            import mediapipe as mp

            # 转换颜色空间从BGR到RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 创建MediaPipe Image对象
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 计算时间戳（毫秒）
            timestamp_ms = int(self._frame_counter * 33.33)  # 假设30fps
            self._frame_counter += 1

            # 进行姿态估计（VIDEO模式需要时间戳）
            results = self.pose.detect_for_video(mp_image, timestamp_ms)

            # 如果没有检测到姿态关键点
            if not results.pose_landmarks:
                return {
                    'success': False,
                    'keypoints': None,
                    'confidence': 0.0
                }

            # 提取第一个检测到的姿态关键点（通常只有一个人）
            pose_keypoints = []
            total_confidence = 0
            landmark_count = 0

            for landmark in results.pose_landmarks[0]:
                # 获取关键点坐标和置信度
                x = landmark.x
                y = landmark.y
                z = landmark.z
                visibility = landmark.visibility if hasattr(landmark, 'visibility') else landmark.presence

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

        except Exception as e:
            logger.error(f"姿态估计处理失败: {e}")
            return {
                'success': False,
                'keypoints': None,
                'confidence': 0.0
            }
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.pose is not None
