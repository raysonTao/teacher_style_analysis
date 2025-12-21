"""视频特征提取模块，负责提取视频中的特征"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from collections import defaultdict

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import VIDEO_CONFIG, logger
from .object_detection import YOLOObjectDetector
from .pose_estimation import MediaPipePoseEstimator
from .action_recognition import PoseActionRecognizer
from .visualization_utils import VisualizationManager

class VideoFeatureExtractor:
    """视频特征提取类，整合目标检测、姿态估计和动作识别"""
    
    def __init__(self):
        """初始化视频特征提取器"""
        self.object_detector = YOLOObjectDetector()
        self.pose_estimator = MediaPipePoseEstimator()
        self.action_recognizer = PoseActionRecognizer()

        # 初始化特征变量
        self.action_sequence = []
        self.action_counts = defaultdict(int)
        self.pose_confidences = []
        self.motion_energy = []
        self.spatial_distribution = defaultdict(int)

        # 可视化管理器（延迟初始化）
        self.visualization_manager = None
        
    def _calculate_motion_energy(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """
        计算两帧之间的运动能量
        
        Args:
            prev_frame: 前一帧
            curr_frame: 当前帧
            
        Returns:
            运动能量值
        """
        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # 计算帧差
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        
        # 二值化
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # 计算运动能量
        motion_energy = np.sum(thresh) / (prev_frame.shape[0] * prev_frame.shape[1])
        
        return motion_energy
    
    def _get_spatial_region(self, center_x: float, center_y: float, frame_width: int, frame_height: int) -> str:
        """
        根据中心点坐标确定空间区域
        
        Args:
            center_x: 中心点x坐标
            center_y: 中心点y坐标
            frame_width: 帧宽度
            frame_height: 帧高度
            
        Returns:
            空间区域名称
        """
        # 将归一化坐标转换为像素坐标
        x = int(center_x * frame_width)
        y = int(center_y * frame_height)
        
        # 定义区域边界
        third_width = frame_width // 3
        third_height = frame_height // 3
        
        # 确定水平区域
        if x < third_width:
            horizontal = "left"
        elif x < 2 * third_width:
            horizontal = "center"
        else:
            horizontal = "right"
        
        # 确定垂直区域
        if y < third_height:
            vertical = "top"
        elif y < 2 * third_height:
            vertical = "middle"
        else:
            vertical = "bottom"
        
        return f"{vertical}_{horizontal}"
    
    def extract_features(self, video_path: str) -> Dict:
        """
        提取视频特征

        Args:
            video_path: 视频文件路径

        Returns:
            视频特征字典
        """
        # 初始化特征字典
        features = {
            "action_sequence": [],
            "action_counts": defaultdict(int),
            "action_frequency": {},
            "pose_confidences": [],
            "motion_energy": [],
            "avg_motion_energy": 0.0,
            "spatial_distribution": defaultdict(int),
            "video_duration": 0.0,
            "total_frames": 0,
            "visualization_output": None
        }

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return features

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"视频信息: FPS={fps}, 总帧数={total_frames}, 宽度={width}, 高度={height}, 时长={duration:.2f}秒")

        features["total_frames"] = total_frames
        features["video_duration"] = duration

        # 初始化可视化管理器
        if VIDEO_CONFIG['enable_visualization']:
            self.visualization_manager = VisualizationManager(video_path)
            self.visualization_manager.init_video_writer(width, height, fps)
            logger.info("可视化管理器已启用")
        
        # 初始化前一帧
        ret, prev_frame = cap.read()
        if not ret:
            logger.error("无法读取视频帧")
            cap.release()
            return features
        
        # 读取视频帧
        frame_count = 1
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # 计算运动能量
            if frame_count % VIDEO_CONFIG["motion_energy_frame_interval"] == 0:
                energy = self._calculate_motion_energy(prev_frame, curr_frame)
                self.motion_energy.append(energy)
                features["motion_energy"].append(energy)
            
            # 每隔几帧进行一次检测
            if frame_count % VIDEO_CONFIG["detection_frame_interval"] == 0:
                logger.debug(f"处理第 {frame_count} 帧")

                # 目标检测（只检测人）
                detections = self.object_detector.detect(
                    curr_frame,
                    confidence_threshold=VIDEO_CONFIG["detection_confidence_threshold"],
                    class_filter=["person"]
                )

                for detection in detections:
                    class_name = detection["class_name"]
                    confidence = detection["confidence"]
                    bbox = detection["bbox"]
                    center = detection["center"]

                    # 只处理person类别
                    if class_name != "person":
                        logger.debug("非人物目标，跳过")
                        continue

                    # 姿态估计
                    pose_result = self.pose_estimator.estimate_pose(curr_frame)

                    if pose_result["success"] and pose_result["keypoints"] is not None:
                        pose_keypoints = pose_result["keypoints"]
                        pose_confidence = pose_result["confidence"]

                        # 动作识别
                        action_name, action_confidence = self.action_recognizer.recognize_action(pose_keypoints)

                        # 更新动作序列
                        action_info = {
                            "frame": frame_count,
                            "time": frame_count / fps,
                            "action": action_name,
                            "confidence": action_confidence,
                            "pose_confidence": pose_confidence,
                            "bbox": bbox,
                            "center": center
                        }

                        self.action_sequence.append(action_info)
                        features["action_sequence"].append(action_info)

                        # 更新动作计数
                        self.action_counts[action_name] += 1
                        features["action_counts"][action_name] += 1

                        # 记录姿态置信度
                        self.pose_confidences.append(pose_confidence)
                        features["pose_confidences"].append(pose_confidence)

                        # 确定空间区域
                        spatial_region = self._get_spatial_region(
                            center[0] / width,  # 归一化x坐标
                            center[1] / height,  # 归一化y坐标
                            width,
                            height
                        )

                        self.spatial_distribution[spatial_region] += 1
                        features["spatial_distribution"][spatial_region] += 1

                        # 可视化：绘制检测框和姿态信息
                        if VIDEO_CONFIG['enable_visualization'] and self.visualization_manager is not None:
                            vis_frame = self.visualization_manager.draw_detection_and_pose(
                                curr_frame,
                                detection,
                                pose_result,
                                action_name,
                                action_confidence,
                                frame_count
                            )
                            self.visualization_manager.save_frame(vis_frame, frame_count)
                    else:
                        # 没有姿态信息，使用默认动作
                        action_name = "unknown"
                        action_confidence = 0.5

                        action_info = {
                            "frame": frame_count,
                            "time": frame_count / fps,
                            "action": action_name,
                            "confidence": action_confidence,
                            "bbox": bbox,
                            "center": center
                        }

                        self.action_sequence.append(action_info)
                        features["action_sequence"].append(action_info)

                        # 可视化：即使没有姿态也绘制检测框
                        if VIDEO_CONFIG['enable_visualization'] and self.visualization_manager is not None:
                            vis_frame = self.visualization_manager.draw_detection_and_pose(
                                curr_frame,
                                detection,
                                pose_result,
                                action_name,
                                action_confidence,
                                frame_count
                            )
                            self.visualization_manager.save_frame(vis_frame, frame_count)
            
            # 更新前一帧
            prev_frame = curr_frame
            frame_count += 1
            
            # 限制处理帧数用于测试
            if VIDEO_CONFIG["test_mode"] and frame_count > VIDEO_CONFIG["test_frame_limit"]:
                logger.info("测试模式，提前结束")
                break
        
        # 释放视频资源
        cap.release()

        # 释放可视化资源并获取输出信息
        if VIDEO_CONFIG['enable_visualization'] and self.visualization_manager is not None:
            self.visualization_manager.release()
            features["visualization_output"] = self.visualization_manager.get_output_summary()

        # 计算平均运动能量
        if features["motion_energy"]:
            features["avg_motion_energy"] = np.mean(features["motion_energy"])
        
        # 计算动作频率
        total_actions = sum(features["action_counts"].values())
        if total_actions > 0:
            for action, count in features["action_counts"].items():
                features["action_frequency"][action] = count / total_actions
        
        return features
    
    def reset(self):
        """重置特征提取器状态"""
        self.action_sequence = []
        self.action_counts = defaultdict(int)
        self.pose_confidences = []
        self.motion_energy = []
        self.spatial_distribution = defaultdict(int)
