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
from .action_recognition import PoseActionRecognizer, STGCNActionRecognizer
from .tracker import TeacherTracker
from .visualization_utils import VisualizationManager

class VideoFeatureExtractor:
    """视频特征提取类，整合目标检测、姿态估计和动作识别"""
    
    # 单例模式实现
    _instance = None
    
    def __new__(cls):
        """控制实例创建，确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super(VideoFeatureExtractor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化视频特征提取器"""
        # 确保只初始化一次
        if not hasattr(self, 'object_detector'):
            self.object_detector = YOLOObjectDetector()
            self.pose_estimator = MediaPipePoseEstimator()
            self.action_recognizer = PoseActionRecognizer()
            self.stgcn_recognizer = STGCNActionRecognizer()
            self.tracker = TeacherTracker()

            # 初始化特征变量
            self.action_sequence = []
            self.action_counts = defaultdict(int)
            self.action_score_sums = defaultdict(float)
            self.action_score_frames = 0
            self.pose_confidences = []
            self.motion_energy = []
            self.spatial_distribution = defaultdict(int)
            self.keypoints_buffer = []
            self.teacher_track_id = None
            self.teacher_track_missing = 0
            self.teacher_track_frames = 0
            self.detection_frames = 0

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
    
    def _select_teacher_from_detections(self, detections: List[Dict], frame_height: int) -> Optional[Dict]:
        """
        从一帧的多个检测结果中选出最可能是教师的一个人

        教师识别策略：
        1. 位置在画面前方（y坐标较小）
        2. 检测框较大（离镜头近）
        3. 排除明显的学生（y > 画面高度的75%）

        Args:
            detections: 检测结果列表
            frame_height: 帧高度

        Returns:
            最可能是教师的检测结果，如果没有则返回None
        """
        if not detections:
            return None

        # 计算bbox面积并添加到每个detection中
        enhanced_detections = []
        for d in detections:
            bbox = d['bbox']
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            enhanced_d = d.copy()
            enhanced_d['bbox_area'] = bbox_area
            enhanced_detections.append(enhanced_d)

        # 步骤1: 过滤掉明显的学生（画面下方75%以后的区域）
        student_threshold_y = frame_height * 0.75
        candidates = [d for d in enhanced_detections if d['center'][1] < student_threshold_y]

        # 如果过滤后没有候选（全是学生），退回到选择y最小的
        if not candidates:
            logger.debug(f"所有检测都在学生区域，选择最靠前的")
            candidates = enhanced_detections

        # 步骤2: 在候选中选择最可能的教师
        # 综合考虑位置（y坐标）和大小（bbox面积）
        # 优先级：位置权重60%，大小权重40%
        def teacher_score(detection):
            # y坐标归一化（越小越好，所以用1减去）
            y_normalized = 1 - (detection['center'][1] / frame_height)
            # bbox面积归一化
            max_area = max(d['bbox_area'] for d in candidates)
            area_normalized = detection['bbox_area'] / max_area if max_area > 0 else 0
            # 加权评分
            return 0.6 * y_normalized + 0.4 * area_normalized

        teacher = max(candidates, key=teacher_score)

        logger.debug(f"选中教师: y={teacher['center'][1]:.1f}, area={teacher['bbox_area']:.0f}, "
                    f"候选人数={len(candidates)}, 总检测={len(detections)}")

        return teacher

    def _select_teacher_from_tracks(self, tracks: List[Dict], frame_height: int) -> Optional[Dict]:
        """从跟踪结果中选择教师轨迹"""
        if not tracks:
            return None

        enhanced_tracks = []
        for t in tracks:
            bbox = t['bbox']
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            center_y = (bbox[1] + bbox[3]) / 2
            enhanced = t.copy()
            enhanced['bbox_area'] = bbox_area
            enhanced['center'] = [
                (bbox[0] + bbox[2]) / 2,
                center_y
            ]
            enhanced_tracks.append(enhanced)

        student_threshold_y = frame_height * 0.75
        candidates = [t for t in enhanced_tracks if t['center'][1] < student_threshold_y]
        if not candidates:
            candidates = enhanced_tracks

        def teacher_score(track):
            y_normalized = 1 - (track['center'][1] / frame_height)
            max_area = max(t['bbox_area'] for t in candidates)
            area_normalized = track['bbox_area'] / max_area if max_area > 0 else 0
            return 0.6 * y_normalized + 0.4 * area_normalized

        return max(candidates, key=teacher_score)

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
            "action_distribution": {},
            "pose_estimation": [],
            "pose_confidences": [],
            "motion_energy": [],
            "avg_motion_energy": 0.0,
            "spatial_distribution": defaultdict(int),
            "keypoints_sequence": [],
            "teacher_track_id": None,
            "track_continuity": 0.0,
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

                self.detection_frames += 1

                # DeepSORT跟踪
                tracks = self.tracker.update(detections, curr_frame)
                teacher_track = None
                if self.teacher_track_id is not None:
                    for t in tracks:
                        if t.get('track_id') == self.teacher_track_id:
                            teacher_track = t
                            break

                if teacher_track is None:
                    teacher_track = self._select_teacher_from_tracks(tracks, height)
                    if teacher_track:
                        self.teacher_track_id = teacher_track.get('track_id')
                        self.teacher_track_missing = 0

                if teacher_track:
                    self.teacher_track_frames += 1
                    self.teacher_track_missing = 0
                else:
                    self.teacher_track_missing += 1
                    if self.teacher_track_missing > VIDEO_CONFIG.get('teacher_track_patience', 45):
                        self.teacher_track_id = None

                if teacher_track:
                    bbox = teacher_track["bbox"]
                    center = teacher_track.get("center")
                    if center is None:
                        center = [
                            (bbox[0] + bbox[2]) / 2,
                            (bbox[1] + bbox[3]) / 2
                        ]
                    teacher_detection = {
                        "class_name": teacher_track.get("class_name", "person"),
                        "confidence": teacher_track.get("confidence", 0.0),
                        "bbox": bbox,
                        "center": center
                    }

                    # 姿态估计（聚焦教师区域）
                    pose_result = self.pose_estimator.estimate_pose(curr_frame, bbox=bbox)

                    action_name = "unknown"
                    action_confidence = 0.0
                    action_scores = {}
                    action_source = "stgcn"

                    if pose_result["success"] and pose_result["keypoints"] is not None:
                        pose_keypoints = pose_result["keypoints"]
                        pose_confidence = pose_result["confidence"]

                        self.keypoints_buffer.append(pose_keypoints)
                        features["keypoints_sequence"].append(pose_keypoints.tolist())
                        features["pose_estimation"].append(pose_keypoints.tolist())
                        max_buffer = VIDEO_CONFIG.get('stgcn_sequence_length', 32) * 2
                        if len(self.keypoints_buffer) > max_buffer:
                            self.keypoints_buffer = self.keypoints_buffer[-max_buffer:]

                        if (len(self.keypoints_buffer) >= VIDEO_CONFIG.get('stgcn_sequence_length', 32) and
                                frame_count % VIDEO_CONFIG.get('stgcn_stride', 8) == 0):
                            seq = np.stack(self.keypoints_buffer[-VIDEO_CONFIG.get('stgcn_sequence_length', 32):])
                            action_name, action_confidence, action_scores = (
                                self.stgcn_recognizer.recognize_action_sequence(seq)
                            )
                        else:
                            action_source = "stgcn_unready"

                        # 更新动作序列
                        action_info = {
                            "frame": frame_count,
                            "time": frame_count / fps,
                            "action": action_name,
                            "confidence": action_confidence,
                            "pose_confidence": pose_confidence,
                            "bbox": bbox,
                            "center": center,
                            "source": action_source,
                            "action_scores": action_scores
                        }

                        self.action_sequence.append(action_info)
                        features["action_sequence"].append(action_info)

                        # 更新动作计数
                        self.action_counts[action_name] += 1
                        features["action_counts"][action_name] += 1

                        if action_scores:
                            for key, value in action_scores.items():
                                self.action_score_sums[key] += value
                            self.action_score_frames += 1

                        # 记录姿态置信度
                        self.pose_confidences.append(pose_confidence)
                        features["pose_confidences"].append(pose_confidence)
                    else:
                        # 没有姿态信息，标记为未知动作
                        action_info = {
                            "frame": frame_count,
                            "time": frame_count / fps,
                            "action": action_name,
                            "confidence": action_confidence,
                            "bbox": bbox,
                            "center": center,
                            "source": "none"
                        }

                        self.action_sequence.append(action_info)
                        features["action_sequence"].append(action_info)

                    # 确定空间区域
                    spatial_region = self._get_spatial_region(
                        center[0] / width,
                        center[1] / height,
                        width,
                        height
                    )

                    self.spatial_distribution[spatial_region] += 1
                    features["spatial_distribution"][spatial_region] += 1

                    # 可视化：绘制检测框和姿态信息
                    if VIDEO_CONFIG['enable_visualization'] and self.visualization_manager is not None:
                        vis_frame = self.visualization_manager.draw_detection_and_pose(
                            curr_frame,
                            teacher_detection,
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

        legacy_map = {
            "gesturing": ["wave", "raise_hand_hold"],
            "pointing": ["pointing"],
            "writing": ["writing"],
            "standing": ["standing"],
            "walking": ["walking"]
        }
        for legacy_action, mapped in legacy_map.items():
            if legacy_action not in features["action_frequency"]:
                features["action_frequency"][legacy_action] = sum(
                    features["action_frequency"].get(a, 0.0) for a in mapped
                )

        if self.action_score_frames > 0:
            features["action_distribution"] = {
                action: score / self.action_score_frames
                for action, score in self.action_score_sums.items()
            }

        # 兼容字段：behavior_frequency / behavior_counts
        features["behavior_frequency"] = dict(features["action_frequency"])
        features["behavior_counts"] = dict(features["action_counts"])

        features["teacher_track_id"] = self.teacher_track_id
        if self.detection_frames > 0:
            features["track_continuity"] = self.teacher_track_frames / self.detection_frames
        
        return features
    
    def reset(self):
        """重置特征提取器状态"""
        self.action_sequence = []
        self.action_counts = defaultdict(int)
        self.action_score_sums = defaultdict(float)
        self.action_score_frames = 0
        self.pose_confidences = []
        self.motion_energy = []
        self.spatial_distribution = defaultdict(int)
        self.keypoints_buffer = []
        self.teacher_track_id = None
        self.teacher_track_missing = 0
        self.teacher_track_frames = 0
        self.detection_frames = 0
