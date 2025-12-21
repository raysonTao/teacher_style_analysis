"""可视化工具模块，用于绘制检测结果和姿态信息"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import VIDEO_CONFIG, VISUALIZATION_DIR, logger

# MediaPipe姿态关键点连接关系（骨架）
POSE_CONNECTIONS = [
    # 面部
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # 躯干
    (9, 10),
    # 左臂
    (11, 13), (13, 15),
    # 右臂
    (12, 14), (14, 16),
    # 左手
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    # 右手
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    # 躯干连接
    (11, 12), (11, 23), (12, 24), (23, 24),
    # 左腿
    (23, 25), (25, 27), (27, 29), (27, 31),
    (29, 31),
    # 右腿
    (24, 26), (26, 28), (28, 30), (28, 32),
    (30, 32)
]


class VisualizationManager:
    """可视化管理器，负责绘制检测结果和姿态信息"""

    def __init__(self, video_path: str):
        """
        初始化可视化管理器

        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.video_name = Path(video_path).stem

        # 使用视频文件的hash作为唯一标识
        self.video_id = self._generate_video_id(video_path)

        # 创建输出目录
        self.output_dir = VISUALIZATION_DIR / self.video_id
        self.frames_dir = self.output_dir / "frames"

        if VIDEO_CONFIG['enable_visualization']:
            self.output_dir.mkdir(exist_ok=True, parents=True)
            if VIDEO_CONFIG['save_visualization_frames']:
                self.frames_dir.mkdir(exist_ok=True, parents=True)

        # 视频写入器
        self.video_writer = None
        self.video_output_path = None

        logger.info(f"初始化可视化管理器: video_id={self.video_id}, output_dir={self.output_dir}")

    def _generate_video_id(self, video_path: str) -> str:
        """
        生成视频唯一ID（使用文件hash）

        Args:
            video_path: 视频文件路径

        Returns:
            视频ID
        """
        try:
            with open(video_path, 'rb') as f:
                # 读取文件的前1MB用于生成hash
                content = f.read(1024 * 1024)
                video_hash = hashlib.sha256(content).hexdigest()[:10]
                return f"{self.video_name}_{video_hash}"
        except Exception as e:
            logger.error(f"生成视频ID失败: {e}")
            # 如果失败，使用文件名作为ID
            return self.video_name

    def init_video_writer(self, frame_width: int, frame_height: int, fps: float):
        """
        初始化视频写入器

        Args:
            frame_width: 视频宽度
            frame_height: 视频高度
            fps: 视频帧率
        """
        if not VIDEO_CONFIG['enable_visualization'] or not VIDEO_CONFIG['save_visualization_video']:
            return

        self.video_output_path = self.output_dir / f"{self.video_id}_visualization.mp4"

        # 使用mp4v编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.video_output_path),
            fourcc,
            fps,
            (frame_width, frame_height)
        )

        if self.video_writer.isOpened():
            logger.info(f"视频写入器初始化成功: {self.video_output_path}")
        else:
            logger.error(f"视频写入器初始化失败: {self.video_output_path}")
            self.video_writer = None

    def draw_detection_and_pose(
        self,
        frame: np.ndarray,
        detection: Dict,
        pose_result: Dict,
        action_name: str = None,
        action_confidence: float = None,
        frame_count: int = None
    ) -> np.ndarray:
        """
        在帧上绘制检测框和姿态信息

        Args:
            frame: 输入帧
            detection: 检测结果字典
            pose_result: 姿态估计结果字典
            action_name: 动作名称
            action_confidence: 动作置信度
            frame_count: 帧编号

        Returns:
            绘制后的帧
        """
        vis_frame = frame.copy()

        # 1. 绘制检测边界框（红色）
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(
            vis_frame,
            (x1, y1),
            (x2, y2),
            VIDEO_CONFIG['bbox_color'],
            VIDEO_CONFIG['bbox_thickness']
        )

        # 在边界框上方显示类别和置信度
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        label_y = max(y1 - 10, 20)
        cv2.putText(
            vis_frame,
            label,
            (x1, label_y),
            VIDEO_CONFIG['text_font'],
            VIDEO_CONFIG['text_font_scale'],
            VIDEO_CONFIG['bbox_color'],
            VIDEO_CONFIG['text_thickness']
        )

        # 2. 绘制姿态关键点和骨架
        if pose_result['success'] and pose_result['keypoints'] is not None:
            keypoints = pose_result['keypoints']
            h, w = frame.shape[:2]

            # 绘制骨架连接线
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_point = keypoints[start_idx]
                    end_point = keypoints[end_idx]

                    # 检查可见性
                    if start_point[3] > 0.5 and end_point[3] > 0.5:
                        # 转换归一化坐标到像素坐标
                        pt1 = (int(start_point[0] * w), int(start_point[1] * h))
                        pt2 = (int(end_point[0] * w), int(end_point[1] * h))

                        cv2.line(
                            vis_frame,
                            pt1,
                            pt2,
                            VIDEO_CONFIG['skeleton_color'],
                            VIDEO_CONFIG['skeleton_thickness']
                        )

            # 绘制关键点
            for kp in keypoints:
                if kp[3] > 0.5:  # 可见性阈值
                    x = int(kp[0] * w)
                    y = int(kp[1] * h)
                    cv2.circle(
                        vis_frame,
                        (x, y),
                        VIDEO_CONFIG['keypoint_radius'],
                        VIDEO_CONFIG['keypoint_color'],
                        -1
                    )

        # 3. 显示姿态和动作信息（蓝色文本）
        info_y = 30
        info_texts = []

        if frame_count is not None:
            info_texts.append(f"Frame: {frame_count}")

        if pose_result['success']:
            info_texts.append(f"Pose Confidence: {pose_result['confidence']:.2f}")

        if action_name is not None:
            action_text = f"Action: {action_name}"
            if action_confidence is not None:
                action_text += f" ({action_confidence:.2f})"
            info_texts.append(action_text)

        # 绘制信息文本
        for text in info_texts:
            cv2.putText(
                vis_frame,
                text,
                (10, info_y),
                VIDEO_CONFIG['text_font'],
                VIDEO_CONFIG['text_font_scale'],
                VIDEO_CONFIG['pose_text_color'],
                VIDEO_CONFIG['text_thickness']
            )
            info_y += 25

        return vis_frame

    def save_frame(self, frame: np.ndarray, frame_count: int):
        """
        保存可视化帧

        Args:
            frame: 要保存的帧
            frame_count: 帧编号
        """
        if not VIDEO_CONFIG['enable_visualization']:
            return

        # 保存到视频文件
        if self.video_writer is not None and self.video_writer.isOpened():
            self.video_writer.write(frame)

        # 保存为图片文件
        if VIDEO_CONFIG['save_visualization_frames']:
            frame_path = self.frames_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

    def release(self):
        """释放资源"""
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"可视化视频已保存: {self.video_output_path}")

        logger.info(f"可视化结果已保存到: {self.output_dir}")

    def get_output_summary(self) -> Dict:
        """
        获取输出摘要信息

        Returns:
            输出摘要字典
        """
        summary = {
            'video_id': self.video_id,
            'output_dir': str(self.output_dir),
            'video_output_path': str(self.video_output_path) if self.video_output_path else None,
            'frames_dir': str(self.frames_dir) if VIDEO_CONFIG['save_visualization_frames'] else None
        }

        return summary
