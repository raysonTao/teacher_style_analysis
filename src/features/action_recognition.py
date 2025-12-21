"""动作识别模块，负责基于姿态估计结果识别动作"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ACTION_CONFIG, logger

class PoseActionRecognizer:
    """基于姿态的动作识别类，用于识别教师的动作"""
    
    def __init__(self):
        """初始化动作识别器"""
        self.action_config = ACTION_CONFIG
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        计算三个点之间的角度
        
        Args:
            a: 第一个点的坐标 [x, y, z, visibility]
            b: 第二个点的坐标 [x, y, z, visibility]
            c: 第三个点的坐标 [x, y, z, visibility]
            
        Returns:
            角度值（度数）
        """
        # 只使用xy坐标进行计算
        a = a[:2]
        b = b[:2]
        c = c[:2]
        
        # 计算向量
        ab = np.array([b[0] - a[0], b[1] - a[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        # 计算点积和模长
        dot_product = np.dot(ab, bc)
        ab_magnitude = np.linalg.norm(ab)
        bc_magnitude = np.linalg.norm(bc)
        
        if ab_magnitude == 0 or bc_magnitude == 0:
            return 0.0
        
        # 计算余弦值和角度
        cos_theta = dot_product / (ab_magnitude * bc_magnitude)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保值在有效范围内
        angle = np.arccos(cos_theta) * (180 / np.pi)
        
        return angle
    
    def recognize_action(self, pose_keypoints: np.ndarray) -> Tuple[str, float]:
        """
        基于姿态关键点识别动作
        
        Args:
            pose_keypoints: 姿态关键点数组，shape为(33, 4)，包含x, y, z, visibility
            
        Returns:
            动作名称和置信度
        """
        if pose_keypoints is None or len(pose_keypoints) < 33:
            return 'unknown', 0.0
        
        try:
            # 提取关键部位的关键点
            left_knee = pose_keypoints[25]  # 左膝盖
            right_knee = pose_keypoints[26]  # 右膝盖
            left_hip = pose_keypoints[23]  # 左髋
            right_hip = pose_keypoints[24]  # 右髋
            left_shoulder = pose_keypoints[11]  # 左肩
            right_shoulder = pose_keypoints[12]  # 右肩
            left_elbow = pose_keypoints[13]  # 左肘
            right_elbow = pose_keypoints[14]  # 右肘
            left_wrist = pose_keypoints[15]  # 左手腕
            right_wrist = pose_keypoints[16]  # 右手腕
            left_ankle = pose_keypoints[27]  # 左脚踝
            right_ankle = pose_keypoints[28]  # 右脚踝
            nose = pose_keypoints[0]  # 鼻子
            
            # 计算关键角度
            left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
            left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # 计算手腕相对于身体的位置
            left_wrist_relative_x = left_wrist[0] - left_shoulder[0]
            right_wrist_relative_x = right_wrist[0] - right_shoulder[0]
            left_wrist_relative_y = left_wrist[1] - left_shoulder[1]
            right_wrist_relative_y = right_wrist[1] - right_shoulder[1]
            
            # 初始化动作和置信度
            action_name = 'standing'
            confidence = 0.7  # 默认置信度
            
            # 1. 检测walking（膝盖弯曲角度变化）
            if (left_knee_angle < 140 or right_knee_angle < 140) and \
               (left_knee_angle > 70 or right_knee_angle > 70):
                action_name = 'walking'
                confidence = 0.8
            
            # 2. 检测writing（一只手在胸前区域，手肘弯曲）
            elif (left_elbow_angle < 90 and right_elbow_angle > 150 and \
                  abs(left_wrist_relative_y) < 0.2 and left_wrist_relative_x < 0.1) or \
                 (right_elbow_angle < 90 and left_elbow_angle > 150 and \
                  abs(right_wrist_relative_y) < 0.2 and right_wrist_relative_x > 0.1):
                action_name = 'writing'
                confidence = 0.85
            
            # 3. 检测gesturing（手臂抬起，手肘弯曲）
            elif (left_elbow_angle < 120 and left_wrist[1] < left_shoulder[1] and \
                  abs(left_wrist_relative_x) > 0.1) or \
                 (right_elbow_angle < 120 and right_wrist[1] < right_shoulder[1] and \
                  abs(right_wrist_relative_x) > 0.1):
                action_name = 'gesturing'
                confidence = 0.8
            
            # 4. 检测pointing（一只手臂伸直，手指指向某个方向）
            elif (left_elbow_angle > 160 and abs(left_wrist_relative_x) > 0.3) or \
                 (right_elbow_angle > 160 and abs(right_wrist_relative_x) > 0.3):
                action_name = 'pointing'
                confidence = 0.75
            
            # 5. 其他情况默认standing
            else:
                action_name = 'standing'
                confidence = 0.7
            
            return action_name, confidence
            
        except Exception as e:
            logger.error(f"动作识别失败: {e}")
            import traceback
            traceback.print_exc()
            return 'unknown', 0.0
    
    def get_action_config(self) -> Dict:
        """获取动作识别配置"""
        return self.action_config
