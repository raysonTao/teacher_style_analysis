"""动作识别模块，负责基于姿态估计结果识别动作"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ACTION_CONFIG, VIDEO_CONFIG, MODEL_CONFIG, logger

try:
    from models.deep_learning.stgcn_model import STGCNModel
except Exception:
    STGCNModel = None

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
            
            # 计算动作评分（连续评分避免硬阈值）
            def score_center(value: float, center: float, width: float) -> float:
                if width <= 0:
                    return 0.0
                return float(np.exp(-((value - center) / width) ** 2))

            def score_range(value: float, low: float, high: float) -> float:
                if low >= high:
                    return 0.0
                return float(np.clip((value - low) / (high - low), 0.0, 1.0))

            knee_flex = max(
                score_center(left_knee_angle, 110.0, 40.0),
                score_center(right_knee_angle, 110.0, 40.0)
            )
            knee_straight = min(
                score_center(left_knee_angle, 175.0, 20.0),
                score_center(right_knee_angle, 175.0, 20.0)
            )

            left_elbow_bent = score_center(left_elbow_angle, 70.0, 35.0)
            right_elbow_bent = score_center(right_elbow_angle, 70.0, 35.0)
            left_elbow_straight = score_center(left_elbow_angle, 175.0, 20.0)
            right_elbow_straight = score_center(right_elbow_angle, 175.0, 20.0)

            left_wrist_raise = score_range(left_shoulder[1] - left_wrist[1], 0.05, 0.4)
            right_wrist_raise = score_range(right_shoulder[1] - right_wrist[1], 0.05, 0.4)
            left_wrist_extend = score_range(abs(left_wrist_relative_x), 0.1, 0.5)
            right_wrist_extend = score_range(abs(right_wrist_relative_x), 0.1, 0.5)

            # 动作得分
            walking_score = knee_flex * score_range(abs(left_knee_angle - right_knee_angle), 5.0, 40.0)
            writing_score = max(
                left_elbow_bent * (1.0 - left_wrist_extend),
                right_elbow_bent * (1.0 - right_wrist_extend)
            )
            gesturing_score = max(
                left_elbow_bent * left_wrist_raise * left_wrist_extend,
                right_elbow_bent * right_wrist_raise * right_wrist_extend
            )
            pointing_score = max(
                left_elbow_straight * left_wrist_extend,
                right_elbow_straight * right_wrist_extend
            )
            standing_score = knee_straight * score_center((left_elbow_angle + right_elbow_angle) / 2, 170.0, 25.0)

            scores = {
                "walking": walking_score,
                "writing": writing_score,
                "gesturing": gesturing_score,
                "pointing": pointing_score,
                "standing": standing_score
            }

            action_name = max(scores.items(), key=lambda x: x[1])[0]
            confidence = float(np.clip(scores[action_name], 0.0, 1.0))

            return action_name, confidence
            
        except Exception as e:
            logger.error(f"动作识别失败: {e}")
            import traceback
            traceback.print_exc()
            return 'unknown', 0.0
    
    def get_action_config(self) -> Dict:
        """获取动作识别配置"""
        return self.action_config


class STGCNActionRecognizer:
    """基于ST-GCN的动作识别器"""

    def __init__(self):
        self.action_labels = VIDEO_CONFIG.get('stgcn_action_labels', [])
        self.model = None
        self.device = None
        self._load_model()

    def _load_model(self):
        if STGCNModel is None:
            logger.error("ST-GCN模型定义不可用，无法加载")
            return

        try:
            import torch

            model_path = MODEL_CONFIG.get('stgcn_model_path')
            if not model_path or not os.path.exists(model_path):
                logger.error("未找到ST-GCN模型权重，无法进行动作识别")
                return

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            num_classes = max(len(self.action_labels), 1)
            self.model = STGCNModel(num_class=num_classes).to(self.device).eval()

            state = torch.load(model_path, map_location=self.device)
            state_dict = state.get('state_dict', state)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"ST-GCN模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"ST-GCN模型加载失败: {e}")
            self.model = None

    def recognize_action_sequence(self, keypoints_sequence: np.ndarray) -> Tuple[str, float, Dict]:
        """
        基于关键点序列识别动作

        Args:
            keypoints_sequence: [T, V, C] 或 [T, V, 4]

        Returns:
            (动作名称, 置信度, 动作分布)
        """
        if self.model is None or keypoints_sequence is None:
            return 'unknown', 0.0, {}

        try:
            import torch
            import torch.nn.functional as F

            sequence = np.array(keypoints_sequence, dtype=np.float32)
            if sequence.ndim != 3:
                return 'unknown', 0.0, {}

            # 仅使用x,y维度
            if sequence.shape[2] >= 2:
                sequence = sequence[:, :, :2]
            else:
                return 'unknown', 0.0, {}

            # 归一化到0-1范围
            sequence = np.clip(sequence, 0.0, 1.0)

            # 输入形状 [N, C, T, V]
            sequence = np.transpose(sequence, (2, 0, 1))
            tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            scores = {}
            for idx, score in enumerate(probs.tolist()):
                label = self.action_labels[idx] if idx < len(self.action_labels) else f"action_{idx}"
                scores[label] = float(score)

            if scores:
                best_label, best_score = max(scores.items(), key=lambda x: x[1])
            else:
                best_label, best_score = 'unknown', 0.0

            return best_label, float(best_score), scores

        except Exception as e:
            logger.warning(f"ST-GCN动作识别失败: {e}")
            return 'unknown', 0.0, {}
