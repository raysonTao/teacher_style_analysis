"""深度学习模型推理包装器"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.deep_learning.mman_model import MMANModel, create_model
from src.models.deep_learning.config import ModelConfig
from src.config.config import logger


class DeepLearningInference:
    """深度学习模型推理器"""

    def __init__(
        self,
        checkpoint_path: str = "./checkpoints/best_model.pth",
        model_config: str = 'default',
        device: Optional[str] = None
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            model_config: 模型配置（default/lightweight/high_accuracy）
            device: 设备（cuda/cpu，默认自动选择）
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_config_name = model_config

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 风格标签映射
        self.style_labels = [
            '理论讲授型', '启发引导型', '互动导向型', '逻辑推导型',
            '题目驱动型', '情感表达型', '耐心细致型'
        ]

        # 加载模型
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        """加载训练好的模型"""
        try:
            if not self.checkpoint_path.exists():
                logger.warning(f"检查点文件不存在: {self.checkpoint_path}")
                logger.warning("深度学习模型未加载，将只能使用规则系统")
                return

            logger.info(f"加载深度学习模型: {self.checkpoint_path}")

            # 创建模型
            self.model = create_model(self.model_config_name)

            # 加载检查点
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 移到设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True

            # 记录模型信息
            epoch = checkpoint.get('epoch', 'unknown')
            best_acc = checkpoint.get('best_val_acc', 'unknown')
            logger.info(f"模型加载成功 - Epoch: {epoch}, 最佳准确率: {best_acc}")

            # 如果有保存的指标，也加载它
            metrics_path = self.checkpoint_path.parent / 'best_metrics.json'
            if metrics_path.exists():
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    self.metrics = json.load(f)
                logger.info(f"模型性能 - Accuracy: {self.metrics.get('accuracy', 'N/A'):.4f}, "
                          f"F1 (macro): {self.metrics.get('f1_macro', 'N/A'):.4f}")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.warning("将使用规则系统作为备用")
            self.is_loaded = False

    def _extract_feature_vector(self, features: Dict) -> Dict[str, torch.Tensor]:
        """
        从完整特征中提取深度学习模型需要的特征向量

        Args:
            features: 完整的多模态特征字典

        Returns:
            模型输入格式的特征字典
        """
        # 提取视频特征（20维）
        video_features = []
        video = features.get('video', {})
        video_features.extend([
            video.get('head_movement_frequency', 0.0),
            video.get('body_movement_frequency', 0.0),
            video.get('behavior_frequency', {}).get('writing', 0.0),
            video.get('behavior_frequency', {}).get('gesturing', 0.0),
            video.get('behavior_frequency', {}).get('pointing', 0.0),
            video.get('behavior_frequency', {}).get('standing', 0.0),
            video.get('behavior_frequency', {}).get('walking', 0.0),
            video.get('eye_contact_score', 0.0),
            video.get('facial_expression_scores', {}).get('neutral', 0.0),
            video.get('facial_expression_scores', {}).get('happy', 0.0),
            video.get('facial_expression_scores', {}).get('surprise', 0.0),
            video.get('facial_expression_scores', {}).get('sad', 0.0),
            video.get('facial_expression_scores', {}).get('angry', 0.0),
            video.get('facial_expression_scores', {}).get('disgust', 0.0),
            video.get('facial_expression_scores', {}).get('fear', 0.0),
        ])
        # 补足到20维
        while len(video_features) < 20:
            video_features.append(0.0)
        video_features = video_features[:20]

        # 提取音频特征（15维）
        audio_features = []
        audio = features.get('audio', {})
        audio_features.extend([
            audio.get('speech_rate', 0.0) / 200.0,  # 归一化
            audio.get('volume_level', 0.0),
            audio.get('pitch_variation', 0.0),
            audio.get('silence_ratio', 0.0),
            audio.get('emotion_scores', {}).get('neutral', 0.0),
            audio.get('emotion_scores', {}).get('happy', 0.0),
            audio.get('emotion_scores', {}).get('sad', 0.0),
            audio.get('emotion_scores', {}).get('angry', 0.0),
            audio.get('emotion_scores', {}).get('fear', 0.0),
            audio.get('emotion_scores', {}).get('disgust', 0.0),
            audio.get('emotion_scores', {}).get('surprise', 0.0),
        ])
        # 补足到15维
        while len(audio_features) < 15:
            audio_features.append(0.0)
        audio_features = audio_features[:15]

        # 提取文本特征（25维）
        text_features = []
        text = features.get('text', {})
        text_features.extend([
            text.get('vocabulary_richness', 0.0),
            text.get('sentence_complexity', 0.0),
            text.get('question_frequency', 0.0),
            text.get('keyword_density', {}).get('definition', 0.0),
            text.get('keyword_density', {}).get('example', 0.0),
            text.get('keyword_density', {}).get('explanation', 0.0),
            text.get('keyword_density', {}).get('summary', 0.0),
            text.get('keyword_density', {}).get('question', 0.0),
            text.get('logical_indicators', {}).get('causality', 0.0),
            text.get('logical_indicators', {}).get('comparison', 0.0),
            text.get('logical_indicators', {}).get('sequence', 0.0),
            text.get('logical_indicators', {}).get('emphasis', 0.0),
            text.get('sentiment_score', 0.0),
        ])
        # 补足到25维
        while len(text_features) < 25:
            text_features.append(0.0)
        text_features = text_features[:25]

        # 转换为tensor
        feature_tensors = {
            'video': torch.tensor(video_features, dtype=torch.float32).unsqueeze(0),
            'audio': torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0),
            'text': torch.tensor(text_features, dtype=torch.float32).unsqueeze(0)
        }

        return feature_tensors

    def _extract_rule_features(self, features: Dict) -> Optional[torch.Tensor]:
        """
        从特征中提取规则系统输出（7维）

        Args:
            features: 完整的多模态特征字典

        Returns:
            规则特征tensor或None
        """
        # 如果features中已经有预计算的风格指标，可以使用它
        fusion = features.get('fusion', {})
        style_metrics = fusion.get('teaching_style_metrics', {})

        if style_metrics:
            rule_features = [
                style_metrics.get('lecturing', 0.0),
                style_metrics.get('guiding', 0.0),
                style_metrics.get('interactive', 0.0),
                style_metrics.get('logical', 0.0),
                style_metrics.get('problem_driven', 0.0),
                style_metrics.get('emotional', 0.0),
                style_metrics.get('patient', 0.0),
            ]
            return torch.tensor(rule_features, dtype=torch.float32).unsqueeze(0)

        return None

    @torch.no_grad()
    def predict(
        self,
        features: Dict,
        use_rule_features: bool = True
    ) -> Dict:
        """
        使用深度学习模型进行预测

        Args:
            features: 多模态特征字典
            use_rule_features: 是否使用规则系统特征

        Returns:
            预测结果字典
        """
        if not self.is_loaded:
            raise RuntimeError("深度学习模型未加载，请检查检查点文件是否存在")

        try:
            # 提取特征向量
            feature_tensors = self._extract_feature_vector(features)

            # 移到设备
            feature_tensors = {
                k: v.to(self.device) for k, v in feature_tensors.items()
            }

            # 提取规则特征（如果需要）
            rule_features = None
            if use_rule_features:
                rule_features = self._extract_rule_features(features)
                if rule_features is not None:
                    rule_features = rule_features.to(self.device)
                elif self.model.config.use_rule_features:
                    # 如果模型训练时使用了规则特征，但现在无法提取，提供全零向量
                    # 这样可以保持维度一致，避免分类器维度不匹配
                    logger.warning("无法提取规则特征，使用全零向量作为回退")
                    rule_features = torch.zeros(
                        1, self.model.config.rule_feature_dim,
                        dtype=torch.float32,
                        device=self.device
                    )
            elif self.model.config.use_rule_features:
                # 即使不使用规则特征，如果模型训练时使用了，也需要提供全零向量
                # 以保持维度一致
                rule_features = torch.zeros(
                    1, self.model.config.rule_feature_dim,
                    dtype=torch.float32,
                    device=self.device
                )

            # 前向传播
            outputs = self.model(feature_tensors, rule_features)

            # 获取预测结果
            logits = outputs['logits']
            probabilities = outputs['probabilities']
            prediction = outputs['predictions']

            # 转换为numpy
            probs = probabilities.cpu().numpy()[0]
            pred_class = prediction.cpu().item()

            # 构建结果
            style_scores = {
                self.style_labels[i]: float(probs[i])
                for i in range(len(self.style_labels))
            }

            # 获取前3个最高分数的风格
            sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
            top_styles = sorted_styles[:3]

            result = {
                'style_scores': style_scores,
                'predicted_style': self.style_labels[pred_class],
                'top_styles': top_styles,
                'confidence': float(probs[pred_class]),
                'probabilities': probs.tolist()
            }

            return result

        except Exception as e:
            logger.error(f"深度学习模型推理失败: {e}")
            raise

    def get_model_info(self) -> Dict:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        if not self.is_loaded:
            return {
                'is_loaded': False,
                'message': '模型未加载'
            }

        info = {
            'is_loaded': True,
            'checkpoint_path': str(self.checkpoint_path),
            'model_config': self.model_config_name,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'style_labels': self.style_labels
        }

        # 如果有加载的指标信息，添加进去
        if hasattr(self, 'metrics'):
            info['performance'] = {
                'accuracy': self.metrics.get('accuracy', 'N/A'),
                'f1_macro': self.metrics.get('f1_macro', 'N/A'),
                'f1_weighted': self.metrics.get('f1_weighted', 'N/A')
            }

        return info


# 创建全局推理器实例（延迟初始化）
_inference_instance = None

def get_inference_instance(
    checkpoint_path: str = "./checkpoints/best_model.pth",
    model_config: str = 'default',
    device: Optional[str] = None,
    force_reload: bool = False
) -> DeepLearningInference:
    """
    获取深度学习推理器单例

    Args:
        checkpoint_path: 检查点路径
        model_config: 模型配置
        device: 设备
        force_reload: 是否强制重新加载

    Returns:
        推理器实例
    """
    global _inference_instance

    if _inference_instance is None or force_reload:
        _inference_instance = DeepLearningInference(
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            device=device
        )

    return _inference_instance


if __name__ == '__main__':
    """测试推理器"""
    print("测试深度学习推理器...")

    # 创建推理器
    inference = DeepLearningInference()

    # 打印模型信息
    info = inference.get_model_info()
    print("\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if inference.is_loaded:
        # 创建测试特征
        test_features = {
            'video': {
                'head_movement_frequency': 0.5,
                'behavior_frequency': {'gesturing': 0.3, 'pointing': 0.2},
                'facial_expression_scores': {'happy': 0.6, 'neutral': 0.4}
            },
            'audio': {
                'speech_rate': 120,
                'volume_level': 0.7,
                'pitch_variation': 0.5,
                'emotion_scores': {'happy': 0.5, 'neutral': 0.5}
            },
            'text': {
                'vocabulary_richness': 0.6,
                'sentence_complexity': 0.5,
                'question_frequency': 0.3,
                'keyword_density': {'explanation': 0.4}
            },
            'fusion': {
                'teaching_style_metrics': {
                    'lecturing': 0.7,
                    'guiding': 0.5,
                    'interactive': 0.6,
                    'logical': 0.4,
                    'problem_driven': 0.3,
                    'emotional': 0.5,
                    'patient': 0.4
                }
            }
        }

        # 进行预测
        print("\n进行预测...")
        result = inference.predict(test_features)

        print("\n预测结果:")
        print(f"  预测风格: {result['predicted_style']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"\n  Top-3 风格:")
        for style, score in result['top_styles']:
            print(f"    {style}: {score:.4f}")

        print("\n所有风格得分:")
        for style, score in result['style_scores'].items():
            print(f"  {style}: {score:.4f}")

    print("\n推理器测试完成!")
