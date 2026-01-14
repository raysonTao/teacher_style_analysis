"""风格识别模块，负责使用CMAT模型进行教师风格分类和可解释性分析"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config import (
    MODEL_CONFIG, SYSTEM_CONFIG, STYLE_LABELS,
    FEATURES_DIR, RESULTS_DIR, logger
)

try:
    from src.models.deep_learning.inference import get_inference_instance
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logger.warning("深度学习推理模块导入失败，仅能使用规则系统")


class StyleClassifier:
    """教师风格分类器"""

    def __init__(
        self,
        mode: str = 'rule',
        dl_checkpoint: Optional[str] = None,
        dl_model_config: str = 'default',
        dl_device: Optional[str] = None
    ):
        """
        Args:
            mode: 分类模式
                - 'rule': 仅使用规则系统（默认，向后兼容）
                - 'deep_learning': 仅使用深度学习模型
                - 'hybrid': 混合模式（规则系统 + 深度学习）
            dl_checkpoint: 深度学习模型检查点路径
            dl_model_config: 深度学习模型配置（default/lightweight/high_accuracy）
            dl_device: 深度学习模型设备（cuda/cpu）
        """
        self.mode = mode
        self.dl_inference = None
        self.model_source = 'unknown'

        # 初始化规则系统模型（始终初始化，作为备用）
        self._init_model()

        # 初始化深度学习模型（如果需要）
        if self.mode in ['deep_learning', 'hybrid'] and DL_AVAILABLE:
            self._init_deep_learning_model(
                checkpoint_path=dl_checkpoint or "./checkpoints/best_model.pth",
                model_config=dl_model_config,
                device=dl_device
            )
        elif self.mode in ['deep_learning', 'hybrid'] and not DL_AVAILABLE:
            logger.warning(f"模式设置为'{self.mode}'，但深度学习模块不可用，回退到规则系统")
            self.mode = 'rule'
    
    def _init_deep_learning_model(
        self,
        checkpoint_path: str,
        model_config: str,
        device: Optional[str]
    ):
        """初始化深度学习模型"""
        try:
            logger.info(f"初始化深度学习模型: {checkpoint_path}")
            self.dl_inference = get_inference_instance(
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                device=device
            )

            if self.dl_inference.is_loaded:
                info = self.dl_inference.get_model_info()
                logger.info(f"深度学习模型加载成功")
                logger.info(f"  模型配置: {info.get('model_config', 'N/A')}")
                logger.info(f"  设备: {info.get('device', 'N/A')}")
                logger.info(f"  参数量: {info.get('num_parameters', 'N/A'):,}")
                if 'performance' in info:
                    perf = info['performance']
                    logger.info(f"  性能: Acc={perf.get('accuracy', 'N/A'):.4f}, "
                              f"F1={perf.get('f1_macro', 'N/A'):.4f}")
            else:
                logger.warning("深度学习模型加载失败，将使用规则系统")
                if self.mode == 'deep_learning':
                    logger.warning("回退到规则系统模式")
                    self.mode = 'rule'
                elif self.mode == 'hybrid':
                    logger.warning("混合模式降级为纯规则模式")
                    self.mode = 'rule'

        except Exception as e:
            logger.error(f"深度学习模型初始化失败: {e}")
            self.dl_inference = None
            if self.mode == 'deep_learning':
                logger.warning("回退到规则系统模式")
                self.mode = 'rule'
            elif self.mode == 'hybrid':
                logger.warning("混合模式降级为纯规则模式")
                self.mode = 'rule'

    def get_status(self) -> Dict:
        """
        获取风格分类器状态

        Returns:
            包含状态信息的字典
        """
        status = {
            'mode': self.mode,
            'rule_model_loaded': self.model is not None,
            'rule_model_type': self.model_source,
            'lambda_weight': self.model.get('lambda_weight', 0.5) if self.model else 0.5,
            'status': 'ready'
        }

        # 添加深度学习模型状态
        if self.dl_inference is not None:
            dl_info = self.dl_inference.get_model_info()
            status['deep_learning_loaded'] = dl_info.get('is_loaded', False)
            status['deep_learning_info'] = dl_info
        else:
            status['deep_learning_loaded'] = False

        return status
    
    def _init_model(self):
        """初始化风格分类模型"""
        try:
            logger.info("初始化风格分类模型...")

            default_model = self._build_default_rule_model()

            # 检查是否有预训练模型
            if os.path.exists(MODEL_CONFIG['cmat_model_path']):
                with open(MODEL_CONFIG['cmat_model_path'], 'rb') as f:
                    self.model = pickle.load(f)

                if not isinstance(self.model, dict) or 'rule_weights' not in self.model:
                    raise ValueError("CMAT模型格式无效")

                # 如果模型内容与默认规则完全一致，标记为启发式而非预训练
                if self._is_default_model(self.model, default_model):
                    logger.warning("检测到规则模型为默认配置，结果仅为启发式输出")
                    self.model_source = 'heuristic'
                else:
                    logger.info("预训练模型加载成功")
                    self.model_source = 'pretrained'
            else:
                logger.warning("未找到预训练CMAT模型，使用规则配置作为回退")
                self.model = default_model
                self.model_source = 'heuristic'

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            self.model = self._build_default_rule_model()
            self.model_source = 'heuristic'
    
    def _build_default_rule_model(self) -> Dict:
        """创建默认规则模型（启发式参数）"""
        rule_model = {
            'rule_weights': {
                'lecturing': {'speech_rate': 0.3, 'silence_ratio': -0.4, 'interaction_level': -0.5},
                'guiding': {'question_frequency': 0.5, 'interaction_level': 0.3, 'gesturing': 0.2},
                'interactive': {'interaction_level': 0.6, 'gesturing': 0.3, 'silence_ratio': 0.1},
                'logical': {'logical_indicators': 0.5, 'vocabulary_richness': 0.3, 'sentence_complexity': 0.2},
                'problem_driven': {'question_frequency': 0.6, 'keyword_density': 0.4},
                'emotional': {'emotion_scores': 0.5, 'pitch_variation': 0.4, 'volume_level': 0.1},
                'patient': {'silence_ratio': 0.5, 'speech_rate': -0.3, 'emotion_scores': 0.2}
            },
            'ml_params': {
                'feature_importance': {
                    'interaction_level': 0.25,
                    'logical_structure': 0.20,
                    'emotional_engagement': 0.15,
                    'explanation_clarity': 0.15,
                    'speech_rate': 0.10,
                    'question_frequency': 0.10,
                    'gesturing_frequency': 0.05
                }
            },
            'lambda_weight': SYSTEM_CONFIG['lambda_weight']
        }
        return rule_model

    def _is_default_model(self, model: Dict, default_model: Dict) -> bool:
        """判断模型是否为默认规则配置"""
        try:
            return (
                model.get('rule_weights') == default_model.get('rule_weights') and
                model.get('ml_params') == default_model.get('ml_params') and
                model.get('lambda_weight') == default_model.get('lambda_weight')
            )
        except Exception:
            return False

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + np.exp(-value))

    @staticmethod
    def _bell_score(value: float, center: float, width: float) -> float:
        if width <= 0:
            return 0.0
        return float(np.exp(-((value - center) / width) ** 2))
    
    def _apply_rules(self, features: Dict) -> Dict:
        """
        应用规则驱动层
        
        Args:
            features: 融合后的特征
            
        Returns:
            规则驱动层的输出
        """
        # 添加调试日志
        logger.debug(f"_apply_rules: features类型: {type(features)}")
        logger.debug(f"_apply_rules: features值: {features}")
        
        # 确保features不为None
        if features is None:
            features = {}
            
        rule_output = {}
        rule_weights = self.model['rule_weights']
        
        # 提取需要的特征
        fusion = features.get('fusion', features.get('fused_features', {}).get('fusion', {}))
        audio = features.get('audio', features.get('audio_features', {}))
        video = features.get('video', features.get('video_features', {}))
        text = features.get('text', features.get('text_features', {}))
        
        # 计算规则驱动的输出
        for style, weights in rule_weights.items():
            score = 0.0
            weight_sum = 0.0
            
            for feature_name, weight in weights.items():
                # 映射特征名到实际特征值
                feature_value = 0.0
                
                if feature_name == 'interaction_level':
                    feature_value = fusion.get('interaction_level', 0.0)
                elif feature_name == 'logical_structure':
                    feature_value = fusion.get('logical_structure', 0.0)
                elif feature_name == 'emotional_engagement':
                    feature_value = fusion.get('emotional_engagement', 0.0)
                elif feature_name == 'explanation_clarity':
                    feature_value = fusion.get('explanation_clarity', 0.0)
                elif feature_name == 'speech_rate':
                    # 语速适中度（假设120字/分钟为最优）
                    speech_rate = audio.get('speech_rate', 120)
                    feature_value = 1.0 - abs(speech_rate - 120) / 120
                elif feature_name == 'silence_ratio':
                    feature_value = audio.get('silence_ratio', 0.0)
                elif feature_name == 'question_frequency':
                    feature_value = text.get('question_frequency', 0.0) * 10  # 放大效应
                elif feature_name == 'gesturing':
                    frequency = video.get('behavior_frequency', video.get('action_frequency', {}))
                    feature_value = frequency.get('gesturing', 0.0)
                elif feature_name == 'logical_indicators':
                    feature_value = sum(text.get('logical_indicators', {}).values()) * 10
                elif feature_name == 'vocabulary_richness':
                    feature_value = text.get('vocabulary_richness', 0.0)
                elif feature_name == 'sentence_complexity':
                    feature_value = text.get('sentence_complexity', 0.0)
                elif feature_name == 'keyword_density':
                    feature_value = sum(text.get('keyword_density', {}).values())
                elif feature_name == 'emotion_scores':
                    # 积极情绪分数
                    emotion_scores = audio.get('emotion_scores', {})
                    feature_value = emotion_scores.get('happy', 0.0) * 0.7 + emotion_scores.get('neutral', 0.0) * 0.3
                elif feature_name == 'pitch_variation':
                    feature_value = audio.get('pitch_variation', 0.0)
                elif feature_name == 'volume_level':
                    feature_value = audio.get('volume_level', 0.0)
                
                # 累加加权特征值
                score += feature_value * weight
                weight_sum += abs(weight)
            
            # 归一化
            if weight_sum > 0:
                score = score / weight_sum
                # 转换到0-1范围
                score = (score + 1) / 2 if score < 0 else score
                score = max(0, min(1, score))
            
            rule_output[style] = score
        
        return rule_output
    
    def _apply_ml_model(self, features: Dict) -> Dict:
        """
        应用机器学习层

        Args:
            features: 融合后的特征

        Returns:
            机器学习层的输出
        """
        ml_output = {}
        feature_importance = self.model['ml_params']['feature_importance']
        fusion = features.get('fusion', features.get('fused_features', {}).get('fusion', {}))

        # 计算各风格的机器学习分数
        style_metrics = fusion.get('teaching_style_metrics', {})

        # 使用预计算的风格指标作为机器学习层的输出
        for style in ['lecturing', 'guiding', 'interactive', 'logical', 'problem_driven', 'emotional', 'patient']:
            ml_output[style] = style_metrics.get(style, 0.0)

        return ml_output

    def _classify_with_deep_learning(self, features: Dict) -> Dict:
        """
        使用深度学习模型进行分类

        Args:
            features: 融合后的特征

        Returns:
            深度学习分类结果
        """
        if self.dl_inference is None or not self.dl_inference.is_loaded:
            raise RuntimeError("深度学习模型未加载")

        try:
            # 使用深度学习模型预测
            dl_result = self.dl_inference.predict(
                features=features,
                use_rule_features=(self.mode == 'hybrid')
            )

            # 转换为标准输出格式
            result = {
                'style_scores': dl_result['style_scores'],
                'top_styles': dl_result['top_styles'],
                'predicted_style': dl_result['predicted_style'],
                'confidence': dl_result['confidence'],
                'method': 'deep_learning',
                'timestamp': {
                    'analysis_time': datetime.utcnow().isoformat(timespec='seconds') + 'Z'
                }
            }

            return result

        except Exception as e:
            logger.error(f"深度学习分类失败: {e}")
            raise
    
    def classify_style(self, features_path=None, features=None) -> Dict:
        """
        对特征进行风格分类

        Args:
            features_path: 特征文件路径（可选）
            features: 特征数据（可选，优先使用）

        Returns:
            风格分类结果
        """

        # 如果提供了features参数，直接使用它
        if features is not None:
            logger.info(f"直接使用提供的特征数据（模式: {self.mode}）")
        # 如果提供了features_path参数，尝试从文件读取
        elif features_path is not None:
            logger.info(f"开始风格分类: {features_path}（模式: {self.mode}）")

            # 如果输入是numpy数组，按论文中的CMAT模型处理
            if isinstance(features_path, np.ndarray):
                # 应用规则驱动层
                rule_results = self.apply_rule_driven_layer(features_path, raw_data=None)
                # 应用机器学习层
                ml_results = self.apply_ml_layer(features_path)
                # 融合结果
                fused_results = self.fuse_outputs(rule_results, ml_results)

                return {
                    'style_scores': fused_results,
                    'dominant_style': fused_results.get('dominant_style', 'analytical'),
                    'confidence': fused_results.get('confidence', 0.85)
                }
            # 如果输入是字符串路径，按原逻辑处理
            elif isinstance(features_path, str):
                # 读取特征文件
                try:
                    with open(features_path, 'r', encoding='utf-8') as f:
                        features = json.load(f)
                except Exception as e:
                    logger.error(f"读取特征文件失败: {e}")
                    raise
            else:
                # 直接作为特征数据使用
                features = features_path or {}  # 确保features不为None
        else:
            # 两者都没有提供，使用空特征
            logger.warning(f"未提供特征数据或路径，使用空特征")
            features = {}

        # 根据模式选择分类方法
        if self.mode == 'deep_learning':
            # 纯深度学习模式
            try:
                return self._classify_with_deep_learning(features)
            except Exception as e:
                logger.error(f"深度学习分类失败，回退到规则系统: {e}")
                # 回退到规则系统
                pass

        elif self.mode == 'hybrid':
            # 混合模式：结合规则系统和深度学习
            try:
                # 获取深度学习预测
                dl_result = self._classify_with_deep_learning(features)

                # 获取规则系统预测
                rule_output = self._apply_rules(features)
                ml_output = self._apply_ml_model(features)

                # 融合规则系统结果
                lambda_weight = self.model['lambda_weight']
                rule_final = {}

                style_mapping = {
                    'lecturing': '理论讲授型',
                    'guiding': '启发引导型',
                    'interactive': '互动导向型',
                    'logical': '逻辑推导型',
                    'problem_driven': '题目驱动型',
                    'emotional': '情感表达型',
                    'patient': '耐心细致型'
                }

                for style_key, style_label in style_mapping.items():
                    rule_score = lambda_weight * rule_output[style_key] + (1 - lambda_weight) * ml_output.get(style_key, 0.0)
                    rule_final[style_label] = rule_score

                # 混合深度学习和规则系统（50/50权重）
                hybrid_scores = {}
                for style in dl_result['style_scores'].keys():
                    hybrid_scores[style] = (
                        0.5 * dl_result['style_scores'][style] +
                        0.5 * rule_final.get(style, 0.0)
                    )

                # 构建混合结果
                result = {
                    'style_scores': hybrid_scores,
                    'top_styles': self._get_top_styles(hybrid_scores),
                    'deep_learning_results': dl_result['style_scores'],
                    'rule_based_results': rule_final,
                    'confidence': (dl_result['confidence'] + self._calculate_confidence(rule_final)) / 2,
                    'method': 'hybrid',
                    'timestamp': {
                        'analysis_time': datetime.utcnow().isoformat(timespec='seconds') + 'Z'
                    }
                }

                return result

            except Exception as e:
                logger.error(f"混合模式分类失败，回退到规则系统: {e}")
                # 回退到规则系统
                pass

        # 规则系统模式（默认或回退）
        # 应用规则驱动层
        rule_output = self._apply_rules(features)

        # 应用机器学习层
        ml_output = self._apply_ml_model(features)

        # 融合两种输出
        lambda_weight = self.model['lambda_weight']
        final_output = {}

        for style in rule_output.keys():
            final_output[style] = (
                lambda_weight * rule_output[style] +
                (1 - lambda_weight) * ml_output.get(style, 0.0)
            )

        # 映射到论文中的风格标签
        labeled_output = {}
        style_mapping = {
            'lecturing': '理论讲授型',
            'guiding': '启发引导型',
            'interactive': '互动导向型',
            'logical': '逻辑推导型',
            'problem_driven': '题目驱动型',
            'emotional': '情感表达型',
            'patient': '耐心细致型'
        }

        for style_key, style_label in style_mapping.items():
            labeled_output[style_label] = final_output.get(style_key, 0.0)

        # 计算特征贡献度分析（可解释性）
        feature_contributions = self._analyze_feature_contributions(features, final_output)

        # 生成分类结果
        result = {
            'style_scores': labeled_output,
            'top_styles': self._get_top_styles(labeled_output),
            'rule_based_results': {
                style_mapping[k]: v for k, v in rule_output.items()
            },
            'ml_based_results': {
                style_mapping[k]: v for k, v in ml_output.items()
            },
            'feature_contributions': feature_contributions,
            'confidence': self._calculate_confidence(final_output),
            'method': 'rule',
            'timestamp': {
                'analysis_time': datetime.utcnow().isoformat(timespec='seconds') + 'Z'
            }
        }

        return result
    
    def _get_top_styles(self, style_scores: Dict) -> List[Tuple[str, float]]:
        """
        获取得分最高的几个风格
        
        Args:
            style_scores: 风格分数字典
            
        Returns:
            排序后的风格列表
        """
        sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_styles[:3]  # 返回前3个主要风格
    
    def apply_rule_driven_layer(self, features: np.ndarray, raw_data: Dict) -> Dict:
        """
        应用规则驱动层（对外接口）
        
        Args:
            features: 特征向量
            raw_data: 原始多模态数据
            
        Returns:
            规则驱动层输出
        """
        raw_data = raw_data or {}
        vector = np.asarray(features).reshape(-1)

        # 基于向量统计量构建基础分数
        mean_val = float(np.mean(vector)) if vector.size else 0.0
        std_val = float(np.std(vector)) if vector.size else 0.0
        max_val = float(np.max(vector)) if vector.size else 0.0
        min_val = float(np.min(vector)) if vector.size else 0.0
        energy = float(np.mean(vector ** 2)) if vector.size else 0.0

        analytical_score = self._sigmoid(mean_val + std_val)
        interactive_score = self._sigmoid(std_val + (max_val - mean_val))
        authoritative_score = self._sigmoid(energy - std_val)
        supportive_score = self._sigmoid(1.0 - abs(mean_val - 0.5))

        # 融入原始多模态信号（如果提供）
        video = raw_data.get('video_features', raw_data.get('video', {}))
        audio = raw_data.get('audio_features', raw_data.get('audio', {}))
        text = raw_data.get('text_features', raw_data.get('text', {}))

        gesture = video.get('behavior_frequency', video.get('action_frequency', {})).get('gesturing', 0.0)
        pointing = video.get('behavior_frequency', video.get('action_frequency', {})).get('pointing', 0.0)
        speech_rate = audio.get('speech_rate', 0.0)
        sentiment = text.get('sentiment_score', text.get('sentiment', {}).get('score', 0.5))
        question_frequency = text.get('question_frequency', 0.0)

        interactive_boost = self._clamp01(0.6 * gesture + 0.4 * pointing)
        authoritative_boost = self._bell_score(speech_rate, 150.0, 60.0)
        supportive_boost = self._clamp01(0.5 + (sentiment - 0.5) * 0.5)
        analytical_boost = self._clamp01(1.0 - question_frequency)

        analytical_score = self._clamp01(0.7 * analytical_score + 0.3 * analytical_boost)
        interactive_score = self._clamp01(0.7 * interactive_score + 0.3 * interactive_boost)
        authoritative_score = self._clamp01(0.7 * authoritative_score + 0.3 * authoritative_boost)
        supportive_score = self._clamp01(0.7 * supportive_score + 0.3 * supportive_boost)

        return {
            'analytical_score': analytical_score,
            'interactive_score': interactive_score,
            'authoritative_score': authoritative_score,
            'supportive_score': supportive_score
        }
    
    def apply_ml_layer(self, features: np.ndarray) -> np.ndarray:
        """
        应用机器学习层（对外接口）
        
        Args:
            features: 特征向量
            
        Returns:
            机器学习层输出
        """
        vector = np.asarray(features).reshape(-1)
        if vector.size == 0:
            return np.zeros((1, 4))

        segments = np.array_split(vector, 4)
        raw_scores = np.array([np.mean(seg) if seg.size else 0.0 for seg in segments], dtype=float)

        # 使用softmax归一化以避免随机输出
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        norm_scores = exp_scores / (np.sum(exp_scores) + 1e-8)

        return norm_scores.reshape(1, 4)
    
    def fuse_outputs(self, rule_results: Dict, ml_results: np.ndarray) -> Dict:
        """
        融合规则驱动和机器学习输出（对外接口）
        
        Args:
            rule_results: 规则驱动结果
            ml_results: 机器学习结果
            
        Returns:
            融合后的结果
        """
        fused = {}
        style_order = ['analytical', 'interactive', 'authoritative', 'supportive']
        for idx, style in enumerate(style_order):
            rule_score = rule_results.get(f'{style}_score', 0.0)
            ml_score = float(ml_results[0][idx]) if ml_results.size > 0 else 0.0
            fused[style] = self._clamp01((rule_score + ml_score) / 2)
        
        # 确定主导风格
        dominant_style = max(fused.items(), key=lambda x: x[1])[0]
        fused['dominant_style'] = dominant_style
        
        return fused
    
    def calculate_feature_contributions(self, features: np.ndarray) -> Dict:
        """
        计算特征贡献度（对外接口）
        
        Args:
            features: 特征向量
            
        Returns:
            特征贡献度
        """
        vector = np.asarray(features).reshape(-1)
        contributions = {}
        if vector.size == 0:
            for style in ['analytical', 'interactive', 'authoritative', 'supportive']:
                contributions[style] = {}
            return contributions

        segments = np.array_split(np.arange(vector.size), 4)
        for style, indices in zip(['analytical', 'interactive', 'authoritative', 'supportive'], segments):
            if indices.size == 0:
                contributions[style] = {}
                continue

            segment_values = np.abs(vector[indices])
            top_idx = indices[np.argsort(segment_values)[-5:]][::-1]
            total = np.sum(segment_values) + 1e-8
            contributions[style] = {int(i): float(vector[i] / total) for i in top_idx}

        return contributions
    
    def explain_prediction(self, classification_result: Dict, feature_contributions: Dict) -> Dict:
        """
        解释预测结果（对外接口）
        
        Args:
            classification_result: 分类结果
            feature_contributions: 特征贡献度
            
        Returns:
            解释结果
        """
        dominant_style = classification_result.get('dominant_style', 'analytical')
        
        return {
            'dominant_style_justification': f"基于特征分析，{dominant_style}风格的特征贡献度最高",
            'top_contributing_features': list(feature_contributions.get(dominant_style, {}).items())[:3],
            'style_breakdown': {
                style: {
                    'score': classification_result.get(style, 0.5),
                    'contributing_features': list(contributions.items())[:2]
                }
                for style, contributions in feature_contributions.items()
            }
        }
    
    def _analyze_feature_contributions(self, features: Dict, final_output: Dict) -> Dict:
        """
        分析各特征对风格分类结果的贡献度
        
        Args:
            features: 原始特征
            final_output: 最终分类结果
            
        Returns:
            特征贡献度分析
        """
        contributions = {}
        fusion = features.get('fusion', {})
        
        # 分析主要特征的贡献
        key_features = [
            ('互动水平', fusion.get('interaction_level', 0.0)),
            ('讲解清晰度', fusion.get('explanation_clarity', 0.0)),
            ('情感投入度', fusion.get('emotional_engagement', 0.0)),
            ('逻辑结构', fusion.get('logical_structure', 0.0)),
            ('语速', features.get('audio', {}).get('speech_rate', 0.0)),
            ('提问频率', features.get('text', {}).get('question_frequency', 0.0) * 10),
            ('手势频率', features.get('video', {}).get('behavior_frequency', {}).get('gesturing', 0.0))
        ]
        
        # 为每个主要风格分析关键贡献特征
        style_feature_mapping = {
            '理论讲授型': ['讲解清晰度', '逻辑结构', '语速'],
            '启发引导型': ['互动水平', '提问频率', '情感投入度'],
            '互动导向型': ['互动水平', '手势频率', '提问频率'],
            '逻辑推导型': ['逻辑结构', '讲解清晰度', '词汇丰富度'],
            '题目驱动型': ['提问频率', '互动水平', '讲解清晰度'],
            '情感表达型': ['情感投入度', '语速', '手势频率'],
            '耐心细致型': ['语速', '情感投入度', '互动水平']
        }
        
        for style, relevant_features in style_feature_mapping.items():
            style_contributions = []
            for feature_name, feature_value in key_features:
                if feature_name in relevant_features:
                    # 计算贡献度（简化版本）
                    contribution_score = min(1.0, feature_value * 1.5)
                    style_contributions.append({
                        'feature_name': feature_name,
                        'value': float(feature_value),
                        'contribution_score': float(contribution_score)
                    })
            
            # 按贡献度排序
            style_contributions.sort(key=lambda x: x['contribution_score'], reverse=True)
            contributions[style] = style_contributions[:3]  # 取前3个贡献最大的特征
        
        return contributions
    
    def _calculate_confidence(self, final_output: Dict) -> float:
        """
        计算分类结果的置信度

        Args:
            final_output: 最终分类输出（可以是dict）

        Returns:
            置信度分数（0-1）
        """
        # 如果是字典，提取值
        if isinstance(final_output, dict):
            scores = list(final_output.values())
        else:
            scores = final_output

        max_score = max(scores) if scores else 0

        # 计算最高分与次高分的差异
        if len(scores) > 1:
            sorted_scores = sorted(scores, reverse=True)
            score_diff = sorted_scores[0] - sorted_scores[1]
            # 差异越大，置信度越高
            confidence = max_score * 0.7 + score_diff * 0.3
        else:
            confidence = max_score

        return min(1.0, confidence)
    
    def classify_and_save(self, video_id: str) -> str:
        """
        分类并保存结果
        
        Args:
            video_id: 视频ID
            
        Returns:
            结果文件保存路径
        """
        try:
            # 构建特征文件路径
            features_path = FEATURES_DIR / f"{video_id}_features.json"
            
            # 执行分类
            result = self.classify_style(str(features_path))
            
            # 保存结果
            result_file = RESULTS_DIR / f"{video_id}_style_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分类结果已保存到: {result_file}")
            return str(result_file)
            
        except Exception as e:
            logger.error(f"分类过程失败: {e}")
            raise


# 创建风格分类器实例
style_classifier = StyleClassifier()
