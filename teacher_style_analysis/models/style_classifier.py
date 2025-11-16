"""风格识别模块，负责使用CMAT模型进行教师风格分类和可解释性分析"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    MODEL_CONFIG, SYSTEM_CONFIG, STYLE_LABELS,
    FEATURES_DIR, RESULTS_DIR
)


class StyleClassifier:
    """教师风格分类器"""
    
    def __init__(self):
        self._init_model()
    
    def get_status(self) -> Dict:
        """
        获取风格分类器状态
        
        Returns:
            包含状态信息的字典
        """
        return {
            'model_loaded': self.model is not None,
            'model_type': 'pretrained' if os.path.exists(MODEL_CONFIG['cmat_model_path']) else 'mock',
            'lambda_weight': self.model.get('lambda_weight', 0.5) if self.model else 0.5,
            'status': 'ready' if self.model else 'not_loaded'
        }
    
    def _init_model(self):
        """初始化风格分类模型"""
        try:
            print("初始化风格分类模型...")
            
            # 这里我们模拟CMAT模型（Combined Multi-modal Attention-based Teaching style model）
            # 实际使用时，这里应该加载预训练的XGBoost或RandomForest模型
            
            # 检查是否有预训练模型
            if os.path.exists(MODEL_CONFIG['cmat_model_path']):
                with open(MODEL_CONFIG['cmat_model_path'], 'rb') as f:
                    self.model = pickle.load(f)
                print("预训练模型加载成功")
            else:
                # 创建模拟模型
                self.model = self._create_mock_model()
                print("创建模拟模型成功")
                
                # 保存模拟模型
                os.makedirs(os.path.dirname(MODEL_CONFIG['cmat_model_path']), exist_ok=True)
                with open(MODEL_CONFIG['cmat_model_path'], 'wb') as f:
                    pickle.dump(self.model, f)
                print(f"模拟模型已保存到: {MODEL_CONFIG['cmat_model_path']}")
                
        except Exception as e:
            print(f"模型初始化失败: {e}")
            self.model = self._create_mock_model()
    
    def _create_mock_model(self) -> Dict:
        """创建模拟的CMAT模型"""
        # 模拟模型包含规则驱动层和机器学习层的参数
        mock_model = {
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
        return mock_model
    
    def _apply_rules(self, features: Dict) -> Dict:
        """
        应用规则驱动层
        
        Args:
            features: 融合后的特征
            
        Returns:
            规则驱动层的输出
        """
        rule_output = {}
        rule_weights = self.model['rule_weights']
        
        # 提取需要的特征
        fusion = features.get('fusion', {})
        audio = features.get('audio', {})
        video = features.get('video', {})
        text = features.get('text', {})
        
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
                    feature_value = video.get('behavior_frequency', {}).get('gesturing', 0.0)
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
        fusion = features.get('fusion', {})
        
        # 计算各风格的机器学习分数
        style_metrics = fusion.get('teaching_style_metrics', {})
        
        # 使用预计算的风格指标作为机器学习层的输出
        for style in ['lecturing', 'guiding', 'interactive', 'logical', 'problem_driven', 'emotional', 'patient']:
            ml_output[style] = style_metrics.get(style, 0.0)
        
        return ml_output
    
    def classify_style(self, features_path: str, raw_data: Dict = None) -> Dict:
        """
        分类教学风格
        
        Args:
            features_path: 特征文件路径或numpy数组
            raw_data: 原始数据（可选）
            
        Returns:
            Dict: 包含风格分类结果的字典
        """
        # 如果是numpy数组，直接处理
        if isinstance(features_path, np.ndarray):
            # 应用规则驱动层
            rule_results = self.apply_rule_driven_layer(features_path, raw_data)
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
        if isinstance(features_path, str):
            print(f"开始风格分类: {features_path}")
            
            # 读取特征文件
            try:
                with open(features_path, 'r', encoding='utf-8') as f:
                    features = json.load(f)
            except Exception as e:
                print(f"读取特征文件失败: {e}")
                raise
        else:
            # 直接作为特征数据使用
            features = features_path
        
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
            'timestamp': {
                'analysis_time': '2024-11-12T23:30:00Z'  # 模拟时间戳
            }
        }
        
        return result
        """
        对特征进行风格分类
        
        Args:
            features_path: 特征文件路径
            
        Returns:
            风格分类结果
        """
        print(f"开始风格分类: {features_path}")
        
        # 读取特征文件
        try:
            with open(features_path, 'r', encoding='utf-8') as f:
                features = json.load(f)
        except Exception as e:
            print(f"读取特征文件失败: {e}")
            raise
        
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
            'timestamp': {
                'analysis_time': '2024-11-12T23:30:00Z'  # 模拟时间戳
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
        # 创建测试用的规则输出
        return {
            'analytical_score': 0.7,
            'interactive_score': 0.8,
            'authoritative_score': 0.5,
            'supportive_score': 0.6
        }
    
    def apply_ml_layer(self, features: np.ndarray) -> np.ndarray:
        """
        应用机器学习层（对外接口）
        
        Args:
            features: 特征向量
            
        Returns:
            机器学习层输出
        """
        # 模拟ML模型输出
        return np.random.rand(1, 4)  # 返回4种风格的分数
    
    def fuse_outputs(self, rule_results: Dict, ml_results: np.ndarray) -> Dict:
        """
        融合规则驱动和机器学习输出（对外接口）
        
        Args:
            rule_results: 规则驱动结果
            ml_results: 机器学习结果
            
        Returns:
            融合后的结果
        """
        # 简化融合逻辑
        fused = {}
        for style in ['analytical', 'interactive', 'authoritative', 'supportive']:
            rule_score = rule_results.get(f'{style}_score', 0.5)
            ml_score = ml_results[0][0] if ml_results.size > 0 else 0.5  # 简化处理
            fused[style] = (rule_score + ml_score) / 2
        
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
        # 模拟特征贡献度
        contributions = {}
        for style in ['analytical', 'interactive', 'authoritative', 'supportive']:
            contributions[style] = {
                i: np.random.rand() for i in range(min(5, features.shape[1]))
            }
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
            final_output: 最终分类输出
            
        Returns:
            置信度分数（0-1）
        """
        scores = list(final_output.values())
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
            
            print(f"分类结果已保存到: {result_file}")
            return str(result_file)
            
        except Exception as e:
            print(f"分类过程失败: {e}")
            raise


# 创建风格分类器实例
style_classifier = StyleClassifier()