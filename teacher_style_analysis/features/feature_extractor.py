"""特征提取模块，负责从视频、音频和文本中提取多模态特征"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    MODEL_CONFIG, SYSTEM_CONFIG, 
    VIDEO_DIR, AUDIO_DIR, TEXT_DIR, FEATURES_DIR
)


class FeatureExtractor:
    """多模态特征提取器"""
    
    def __init__(self):
        self._init_models()
    
    def _init_models(self):
        """初始化各种特征提取模型"""
        try:
            # 这里我们使用模拟的模型接口，实际使用时需要替换为真实的模型加载代码
            print("初始化特征提取模型...")
            
            # YOLOv8动作检测模型
            # 实际使用时: from ultralytics import YOLO
            # self.yolo_model = YOLO(MODEL_CONFIG['yolo_model_path'])
            self.yolo_model = None
            
            # OpenPose姿态估计模型
            # 实际使用时需要加载OpenPose模型
            self.openpose_model = None
            
            # Whisper语音识别模型
            # 实际使用时: import whisper
            # self.whisper_model = whisper.load_model(MODEL_CONFIG['whisper_model'])
            self.whisper_model = None
            
            # BERT文本分析模型
            # 实际使用时: from transformers import BertModel, BertTokenizer
            # self.bert_tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['bert_model'])
            # self.bert_model = BertModel.from_pretrained(MODEL_CONFIG['bert_model'])
            self.bert_model = None
            self.bert_tokenizer = None
            
            print("模型初始化完成")
            
        except Exception as e:
            print(f"模型初始化失败: {e}")
    
    def get_status(self) -> Dict:
        """
        获取特征提取器状态
        
        Returns:
            包含状态信息的字典
        """
        return {
            'yolo_model_loaded': self.yolo_model is not None,
            'pose_model_loaded': self.pose_model is not None,
            'whisper_model_loaded': self.whisper_model is not None,
            'bert_model_loaded': self.bert_model is not None,
            'status': 'ready' if all([
                self.yolo_model is not None,
                self.pose_model is not None,
                self.whisper_model is not None,
                self.bert_model is not None
            ]) else 'partially_loaded'
        }
    
    def extract_video_features(self, video_path: str) -> Dict:
        """
        提取视频特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含视频特征的字典
        """
        print(f"提取视频特征: {video_path}")
        
        # 模拟视频特征提取
        # 在实际应用中，这里需要使用YOLOv8和OpenPose等模型进行真实的特征提取
        
        features = {
            'action_sequence': [],  # 动作序列
            'pose_estimation': [],  # 姿态估计数据
            'motion_energy': [],  # 动作能量
            'spatial_distribution': {},  # 空间分布
            'behavior_frequency': {},  # 行为频率统计
        }
        
        # 模拟一些行为频率数据
        features['behavior_frequency'] = {
            'standing': 0.85,  # 站立比例
            'walking': 0.10,  # 行走比例
            'gesturing': 0.65,  # 手势比例
            'writing': 0.40,  # 书写比例
            'pointing': 0.30   # 指向比例
        }
        
        # 模拟空间分布数据
        features['spatial_distribution'] = {
            'front': 0.60,  # 讲台前区域
            'middle': 0.30,  # 教室中间区域
            'side': 0.10     # 教室两侧区域
        }
        
        return features
    
    def fuse_multimodal_features(self, video_features: Dict, audio_features: Dict, text_features: Dict) -> Dict:
        """
        融合多模态特征
        
        Args:
            video_features: 视频特征
            audio_features: 音频特征
            text_features: 文本特征
            
        Returns:
            融合后的特征字典
        """
        print("融合多模态特征...")
        
        # 计算教学风格指标
        fusion = {
            'interaction_level': 0.0,  # 互动水平
            'explanation_clarity': 0.0,  # 讲解清晰度
            'emotional_engagement': 0.0,  # 情感投入度
            'logical_structure': 0.0,  # 逻辑结构
            'teaching_style_metrics': {
                'lecturing': 0.0,  # 讲授型
                'guiding': 0.0,  # 引导型
                'interactive': 0.0,  # 互动型
                'logical': 0.0,  # 逻辑型
                'problem_driven': 0.0,  # 题目驱动型
                'emotional': 0.0,  # 情感型
                'patient': 0.0  # 耐心型
            }
        }
        
        # 计算互动水平（基于提问频率、手势频率等）
        question_freq = text_features.get('question_frequency', 0.0)
        gesture_freq = video_features.get('behavior_frequency', {}).get('gesturing', 0.0)
        fusion['interaction_level'] = min(1.0, (question_freq * 5 + gesture_freq * 2) / 2)
        
        # 计算讲解清晰度（基于语速、词汇丰富度、逻辑结构）
        speech_rate = audio_features.get('speech_rate', 120)
        vocab_richness = text_features.get('vocabulary_richness', 0.0)
        logical_indicators = sum(text_features.get('logical_indicators', {}).values())
        
        # 假设120字/分钟为最佳语速
        speech_quality = 1.0 - abs(speech_rate - 120) / 120
        fusion['explanation_clarity'] = (speech_quality * 0.4 + vocab_richness * 0.3 + logical_indicators * 0.3)
        
        # 计算情感投入度（基于情绪分数、语调变化、手势频率）
        emotion_scores = audio_features.get('emotion_scores', {})
        positive_emotion = emotion_scores.get('happy', 0.0) * 0.7 + emotion_scores.get('neutral', 0.0) * 0.3
        pitch_variation = audio_features.get('pitch_variation', 0.0)
        
        fusion['emotional_engagement'] = (positive_emotion * 0.5 + pitch_variation * 0.3 + gesture_freq * 0.2)
        
        # 计算逻辑结构（基于逻辑指示词、句子复杂度）
        sentence_complexity = text_features.get('sentence_complexity', 0.0)
        fusion['logical_structure'] = (logical_indicators * 0.6 + sentence_complexity * 0.4)
        
        # 计算各教学风格指标
        metrics = fusion['teaching_style_metrics']
        
        # 讲授型：高语速、高词汇丰富度、低互动
        metrics['lecturing'] = (speech_quality * 0.4 + vocab_richness * 0.3 + 
                               (1 - fusion['interaction_level']) * 0.3)
        
        # 引导型：高互动、中语速、高提问频率
        metrics['guiding'] = (fusion['interaction_level'] * 0.4 + speech_quality * 0.3 + 
                             question_freq * 0.3)
        
        # 互动型：高互动、高提问频率、高情感投入
        metrics['interactive'] = (fusion['interaction_level'] * 0.5 + question_freq * 0.3 + 
                               fusion['emotional_engagement'] * 0.2)
        
        # 逻辑型：高逻辑结构、中语速、高讲解清晰度
        metrics['logical'] = (fusion['logical_structure'] * 0.4 + speech_quality * 0.3 + 
                           fusion['explanation_clarity'] * 0.3)
        
        # 题目驱动型：高提问频率、高互动、中讲解清晰度
        metrics['problem_driven'] = (question_freq * 0.5 + fusion['interaction_level'] * 0.3 + 
                                    fusion['explanation_clarity'] * 0.2)
        
        # 情感型：高情感投入、高语调变化、高手势频率
        metrics['emotional'] = (fusion['emotional_engagement'] * 0.5 + pitch_variation * 0.3 + 
                               gesture_freq * 0.2)
        
        # 耐心型：低语速、高情感投入、中互动
        slow_speech_bonus = max(0, (140 - speech_rate) / 140)  # 语速越慢越耐心
        metrics['patient'] = (slow_speech_bonus * 0.4 + fusion['emotional_engagement'] * 0.4 + 
                           fusion['interaction_level'] * 0.2)
        
        # 确保所有指标都在0-1范围内
        for key in metrics:
            metrics[key] = min(1.0, max(0.0, metrics[key]))
        
        return {
            'video': video_features,
            'audio': audio_features,
            'text': text_features,
            'fusion': fusion
        }
    
    def extract_audio_features(self, audio_path: str) -> Dict:
        """
        提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            包含音频特征的字典
        """
        print(f"提取音频特征: {audio_path}")
        
        # 模拟音频特征提取
        # 实际应用中，这里需要使用Whisper模型进行语音识别和特征提取
        
        features = {
            'speech_rate': 0.0,  # 语速
            'pitch_variation': 0.0,  # 语调变化
            'emotion_scores': {},  # 情绪分数
            'volume_level': 0.0,  # 音量水平
            'silence_ratio': 0.0,  # 沉默比例
        }
        
        # 模拟一些音频特征数据
        features['speech_rate'] = 120  # 平均语速（字/分钟）
        features['pitch_variation'] = 0.45  # 语调变化程度
        features['emotion_scores'] = {
            'happy': 0.3,
            'neutral': 0.5,
            'serious': 0.2
        }
        features['volume_level'] = 0.75  # 音量水平（0-1）
        features['silence_ratio'] = 0.15  # 沉默比例
        
        return features
    
    def extract_text_features(self, transcript_path: str) -> Dict:
        """
        提取文本特征
        
        Args:
            transcript_path: 文本转录文件路径
            
        Returns:
            包含文本特征的字典
        """
        print(f"提取文本特征: {transcript_path}")
        
        # 读取转录文本
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
        except Exception as e:
            print(f"读取转录文本失败: {e}")
            # 使用模拟文本
            transcript = "这是一个模拟的教学转录文本，包含了教师的讲解内容和提问。"
        
        # 模拟文本特征提取
        # 实际应用中，这里需要使用BERT等模型进行文本分析
        
        features = {
            'vocabulary_richness': 0.0,  # 词汇丰富度
            'sentence_complexity': 0.0,  # 句子复杂度
            'question_frequency': 0.0,  # 提问频率
            'keyword_density': {},  # 关键词密度
            'logical_indicators': {},  # 逻辑指示词统计
            'semantic_features': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # 语义特征
            'sentiment_score': 0.0,  # 情感分数
            'teaching_terms': [],  # 教学术语
        }
        
        # 模拟一些文本特征数据
        features['vocabulary_richness'] = 0.65  # 词汇丰富度
        features['sentence_complexity'] = 0.50  # 句子复杂度
        features['question_frequency'] = 0.12  # 提问频率（每100字）
        features['keyword_density'] = {
            '概念': 0.05,
            '原理': 0.04,
            '例子': 0.03
        }
        features['logical_indicators'] = {
            '因为': 0.02,
            '所以': 0.03,
            '因此': 0.01,
            '首先': 0.015,
            '其次': 0.01,
            '最后': 0.005
        }
        features['semantic_features'] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 语义特征向量
        features['sentiment_score'] = 0.75  # 情感分数
        features['teaching_terms'] = ['教学', '讲解', '提问', '分析', '总结']  # 教学术语
        
        return features
    
    def fuse_features(self, video_features: Dict, audio_features: Dict, 
                     text_features: Dict) -> Dict:
        """
        融合多模态特征
        
        Args:
            video_features: 视频特征
            audio_features: 音频特征
            text_features: 文本特征
            
        Returns:
            融合后的特征字典
        """
        print("融合多模态特征")
        
        # 创建融合特征字典
        fused_features = {
            'video': video_features,
            'audio': audio_features,
            'text': text_features,
            'fusion': {
                'teaching_style_metrics': {},  # 教学风格指标
                'interaction_level': 0.0,  # 互动水平
                'explanation_clarity': 0.0,  # 讲解清晰度
                'emotional_engagement': 0.0,  # 情感投入度
                'logical_structure': 0.0,  # 逻辑结构
            }
        }
        
        # 计算一些融合指标（模拟）
        # 互动水平 = 文本提问频率 + 视频手势频率 + 音频情绪变化
        interaction_level = (
            text_features.get('question_frequency', 0) * 2.0 +
            video_features.get('behavior_frequency', {}).get('gesturing', 0) * 1.5 +
            audio_features.get('pitch_variation', 0) * 1.0
        ) / 4.5
        
        # 讲解清晰度 = 文本逻辑指示词密度 + 语速适中度
        logical_indicators = sum(text_features.get('logical_indicators', {}).values())
        speech_rate_factor = 1.0 - abs(audio_features.get('speech_rate', 0) - 120) / 120  # 语速适中度
        explanation_clarity = (logical_indicators * 10 + speech_rate_factor) / 2
        
        # 情感投入度 = 音频情绪积极度 + 语调变化
        positive_emotion = sum(audio_features.get('emotion_scores', {}).values()) * 0.5
        emotional_engagement = (positive_emotion + audio_features.get('pitch_variation', 0)) / 2
        
        # 逻辑结构 = 文本逻辑指示词 + 词汇丰富度
        logical_structure = (logical_indicators * 10 + text_features.get('vocabulary_richness', 0)) / 2
        
        # 更新融合特征
        fused_features['fusion']['interaction_level'] = float(interaction_level)
        fused_features['fusion']['explanation_clarity'] = float(explanation_clarity)
        fused_features['fusion']['emotional_engagement'] = float(emotional_engagement)
        fused_features['fusion']['logical_structure'] = float(logical_structure)
        
        # 计算教学风格指标
        fused_features['fusion']['teaching_style_metrics'] = {
            'lecturing': 0.0,
            'guiding': 0.0,
            'interactive': 0.0,
            'logical': 0.0,
            'problem_driven': 0.0,
            'emotional': 0.0,
            'patient': 0.0
        }
        
        # 填充风格指标（模拟计算）
        metrics = fused_features['fusion']['teaching_style_metrics']
        metrics['lecturing'] = 0.6 - interaction_level * 0.5  # 讲授型与互动负相关
        metrics['guiding'] = interaction_level * 0.6 + explanation_clarity * 0.4
        metrics['interactive'] = interaction_level
        metrics['logical'] = logical_structure * 0.8 + explanation_clarity * 0.2
        metrics['problem_driven'] = text_features.get('question_frequency', 0) * 3.0
        metrics['emotional'] = emotional_engagement
        metrics['patient'] = audio_features.get('silence_ratio', 0) * 2.0 + (1.0 - audio_features.get('speech_rate', 0) / 150)
        
        # 确保所有值在0-1范围内
        for key in metrics:
            metrics[key] = max(0, min(1, metrics[key]))
        
        return fused_features
    
    def process_video(self, video_path: str) -> Dict:
        """
        处理视频文件，提取所有模态的特征并融合
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含所有特征的字典
        """
        print(f"处理视频文件: {video_path}")
        
        # 提取各模态特征
        video_features = self.extract_video_features(video_path)
        audio_features = self.extract_audio_features(video_path)
        
        # 生成对应的转录文件路径
        transcript_path = video_path.replace('.mp4', '_transcript.txt').replace('.avi', '_transcript.txt')
        text_features = self.extract_text_features(transcript_path)
        
        # 融合特征
        fused_features = self.fuse_multimodal_features(video_features, audio_features, text_features)
        
        # 返回完整的特征结果
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'text_features': text_features,
            'fused_features': fused_features
        }
    
    def extract_and_save_features(self, video_id: str) -> str:
        """
        提取并保存所有特征
        
        Args:
            video_id: 视频ID
            
        Returns:
            特征文件保存路径
        """
        try:
            # 构建文件路径
            video_path = str(VIDEO_DIR / f"{video_id}_*.mp4")  # 简化处理
            audio_path = str(AUDIO_DIR / f"{video_id}.wav")
            transcript_path = str(TEXT_DIR / f"{video_id}_transcript.txt")
            
            # 提取各模态特征
            video_features = self.extract_video_features(video_path)
            audio_features = self.extract_audio_features(audio_path)
            text_features = self.extract_text_features(transcript_path)
            
            # 融合特征
            fused_features = self.fuse_features(video_features, audio_features, text_features)
            
            # 保存特征到文件
            features_file = FEATURES_DIR / f"{video_id}_features.json"
            with open(features_file, 'w', encoding='utf-8') as f:
                json.dump(fused_features, f, ensure_ascii=False, indent=2)
            
            print(f"特征已保存到: {features_file}")
            return str(features_file)
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            raise


# 创建特征提取器实例
feature_extractor = FeatureExtractor()