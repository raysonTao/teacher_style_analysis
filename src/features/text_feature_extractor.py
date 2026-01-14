"""文本特征提取模块，负责提取文本中的特征"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import re
import jieba
import jieba.analyse

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import TEXT_CONFIG, logger

class TextFeatureExtractor:
    """文本特征提取类，用于提取文本中的特征"""

    # 单例模式实现
    _instance = None

    def __new__(cls):
        """控制实例创建，确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super(TextFeatureExtractor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化文本特征提取器"""
        # 确保模型只加载一次
        if not hasattr(self, 'bert_model'):
            self.bert_model = None
            self.tokenizer = None
            self._load_model()

    def _load_model(self):
        """加载BERT模型"""
        try:
            from transformers import BertTokenizer, BertModel
            import warnings
            
            logger.info("初始化BERT模型...")
            
            # 过滤huggingface_hub的resume_download弃用警告
            warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
            
            # 加载预训练的BERT模型和分词器
            # force_download=False：避免使用已弃用的resume_download参数
            # ignore_mismatched_sizes=True：忽略未使用的CLS层权重警告，因为我们只使用编码器部分
            self.tokenizer = BertTokenizer.from_pretrained(
                TEXT_CONFIG['bert_model_name'],
                force_download=False
            )
            self.bert_model = BertModel.from_pretrained(
                TEXT_CONFIG['bert_model_name'],
                force_download=False,
                ignore_mismatched_sizes=True
            )
            
            logger.info(f"BERT模型加载成功，名称: {TEXT_CONFIG['bert_model_name']}")
            logger.debug(f"分词器类型: {type(self.tokenizer)}")
            logger.debug(f"模型类型: {type(self.bert_model)}")
            
        except Exception as e:
            logger.error(f"BERT模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.bert_model = None
            self.tokenizer = None
    
    def extract_features(self, text: str) -> Dict:
        """
        提取文本特征
        
        Args:
            text: 输入文本
            
        Returns:
            文本特征字典
        """
        features = {
            "text": text,
            "embedding": None,
            "semantic_features": None,
            "sentiment": {
                "score": 0.0,
                "label": "neutral"
            },
            "sentiment_score": 0.0,
            "vocabulary_richness": 0.0,
            "sentence_complexity": 0.0,
            "question_frequency": 0.0,
            "keyword_density": {},
            "logical_indicators": {},
            "keywords": [],
            "teaching_terms": [],
            "sentence_count": 0,
            "word_count": 0
        }
        
        if not text:
            return features
        
        try:
            # 基础分句与分词
            sentences = [s.strip() for s in re.split(r'[。！？!?]+', text) if s.strip()]
            tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]

            features["sentence_count"] = len(sentences)
            features["word_count"] = len(tokens)

            # 词汇丰富度与句子复杂度
            if tokens:
                unique_tokens = len(set(tokens))
                features["vocabulary_richness"] = unique_tokens / max(len(tokens), 1)

            avg_words_per_sentence = (
                len(tokens) / max(len(sentences), 1) if sentences else 0.0
            )
            features["sentence_complexity"] = min(avg_words_per_sentence / 30.0, 1.0)

            # 提问频率（按句子归一化）
            question_count = text.count('？') + text.count('?')
            features["question_frequency"] = (
                question_count / max(len(sentences), 1) if sentences else 0.0
            )
            
            # 提取BERT嵌入
            if self.bert_model and self.tokenizer:
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    
                    # 获取CLS嵌入
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    features["embedding"] = embedding.tolist()
                    features["semantic_features"] = embedding.squeeze(0)
                    
                except Exception as e:
                        logger.error(f"BERT嵌入提取失败: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 情绪分析：基于情感词覆盖率计算连续分数
            positive_words = ["好", "棒", "优秀", "正确", "精彩", "清晰", "有效", "积极", "鼓励", "喜欢"]
            negative_words = ["不好", "错误", "糟糕", "差", "问题", "混乱", "消极", "不清晰", "失误"]

            positive_count = sum(1 for word in tokens if word in positive_words)
            negative_count = sum(1 for word in tokens if word in negative_words)
            total_sentiment = positive_count + negative_count

            if total_sentiment > 0:
                sentiment_score = 0.5 + 0.5 * (positive_count - negative_count) / total_sentiment
            else:
                sentiment_score = 0.5

            sentiment_score = max(0.0, min(1.0, float(sentiment_score)))
            if sentiment_score > 0.55:
                sentiment_label = "positive"
            elif sentiment_score < 0.45:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            features["sentiment"] = {
                "score": sentiment_score,
                "label": sentiment_label
            }
            features["sentiment_score"] = sentiment_score

            # 关键词提取
            keywords = jieba.analyse.extract_tags(text, topK=8) if text.strip() else []
            if not keywords:
                stopwords = {"的", "了", "是", "在", "和", "与", "及", "或", "也", "都", "而", "我们", "你们", "他们"}
                filtered = [t for t in tokens if t not in stopwords and len(t) > 1]
                counts = {}
                for token in filtered:
                    counts[token] = counts.get(token, 0) + 1
                keywords = [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]]
            features["keywords"] = keywords

            # 教学术语与关键词密度
            teaching_terms = {
                'definition': ["定义", "概念", "本质", "性质"],
                'example': ["例如", "比如", "举例", "实例"],
                'explanation': ["解释", "说明", "分析", "推导"],
                'summary': ["总结", "归纳", "结论", "概括"],
                'question': ["为什么", "如何", "是否", "能否"],
                'problem': ["题", "练习", "作业", "求解"]
            }

            term_hits = []
            keyword_density = {}
            for category, terms in teaching_terms.items():
                count = sum(1 for t in tokens if t in terms)
                keyword_density[category] = count / max(len(tokens), 1)
                term_hits.extend([t for t in tokens if t in terms])

            features["keyword_density"] = keyword_density
            features["teaching_terms"] = sorted(set(term_hits))

            # 逻辑连接词统计
            logical_markers = {
                'causality': ["因为", "所以", "因此", "由于", "从而"],
                'comparison': ["相比", "类似", "不同", "相同", "对比"],
                'sequence': ["首先", "其次", "然后", "最后", "接着"],
                'emphasis': ["重点", "关键", "特别", "必须", "注意"]
            }

            logical_indicators = {}
            for category, markers in logical_markers.items():
                count = sum(1 for t in tokens if t in markers)
                logical_indicators[category] = count / max(len(tokens), 1)
            features["logical_indicators"] = logical_indicators

        except Exception as e:
            logger.error(f"文本特征提取失败: {e}")
            import traceback
            traceback.print_exc()

        if features["semantic_features"] is None:
            features["semantic_features"] = np.array([])

        return features
    
    def process_transcription(self, transcription: str) -> Dict:
        """
        处理语音识别转录结果
        
        Args:
            transcription: 语音识别转录结果
            
        Returns:
            处理后的文本特征
        """
        return self.extract_features(transcription)
