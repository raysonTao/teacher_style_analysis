"""文本特征提取模块，负责提取文本中的特征"""
import os
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import TEXT_CONFIG

class TextFeatureExtractor:
    """文本特征提取类，用于提取文本中的特征"""
    
    def __init__(self):
        """初始化文本特征提取器"""
        self.bert_model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载BERT模型"""
        try:
            from transformers import BertTokenizer, BertModel
            print("初始化BERT模型...")
            
            # 加载预训练的BERT模型和分词器
            self.tokenizer = BertTokenizer.from_pretrained(TEXT_CONFIG['bert_model_name'])
            self.bert_model = BertModel.from_pretrained(TEXT_CONFIG['bert_model_name'])
            
            print(f"BERT模型加载成功，名称: {TEXT_CONFIG['bert_model_name']}")
            print(f"分词器类型: {type(self.tokenizer)}")
            print(f"模型类型: {type(self.bert_model)}")
            
        except Exception as e:
            print(f"BERT模型加载失败: {e}")
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
            "sentiment": {
                "score": 0.0,
                "label": "neutral"
            },
            "keywords": [],
            "sentence_count": 0,
            "word_count": 0
        }
        
        if not text:
            return features
        
        try:
            # 计算句子数和单词数
            features["sentence_count"] = len(text.split('。'))
            features["word_count"] = len(text)
            
            # 提取BERT嵌入
            if self.bert_model and self.tokenizer:
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    
                    # 获取CLS嵌入
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    features["embedding"] = embedding.tolist()
                    
                except Exception as e:
                    print(f"BERT嵌入提取失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 模拟情绪分析（实际项目中可以使用专业的情绪分析模型）
            # 简单的基于关键词的情绪分析
            positive_words = ["好", "棒", "优秀", "正确", "精彩"]
            negative_words = ["不好", "错误", "糟糕", "差", "问题"]
            
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            if positive_count > negative_count:
                features["sentiment"] = {
                    "score": 0.7,
                    "label": "positive"
                }
            elif negative_count > positive_count:
                features["sentiment"] = {
                    "score": 0.3,
                    "label": "negative"
                }
            
            # 简单的关键词提取（实际项目中可以使用专业的关键词提取模型）
            features["keywords"] = positive_words[:3]  # 示例关键词
            
        except Exception as e:
            print(f"文本特征提取失败: {e}")
            import traceback
            traceback.print_exc()
        
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
