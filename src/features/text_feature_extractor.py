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
            self.dialogue_act_model = None
            self.dialogue_act_tokenizer = None
            self.dialogue_act_device = None
            self._load_model()

    def _load_model(self):
        """加载BERT模型（语义表征 + 对话行为识别）"""
        try:
            from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification
            import warnings
            import torch
            
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

            # 对话行为识别模型
            dialogue_act_name = TEXT_CONFIG.get('dialogue_act_model_name', TEXT_CONFIG['bert_model_name'])
            self.dialogue_act_tokenizer = BertTokenizer.from_pretrained(
                dialogue_act_name,
                force_download=False
            )
            self.dialogue_act_model = AutoModelForSequenceClassification.from_pretrained(
                dialogue_act_name,
                force_download=False,
                num_labels=len(TEXT_CONFIG.get('dialogue_act_labels', [])) or 4
            )
            self.dialogue_act_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dialogue_act_model = self.dialogue_act_model.to(self.dialogue_act_device).eval()
            
            logger.info(f"BERT模型加载成功，名称: {TEXT_CONFIG['bert_model_name']}")
            logger.debug(f"分词器类型: {type(self.tokenizer)}")
            logger.debug(f"模型类型: {type(self.bert_model)}")
            logger.info(f"对话行为模型加载成功，名称: {dialogue_act_name}")
            
        except Exception as e:
            logger.error(f"BERT模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.bert_model = None
            self.tokenizer = None
            self.dialogue_act_model = None
            self.dialogue_act_tokenizer = None
            self.dialogue_act_device = None
    
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
            "dialogue_act": "unknown",
            "dialogue_act_scores": {},
            "dialogue_act_sequence": [],
            "error": None,
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

            # 对话行为识别
            dialogue_act, act_scores, act_sequence = self._recognize_dialogue_act(text)
            features["dialogue_act"] = dialogue_act
            features["dialogue_act_scores"] = act_scores
            features["dialogue_act_sequence"] = act_sequence
            if dialogue_act == "unknown" and features.get("error") is None:
                features["error"] = "对话行为识别失败或模型未加载"
            
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

    def _split_sentences(self, text: str) -> List[str]:
        """按标点切分句子"""
        if not text:
            return []
        sentences = [s.strip() for s in re.split(r'[。！？!?]+', text) if s.strip()]
        return sentences if sentences else [text.strip()]

    def _recognize_dialogue_act(self, text: str) -> Tuple[str, Dict, List[Dict]]:
        """基于BERT的对话行为识别（提问/指令/讲解/反馈）"""
        labels = TEXT_CONFIG.get('dialogue_act_labels', ['question', 'instruction', 'explanation', 'feedback'])
        label_map = TEXT_CONFIG.get('dialogue_act_label_map', {})
        default_label = 'unknown'

        if not text:
            return label_map.get(default_label, default_label), {}, []

        # 使用模型预测
        if not self.dialogue_act_model or not self.dialogue_act_tokenizer:
            logger.error("对话行为模型未加载，无法进行BERT识别")
            return default_label, {}, []

        try:
            import torch
            import torch.nn.functional as F
            sentences = self._split_sentences(text)
            scores_list = []
            sequence = []

            for sentence in sentences:
                inputs = self.dialogue_act_tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=TEXT_CONFIG.get('max_length', 256)
                )
                inputs = {k: v.to(self.dialogue_act_device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.dialogue_act_model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
                scores_list.append(probs)

                best_idx = int(np.argmax(probs))
                best_label = labels[best_idx] if best_idx < len(labels) else default_label
                sequence.append({
                    "text": sentence,
                    "label": label_map.get(best_label, best_label),
                    "score": float(probs[best_idx])
                })

            mean_scores = np.mean(scores_list, axis=0) if scores_list else np.zeros(len(labels))
            act_scores = {label: float(mean_scores[i]) for i, label in enumerate(labels)}
            best_idx = int(np.argmax(mean_scores)) if mean_scores.size > 0 else 0
            best_label = labels[best_idx] if best_idx < len(labels) else default_label
            return label_map.get(best_label, best_label), act_scores, sequence

        except Exception as e:
            logger.error(f"对话行为识别失败: {e}")
            return default_label, {}, []
