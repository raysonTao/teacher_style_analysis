"""
特征编码器 - 知识蒸馏的核心桥接层
将复杂的多模态特征结构转换为模型可用的固定维度向量

作用：
- 视频特征 → 20维向量
- 音频特征 → 15维向量
- 文本特征 → 25维向量

用于：
1. VLM标注时提供量化特征解读
2. 训练数据转换时生成模型输入
"""

import numpy as np
from typing import Dict, List, Any
import logging
import os
import sys

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import VIDEO_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEncoder:
    """多模态特征编码器"""

    def __init__(self):
        """初始化编码器"""
        # 动作类型映射
        self.action_types = VIDEO_CONFIG.get(
            'stgcn_action_labels',
            ['standing', 'walking', 'gesturing', 'writing', 'pointing']
        )

        # 情感标签映射
        self.sentiment_labels = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear']

    def encode_video_features(self, raw_features: Dict) -> np.ndarray:
        """
        编码视频特征 → 20维向量

        维度分配：
        1-6维: ST-GCN动作分布
        7维: 平均运动能量
        8-16维: 空间分布 (9个区域)
        17维: 教师轨迹连续性
        18维: 视频时长
        19维: 总帧数
        20维: 姿态平均置信度

        Args:
            raw_features: 原始视频特征字典

        Returns:
            20维numpy数组
        """
        features = np.zeros(20)

        try:
            # 1-6维: 动作分布
            action_distribution = raw_features.get('action_distribution') or raw_features.get('action_frequency', {})
            for i, action in enumerate(self.action_types[:6]):
                features[i] = action_distribution.get(action, 0.0)

            # 7维: 平均运动能量（归一化）
            avg_motion = raw_features.get('avg_motion_energy', 0.0)
            features[6] = min(avg_motion * 10, 1.0)  # 假设正常范围0-0.1，映射到0-1

            # 8-16维: 空间分布（9个区域）
            spatial_dist = raw_features.get('spatial_distribution', {})
            regions = ['top_left', 'top_center', 'top_right',
                      'middle_left', 'middle_center', 'middle_right',
                      'bottom_left', 'bottom_center', 'bottom_right']
            for i, region in enumerate(regions):
                features[7 + i] = spatial_dist.get(region, 0.0)

            # 17维: 教师轨迹连续性
            track_continuity = raw_features.get('track_continuity', 0.0)
            features[16] = min(max(track_continuity, 0.0), 1.0)

            # 18维: 视频时长（归一化，假设最长60分钟）
            duration = raw_features.get('video_duration', 0.0)
            features[17] = min(duration / 3600, 1.0)

            # 19维: 总帧数（归一化，假设最多180000帧=1小时@30fps）
            total_frames = raw_features.get('total_frames', 0)
            features[18] = min(total_frames / 180000, 1.0)

            # 20维: 姿态平均置信度
            pose_confidences = raw_features.get('pose_confidences', [])
            if pose_confidences:
                features[19] = np.mean(pose_confidences)
            else:
                features[19] = 0.0

        except Exception as e:
            logger.error(f"编码视频特征失败: {e}")
            # 返回零向量作为后备

        return features

    def encode_audio_features(self, raw_features: Dict) -> np.ndarray:
        """
        编码音频特征 → 15维向量

        维度分配：
        1-6维: 情感分布
        7维: 语速
        8维: 语音活动比例
        9维: 静音比例
        10维: 音量水平
        11维: 音高变化
        12维: 情感极性分数
        13-15维: Wav2Vec2嵌入压缩

        Args:
            raw_features: 原始音频特征字典

        Returns:
            15维numpy数组
        """
        features = np.zeros(15)

        try:
            # 1-6维: 情感分布
            emotion_scores = raw_features.get('emotion_scores', {})
            for i, label in enumerate(self.emotion_labels):
                if i >= 6:
                    break
                features[i] = emotion_scores.get(label, 0.0)

            # 7维: 语速（归一化）
            speech_rate = raw_features.get('speech_rate', 0.0)
            features[6] = min(speech_rate / 200.0, 1.0)

            # 8维: 语音活动比例
            features[7] = raw_features.get('voice_activity_ratio', 0.0)

            # 9维: 静音比例
            features[8] = raw_features.get('silence_ratio', 0.0)

            # 10维: 音量水平
            features[9] = raw_features.get('volume_level', 0.0)

            # 11维: 音高变化
            features[10] = raw_features.get('pitch_variation', 0.0)

            # 12维: 情感极性分数
            features[11] = raw_features.get('sentiment_score', 0.0)

            # 13-15维: Wav2Vec2嵌入压缩
            embedding = raw_features.get('wav2vec2_embedding', [])
            if embedding:
                segment = max(len(embedding) // 3, 1)
                for i in range(3):
                    start = i * segment
                    end = len(embedding) if i == 2 else min((i + 1) * segment, len(embedding))
                    features[12 + i] = float(np.mean(embedding[start:end]))

        except Exception as e:
            logger.error(f"编码音频特征失败: {e}")

        return features

    def encode_text_features(self, raw_features: Dict) -> np.ndarray:
        """
        编码文本特征 → 25维向量

        维度分配：
        1-4维: 对话行为分布
        5-14维: BERT嵌入降维
        15维: 情感分数
        16维: 词汇丰富度
        17维: 句子复杂度
        18维: 提问频率
        19-22维: 教学关键词密度
        23-25维: 逻辑连接词指标

        Args:
            raw_features: 原始文本特征字典

        Returns:
            25维numpy数组
        """
        features = np.zeros(25)

        try:
            # 1-4维: 对话行为分布
            act_scores = raw_features.get('dialogue_act_scores', {})
            labels = ['question', 'instruction', 'explanation', 'feedback']
            for i, label in enumerate(labels):
                features[i] = act_scores.get(label, 0.0)

            # 5-14维: BERT嵌入降维
            embedding = raw_features.get('embedding', [])
            if embedding:
                emb_array = np.array(embedding).flatten()
                seg_size = max(len(emb_array) // 10, 1)
                for i in range(10):
                    start = i * seg_size
                    end = len(emb_array) if i == 9 else min((i + 1) * seg_size, len(emb_array))
                    features[4 + i] = float(np.mean(emb_array[start:end]))

            # 15维: 情感分数
            features[14] = raw_features.get('sentiment_score', 0.0)

            # 16维: 词汇丰富度
            features[15] = raw_features.get('vocabulary_richness', 0.0)

            # 17维: 句子复杂度
            features[16] = raw_features.get('sentence_complexity', 0.0)

            # 18维: 提问频率
            features[17] = raw_features.get('question_frequency', 0.0)

            # 19-22维: 教学关键词密度
            keyword_density = raw_features.get('keyword_density', {})
            keys = ['definition', 'example', 'explanation', 'summary']
            for i, key in enumerate(keys):
                features[18 + i] = keyword_density.get(key, 0.0)

            # 23-25维: 逻辑连接词指标
            logical = raw_features.get('logical_indicators', {})
            logical_keys = ['causality', 'sequence', 'emphasis']
            for i, key in enumerate(logical_keys):
                features[22 + i] = logical.get(key, 0.0)

        except Exception as e:
            logger.error(f"编码文本特征失败: {e}")

        return features

    def encode_all(self, raw_features: Dict) -> Dict[str, np.ndarray]:
        """
        编码所有模态的特征

        Args:
            raw_features: 包含video_features, audio_features, text_features的字典

        Returns:
            编码后的特征字典
        """
        encoded = {}

        # 编码视频特征
        video_features = raw_features.get('video_features', {})
        encoded['video'] = self.encode_video_features(video_features)

        # 编码音频特征
        audio_features = raw_features.get('audio_features', {})
        encoded['audio'] = self.encode_audio_features(audio_features)

        # 编码文本特征
        text_features = raw_features.get('text_features', {})
        encoded['text'] = self.encode_text_features(text_features)

        # 验证维度
        assert encoded['video'].shape == (20,), f"视频特征维度错误: {encoded['video'].shape}"
        assert encoded['audio'].shape == (15,), f"音频特征维度错误: {encoded['audio'].shape}"
        assert encoded['text'].shape == (25,), f"文本特征维度错误: {encoded['text'].shape}"

        logger.debug(f"特征编码完成: video(20), audio(15), text(25)")

        return encoded

    def get_feature_interpretation(self, encoded_features: Dict[str, np.ndarray]) -> str:
        """
        生成人类可读的特征解读（用于VLM prompt）

        Args:
            encoded_features: 编码后的特征字典

        Returns:
            特征解读文本
        """
        video = encoded_features['video']
        audio = encoded_features['audio']
        text = encoded_features['text']

        interpretation = []
        interpretation.append("### 量化特征分析：\n")

        # 视频特征解读
        interpretation.append("**视频行为特征：**")
        action_desc = []
        for i, action in enumerate(self.action_types):
            freq = video[i]
            if freq > 0.1:  # 只显示频率>10%的动作
                level = "高" if freq > 0.3 else "中" if freq > 0.15 else "低"
                action_desc.append(f"{action}({level}: {freq:.1%})")
        interpretation.append("- 动作分布：" + ", ".join(action_desc) if action_desc else "- 动作分布：主要为standing")

        motion = video[6]
        motion_level = "高" if motion > 0.6 else "中" if motion > 0.3 else "低"
        interpretation.append(f"- 运动能量：{motion_level} ({motion:.2f}) → 教师{'活跃度高' if motion > 0.6 else '活跃度中等' if motion > 0.3 else '相对静态'}")

        # 音频特征解读
        interpretation.append("\n**音频特征：**")
        volume_level = audio[9]
        volume_desc = "响亮" if volume_level > 0.6 else "适中" if volume_level > 0.4 else "偏小"
        interpretation.append(f"- 音量：{volume_desc} (水平={volume_level:.2f})")

        pitch_variation = audio[10]
        pitch_desc = "音调变化丰富" if pitch_variation > 0.3 else "音调相对平稳"
        interpretation.append(f"- 音调：{pitch_desc} (变化={pitch_variation:.2f})")

        voice_ratio = audio[7]
        interpretation.append(f"- 语音活动比例：{voice_ratio:.1%} → {'讲话为主' if voice_ratio > 0.7 else '互动较多' if voice_ratio > 0.5 else '停顿较多'}")

        # 文本特征解读
        interpretation.append("\n**文本特征：**")
        vocab_richness = text[15]
        sentence_complexity = text[16]
        interpretation.append(f"- 词汇丰富度：{vocab_richness:.2f}, 句子复杂度：{sentence_complexity:.2f}")

        sentiment_score = text[14]
        sentiment_desc = "积极正向" if sentiment_score > 0.6 else "中性客观" if sentiment_score > 0.4 else "消极或批评性"
        interpretation.append(f"- 情感倾向：{sentiment_desc} (分数={sentiment_score:.2f})")

        return "\n".join(interpretation)


def test_encoder():
    """测试编码器功能"""
    # 创建模拟特征
    raw_features = {
        'video_features': {
            'action_frequency': {
                'standing': 0.4,
                'walking': 0.1,
                'gesturing': 0.3,
                'writing': 0.15,
                'pointing': 0.05
            },
            'avg_motion_energy': 0.06,
            'spatial_distribution': {
                'middle_center': 0.6,
                'middle_left': 0.2,
                'middle_right': 0.2
            },
            'video_duration': 1800,  # 30分钟
            'total_frames': 54000,
            'pose_confidences': [0.85, 0.9, 0.88]
        },
        'audio_features': {
            'volume': [0.6, 0.65, 0.7, 0.68],
            'pitch': [150, 160, 155, 158],
            'voice_activity': [1, 1, 0, 1, 1],
            'sentiment': {'score': 0.7, 'label': 'positive'},
            'transcription': '今天我们学习一个新的概念。这个概念很重要。'
        },
        'text_features': {
            'embedding': np.random.randn(768).tolist(),
            'sentiment': {'score': 0.65, 'label': 'positive'},
            'word_count': 250,
            'sentence_count': 15,
            'keywords': ['概念', '学习', '重要']
        }
    }

    # 测试编码
    encoder = FeatureEncoder()
    encoded = encoder.encode_all(raw_features)

    print("=" * 60)
    print("特征编码测试")
    print("=" * 60)
    print(f"视频特征: {encoded['video'].shape} - {encoded['video'][:5]}")
    print(f"音频特征: {encoded['audio'].shape} - {encoded['audio'][:5]}")
    print(f"文本特征: {encoded['text'].shape} - {encoded['text'][:5]}")

    print("\n" + "=" * 60)
    print("人类可读解读")
    print("=" * 60)
    print(encoder.get_feature_interpretation(encoded))

    print("\n测试通过！")


if __name__ == '__main__':
    test_encoder()
