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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEncoder:
    """多模态特征编码器"""

    def __init__(self):
        """初始化编码器"""
        # 动作类型映射
        self.action_types = ['standing', 'walking', 'gesturing', 'writing', 'pointing']

        # 情感标签映射
        self.sentiment_labels = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}

    def encode_video_features(self, raw_features: Dict) -> np.ndarray:
        """
        编码视频特征 → 20维向量

        维度分配：
        1-5维: 动作频率 (standing, walking, gesturing, writing, pointing)
        6维: 平均运动能量
        7-15维: 空间分布 (9个区域)
        16-20维: 统计量 (持续时间、帧数等)

        Args:
            raw_features: 原始视频特征字典

        Returns:
            20维numpy数组
        """
        features = np.zeros(20)

        try:
            # 1-5维: 动作频率
            action_frequency = raw_features.get('action_frequency', {})
            for i, action in enumerate(self.action_types):
                features[i] = action_frequency.get(action, 0.0)

            # 6维: 平均运动能量（归一化）
            avg_motion = raw_features.get('avg_motion_energy', 0.0)
            features[5] = min(avg_motion * 10, 1.0)  # 假设正常范围0-0.1，映射到0-1

            # 7-15维: 空间分布（9个区域）
            spatial_dist = raw_features.get('spatial_distribution', {})
            regions = ['top_left', 'top_center', 'top_right',
                      'middle_left', 'middle_center', 'middle_right',
                      'bottom_left', 'bottom_center', 'bottom_right']
            for i, region in enumerate(regions):
                features[6 + i] = spatial_dist.get(region, 0.0)

            # 16维: 视频时长（归一化，假设最长60分钟）
            duration = raw_features.get('video_duration', 0.0)
            features[15] = min(duration / 3600, 1.0)

            # 17维: 总帧数（归一化，假设最多180000帧=1小时@30fps）
            total_frames = raw_features.get('total_frames', 0)
            features[16] = min(total_frames / 180000, 1.0)

            # 18维: 姿态平均置信度
            pose_confidences = raw_features.get('pose_confidences', [])
            if pose_confidences:
                features[17] = np.mean(pose_confidences)
            else:
                features[17] = 0.5  # 默认中等置信度

            # 19-20维: 保留维度（可扩展）
            features[18] = 0.0
            features[19] = 0.0

        except Exception as e:
            logger.error(f"编码视频特征失败: {e}")
            # 返回零向量作为后备

        return features

    def encode_audio_features(self, raw_features: Dict) -> np.ndarray:
        """
        编码音频特征 → 15维向量

        维度分配：
        1-4维: 音量统计 (均值, 标准差, 最小值, 最大值)
        5-8维: 音调统计 (均值, 标准差, 最小值, 最大值)
        9维: 语音活动比例
        10-12维: 情感分析 (分数, positive/negative概率)
        13-15维: 文本统计 (词数, 句数, 字符数)

        Args:
            raw_features: 原始音频特征字典

        Returns:
            15维numpy数组
        """
        features = np.zeros(15)

        try:
            # 1-4维: 音量统计
            volume = raw_features.get('volume', [])
            if volume:
                # 归一化到0-1（假设RMS范围0-1）
                features[0] = np.mean(volume)
                features[1] = np.std(volume)
                features[2] = np.min(volume)
                features[3] = np.max(volume)

            # 5-8维: 音调统计
            pitch = raw_features.get('pitch', [])
            if pitch:
                # 过滤无效值（0或负值）
                valid_pitch = [p for p in pitch if p > 0]
                if valid_pitch:
                    # 归一化到0-1（假设音调范围50-500Hz）
                    norm_pitch = [(p - 50) / 450 for p in valid_pitch]
                    features[4] = np.mean(norm_pitch)
                    features[5] = np.std(norm_pitch)
                    features[6] = np.min(norm_pitch)
                    features[7] = np.max(norm_pitch)

            # 9维: 语音活动比例
            voice_activity = raw_features.get('voice_activity', [])
            if voice_activity:
                features[8] = np.mean(voice_activity)

            # 10-12维: 情感分析
            sentiment = raw_features.get('sentiment', {})
            if sentiment:
                features[9] = sentiment.get('score', 0.5)  # 情感分数
                label = sentiment.get('label', 'neutral')
                features[10] = self.sentiment_labels.get(label, 0.0)  # 标签编码
                # 11维: 正负情感差异（可用于区分强烈程度）
                features[11] = abs(features[9] - 0.5) * 2  # 映射到0-1

            # 13-15维: 文本统计（从转录得到）
            transcription = raw_features.get('transcription', '')
            if transcription:
                # 中文：按字符计数
                char_count = len(transcription)
                # 句子数（按标点分割）
                sentence_count = transcription.count('。') + transcription.count('？') + transcription.count('！')
                # 词数（中文按字计，英文按空格分）
                word_count = len(transcription.split())

                # 归一化（假设最多5000字）
                features[12] = min(word_count / 1000, 1.0)
                features[13] = min(sentence_count / 50, 1.0)
                features[14] = min(char_count / 5000, 1.0)

        except Exception as e:
            logger.error(f"编码音频特征失败: {e}")

        return features

    def encode_text_features(self, raw_features: Dict) -> np.ndarray:
        """
        编码文本特征 → 25维向量

        维度分配：
        1-20维: BERT嵌入降维 (768→20, 分段平均池化)
        21-22维: 情感分析 (分数, label编码)
        23-25维: 统计量 (词数, 句数, 关键词数)

        Args:
            raw_features: 原始文本特征字典

        Returns:
            25维numpy数组
        """
        features = np.zeros(25)

        try:
            # 1-20维: BERT嵌入降维
            embedding = raw_features.get('embedding', [])
            if embedding and len(embedding) == 768:
                # 分4段平均池化：768 → 192 → 48 → 20
                # 方法：每38.4个维度合并为1维
                segment_size = 768 // 20
                for i in range(20):
                    start = i * segment_size
                    end = start + segment_size if i < 19 else 768
                    features[i] = np.mean(embedding[start:end])
            elif embedding:
                # 如果维度不是768，做padding或截断
                emb_array = np.array(embedding[:768])
                if len(emb_array) < 768:
                    emb_array = np.pad(emb_array, (0, 768 - len(emb_array)))
                segment_size = 768 // 20
                for i in range(20):
                    start = i * segment_size
                    end = start + segment_size if i < 19 else 768
                    features[i] = np.mean(emb_array[start:end])

            # 21-22维: 情感分析
            sentiment = raw_features.get('sentiment', {})
            if sentiment:
                features[20] = sentiment.get('score', 0.5)
                label = sentiment.get('label', 'neutral')
                features[21] = self.sentiment_labels.get(label, 0.0)

            # 23-25维: 统计量
            word_count = raw_features.get('word_count', 0)
            sentence_count = raw_features.get('sentence_count', 0)
            keywords = raw_features.get('keywords', [])

            # 归一化
            features[22] = min(word_count / 1000, 1.0)
            features[23] = min(sentence_count / 50, 1.0)
            features[24] = min(len(keywords) / 20, 1.0)

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

        motion = video[5]
        motion_level = "高" if motion > 0.6 else "中" if motion > 0.3 else "低"
        interpretation.append(f"- 运动能量：{motion_level} ({motion:.2f}) → 教师{'活跃度高' if motion > 0.6 else '活跃度中等' if motion > 0.3 else '相对静态'}")

        # 音频特征解读
        interpretation.append("\n**音频特征：**")
        volume_mean = audio[0]
        volume_std = audio[1]
        volume_desc = "响亮且稳定" if volume_mean > 0.6 and volume_std < 0.2 else \
                     "音量适中" if volume_mean > 0.4 else "音量较小"
        interpretation.append(f"- 音量：{volume_desc} (均值={volume_mean:.2f}, 变化={volume_std:.2f})")

        pitch_mean = audio[4]
        pitch_std = audio[5]
        pitch_desc = "音调变化丰富" if pitch_std > 0.15 else "音调相对平稳"
        interpretation.append(f"- 音调：{pitch_desc} (均值={pitch_mean:.2f}, 变化={pitch_std:.2f})")

        voice_ratio = audio[8]
        interpretation.append(f"- 语音活动比例：{voice_ratio:.1%} → {'讲话为主' if voice_ratio > 0.7 else '互动较多' if voice_ratio > 0.5 else '停顿较多'}")

        # 文本特征解读
        interpretation.append("\n**文本特征：**")
        word_count = int(text[22] * 1000)  # 反归一化
        sentence_count = int(text[23] * 50)
        interpretation.append(f"- 讲话内容量：约{word_count}词，{sentence_count}句")

        if word_count > 0 and sentence_count > 0:
            avg_sentence_len = word_count / sentence_count
            sentence_desc = "句子较长，讲解详细" if avg_sentence_len > 25 else \
                           "句子适中" if avg_sentence_len > 15 else "句子简短，节奏快"
            interpretation.append(f"- 表达风格：{sentence_desc} (平均{avg_sentence_len:.1f}词/句)")

        sentiment_score = text[20]
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
