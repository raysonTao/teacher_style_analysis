"""
MM-TBA 数据集转换脚本
将 MM-TBA 的讲课评价数据转换为训练格式
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import re

# 教学风格标签映射
STYLE_LABELS = {
    '理论讲授型': 0,
    '启发引导型': 1,
    '互动导向型': 2,
    '逻辑推导型': 3,
    '题目驱动型': 4,
    '情感表达型': 5,
    '耐心细致型': 6
}

def extract_features_from_text(text: str) -> Dict:
    """从讲课文本中提取简单的特征"""

    # 文本特征提取（简化版）
    words = text.split()
    sentences = text.split('。')

    # 计算基本统计特征
    text_length = len(text)
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_length = word_count / max(sentence_count, 1)

    # 关键词检测
    question_keywords = ['什么', '为什么', '怎么', '如何', '?', '？']
    explanation_keywords = ['因为', '所以', '就是说', '也就是', '比如']
    interaction_keywords = ['同学们', '大家', '我们', '你们']
    logical_keywords = ['首先', '其次', '然后', '接下来', '总之']

    question_count = sum(text.count(kw) for kw in question_keywords)
    explanation_count = sum(text.count(kw) for kw in explanation_keywords)
    interaction_count = sum(text.count(kw) for kw in interaction_keywords)
    logical_count = sum(text.count(kw) for kw in logical_keywords)

    # 归一化特征
    text_features = [
        min(text_length / 5000, 1.0),  # 文本长度
        min(word_count / 1000, 1.0),    # 词数
        min(avg_sentence_length / 50, 1.0),  # 平均句长
        min(question_count / 20, 1.0),  # 提问频率
        min(explanation_count / 30, 1.0),  # 解释频率
        min(interaction_count / 20, 1.0),  # 互动词频率
        min(logical_count / 15, 1.0),  # 逻辑词频率
    ]

    # 补充到25维
    while len(text_features) < 25:
        text_features.append(np.random.uniform(0.3, 0.7))

    return text_features[:25]

def extract_style_from_evaluation(evaluation: str) -> int:
    """从评价文本中推断教学风格"""

    # 根据评价关键词推断风格
    if '讲述' in evaluation or '讲解' in evaluation or '理论' in evaluation:
        return STYLE_LABELS['理论讲授型']
    elif '引导' in evaluation or '启发' in evaluation or '思考' in evaluation:
        return STYLE_LABELS['启发引导型']
    elif '互动' in evaluation or '交流' in evaluation or '参与' in evaluation:
        return STYLE_LABELS['互动导向型']
    elif '逻辑' in evaluation or '推理' in evaluation or '严谨' in evaluation:
        return STYLE_LABELS['逻辑推导型']
    elif '题目' in evaluation or '练习' in evaluation or '解题' in evaluation:
        return STYLE_LABELS['题目驱动型']
    elif '情感' in evaluation or '热情' in evaluation or '鼓励' in evaluation:
        return STYLE_LABELS['情感表达型']
    elif '耐心' in evaluation or '细致' in evaluation or '详细' in evaluation:
        return STYLE_LABELS['耐心细致型']
    else:
        # 默认基于评分等级
        if 'A' in evaluation or '优秀' in evaluation:
            return STYLE_LABELS['启发引导型']
        elif 'B' in evaluation or '良好' in evaluation:
            return STYLE_LABELS['理论讲授型']
        elif 'C' in evaluation:
            return STYLE_LABELS['理论讲授型']
        else:
            return STYLE_LABELS['理论讲授型']

def extract_rule_features_from_evaluation(evaluation: str) -> List[float]:
    """从评价中提取规则特征（7维）"""

    # 评分映射
    score_map = {'A': 0.9, 'B': 0.7, 'C': 0.5, 'D': 0.3}

    # 提取各维度分数
    dimensions = {
        'lecturing': 0.5,      # 理论讲授
        'guiding': 0.5,        # 启发引导
        'interactive': 0.5,    # 互动导向
        'logical': 0.5,        # 逻辑推导
        'problem_driven': 0.5, # 题目驱动
        'emotional': 0.5,      # 情感表达
        'patient': 0.5         # 耐心细致
    }

    # 根据评价文本调整分数
    if '互动' in evaluation:
        match = re.search(r'等级：([A-D])', evaluation)
        if match:
            dimensions['interactive'] = score_map.get(match.group(1), 0.5)

    if '引导' in evaluation or '启发' in evaluation:
        dimensions['guiding'] = 0.7

    if '逻辑' in evaluation or '严谨' in evaluation:
        dimensions['logical'] = 0.7

    if '情感' in evaluation or '热情' in evaluation:
        dimensions['emotional'] = 0.7

    if '耐心' in evaluation or '细致' in evaluation:
        dimensions['patient'] = 0.7

    return list(dimensions.values())

def convert_mmtba_to_training_format(
    mmtba_train_path: str,
    mmtba_eval_path: str,
    output_path: str
):
    """转换 MM-TBA 数据为训练格式"""

    print("开始转换 MM-TBA 数据集...")

    # 读取训练和评估数据
    with open(mmtba_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(mmtba_eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    all_data = train_data + eval_data
    print(f"总共 {len(all_data)} 个样本")

    converted_samples = []

    # 划分数据集：70% 训练，15% 验证，15% 测试
    total_samples = len(all_data)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    # test_size = total_samples - train_size - val_size

    for idx, item in enumerate(all_data):
        conversations = item['conversations']

        # 提取讲课文本和评价
        lecture_text = conversations[0]['value']  # human
        evaluation_text = conversations[1]['value']  # gpt

        # 从文本提取特征
        text_features = extract_features_from_text(lecture_text)

        # 生成模拟的视频和音频特征（基于文本长度和内容）
        text_len = len(lecture_text)
        video_features = [
            min(text_len / 5000, 1.0) * 0.6,  # 基于文本长度估算活动度
            np.random.uniform(0.3, 0.7),
        ]
        while len(video_features) < 20:
            video_features.append(np.random.uniform(0.3, 0.7))

        audio_features = [
            min(len(lecture_text.split()) / 500, 1.0),  # 估算语速
            np.random.uniform(0.5, 0.8),
        ]
        while len(audio_features) < 15:
            audio_features.append(np.random.uniform(0.3, 0.7))

        # 提取标签
        label = extract_style_from_evaluation(evaluation_text)

        # 提取规则特征
        rule_features = extract_rule_features_from_evaluation(evaluation_text)

        # 确定数据集划分
        if idx < train_size:
            split = 'train'
        elif idx < train_size + val_size:
            split = 'val'
        else:
            split = 'test'

        # 构建样本
        sample = {
            'sample_id': f'mmtba_{idx:04d}',
            'video_features': video_features,
            'audio_features': audio_features,
            'text_features': text_features,
            'rule_features': rule_features,
            'label': label,
            'split': split,
            'metadata': {
                'source': 'mm-tba',
                'original_text_length': len(lecture_text),
                'evaluation': evaluation_text[:200] + '...' if len(evaluation_text) > 200 else evaluation_text
            }
        }

        converted_samples.append(sample)

    # 保存转换后的数据
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, ensure_ascii=False, indent=2)

    print(f"\n转换完成！")
    print(f"总样本数: {len(converted_samples)}")
    print(f"训练集: {sum(1 for s in converted_samples if s['split'] == 'train')} 样本 (70%)")
    print(f"验证集: {sum(1 for s in converted_samples if s['split'] == 'val')} 样本 (15%)")
    print(f"测试集: {sum(1 for s in converted_samples if s['split'] == 'test')} 样本 (15%)")
    print(f"输出文件: {output_file}")

    # 统计标签分布
    from collections import Counter
    label_dist = Counter([s['label'] for s in converted_samples])
    style_names = {v: k for k, v in STYLE_LABELS.items()}

    print("\n标签分布:")
    for label_id, count in sorted(label_dist.items()):
        style_name = style_names[label_id]
        print(f"  {style_name} ({label_id}): {count} 样本")

    return output_file

if __name__ == '__main__':
    # 转换数据
    mmtba_dir = Path("/home/rayson/code/teacher_style_analysis/data/mm-tba/MM-TBA/Teacher_Lecture_Evaluation/finetune_data")

    output_path = convert_mmtba_to_training_format(
        mmtba_train_path=str(mmtba_dir / "train.json"),
        mmtba_eval_path=str(mmtba_dir / "eval.json"),
        output_path="data/mm-tba/mmtba_converted.json"
    )

    print(f"\n✅ 数据转换完成！可以使用以下命令训练：")
    print(f"python -m src.models.deep_learning.train --data_path {output_path} --device cuda")
