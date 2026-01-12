"""
测试VLM标注单个视频
"""
import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.annotation.vlm_annotator import VLMStyleAnnotator
from src.features.feature_encoder import FeatureEncoder

# 加载提取的特征
features_path = "data/custom/extracted_features/features_test.json"
with open(features_path, 'r', encoding='utf-8') as f:
    features_list = json.load(f)

sample = features_list[0]

print("="*60)
print("开始VLM标注测试")
print("="*60)
print(f"样本ID: {sample.get('sample_id')}")
print(f"视频名称: {sample.get('video_name')}")
print(f"关键帧数量: {len(sample.get('keyframes', []))}")
print(f"是否有raw_features: {'raw_features' in sample}")
print()

# 初始化VLM标注器
api_key = os.environ.get('ANTHROPIC_AUTH_TOKEN')
base_url = os.environ.get('ANTHROPIC_BASE_URL', 'https://aidev.deyecloud.com/api')

print(f"API Base URL: {base_url}")
print(f"API Key: {api_key[:20]}..." if api_key else "未设置")
print()

annotator = VLMStyleAnnotator(
    api_key=api_key,
    base_url=base_url,
    model="claude-opus-4-5-20251101"
)

# 准备VLM输入
raw_features = sample.get('raw_features', {})
keyframes = sample.get('keyframes', [])
metadata = sample.get('metadata', {})

print("准备VLM标注输入...")
print(f"- 关键帧: {len(keyframes)} 张")
print(f"- 动作序列长度: {len(raw_features.get('pose_tracking', {}).get('action_sequence', []))}")
print(f"- 空间分布区域: {len(raw_features.get('spatial_distribution', {}))}")
print(f"- 学科: {metadata.get('discipline')}")
print(f"- 年级: {metadata.get('grade')}")
print()

# 测试FeatureEncoder
print("测试FeatureEncoder...")
encoder = FeatureEncoder()
try:
    encoded = encoder.encode_all(raw_features)
    print(f"✓ 视频特征维度: {encoded['video'].shape}")
    print(f"✓ 音频特征维度: {encoded['audio'].shape}")
    print(f"✓ 文本特征维度: {encoded['text'].shape}")

    # 生成人类可读解读
    interpretation = encoder.get_feature_interpretation(encoded)
    print("\n量化特征解读:")
    print(interpretation)
except Exception as e:
    print(f"✗ FeatureEncoder错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("开始调用VLM进行标注...")
print("="*60)

try:
    # 调用VLM标注
    result = annotator.annotate_single_sample(
        video_frames=keyframes,
        audio_transcript=raw_features.get('transcription', ''),
        lecture_text=raw_features.get('transcription', ''),
        metadata=metadata,
        raw_features=raw_features
    )

    print("\n标注结果:")
    print("="*60)
    print(f"教学风格: {result['style']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"\n判断理由:\n{result['reasoning']}")

    if 'secondary_style' in result and result['secondary_style']:
        print(f"\n次要风格: {result['secondary_style']}")

    if 'key_features' in result:
        print(f"\n关键特征:")
        for feature in result['key_features']:
            print(f"  - {feature}")

    # 保存结果
    output_path = "data/custom/vlm_annotations/test_annotation.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    annotation_result = {
        "sample_id": sample.get('sample_id'),
        "annotation": result,
        "source_data": {
            "video_name": sample.get('video_name'),
            "metadata": metadata,
            "keyframes": keyframes[:3]  # 只保存前3帧路径
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([annotation_result], f, ensure_ascii=False, indent=2)

    print(f"\n标注结果已保存到: {output_path}")

except Exception as e:
    print(f"\n✗ VLM标注失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试完成")
print("="*60)
