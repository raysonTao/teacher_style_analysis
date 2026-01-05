"""测试深度学习模型与现有系统的集成"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.core.style_classifier import StyleClassifier
from src.config.config import logger


def print_separator(title=""):
    """打印分隔线"""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)


def test_classification(classifier, features, mode_name):
    """测试分类功能"""
    print_separator(f"测试 {mode_name} 模式")

    # 获取状态
    status = classifier.get_status()
    print("\n分类器状态:")
    print(f"  模式: {status.get('mode')}")
    print(f"  规则模型已加载: {status.get('rule_model_loaded')}")
    print(f"  深度学习模型已加载: {status.get('deep_learning_loaded')}")

    # 执行分类
    print("\n执行分类...")
    try:
        result = classifier.classify_style(features=features)

        print("\n分类结果:")
        print(f"  方法: {result.get('method', 'N/A')}")
        print(f"  置信度: {result.get('confidence', 0):.4f}")

        print("\n  Top-3 风格:")
        for style, score in result['top_styles'][:3]:
            print(f"    {style}: {score:.4f}")

        # 如果是混合模式，显示详细信息
        if result.get('method') == 'hybrid':
            print("\n  深度学习结果:")
            dl_results = result.get('deep_learning_results', {})
            sorted_dl = sorted(dl_results.items(), key=lambda x: x[1], reverse=True)[:3]
            for style, score in sorted_dl:
                print(f"    {style}: {score:.4f}")

            print("\n  规则系统结果:")
            rule_results = result.get('rule_based_results', {})
            sorted_rule = sorted(rule_results.items(), key=lambda x: x[1], reverse=True)[:3]
            for style, score in sorted_rule:
                print(f"    {style}: {score:.4f}")

        return True

    except Exception as e:
        print(f"\n分类失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print_separator("深度学习模型集成测试")

    # 准备测试特征
    test_features = {
        'video': {
            'head_movement_frequency': 0.6,
            'body_movement_frequency': 0.4,
            'behavior_frequency': {
                'writing': 0.2,
                'gesturing': 0.5,
                'pointing': 0.3,
                'standing': 0.8,
                'walking': 0.4
            },
            'eye_contact_score': 0.7,
            'facial_expression_scores': {
                'neutral': 0.3,
                'happy': 0.6,
                'surprise': 0.05,
                'sad': 0.02,
                'angry': 0.01,
                'disgust': 0.01,
                'fear': 0.01
            }
        },
        'audio': {
            'speech_rate': 130.0,
            'volume_level': 0.7,
            'pitch_variation': 0.5,
            'silence_ratio': 0.15,
            'emotion_scores': {
                'neutral': 0.4,
                'happy': 0.5,
                'sad': 0.05,
                'angry': 0.02,
                'fear': 0.01,
                'disgust': 0.01,
                'surprise': 0.01
            }
        },
        'text': {
            'vocabulary_richness': 0.65,
            'sentence_complexity': 0.55,
            'question_frequency': 0.08,
            'keyword_density': {
                'definition': 0.05,
                'example': 0.08,
                'explanation': 0.12,
                'summary': 0.03,
                'question': 0.08
            },
            'logical_indicators': {
                'causality': 0.06,
                'comparison': 0.04,
                'sequence': 0.05,
                'emphasis': 0.07
            },
            'sentiment_score': 0.6
        },
        'fusion': {
            'interaction_level': 0.7,
            'explanation_clarity': 0.6,
            'emotional_engagement': 0.65,
            'logical_structure': 0.55,
            'teaching_style_metrics': {
                'lecturing': 0.5,
                'guiding': 0.7,
                'interactive': 0.75,
                'logical': 0.6,
                'problem_driven': 0.55,
                'emotional': 0.7,
                'patient': 0.6
            }
        }
    }

    # 测试1: 规则系统模式（默认）
    print_separator("测试 1: 规则系统模式")
    classifier_rule = StyleClassifier(mode='rule')
    success_rule = test_classification(classifier_rule, test_features, "规则系统")

    # 测试2: 深度学习模式
    print_separator("测试 2: 深度学习模式")
    checkpoint_path = "./checkpoints/best_model.pth"

    if Path(checkpoint_path).exists():
        classifier_dl = StyleClassifier(
            mode='deep_learning',
            dl_checkpoint=checkpoint_path,
            dl_device='cpu'  # 使用CPU避免cuDNN问题
        )
        success_dl = test_classification(classifier_dl, test_features, "深度学习")
    else:
        print(f"\n检查点文件不存在: {checkpoint_path}")
        print("跳过深度学习模式测试")
        success_dl = False

    # 测试3: 混合模式
    print_separator("测试 3: 混合模式")

    if Path(checkpoint_path).exists():
        classifier_hybrid = StyleClassifier(
            mode='hybrid',
            dl_checkpoint=checkpoint_path,
            dl_device='cpu'  # 使用CPU避免cuDNN问题
        )
        success_hybrid = test_classification(classifier_hybrid, test_features, "混合")
    else:
        print(f"\n检查点文件不存在: {checkpoint_path}")
        print("跳过混合模式测试")
        success_hybrid = False

    # 总结
    print_separator("测试总结")
    print(f"\n规则系统模式: {'✓ 成功' if success_rule else '✗ 失败'}")
    print(f"深度学习模式: {'✓ 成功' if success_dl else '✗ 跳过/失败'}")
    print(f"混合模式: {'✓ 成功' if success_hybrid else '✗ 跳过/失败'}")

    all_success = success_rule and (not Path(checkpoint_path).exists() or (success_dl and success_hybrid))
    print(f"\n总体结果: {'✓ 所有测试通过' if all_success else '✗ 部分测试失败'}")

    print("\n集成测试完成!")


if __name__ == '__main__':
    main()
