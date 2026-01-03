#!/usr/bin/env python3
"""CMAT深度学习模型测试脚本"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import json
from pathlib import Path

# 导入相关模块
from config.config import MODEL_CONFIG, STYLE_LABELS, logger
from models.deep_learning.cmat_model import CMATModel, CMATTrainer
from models.core.style_classifier import StyleClassifier


def test_cmat_model_creation():
    """测试CMAT模型创建"""
    print("=== 测试CMAT模型创建 ===")
    
    try:
        # 定义输入维度
        input_dims = {
            'audio': MODEL_CONFIG['audio_dim'],
            'video': MODEL_CONFIG['video_dim'], 
            'text': MODEL_CONFIG['text_dim']
        }
        
        # 创建CMAT模型
        model = CMATModel(
            input_dims=input_dims,
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_heads=MODEL_CONFIG['num_heads'],
            num_styles=MODEL_CONFIG['num_styles']
        )
        
        print(f"✓ CMAT模型创建成功")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")
        print(f"  可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        return model
        
    except Exception as e:
        print(f"✗ CMAT模型创建失败: {e}")
        return None


def test_model_forward_pass(model):
    """测试模型前向传播"""
    print("\n=== 测试模型前向传播 ===")
    
    try:
        model.eval()
        
        # 创建测试输入
        batch_size = 2
        test_features = {
            'audio': torch.randn(batch_size, 100, MODEL_CONFIG['audio_dim']),
            'video': torch.randn(batch_size, 100, MODEL_CONFIG['video_dim']),
            'text': torch.randn(batch_size, 100, MODEL_CONFIG['text_dim'])
        }
        
        # 前向传播
        with torch.no_grad():
            results = model(test_features)
        
        print(f"✓ 前向传播成功")
        print(f"  风格分数形状: {results['style_scores'].shape}")
        print(f"  SMI分数形状: {results['smi_score'].shape}")
        print(f"  融合特征形状: {results['fused_features'].shape}")
        
        return results
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return None


def test_model_prediction(model):
    """测试模型预测功能"""
    print("\n=== 测试模型预测功能 ===")
    
    try:
        # 创建测试特征（单个样本）
        test_features = {
            'audio': torch.randn(1, 100, MODEL_CONFIG['audio_dim']),
            'video': torch.randn(1, 100, MODEL_CONFIG['video_dim']),
            'text': torch.randn(1, 100, MODEL_CONFIG['text_dim'])
        }
        
        # 使用预测接口
        prediction = model.predict(test_features)
        
        print(f"✓ 预测功能成功")
        print(f"  主导风格: {prediction['dominant_style']}")
        print(f"  置信度: {prediction['confidence']:.4f}")
        print(f"  SMI分数: {prediction['smi_score']:.2f}")
        
        # 显示所有风格分数
        print(f"  风格分数:")
        for style, score in prediction['style_scores'].items():
            print(f"    {style}: {score:.4f}")
        
        return prediction
        
    except Exception as e:
        print(f"✗ 预测功能失败: {e}")
        return None


def test_trainer_creation(model):
    """测试训练器创建"""
    print("\n=== 测试训练器创建 ===")
    
    try:
        # 创建设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建训练器
        trainer = CMATTrainer(
            model=model,
            device=device,
            learning_rate=MODEL_CONFIG['learning_rate'],
            config=MODEL_CONFIG
        )
        
        print(f"✓ 训练器创建成功")
        print(f"  优化器类型: {type(trainer.optimizer).__name__}")
        print(f"  学习率: {trainer.optimizer.param_groups[0]['lr']}")
        
        return trainer
        
    except Exception as e:
        print(f"✗ 训练器创建失败: {e}")
        return None


def test_dummy_data_creation(trainer):
    """测试虚拟数据创建"""
    print("\n=== 测试虚拟数据创建 ===")
    
    try:
        # 创建虚拟数据
        features, targets = trainer.create_dummy_data(batch_size=8, sequence_length=50)
        
        print(f"✓ 虚拟数据创建成功")
        print(f"  音频特征形状: {features['audio'].shape}")
        print(f"  视频特征形状: {features['video'].shape}")
        print(f"  文本特征形状: {features['text'].shape}")
        print(f"  风格标签形状: {targets['style_labels'].shape}")
        print(f"  SMI分数形状: {targets['smi_scores'].shape}")
        
        return features, targets
        
    except Exception as e:
        print(f"✗ 虚拟数据创建失败: {e}")
        return None, None


def test_style_classifier_integration():
    """测试风格分类器集成"""
    print("\n=== 测试风格分类器集成 ===")
    
    try:
        # 创建风格分类器
        classifier = StyleClassifier()
        
        # 获取状态
        status = classifier.get_status()
        print(f"✓ 风格分类器创建成功")
        print(f"  模型类型: {status['model_type']}")
        print(f"  模型加载状态: {status['model_loaded']}")
        print(f"  设备: {status['device']}")
        
        # 获取模型信息
        model_info = classifier.get_model_info()
        print(f"  模型架构: {model_info.get('model_architecture', 'Unknown')}")
        
        return classifier
        
    except Exception as e:
        print(f"✗ 风格分类器集成失败: {e}")
        return None


def test_classifier_with_dummy_data(classifier):
    """测试分类器与虚拟数据"""
    print("\n=== 测试分类器预测 ===")
    
    try:
        # 创建虚拟特征数据
        dummy_features = {
            'audio': {
                'speech_rate': 120.0,
                'silence_ratio': 0.1,
                'pitch_variation': 0.3,
                'emotion_scores': {'happy': 0.7, 'neutral': 0.3}
            },
            'video': {
                'behavior_frequency': {
                    'gesturing': 0.5,
                    'facing_board': 0.3
                }
            },
            'text': {
                'question_frequency': 0.2,
                'logical_indicators': {
                    'causal': 0.3,
                    'temporal': 0.2
                },
                'vocabulary_richness': 0.6
            },
            'fusion': {
                'interaction_level': 0.4,
                'logical_structure': 0.7,
                'emotional_engagement': 0.5,
                'explanation_clarity': 0.6,
                'teaching_style_metrics': {
                    'lecturing': 0.3,
                    'guiding': 0.5,
                    'interactive': 0.4,
                    'logical': 0.7,
                    'problem_driven': 0.2,
                    'emotional': 0.5,
                    'patient': 0.6
                }
            }
        }
        
        # 进行分类
        result = classifier.classify_style(features=dummy_features)
        
        print(f"✓ 分类器预测成功")
        print(f"  模型类型: {result['model_type']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  主导风格: {result['top_styles'][0][0]}")
        
        # 显示前3个风格
        print(f"  前3个风格:")
        for style, score in result['top_styles']:
            print(f"    {style}: {score:.4f}")
        
        return result
        
    except Exception as e:
        print(f"✗ 分类器预测失败: {e}")
        return None


def test_model_saving_loading(model, trainer):
    """测试模型保存和加载"""
    print("\n=== 测试模型保存和加载 ===")
    
    try:
        save_path = 'models/test_cmat_model.pth'
        
        # 保存模型
        trainer.save_model(save_path)
        print(f"✓ 模型保存成功: {save_path}")
        
        # 创建新模型和训练器
        input_dims = {
            'audio': MODEL_CONFIG['audio_dim'],
            'video': MODEL_CONFIG['video_dim'], 
            'text': MODEL_CONFIG['text_dim']
        }
        
        new_model = CMATModel(
            input_dims=input_dims,
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_heads=MODEL_CONFIG['num_heads'],
            num_styles=MODEL_CONFIG['num_styles']
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        new_trainer = CMATTrainer(
            model=new_model,
            device=device,
            learning_rate=MODEL_CONFIG['learning_rate'],
            config=MODEL_CONFIG
        )
        
        # 加载模型
        new_trainer.load_model(save_path)
        print(f"✓ 模型加载成功")
        
        # 验证加载的模型
        new_model.eval()
        test_features = {
            'audio': torch.randn(1, 100, MODEL_CONFIG['audio_dim']),
            'video': torch.randn(1, 100, MODEL_CONFIG['video_dim']),
            'text': torch.randn(1, 100, MODEL_CONFIG['text_dim'])
        }
        
        with torch.no_grad():
            new_results = new_model(test_features)
        
        print(f"✓ 加载的模型预测成功")
        print(f"  风格分数形状: {new_results['style_scores'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型保存/加载失败: {e}")
        return False


def main():
    """主测试函数"""
    print("CMAT深度学习模型测试开始")
    print("=" * 50)
    
    # 1. 测试CMAT模型创建
    model = test_cmat_model_creation()
    if model is None:
        print("模型创建失败，退出测试")
        return
    
    # 2. 测试前向传播
    results = test_model_forward_pass(model)
    if results is None:
        print("前向传播失败，退出测试")
        return
    
    # 3. 测试预测功能
    prediction = test_model_prediction(model)
    if prediction is None:
        print("预测功能失败，退出测试")
        return
    
    # 4. 测试训练器创建
    trainer = test_trainer_creation(model)
    if trainer is None:
        print("训练器创建失败，退出测试")
        return
    
    # 5. 测试虚拟数据创建
    features, targets = test_dummy_data_creation(trainer)
    if features is None:
        print("虚拟数据创建失败")
    
    # 6. 测试风格分类器集成
    classifier = test_style_classifier_integration()
    if classifier is None:
        print("风格分类器集成失败")
    else:
        # 7. 测试分类器预测
        test_classifier_with_dummy_data(classifier)
    
    # 8. 测试模型保存和加载
    test_model_saving_loading(model, trainer)
    
    print("\n" + "=" * 50)
    print("CMAT深度学习模型测试完成")


if __name__ == "__main__":
    main()