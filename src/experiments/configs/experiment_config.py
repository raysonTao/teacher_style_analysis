"""实验配置文件，定义实验参数和设置"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 实验配置
EXPERIMENTS_CONFIG = {
    # 数据配置
    'data': {
        'dataset_path': str(PROJECT_ROOT / 'experiments' / 'data'),
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'batch_size': 32,
        'shuffle': True,
        'seed': 42
    },
    
    # 模型配置
    'models': {
        # 风格分类模型比较
        'classifier_comparison': {
            'models': ['cmat', 'xgboost', 'random_forest', 'svm', 'neural_network'],
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'],
            'cross_validation': 5
        },
        
        # CMAT模型参数
        'cmat_model': {
            'lambda_weights': [0.1, 0.3, 0.5, 0.7, 0.9],  # 规则驱动与机器学习的权重
            'feature_importance_threshold': 0.05,
            'max_depth': 10,
            'n_estimators': 100,
            'learning_rate': 0.1
        },
        
        # 多模态融合策略
        'fusion_strategies': {
            'strategies': ['early_fusion', 'late_fusion', 'hybrid_fusion', 'attention_fusion'],
            'modalities': ['video', 'audio', 'text', 'all']
        }
    },
    
    # 实验配置
    'experiments': {
        # 实验1: 模型性能对比
        'exp_1_model_performance': {
            'name': '模型性能对比实验',
            'description': '比较不同分类算法在教师风格识别任务上的性能',
            'enable': True
        },
        
        # 实验2: 多模态特征融合效果
        'exp_2_modality_fusion': {
            'name': '多模态特征融合效果实验',
            'description': '比较不同模态组合和融合策略的效果',
            'enable': True
        },
        
        # 实验3: 规则驱动与机器学习融合
        'exp_3_rule_ml_fusion': {
            'name': '规则与机器学习融合效果实验',
            'description': '研究lambda权重对融合效果的影响',
            'enable': True
        },
        
        # 实验4: SMI计算方法验证
        'exp_4_smi_validation': {
            'name': '风格匹配度指数验证实验',
            'description': '验证SMI计算方法的有效性和稳定性',
            'enable': True
        },
        
        # 实验5: 跨学科适应性
        'exp_5_cross_discipline': {
            'name': '跨学科适应性实验',
            'description': '评估模型在不同学科教学视频上的泛化能力',
            'enable': True
        },
        
        # 实验6: 可解释性分析
        'exp_6_explainability': {
            'name': '模型可解释性分析实验',
            'description': '分析特征贡献度和规则重要性',
            'enable': True
        }
    },
    
    # 评估配置
    'evaluation': {
        'metrics': {
            'accuracy': True,
            'precision': True,
            'recall': True,
            'f1': True,
            'auc_roc': True,
            'confusion_matrix': True,
            'feature_importance': True,
            'shap_analysis': False  # 可选，计算成本较高
        },
        'visualization': {
            'enable': True,
            'plot_types': ['bar', 'line', 'heatmap', 'radar'],
            'save_format': 'png',
            'dpi': 300
        }
    },
    
    # 路径配置
    'paths': {
        'results_dir': str(PROJECT_ROOT / 'experiments' / 'results'),
        'models_dir': str(PROJECT_ROOT / 'experiments' / 'models'),
        'visualizations_dir': str(PROJECT_ROOT / 'experiments' / 'visualizations'),
        'logs_dir': str(PROJECT_ROOT / 'experiments' / 'logs')
    },
    
    # 运行配置
    'runtime': {
        'num_workers': 4,
        'use_gpu': True,
        'debug': False,
        'verbose': True,
        'save_interval': 50  # 保存检查点的间隔
    }
}

# 风格标签配置 (与主系统保持一致)
STYLE_LABELS = {
    '理论讲授型': 0,
    '启发引导型': 1,
    '互动导向型': 2,
    '逻辑推导型': 3,
    '题目驱动型': 4,
    '情感表达型': 5,
    '耐心细致型': 6
}

# 反向映射
ID_TO_STYLE = {v: k for k, v in STYLE_LABELS.items()}

# 学科和年级配置
DISCIPLINES = ['数学', '语文', '英语', '物理', '化学', '生物', '历史', '地理', '政治']
GRADES = ['初中', '高中', '大学']

# 实验数据生成配置（用于生成模拟数据）
SIMULATION_CONFIG = {
    'sample_size': 1000,
    'noise_level': 0.1,
    'feature_dimensions': {
        'video': 20,
        'audio': 15,
        'text': 25,
        'fusion': 30
    },
    'style_correlation': 0.7,  # 特征与风格的相关程度
    'inter_class_variance': 0.8  # 类间方差
}

# 确保目录存在
for path_key, path_value in EXPERIMENTS_CONFIG['paths'].items():
    os.makedirs(path_value, exist_ok=True)

# 导出配置
__all__ = ['EXPERIMENTS_CONFIG', 'STYLE_LABELS', 'ID_TO_STYLE', 'DISCIPLINES', 'GRADES', 'SIMULATION_CONFIG']