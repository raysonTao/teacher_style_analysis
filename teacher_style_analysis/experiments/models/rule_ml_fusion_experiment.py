"""规则与机器学习融合实验，研究lambda权重对融合效果的影响"""
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from ..configs.experiment_config import EXPERIMENTS_CONFIG, STYLE_LABELS
from ..data.data_generator import data_generator

class RuleMLFusionExperiment:
    """规则与机器学习融合实验类"""
    
    def __init__(self):
        """初始化融合实验"""
        self.config = EXPERIMENTS_CONFIG
        self.cmat_config = self.config['models']['cmat_model']
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.visualizations_dir = Path(self.config['paths']['visualizations_dir'])
        
        # 创建目录
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.visualizations_dir.mkdir(exist_ok=True, parents=True)
        
        # 准备数据
        self.train_data, self.val_data, self.test_data = data_generator.load_dataset()
        
        # 提取特征
        self.X_train, self.y_train = self._extract_features(self.train_data)
        self.X_test, self.y_test = self._extract_features(self.test_data)
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # 融合权重列表
        self.lambda_weights = self.cmat_config['lambda_weights']
        
        # 结果存储
        self.results = {}
    
    def _extract_features(self, data):
        """
        从数据中提取特征
        
        Args:
            data: 数据集
            
        Returns:
            tuple: (X, y)
        """
        feature_columns = [col for col in data.columns if col.startswith('fusion_feature_')]
        X = data[feature_columns].values
        y = data['style_id'].values
        return X, y
    
    def _simulate_rule_based_system(self, X, true_labels):
        """
        模拟基于规则的分类系统
        
        Args:
            X: 特征向量
            true_labels: 真实标签（用于模拟规则系统的部分准确性）
            
        Returns:
            tuple: (rule_predictions, rule_probabilities)
        """
        n_samples = X.shape[0]
        n_classes = len(STYLE_LABELS)
        
        # 生成规则系统预测
        rule_predictions = np.zeros(n_samples, dtype=int)
        rule_probabilities = np.zeros((n_samples, n_classes))
        
        # 模拟规则系统有一定的准确率（70-85%）
        base_accuracy = np.random.uniform(0.7, 0.85)
        correct_predictions = int(n_samples * base_accuracy)
        
        # 随机选择正确预测的样本
        correct_indices = np.random.choice(n_samples, size=correct_predictions, replace=False)
        incorrect_indices = np.setdiff1d(np.arange(n_samples), correct_indices)
        
        # 正确预测的样本
        rule_predictions[correct_indices] = true_labels[correct_indices]
        for idx in correct_indices:
            # 为正确类别分配高概率
            correct_class = true_labels[idx]
            rule_probabilities[idx, correct_class] = 0.7 + np.random.random() * 0.3  # 0.7-1.0
            # 为其他类别分配剩余概率
            other_classes = np.setdiff1d(np.arange(n_classes), [correct_class])
            rule_probabilities[idx, other_classes] = np.random.dirichlet(np.ones(n_classes-1)) * (1 - rule_probabilities[idx, correct_class])
        
        # 错误预测的样本
        for idx in incorrect_indices:
            # 随机选择一个错误类别
            incorrect_class = np.random.choice(np.setdiff1d(np.arange(n_classes), [true_labels[idx]]))
            rule_predictions[idx] = incorrect_class
            # 为错误类别分配较高概率
            rule_probabilities[idx, incorrect_class] = 0.6 + np.random.random() * 0.3  # 0.6-0.9
            # 为其他类别分配剩余概率
            other_classes = np.setdiff1d(np.arange(n_classes), [incorrect_class])
            rule_probabilities[idx, other_classes] = np.random.dirichlet(np.ones(n_classes-1)) * (1 - rule_probabilities[idx, incorrect_class])
        
        return rule_predictions, rule_probabilities
    
    def _simulate_machine_learning_system(self, X, y_train, X_train):
        """
        模拟机器学习分类系统
        
        Args:
            X: 测试特征向量
            y_train: 训练标签
            X_train: 训练特征向量
            
        Returns:
            tuple: (ml_predictions, ml_probabilities)
        """
        # 使用简单的最近邻方法模拟机器学习系统
        from sklearn.neighbors import KNeighborsClassifier
        
        # 训练KNN模型
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(X_train, y_train)
        
        # 预测
        ml_predictions = knn.predict(X)
        ml_probabilities = knn.predict_proba(X)
        
        return ml_predictions, ml_probabilities
    
    def _fuse_predictions(self, rule_prob, ml_prob, lambda_weight):
        """
        融合规则系统和机器学习系统的预测
        
        Args:
            rule_prob: 规则系统的概率预测
            ml_prob: 机器学习系统的概率预测
            lambda_weight: 规则系统的权重 (0-1)
            
        Returns:
            tuple: (fused_predictions, fused_probabilities)
        """
        # 融合概率
        fused_prob = lambda_weight * rule_prob + (1 - lambda_weight) * ml_prob
        
        # 归一化概率
        fused_prob = fused_prob / np.sum(fused_prob, axis=1, keepdims=True)
        
        # 获取预测类别
        fused_predictions = np.argmax(fused_prob, axis=1)
        
        return fused_predictions, fused_prob
    
    def run_fusion_experiment(self, save_results=True):
        """
        运行融合实验，测试不同lambda权重的效果
        
        Args:
            save_results: 是否保存结果
            
        Returns:
            dict: 实验结果
        """
        print(f"\n{'='*60}")
        print(f"开始规则与机器学习融合效果实验")
        print(f"{'='*60}")
        
        # 模拟规则系统预测
        print("模拟基于规则的分类系统...")
        rule_pred_train, rule_prob_train = self._simulate_rule_based_system(self.X_train, self.y_train)
        rule_pred_test, rule_prob_test = self._simulate_rule_based_system(self.X_test, self.y_test)
        
        # 计算规则系统性能
        rule_accuracy = accuracy_score(self.y_test, rule_pred_test)
        rule_f1 = f1_score(self.y_test, rule_pred_test, average='weighted')
        print(f"规则系统性能: 准确率={rule_accuracy:.4f}, F1={rule_f1:.4f}")
        
        # 模拟机器学习系统预测
        print("模拟机器学习分类系统...")
        ml_pred_test, ml_prob_test = self._simulate_machine_learning_system(
            self.X_test, self.y_train, self.X_train
        )
        
        # 计算机器学习系统性能
        ml_accuracy = accuracy_score(self.y_test, ml_pred_test)
        ml_f1 = f1_score(self.y_test, ml_pred_test, average='weighted')
        print(f"机器学习系统性能: 准确率={ml_accuracy:.4f}, F1={ml_f1:.4f}")
        
        # 测试不同的lambda权重
        print(f"\n测试不同的lambda权重 ({self.lambda_weights}):")
        
        for lambda_weight in self.lambda_weights:
            print(f"\nLambda权重: {lambda_weight}")
            
            # 融合预测
            fused_pred, fused_prob = self._fuse_predictions(
                rule_prob_test, ml_prob_test, lambda_weight
            )
            
            # 评估性能
            metrics = {
                'accuracy': accuracy_score(self.y_test, fused_pred),
                'precision': precision_score(self.y_test, fused_pred, average='weighted'),
                'recall': recall_score(self.y_test, fused_pred, average='weighted'),
                'f1': f1_score(self.y_test, fused_pred, average='weighted')
            }
            
            # 保存详细报告
            metrics['classification_report'] = classification_report(
                self.y_test, fused_pred, target_names=list(STYLE_LABELS.keys()), output_dict=True
            )
            metrics['confusion_matrix'] = confusion_matrix(self.y_test, fused_pred)
            
            # 记录结果
            self.results[lambda_weight] = metrics
            
            print(f"融合系统性能: 准确率={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        # 添加单系统结果作为参考
        self.results['rule_only'] = {
            'accuracy': rule_accuracy,
            'f1': rule_f1
        }
        
        self.results['ml_only'] = {
            'accuracy': ml_accuracy,
            'f1': ml_f1
        }
        
        # 保存和可视化结果
        if save_results:
            self._save_results()
            self._visualize_results()
        
        # 找出最佳lambda权重
        best_lambda = None
        best_accuracy = 0
        
        for lambda_weight in self.lambda_weights:
            if self.results[lambda_weight]['accuracy'] > best_accuracy:
                best_accuracy = self.results[lambda_weight]['accuracy']
                best_lambda = lambda_weight
        
        print(f"\n{'='*60}")
        print(f"实验完成")
        print(f"最佳Lambda权重: {best_lambda} (准确率: {best_accuracy:.4f})")
        print(f"{'='*60}")
        
        return self.results
    
    def _save_results(self):
        """
        保存实验结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"rule_ml_fusion_{timestamp}.csv"
        
        # 准备数据
        data = []
        
        # 添加融合结果
        for lambda_weight in self.lambda_weights:
            metrics = self.results[lambda_weight]
            data.append({
                'type': f'fusion_lambda_{lambda_weight}',
                'lambda_weight': lambda_weight,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            })
        
        # 添加单系统结果
        data.append({
            'type': 'rule_only',
            'lambda_weight': 1.0,
            'accuracy': self.results['rule_only']['accuracy'],
            'precision': None,
            'recall': None,
            'f1': self.results['rule_only']['f1']
        })
        
        data.append({
            'type': 'ml_only',
            'lambda_weight': 0.0,
            'accuracy': self.results['ml_only']['accuracy'],
            'precision': None,
            'recall': None,
            'f1': self.results['ml_only']['f1']
        })
        
        # 保存为CSV
        df = pd.DataFrame(data)
        df.to_csv(results_file, index=False, encoding='utf-8')
        print(f"\n融合实验结果已保存到: {results_file}")
        
        # 保存完整结果
        full_results_file = self.results_dir / f"rule_ml_fusion_full_{timestamp}.json"
        pd.Series(self.results).to_json(full_results_file)
    
    def _visualize_results(self):
        """
        可视化实验结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置绘图风格
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 准备数据
        lambdas = self.lambda_weights
        accuracies = [self.results[lam]['accuracy'] for lam in lambdas]
        f1_scores = [self.results[lam]['f1'] for lam in lambdas]
        
        # 添加边界值
        lambdas = [0.0] + lambdas + [1.0]
        accuracies = [self.results['ml_only']['accuracy']] + accuracies + [self.results['rule_only']['accuracy']]
        f1_scores = [self.results['ml_only']['f1']] + f1_scores + [self.results['rule_only']['f1']]
        
        # 绘制性能曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率曲线
        ax1.plot(lambdas, accuracies, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_title('Lambda权重对系统准确率的影响')
        ax1.set_xlabel('Lambda权重 (规则系统权重)')
        ax1.set_ylabel('准确率')
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([min(accuracies) * 0.95, max(accuracies) * 1.05])
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 在曲线上标注数值
        for i, (x, y) in enumerate(zip(lambdas, accuracies)):
            ax1.text(x, y + 0.005, f'{y:.4f}', ha='center', va='bottom')
        
        # F1分数曲线
        ax2.plot(lambdas, f1_scores, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_title('Lambda权重对系统F1分数的影响')
        ax2.set_xlabel('Lambda权重 (规则系统权重)')
        ax2.set_ylabel('F1分数')
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([min(f1_scores) * 0.95, max(f1_scores) * 1.05])
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 在曲线上标注数值
        for i, (x, y) in enumerate(zip(lambdas, f1_scores)):
            ax2.text(x, y + 0.005, f'{y:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"rule_ml_fusion_{timestamp}.png", dpi=300)
        print(f"可视化结果已保存")
        plt.close()
        
        # 绘制热力图比较不同权重下的各类别性能
        self._plot_category_performance_heatmap()
    
    def _plot_category_performance_heatmap(self):
        """
        绘制各类别在不同权重下的性能热力图
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 提取各类别的F1分数
        category_f1 = []
        style_names = list(STYLE_LABELS.keys())
        
        for lambda_weight in self.lambda_weights:
            report = self.results[lambda_weight]['classification_report']
            for style_idx, style_name in enumerate(style_names):
                category_f1.append({
                    'lambda_weight': lambda_weight,
                    'style': style_name,
                    'f1_score': report[str(style_idx)]['f1-score']
                })
        
        df = pd.DataFrame(category_f1)
        pivot_df = df.pivot(index='style', columns='lambda_weight', values='f1_score')
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'F1分数'})
        plt.title('不同Lambda权重下各类别的F1分数')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"category_performance_heatmap_{timestamp}.png", dpi=300)
        print(f"类别性能热力图已保存")
        plt.close()

# 融合实验实例
fusion_experiment = RuleMLFusionExperiment()