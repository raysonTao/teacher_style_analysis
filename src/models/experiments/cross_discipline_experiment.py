"""跨学科适应性实验，评估模型在不同学科教学视频上的泛化能力"""
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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from ...experiments.configs.experiment_config import EXPERIMENTS_CONFIG, STYLE_LABELS, DISCIPLINES
from ...experiments.data.data_generator import data_generator

class CrossDisciplineExperiment:
    """跨学科适应性实验类"""
    
    def __init__(self):
        """初始化跨学科实验"""
        self.config = EXPERIMENTS_CONFIG
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.models_dir = Path(self.config['paths']['models_dir'])
        self.visualizations_dir = Path(self.config['paths']['visualizations_dir'])
        
        # 创建目录
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.visualizations_dir.mkdir(exist_ok=True, parents=True)
        
        # 生成或加载跨学科数据
        self.discipline_data = data_generator.generate_cross_discipline_data()
        
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
    
    def evaluate_on_single_discipline(self, discipline, save_results=True):
        """
        在单个学科上评估模型性能
        
        Args:
            discipline: 学科名称
            save_results: 是否保存结果
            
        Returns:
            dict: 评估结果
        """
        print(f"\n{'='*60}")
        print(f"评估模型在{discipline}学科上的性能")
        print(f"{'='*60}")
        
        if discipline not in self.discipline_data:
            print(f"错误: 找不到{discipline}学科的数据")
            return None
        
        # 获取该学科数据
        data = self.discipline_data[discipline]
        
        # 分割训练集和测试集
        X, y = self._extract_features(data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 训练模型
        model = XGBClassifier(random_state=42, n_jobs=-1)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 评估性能
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'train_time': train_time,
            'sample_count': len(data)
        }
        
        # 保存详细报告
        metrics['classification_report'] = classification_report(
            y_test, y_pred, target_names=list(STYLE_LABELS.keys()), output_dict=True
        )
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # 保存模型
        if save_results:
            model_path = self.models_dir / f"xgb_{discipline}.joblib"
            joblib.dump(model, model_path)
            
            # 保存scaler
            scaler_path = self.models_dir / f"scaler_{discipline}.joblib"
            joblib.dump(scaler, scaler_path)
        
        # 打印结果
        print(f"\n{discipline}学科评估结果:")
        print(f"- 样本数量: {len(data)}")
        print(f"- 训练时间: {train_time:.2f}秒")
        print(f"- 准确率: {metrics['accuracy']:.4f}")
        print(f"- F1分数: {metrics['f1']:.4f}")
        
        # 按风格类别显示性能
        print(f"\n各风格类别的准确率:")
        for style_name, style_id in STYLE_LABELS.items():
            style_mask = y_test == style_id
            if np.sum(style_mask) > 0:
                style_acc = accuracy_score(y_test[style_mask], y_pred[style_mask])
                print(f"  - {style_name}: {style_acc:.4f} ({np.sum(style_mask)}个样本)")
        
        return metrics
    
    def run_cross_discipline_evaluation(self, save_results=True):
        """
        运行跨学科评估，在所有学科上评估模型性能
        
        Args:
            save_results: 是否保存结果
            
        Returns:
            dict: 跨学科评估结果
        """
        print(f"\n{'='*60}")
        print(f"开始跨学科适应性实验")
        print(f"{'='*60}")
        
        discipline_results = {}
        
        # 在每个学科上评估
        for discipline in DISCIPLINES:
            if discipline in self.discipline_data:
                metrics = self.evaluate_on_single_discipline(discipline, save_results)
                discipline_results[discipline] = metrics
        
        # 计算平均性能
        avg_metrics = {
            'accuracy': np.mean([r['accuracy'] for r in discipline_results.values()]),
            'precision': np.mean([r['precision'] for r in discipline_results.values()]),
            'recall': np.mean([r['recall'] for r in discipline_results.values()]),
            'f1': np.mean([r['f1'] for r in discipline_results.values()]),
            'std_accuracy': np.std([r['accuracy'] for r in discipline_results.values()]),
            'std_f1': np.std([r['f1'] for r in discipline_results.values()])
        }
        
        print(f"\n{'='*60}")
        print(f"跨学科评估汇总结果:")
        print(f"- 平均准确率: {avg_metrics['accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
        print(f"- 平均F1分数: {avg_metrics['f1']:.4f} ± {avg_metrics['std_f1']:.4f}")
        print(f"{'='*60}")
        
        # 记录结果
        self.results['by_discipline'] = discipline_results
        self.results['average'] = avg_metrics
        
        # 保存和可视化结果
        if save_results:
            self._save_results()
            self._visualize_results()
        
        return self.results
    
    def run_transfer_learning_experiment(self, source_discipline, target_disciplines=None, save_results=True):
        """
        运行迁移学习实验，评估从源学科到目标学科的知识迁移效果
        
        Args:
            source_discipline: 源学科
            target_disciplines: 目标学科列表，默认为所有其他学科
            save_results: 是否保存结果
            
        Returns:
            dict: 迁移学习结果
        """
        print(f"\n{'='*60}")
        print(f"开始迁移学习实验: 从{source_discipline}迁移到其他学科")
        print(f"{'='*60}")
        
        if source_discipline not in self.discipline_data:
            print(f"错误: 找不到{source_discipline}学科的数据")
            return None
        
        # 如果未指定目标学科，使用所有其他学科
        if target_disciplines is None:
            target_disciplines = [d for d in DISCIPLINES if d != source_discipline and d in self.discipline_data]
        
        # 获取源学科数据并训练模型
        source_data = self.discipline_data[source_discipline]
        X_source, y_source = self._extract_features(source_data)
        
        # 标准化特征
        scaler = StandardScaler()
        X_source_scaled = scaler.fit_transform(X_source)
        
        # 训练源模型
        source_model = XGBClassifier(random_state=42, n_jobs=-1)
        source_model.fit(X_source_scaled, y_source)
        
        transfer_results = {}
        
        # 在每个目标学科上测试
        for target_discipline in target_disciplines:
            print(f"\n迁移到{target_discipline}学科:")
            
            target_data = self.discipline_data[target_discipline]
            X_target, y_target = self._extract_features(target_data)
            
            # 使用源学科的scaler标准化目标学科特征
            X_target_scaled = scaler.transform(X_target)
            
            # 直接使用源模型预测（零样本迁移）
            y_pred_transfer = source_model.predict(X_target_scaled)
            
            # 作为对比，在目标学科上训练新模型
            X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
                X_target_scaled, y_target, test_size=0.3, random_state=42, stratify=y_target
            )
            
            target_model = XGBClassifier(random_state=42, n_jobs=-1)
            target_model.fit(X_target_train, y_target_train)
            y_pred_target = target_model.predict(X_target_test)
            
            # 评估迁移性能
            transfer_metrics = {
                'transfer_accuracy': accuracy_score(y_target, y_pred_transfer),
                'transfer_f1': f1_score(y_target, y_pred_transfer, average='weighted'),
                'native_accuracy': accuracy_score(y_target_test, y_pred_target),
                'native_f1': f1_score(y_target_test, y_pred_target, average='weighted'),
                'transfer_efficiency': accuracy_score(y_target, y_pred_transfer) / accuracy_score(y_target_test, y_pred_target)
            }
            
            transfer_results[target_discipline] = transfer_metrics
            
            print(f"- 迁移准确率: {transfer_metrics['transfer_accuracy']:.4f}")
            print(f"- 目标学科本地准确率: {transfer_metrics['native_accuracy']:.4f}")
            print(f"- 迁移效率: {transfer_metrics['transfer_efficiency']:.4f}")
        
        # 记录迁移学习结果
        self.results['transfer_learning'] = {
            'source_discipline': source_discipline,
            'results': transfer_results
        }
        
        # 保存和可视化结果
        if save_results:
            self._save_transfer_results()
            self._visualize_transfer_results()
        
        return transfer_results
    
    def analyze_style_distribution_across_disciplines(self):
        """
        分析不同学科中风格分布的差异
        
        Returns:
            dict: 风格分布分析结果
        """
        print(f"\n{'='*60}")
        print(f"分析不同学科中风格分布的差异")
        print(f"{'='*60}")
        
        # 统计每个学科的风格分布
        style_distributions = {}
        
        for discipline, data in self.discipline_data.items():
            style_counts = data['style'].value_counts().sort_index()
            style_percentages = (style_counts / len(data)) * 100
            
            style_distributions[discipline] = {
                'counts': style_counts.to_dict(),
                'percentages': style_percentages.to_dict(),
                'total_samples': len(data)
            }
            
            print(f"\n{discipline}学科风格分布:")
            for style, percentage in style_percentages.items():
                print(f"  - {style}: {percentage:.1f}%")
        
        # 记录风格分布结果
        self.results['style_distributions'] = style_distributions
        
        # 可视化风格分布
        self._visualize_style_distributions()
        
        return style_distributions
    
    def _save_results(self):
        """
        保存实验结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存跨学科评估结果
        cross_discipline_file = self.results_dir / f"cross_discipline_evaluation_{timestamp}.csv"
        
        data = []
        for discipline, metrics in self.results['by_discipline'].items():
            data.append({
                'discipline': discipline,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'sample_count': metrics['sample_count']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(cross_discipline_file, index=False, encoding='utf-8')
        
        # 保存平均结果
        avg_file = self.results_dir / f"cross_discipline_average_{timestamp}.csv"
        avg_df = pd.DataFrame([self.results['average']])
        avg_df.to_csv(avg_file, index=False, encoding='utf-8')
        
        print(f"\n跨学科评估结果已保存")
    
    def _save_transfer_results(self):
        """
        保存迁移学习结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'transfer_learning' not in self.results:
            return
        
        transfer_data = []
        source_discipline = self.results['transfer_learning']['source_discipline']
        
        for target_discipline, metrics in self.results['transfer_learning']['results'].items():
            transfer_data.append({
                'source_discipline': source_discipline,
                'target_discipline': target_discipline,
                **metrics
            })
        
        transfer_df = pd.DataFrame(transfer_data)
        transfer_file = self.results_dir / f"transfer_learning_{source_discipline}_{timestamp}.csv"
        transfer_df.to_csv(transfer_file, index=False, encoding='utf-8')
        
        print(f"\n迁移学习结果已保存")
    
    def _visualize_results(self):
        """
        可视化跨学科评估结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置绘图风格
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 准备数据
        disciplines = []
        accuracies = []
        f1_scores = []
        
        for discipline, metrics in self.results['by_discipline'].items():
            disciplines.append(discipline)
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1'])
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率对比
        bars1 = ax1.bar(disciplines, accuracies, color='skyblue')
        ax1.axhline(self.results['average']['accuracy'], color='red', linestyle='--', label='平均准确率')
        ax1.set_title('不同学科的模型准确率')
        ax1.set_xlabel('学科')
        ax1.set_ylabel('准确率')
        ax1.set_ylim([0.5, 1.0])
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax1.legend()
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        # F1分数对比
        bars2 = ax2.bar(disciplines, f1_scores, color='lightgreen')
        ax2.axhline(self.results['average']['f1'], color='red', linestyle='--', label='平均F1分数')
        ax2.set_title('不同学科的模型F1分数')
        ax2.set_xlabel('学科')
        ax2.set_ylabel('F1分数')
        ax2.set_ylim([0.5, 1.0])
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax2.legend()
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"cross_discipline_comparison_{timestamp}.png", dpi=300)
        print(f"跨学科对比可视化已保存")
        plt.close()
    
    def _visualize_transfer_results(self):
        """
        可视化迁移学习结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'transfer_learning' not in self.results:
            return
        
        # 设置绘图风格
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准备数据
        transfer_data = self.results['transfer_learning']['results']
        source_discipline = self.results['transfer_learning']['source_discipline']
        
        target_disciplines = list(transfer_data.keys())
        transfer_accuracies = [v['transfer_accuracy'] for v in transfer_data.values()]
        native_accuracies = [v['native_accuracy'] for v in transfer_data.values()]
        transfer_efficiencies = [v['transfer_efficiency'] for v in transfer_data.values()]
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 迁移准确率 vs 本地准确率
        x = np.arange(len(target_disciplines))
        width = 0.35
        
        ax1.bar(x - width/2, transfer_accuracies, width, label='迁移准确率')
        ax1.bar(x + width/2, native_accuracies, width, label='本地准确率')
        
        ax1.set_title(f'从{source_discipline}到各学科的迁移效果')
        ax1.set_xlabel('目标学科')
        ax1.set_ylabel('准确率')
        ax1.set_ylim([0.4, 1.0])
        ax1.set_xticks(x)
        ax1.set_xticklabels(target_disciplines, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 2. 迁移效率
        bars = ax2.bar(target_disciplines, transfer_efficiencies, color='orange')
        ax2.axhline(1.0, color='red', linestyle='--', label='效率=1')
        
        ax2.set_title('迁移学习效率')
        ax2.set_xlabel('目标学科')
        ax2.set_ylabel('迁移效率 (迁移准确率/本地准确率)')
        ax2.set_ylim([0.5, 1.2])
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax2.legend()
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"transfer_learning_{source_discipline}_{timestamp}.png", dpi=300)
        print(f"迁移学习可视化已保存")
        plt.close()
    
    def _visualize_style_distributions(self):
        """
        可视化不同学科的风格分布
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'style_distributions' not in self.results:
            return
        
        # 设置绘图风格
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准备数据
        style_names = list(STYLE_LABELS.keys())
        disciplines = list(self.results['style_distributions'].keys())
        
        # 创建数据透视表
        data = []
        for discipline, dist in self.results['style_distributions'].items():
            for style in style_names:
                percentage = dist['percentages'].get(style, 0)
                data.append({
                    'discipline': discipline,
                    'style': style,
                    'percentage': percentage
                })
        
        df = pd.DataFrame(data)
        pivot_df = df.pivot(index='style', columns='discipline', values='percentage')
        
        # 绘制热力图
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1f', cbar_kws={'label': '百分比 (%)'})
        plt.title('不同学科的风格分布')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"style_distribution_heatmap_{timestamp}.png", dpi=300)
        print(f"风格分布热力图已保存")
        plt.close()

# 跨学科实验实例
cross_discipline_experiment = CrossDisciplineExperiment()