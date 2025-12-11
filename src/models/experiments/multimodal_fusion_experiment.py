"""多模态融合效果实验，评估不同模态特征组合对分类性能的影响"""
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from ...experiments.configs.experiment_config import EXPERIMENTS_CONFIG, STYLE_LABELS
from ...experiments.data.data_generator import data_generator

class MultimodalFusionExperiment:
    """多模态融合效果实验类"""
    
    def __init__(self):
        """初始化多模态融合实验"""
        self.config = EXPERIMENTS_CONFIG
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.models_dir = Path(self.config['paths']['models_dir'])
        self.visualizations_dir = Path(self.config['paths']['visualizations_dir'])
        
        # 创建目录
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.visualizations_dir.mkdir(exist_ok=True, parents=True)
        
        # 生成或加载融合特征数据
        self.multi_modal_data = data_generator.generate_fusion_data()
        
        # 定义模态类型
        self.modality_types = {
            'visual': [col for col in self.multi_modal_data.columns if col.startswith('visual_feature_')],
            'audio': [col for col in self.multi_modal_data.columns if col.startswith('audio_feature_')],
            'text': [col for col in self.multi_modal_data.columns if col.startswith('text_feature_')],
            'fusion': [col for col in self.multi_modal_data.columns if col.startswith('fusion_feature_')]
        }
        
        # 模态组合配置
        self.modality_combinations = [
            ('visual',),
            ('audio',),
            ('text',),
            ('visual', 'audio'),
            ('visual', 'text'),
            ('audio', 'text'),
            ('visual', 'audio', 'text'),
            ('fusion',)
        ]
        
        # 结果存储
        self.results = {}
    
    def _get_feature_columns(self, modalities):
        """
        获取指定模态组合的特征列
        
        Args:
            modalities: 模态元组，如('visual', 'audio')
            
        Returns:
            list: 特征列名列表
        """
        if 'fusion' in modalities:
            return self.modality_types['fusion']
        
        feature_columns = []
        for modality in modalities:
            if modality in self.modality_types:
                feature_columns.extend(self.modality_types[modality])
        
        return feature_columns
    
    def _extract_features(self, modalities):
        """
        从数据中提取指定模态的特征
        
        Args:
            modalities: 模态元组
            
        Returns:
            tuple: (X, y)
        """
        feature_columns = self._get_feature_columns(modalities)
        X = self.multi_modal_data[feature_columns].values
        y = self.multi_modal_data['style_id'].values
        return X, y
    
    def _train_and_evaluate(self, X_train, y_train, X_test, y_test, model_name='xgb'):
        """
        训练模型并评估性能
        
        Args:
            X_train, y_train: 训练数据和标签
            X_test, y_test: 测试数据和标签
            model_name: 模型名称
            
        Returns:
            tuple: (模型, 评估指标)
        """
        # 选择模型
        if model_name == 'xgb':
            model = XGBClassifier(random_state=42, n_jobs=-1)
        elif model_name == 'rf':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_name == 'svm':
            model = SVC(random_state=42, probability=True)
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
        
        # 训练模型
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # 评估性能
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'train_time': train_time,
            'feature_dimension': X_train.shape[1]
        }
        
        # 保存详细报告
        metrics['classification_report'] = classification_report(
            y_test, y_pred, target_names=list(STYLE_LABELS.keys()), output_dict=True
        )
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        return model, metrics
    
    def evaluate_modality_combination(self, modalities, model_name='xgb', save_results=True):
        """
        评估特定模态组合的性能
        
        Args:
            modalities: 模态元组
            model_name: 模型名称
            save_results: 是否保存结果
            
        Returns:
            dict: 评估指标
        """
        modalities_str = '+'.join(modalities)
        print(f"\n{'='*60}")
        print(f"评估模态组合: {modalities_str}，使用模型: {model_name}")
        print(f"{'='*60}")
        
        # 提取特征
        X, y = self._extract_features(modalities)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练和评估
        model, metrics = self._train_and_evaluate(
            X_train_scaled, y_train, X_test_scaled, y_test, model_name
        )
        
        # 交叉验证
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1
        )
        metrics['cv_accuracy'] = np.mean(cv_scores)
        metrics['cv_std'] = np.std(cv_scores)
        
        # 保存模型和scaler
        if save_results:
            model_path = self.models_dir / f"{model_name}_{modalities_str}.joblib"
            joblib.dump(model, model_path)
            
            scaler_path = self.models_dir / f"scaler_{modalities_str}.joblib"
            joblib.dump(scaler, scaler_path)
        
        # 打印结果
        print(f"\n模态组合 {modalities_str} 评估结果:")
        print(f"- 特征维度: {X.shape[1]}")
        print(f"- 训练时间: {metrics['train_time']:.2f}秒")
        print(f"- 测试集准确率: {metrics['accuracy']:.4f}")
        print(f"- 交叉验证平均准确率: {metrics['cv_accuracy']:.4f} ± {metrics['cv_std']:.4f}")
        print(f"- F1分数: {metrics['f1']:.4f}")
        
        return metrics
    
    def run_all_combinations(self, model_name='xgb', save_results=True):
        """
        运行所有模态组合的实验
        
        Args:
            model_name: 模型名称
            save_results: 是否保存结果
            
        Returns:
            dict: 所有组合的评估结果
        """
        print(f"\n{'='*60}")
        print(f"开始多模态融合实验，使用模型: {model_name}")
        print(f"{'='*60}")
        
        combination_results = {}
        
        # 评估每个模态组合
        for modalities in self.modality_combinations:
            modalities_str = '+'.join(modalities)
            metrics = self.evaluate_modality_combination(modalities, model_name, save_results)
            combination_results[modalities_str] = metrics
        
        # 记录结果
        self.results['modality_combinations'] = combination_results
        self.results['model_name'] = model_name
        
        # 保存和可视化结果
        if save_results:
            self._save_results()
            self._visualize_results()
        
        # 输出最佳组合
        best_combination = max(
            combination_results.items(),
            key=lambda x: (x[1]['accuracy'], x[1]['f1'])
        )
        
        print(f"\n{'='*60}")
        print(f"最佳模态组合: {best_combination[0]}")
        print(f"- 准确率: {best_combination[1]['accuracy']:.4f}")
        print(f"- F1分数: {best_combination[1]['f1']:.4f}")
        print(f"{'='*60}")
        
        return combination_results
    
    def analyze_modality_contribution(self, model_name='xgb'):
        """
        分析每个模态对最终性能的贡献
        
        Args:
            model_name: 模型名称
            
        Returns:
            dict: 模态贡献分析结果
        """
        print(f"\n{'='*60}")
        print(f"分析各模态对最终性能的贡献，使用模型: {model_name}")
        print(f"{'='*60}")
        
        if 'modality_combinations' not in self.results:
            self.run_all_combinations(model_name, save_results=False)
        
        # 获取所有模态组合的结果
        results = self.results['modality_combinations']
        
        # 计算单模态性能
        single_modality_results = {
            'visual': results['visual'],
            'audio': results['audio'],
            'text': results['text']
        }
        
        # 计算双模态组合性能
        dual_modality_results = {
            'visual+audio': results['visual+audio'],
            'visual+text': results['visual+text'],
            'audio+text': results['audio+text']
        }
        
        # 获取三模态组合性能
        triple_modality_results = results['visual+audio+text']
        
        # 获取融合模型性能
        fusion_model_results = results['fusion']
        
        # 计算模态贡献
        contributions = {}
        
        # 计算相对增益
        for dual_modality, dual_results in dual_modality_results.items():
            modalities = dual_modality.split('+')
            single1_results = single_modality_results[modalities[0]]
            single2_results = single_modality_results[modalities[1]]
            
            # 计算相对于两个单模态的平均性能的增益
            avg_single_accuracy = (single1_results['accuracy'] + single2_results['accuracy']) / 2
            gain_accuracy = dual_results['accuracy'] - avg_single_accuracy
            
            contributions[dual_modality] = {
                'accuracy': dual_results['accuracy'],
                'avg_single_accuracy': avg_single_accuracy,
                'gain_accuracy': gain_accuracy,
                'gain_percentage': (gain_accuracy / avg_single_accuracy) * 100 if avg_single_accuracy > 0 else 0
            }
        
        # 计算三模态相对增益
        max_dual_accuracy = max([r['accuracy'] for r in dual_modality_results.values()])
        triple_gain = triple_modality_results['accuracy'] - max_dual_accuracy
        
        contributions['visual+audio+text'] = {
            'accuracy': triple_modality_results['accuracy'],
            'max_dual_accuracy': max_dual_accuracy,
            'gain_accuracy': triple_gain,
            'gain_percentage': (triple_gain / max_dual_accuracy) * 100 if max_dual_accuracy > 0 else 0
        }
        
        # 计算融合模型相对增益
        fusion_gain = fusion_model_results['accuracy'] - triple_modality_results['accuracy']
        
        contributions['fusion'] = {
            'accuracy': fusion_model_results['accuracy'],
            'triple_accuracy': triple_modality_results['accuracy'],
            'gain_accuracy': fusion_gain,
            'gain_percentage': (fusion_gain / triple_modality_results['accuracy']) * 100 if triple_modality_results['accuracy'] > 0 else 0
        }
        
        # 打印模态贡献分析
        print(f"\n模态组合性能增益分析:")
        for modality, stats in contributions.items():
            print(f"\n{modality}:")
            print(f"- 准确率: {stats['accuracy']:.4f}")
            if 'avg_single_accuracy' in stats:
                print(f"- 两单模态平均准确率: {stats['avg_single_accuracy']:.4f}")
            elif 'max_dual_accuracy' in stats:
                print(f"- 双模态最高准确率: {stats['max_dual_accuracy']:.4f}")
            elif 'triple_accuracy' in stats:
                print(f"- 三模态准确率: {stats['triple_accuracy']:.4f}")
            print(f"- 增益准确率: {stats['gain_accuracy']:.4f}")
            print(f"- 增益百分比: {stats['gain_percentage']:.2f}%")
        
        # 记录模态贡献结果
        self.results['modality_contributions'] = contributions
        
        # 可视化模态贡献
        self._visualize_modality_contributions()
        
        return contributions
    
    def run_ablation_study(self, model_name='xgb', save_results=True):
        """
        运行消融实验，评估每个模态在完整模型中的必要性
        
        Args:
            model_name: 模型名称
            save_results: 是否保存结果
            
        Returns:
            dict: 消融实验结果
        """
        print(f"\n{'='*60}")
        print(f"开始消融实验，使用模型: {model_name}")
        print(f"{'='*60}")
        
        # 定义消融实验的模态组合
        ablation_combinations = [
            ('visual', 'audio', 'text'),  # 完整模型
            ('audio', 'text'),           # 移除视觉模态
            ('visual', 'text'),          # 移除音频模态
            ('visual', 'audio')          # 移除文本模态
        ]
        
        ablation_results = {}
        
        # 评估每个消融组合
        for modalities in ablation_combinations:
            modalities_str = '+'.join(modalities)
            metrics = self.evaluate_modality_combination(modalities, model_name, save_results)
            ablation_results[modalities_str] = metrics
        
        # 获取完整模型性能
        full_model_results = ablation_results['visual+audio+text']
        
        # 计算消融影响
        ablation_impact = {}
        
        for modalities_str, results in ablation_results.items():
            if modalities_str != 'visual+audio+text':  # 跳过完整模型
                # 确定被移除的模态
                removed_modality = 'visual'
                if modalities_str == 'audio+text':
                    removed_modality = '视觉'
                elif modalities_str == 'visual+text':
                    removed_modality = '音频'
                elif modalities_str == 'visual+audio':
                    removed_modality = '文本'
                
                # 计算性能下降
                accuracy_drop = full_model_results['accuracy'] - results['accuracy']
                f1_drop = full_model_results['f1'] - results['f1']
                
                ablation_impact[removed_modality] = {
                    'removed_modality': removed_modality,
                    'remaining_modalities': modalities_str,
                    'accuracy_without': results['accuracy'],
                    'f1_without': results['f1'],
                    'accuracy_drop': accuracy_drop,
                    'f1_drop': f1_drop,
                    'accuracy_drop_percentage': (accuracy_drop / full_model_results['accuracy']) * 100 if full_model_results['accuracy'] > 0 else 0
                }
        
        # 打印消融实验结果
        print(f"\n消融实验结果:")
        print(f"完整模型准确率: {full_model_results['accuracy']:.4f}")
        
        for modality, impact in ablation_impact.items():
            print(f"\n移除{modality}模态:")
            print(f"- 剩余模态: {impact['remaining_modalities']}")
            print(f"- 准确率: {impact['accuracy_without']:.4f}")
            print(f"- 准确率下降: {impact['accuracy_drop']:.4f} ({impact['accuracy_drop_percentage']:.2f}%)")
            print(f"- F1下降: {impact['f1_drop']:.4f}")
        
        # 确定最关键的模态（准确率下降最大）
        most_critical_modality = max(ablation_impact.items(), key=lambda x: x[1]['accuracy_drop'])
        
        print(f"\n{'='*60}")
        print(f"最关键模态: {most_critical_modality[0]}")
        print(f"- 准确率下降: {most_critical_modality[1]['accuracy_drop']:.4f} ({most_critical_modality[1]['accuracy_drop_percentage']:.2f}%)")
        print(f"{'='*60}")
        
        # 记录消融实验结果
        self.results['ablation_study'] = {
            'full_model': full_model_results,
            'ablation_impact': ablation_impact,
            'most_critical_modality': most_critical_modality[0]
        }
        
        # 保存和可视化结果
        if save_results:
            self._save_ablation_results()
            self._visualize_ablation_results()
        
        return ablation_impact
    
    def _save_results(self):
        """
        保存多模态融合实验结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'modality_combinations' not in self.results:
            return
        
        # 保存模态组合结果
        fusion_file = self.results_dir / f"multimodal_fusion_{self.results['model_name']}_{timestamp}.csv"
        
        data = []
        for modality_str, metrics in self.results['modality_combinations'].items():
            data.append({
                'modality_combination': modality_str,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'cv_accuracy': metrics['cv_accuracy'],
                'feature_dimension': metrics['feature_dimension'],
                'train_time': metrics['train_time']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(fusion_file, index=False, encoding='utf-8')
        
        print(f"多模态融合实验结果已保存")
    
    def _save_ablation_results(self):
        """
        保存消融实验结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'ablation_study' not in self.results:
            return
        
        ablation_file = self.results_dir / f"ablation_study_{self.results['model_name']}_{timestamp}.csv"
        
        data = []
        for modality, impact in self.results['ablation_study']['ablation_impact'].items():
            data.append(impact)
        
        df = pd.DataFrame(data)
        df.to_csv(ablation_file, index=False, encoding='utf-8')
        
        print(f"消融实验结果已保存")
    
    def _visualize_results(self):
        """
        可视化多模态融合实验结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'modality_combinations' not in self.results:
            return
        
        # 设置绘图风格
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 准备数据
        combinations = []
        accuracies = []
        f1_scores = []
        
        for modality_str, metrics in self.results['modality_combinations'].items():
            combinations.append(modality_str)
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1'])
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率对比
        bars1 = ax1.bar(combinations, accuracies, color='skyblue')
        ax1.set_title(f"不同模态组合的准确率 ({self.results['model_name']})")
        ax1.set_xlabel('模态组合')
        ax1.set_ylabel('准确率')
        ax1.set_ylim([0.5, 1.0])
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        # F1分数对比
        bars2 = ax2.bar(combinations, f1_scores, color='lightgreen')
        ax2.set_title(f"不同模态组合的F1分数 ({self.results['model_name']})")
        ax2.set_xlabel('模态组合')
        ax2.set_ylabel('F1分数')
        ax2.set_ylim([0.5, 1.0])
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"multimodal_fusion_comparison_{timestamp}.png", dpi=300)
        print(f"多模态融合对比可视化已保存")
        plt.close()
    
    def _visualize_modality_contributions(self):
        """
        可视化模态贡献分析结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'modality_contributions' not in self.results:
            return
        
        # 设置绘图风格
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准备数据
        contributions = self.results['modality_contributions']
        modalities = list(contributions.keys())
        gains = [c['gain_accuracy'] for c in contributions.values()]
        percentages = [c['gain_percentage'] for c in contributions.values()]
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 增益准确率
        bars1 = ax1.bar(modalities, gains, color='lightcoral')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_title('模态组合性能增益')
        ax1.set_xlabel('模态组合')
        ax1.set_ylabel('增益准确率')
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005 if height >= 0 else height - 0.02,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        # 增益百分比
        bars2 = ax2.bar(modalities, percentages, color='lightpink')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_title('模态组合性能增益百分比')
        ax2.set_xlabel('模态组合')
        ax2.set_ylabel('增益百分比 (%)')
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1 if height >= 0 else height - 3,
                    f'{height:.1f}%', ha='center', va='bottom', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"modality_contributions_{timestamp}.png", dpi=300)
        print(f"模态贡献可视化已保存")
        plt.close()
    
    def _visualize_ablation_results(self):
        """
        可视化消融实验结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if 'ablation_study' not in self.results:
            return
        
        # 设置绘图风格
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准备数据
        ablation_impact = self.results['ablation_study']['ablation_impact']
        modalities = list(ablation_impact.keys())
        accuracy_drops = [impact['accuracy_drop'] for impact in ablation_impact.values()]
        accuracy_without = [impact['accuracy_without'] for impact in ablation_impact.values()]
        
        # 添加完整模型数据
        modalities.append('完整模型')
        accuracy_drops.append(0)  # 完整模型没有下降
        accuracy_without.append(self.results['ablation_study']['full_model']['accuracy'])
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率对比
        bars1 = ax1.bar(modalities, accuracy_without, color='lightblue')
        ax1.set_title('消融实验准确率对比')
        ax1.set_xlabel('移除的模态 (或完整模型)')
        ax1.set_ylabel('准确率')
        ax1.set_ylim([0.5, 1.0])
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        # 准确率下降
        ablation_modalities = list(ablation_impact.keys())  # 只包含消融模态，不包含完整模型
        ablation_drops = [impact['accuracy_drop'] for impact in ablation_impact.values()]
        
        bars2 = ax2.bar(ablation_modalities, ablation_drops, color='lightcoral')
        ax2.set_title('移除各模态导致的准确率下降')
        ax2.set_xlabel('移除的模态')
        ax2.set_ylabel('准确率下降')
        ax2.set_ylim([0, 0.15])  # 根据实际数据调整
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"ablation_study_{timestamp}.png", dpi=300)
        print(f"消融实验可视化已保存")
        plt.close()

# 多模态融合实验实例
multimodal_fusion_experiment = MultimodalFusionExperiment()