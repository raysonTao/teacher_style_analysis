"""SMI验证实验，验证风格匹配度指数计算方法的有效性"""
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

from ..configs.experiment_config import EXPERIMENTS_CONFIG, STYLE_LABELS
from ..data.data_generator import data_generator

class SMIVerificationExperiment:
    """SMI验证实验类"""
    
    def __init__(self):
        """初始化SMI验证实验"""
        self.config = EXPERIMENTS_CONFIG
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.visualizations_dir = Path(self.config['paths']['visualizations_dir'])
        
        # 创建目录
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.visualizations_dir.mkdir(exist_ok=True, parents=True)
        
        # 生成或加载SMI验证数据
        self.smi_data = data_generator.generate_smi_validation_data()
        
        # 结果存储
        self.results = {}
    
    def calculate_smi(self, features, target_style_id):
        """
        计算风格匹配度指数(SMI)
        
        Args:
            features: 特征向量
            target_style_id: 目标风格ID
            
        Returns:
            float: SMI值 (0-1之间)
        """
        # 提取所有模态特征
        video_features = np.array([features[f'video_feature_{i}'] for i in range(20)])
        audio_features = np.array([features[f'audio_feature_{i}'] for i in range(15)])
        text_features = np.array([features[f'text_feature_{i}'] for i in range(25)])
        
        # 计算各模态内部特征的一致性
        def calculate_consistency(features):
            # 计算特征向量的方差，方差越小表示一致性越高
            if len(features) <= 1:
                return 1.0
            std_dev = np.std(features)
            max_std = np.max(np.abs(features))
            if max_std == 0:
                return 1.0
            consistency = 1 - min(std_dev / max_std, 1.0)
            return consistency
        
        # 计算模态权重（基于特征一致性）
        video_consistency = calculate_consistency(video_features)
        audio_consistency = calculate_consistency(audio_features)
        text_consistency = calculate_consistency(text_features)
        
        # 归一化权重
        total_consistency = video_consistency + audio_consistency + text_consistency
        if total_consistency == 0:
            video_weight = audio_weight = text_weight = 1/3
        else:
            video_weight = video_consistency / total_consistency
            audio_weight = audio_consistency / total_consistency
            text_weight = text_consistency / total_consistency
        
        # 计算特征与目标风格的匹配度（这里使用简化的方法）
        # 实际应用中，这应该基于训练好的模型
        def calculate_modality_match(features, target_style_id):
            # 基于特征统计计算匹配度
            mean_val = np.mean(features)
            std_val = np.std(features)
            
            # 添加风格ID的影响
            style_influence = (target_style_id + 1) / (len(STYLE_LABELS) + 1)
            
            # 计算综合匹配度
            # 这里使用简单的公式，实际应该基于模型预测
            match_score = 0.5 + 0.3 * np.tanh(mean_val) + 0.2 * style_influence
            
            # 确保在0-1范围内
            match_score = max(0.0, min(1.0, match_score))
            
            return match_score
        
        # 计算各模态匹配度
        video_match = calculate_modality_match(video_features, target_style_id)
        audio_match = calculate_modality_match(audio_features, target_style_id)
        text_match = calculate_modality_match(text_features, target_style_id)
        
        # 计算加权SMI
        smi = (video_weight * video_match + 
               audio_weight * audio_match + 
               text_weight * text_match)
        
        # 应用非线性调整，使结果更符合直觉
        smi = 0.5 * (1 + np.tanh(2 * (smi - 0.5)))
        
        return smi
    
    def verify_smi_calculation(self, save_results=True):
        """
        验证SMI计算方法的有效性
        
        Args:
            save_results: 是否保存结果
            
        Returns:
            dict: 验证结果
        """
        print(f"\n{'='*60}")
        print(f"开始风格匹配度指数(SMI)验证实验")
        print(f"{'='*60}")
        
        # 计算每个样本的SMI
        print("计算样本的SMI值...")
        true_match_degrees = []
        calculated_smis = []
        style_ids = []
        
        start_time = time.time()
        for idx, row in self.smi_data.iterrows():
            if idx % 100 == 0 and idx > 0:
                print(f"已处理{idx}个样本...")
            
            # 计算SMI
            smi = self.calculate_smi(row, row['style_id'])
            
            # 记录结果
            true_match_degrees.append(row['true_match_degree'])
            calculated_smis.append(smi)
            style_ids.append(row['style_id'])
        
        calculation_time = time.time() - start_time
        print(f"SMI计算完成，耗时: {calculation_time:.2f}秒")
        
        # 添加计算的SMI到数据中
        self.smi_data['calculated_smi'] = calculated_smis
        
        # 计算整体相关性
        pearson_corr, pearson_p = pearsonr(true_match_degrees, calculated_smis)
        spearman_corr, spearman_p = spearmanr(true_match_degrees, calculated_smis)
        
        # 计算均方误差
        mse = np.mean((np.array(true_match_degrees) - np.array(calculated_smis)) ** 2)
        
        # 计算绝对误差
        mae = np.mean(np.abs(np.array(true_match_degrees) - np.array(calculated_smis)))
        
        # 整体结果
        overall_results = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'mse': mse,
            'mae': mae,
            'sample_count': len(self.smi_data)
        }
        
        print(f"\n整体验证结果:")
        print(f"- Pearson相关系数: {pearson_corr:.4f} (p值: {pearson_p:.4e})")
        print(f"- Spearman相关系数: {spearman_corr:.4f} (p值: {spearman_p:.4e})")
        print(f"- 均方误差 (MSE): {mse:.4f}")
        print(f"- 平均绝对误差 (MAE): {mae:.4f}")
        
        # 按风格分类统计
        style_results = {}
        for style_id in range(len(STYLE_LABELS)):
            style_mask = np.array(style_ids) == style_id
            if np.sum(style_mask) > 1:
                style_true = np.array(true_match_degrees)[style_mask]
                style_calculated = np.array(calculated_smis)[style_mask]
                
                style_pearson, _ = pearsonr(style_true, style_calculated)
                style_spearman, _ = spearmanr(style_true, style_calculated)
                style_mse = np.mean((style_true - style_calculated) ** 2)
                style_mae = np.mean(np.abs(style_true - style_calculated))
                
                style_results[style_id] = {
                    'pearson_correlation': style_pearson,
                    'spearman_correlation': style_spearman,
                    'mse': style_mse,
                    'mae': style_mae,
                    'sample_count': np.sum(style_mask)
                }
        
        # 记录结果
        self.results['overall'] = overall_results
        self.results['by_style'] = style_results
        
        # 保存和可视化结果
        if save_results:
            self._save_results()
            self._visualize_results(true_match_degrees, calculated_smis, style_ids)
        
        print(f"\n{'='*60}")
        print(f"SMI验证实验完成")
        print(f"{'='*60}")
        
        return self.results
    
    def analyze_smi_accuracy_by_threshold(self, save_results=True):
        """
        分析不同SMI阈值下的准确率
        
        Args:
            save_results: 是否保存结果
            
        Returns:
            dict: 阈值分析结果
        """
        print(f"\n分析不同SMI阈值下的准确率...")
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        true_positive_rates = []
        false_positive_rates = []
        accuracy_scores = []
        
        for threshold in thresholds:
            # 定义正例为匹配度高的样本
            high_match_threshold = 0.7
            
            # 计算混淆矩阵的四个部分
            true_positives = 0  # 真实高匹配，预测高匹配
            false_positives = 0  # 真实低匹配，预测高匹配
            true_negatives = 0  # 真实低匹配，预测低匹配
            false_negatives = 0  # 真实高匹配，预测低匹配
            
            for _, row in self.smi_data.iterrows():
                is_high_match = row['true_match_degree'] >= high_match_threshold
                predicted_high_match = row['calculated_smi'] >= threshold
                
                if is_high_match and predicted_high_match:
                    true_positives += 1
                elif not is_high_match and predicted_high_match:
                    false_positives += 1
                elif not is_high_match and not predicted_high_match:
                    true_negatives += 1
                else:  # is_high_match and not predicted_high_match
                    false_negatives += 1
            
            # 计算比率
            total_positives = true_positives + false_negatives
            total_negatives = false_positives + true_negatives
            
            if total_positives > 0:
                tpr = true_positives / total_positives
            else:
                tpr = 0
            
            if total_negatives > 0:
                fpr = false_positives / total_negatives
            else:
                fpr = 0
            
            total_samples = total_positives + total_negatives
            if total_samples > 0:
                accuracy = (true_positives + true_negatives) / total_samples
            else:
                accuracy = 0
            
            true_positive_rates.append(tpr)
            false_positive_rates.append(fpr)
            accuracy_scores.append(accuracy)
        
        # 找出最佳阈值
        best_threshold_idx = np.argmax(accuracy_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_accuracy = accuracy_scores[best_threshold_idx]
        
        print(f"最佳SMI阈值: {best_threshold:.2f} (准确率: {best_accuracy:.4f})")
        
        # 记录阈值分析结果
        threshold_results = {
            'thresholds': thresholds.tolist(),
            'tpr': true_positive_rates,
            'fpr': false_positive_rates,
            'accuracy': accuracy_scores,
            'best_threshold': best_threshold,
            'best_accuracy': best_accuracy
        }
        
        self.results['threshold_analysis'] = threshold_results
        
        # 可视化结果
        if save_results:
            self._visualize_threshold_analysis(thresholds, true_positive_rates, false_positive_rates, accuracy_scores)
        
        return threshold_results
    
    def _save_results(self):
        """
        保存实验结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存整体结果
        overall_results_file = self.results_dir / f"smi_verification_overall_{timestamp}.csv"
        overall_df = pd.DataFrame([self.results['overall']])
        overall_df.to_csv(overall_results_file, index=False, encoding='utf-8')
        
        # 保存按风格分类的结果
        style_results_file = self.results_dir / f"smi_verification_by_style_{timestamp}.csv"
        style_data = []
        for style_id, metrics in self.results['by_style'].items():
            style_name = list(STYLE_LABELS.keys())[style_id]
            style_data.append({
                'style_id': style_id,
                'style_name': style_name,
                **metrics
            })
        
        style_df = pd.DataFrame(style_data)
        style_df.to_csv(style_results_file, index=False, encoding='utf-8')
        
        # 保存SMI数据
        smi_data_file = self.results_dir / f"smi_verification_data_{timestamp}.csv"
        self.smi_data.to_csv(smi_data_file, index=False, encoding='utf-8')
        
        print(f"\nSMI验证结果已保存")
    
    def _visualize_results(self, true_match_degrees, calculated_smis, style_ids):
        """
        可视化验证结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置绘图风格
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 1. 散点图：真实匹配度 vs 计算的SMI
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(true_match_degrees, calculated_smis, 
                             c=style_ids, cmap='tab10', alpha=0.6, s=50)
        plt.colorbar(scatter, label='风格ID')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2)  # 理想线
        plt.title(f'SMI验证结果 (Pearson相关性: {self.results["overall"]["pearson_correlation"]:.4f})')
        plt.xlabel('真实匹配度')
        plt.ylabel('计算的SMI值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"smi_scatter_{timestamp}.png", dpi=300)
        plt.close()
        
        # 2. 误差分布直方图
        errors = np.array(true_match_degrees) - np.array(calculated_smis)
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, color='skyblue')
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.title(f'SMI计算误差分布 (MAE: {self.results["overall"]["mae"]:.4f})')
        plt.xlabel('误差 (真实匹配度 - 计算的SMI)')
        plt.ylabel('频数')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"smi_error_distribution_{timestamp}.png", dpi=300)
        plt.close()
        
        # 3. 按风格分类的相关性柱状图
        if 'by_style' in self.results:
            style_names = list(STYLE_LABELS.keys())
            correlations = []
            
            for style_id in range(len(style_names)):
                if style_id in self.results['by_style']:
                    correlations.append(self.results['by_style'][style_id]['pearson_correlation'])
                else:
                    correlations.append(0)
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(style_names)), correlations, color='lightgreen')
            plt.axhline(self.results['overall']['pearson_correlation'], color='red', linestyle='--', label='整体相关性')
            plt.title('不同风格的SMI计算相关性')
            plt.xlabel('风格')
            plt.ylabel('Pearson相关系数')
            plt.xticks(range(len(style_names)), style_names, rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / f"smi_by_style_correlation_{timestamp}.png", dpi=300)
            plt.close()
    
    def _visualize_threshold_analysis(self, thresholds, tpr, fpr, accuracy):
        """
        可视化阈值分析结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. ROC曲线
        ax1.plot(fpr, tpr, 'b-', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=1)
        ax1.set_title('SMI阈值ROC曲线')
        ax1.set_xlabel('假阳性率 (FPR)')
        ax1.set_ylabel('真阳性率 (TPR)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # 2. 准确率曲线
        ax2.plot(thresholds, accuracy, 'g-', linewidth=2)
        best_idx = np.argmax(accuracy)
        ax2.scatter(thresholds[best_idx], accuracy[best_idx], 
                   color='red', s=100, zorder=5, label=f'最佳阈值 ({thresholds[best_idx]:.2f})')
        ax2.set_title('SMI阈值准确率曲线')
        ax2.set_xlabel('SMI阈值')
        ax2.set_ylabel('准确率')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlim([0.1, 0.95])
        ax2.set_ylim([min(accuracy) * 0.95, max(accuracy) * 1.02])
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"smi_threshold_analysis_{timestamp}.png", dpi=300)
        plt.close()

# SMI验证实验实例
smi_experiment = SMIVerificationExperiment()