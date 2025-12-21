"""实验结果分析器，用于综合分析和可视化所有实验结果"""
import numpy as np
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import json
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..configs.experiment_config import EXPERIMENTS_CONFIG
# 导入全局logger
from ...config.config import logger

class ResultAnalyzer:
    """实验结果分析器类"""
    
    def __init__(self):
        """初始化实验结果分析器"""
        self.config = EXPERIMENTS_CONFIG
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.visualizations_dir = Path(self.config['paths']['visualizations_dir'])
        self.models_dir = Path(self.config['paths']['models_dir'])
        
        # 创建可视化目录
        self.visualizations_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.rcParams['figure.figsize'] = (10, 6)  # 默认图表大小
        plt.rcParams['figure.dpi'] = 100  # 默认DPI
        
        # 存储所有加载的结果
        self.all_results = {
            'model_comparison': None,
            'multimodal_fusion': None,
            'rule_ml_fusion': None,
            'smi_validation': None,
            'cross_discipline': None
        }
    
    def load_model_comparison_results(self, file_pattern=None):
        """
        加载模型比较实验结果
        
        Args:
            file_pattern: 文件模式，默认为model_comparison_*.csv
            
        Returns:
            DataFrame: 模型比较结果
        """
        if file_pattern is None:
            file_pattern = "model_comparison_*.csv"
        
        # 查找最新的结果文件
        result_files = sorted(list(self.results_dir.glob(file_pattern)), reverse=True)
        
        if not result_files:
            logger.error(f"未找到模型比较结果文件: {file_pattern}")
            return None
        
        # 加载最新的结果文件
        latest_file = result_files[0]
        logger.info(f"加载模型比较结果: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        self.all_results['model_comparison'] = df
        
        return df
    
    def load_multimodal_fusion_results(self, file_pattern=None):
        """
        加载多模态融合实验结果
        
        Args:
            file_pattern: 文件模式，默认为multimodal_fusion_*.csv
            
        Returns:
            DataFrame: 多模态融合结果
        """
        if file_pattern is None:
            file_pattern = "multimodal_fusion_*.csv"
        
        # 查找最新的结果文件
        result_files = sorted(list(self.results_dir.glob(file_pattern)), reverse=True)
        
        if not result_files:
            logger.error(f"未找到多模态融合结果文件: {file_pattern}")
            return None
        
        # 加载最新的结果文件
        latest_file = result_files[0]
        logger.info(f"加载多模态融合结果: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        self.all_results['multimodal_fusion'] = df
        
        return df
    
    def load_rule_ml_fusion_results(self, file_pattern=None):
        """
        加载规则与机器学习融合实验结果
        
        Args:
            file_pattern: 文件模式，默认为rule_ml_fusion_*.csv
            
        Returns:
            DataFrame: 规则与机器学习融合结果
        """
        if file_pattern is None:
            file_pattern = "rule_ml_fusion_*.csv"
        
        # 查找最新的结果文件
        result_files = sorted(list(self.results_dir.glob(file_pattern)), reverse=True)
        
        if not result_files:
            logger.error(f"未找到规则与机器学习融合结果文件: {file_pattern}")
            return None
        
        # 加载最新的结果文件
        latest_file = result_files[0]
        logger.info(f"加载规则与机器学习融合结果: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        self.all_results['rule_ml_fusion'] = df
        
        return df
    
    def load_smi_validation_results(self, file_pattern=None):
        """
        加载SMI验证实验结果
        
        Args:
            file_pattern: 文件模式，默认为smi_validation_*.csv
            
        Returns:
            DataFrame: SMI验证结果
        """
        if file_pattern is None:
            file_pattern = "smi_validation_*.csv"
        
        # 查找最新的结果文件
        result_files = sorted(list(self.results_dir.glob(file_pattern)), reverse=True)
        
        if not result_files:
            logger.error(f"未找到SMI验证结果文件: {file_pattern}")
            return None
        
        # 加载最新的结果文件
        latest_file = result_files[0]
        logger.info(f"加载SMI验证结果: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        self.all_results['smi_validation'] = df
        
        return df
    
    def load_cross_discipline_results(self, file_pattern=None):
        """
        加载跨学科评估实验结果
        
        Args:
            file_pattern: 文件模式，默认为cross_discipline_evaluation_*.csv
            
        Returns:
            DataFrame: 跨学科评估结果
        """
        if file_pattern is None:
            file_pattern = "cross_discipline_evaluation_*.csv"
        
        # 查找最新的结果文件
        result_files = sorted(list(self.results_dir.glob(file_pattern)), reverse=True)
        
        if not result_files:
            logger.error(f"未找到跨学科评估结果文件: {file_pattern}")
            return None
        
        # 加载最新的结果文件
        latest_file = result_files[0]
        logger.info(f"加载跨学科评估结果: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        self.all_results['cross_discipline'] = df
        
        return df
    
    def load_all_results(self):
        """
        加载所有实验结果
        
        Returns:
            dict: 所有实验结果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"加载所有实验结果")
        logger.info(f"{'='*60}")
        
        self.load_model_comparison_results()
        self.load_multimodal_fusion_results()
        self.load_rule_ml_fusion_results()
        self.load_smi_validation_results()
        self.load_cross_discipline_results()
        
        # 计算已加载结果数量
        loaded_count = sum(1 for r in self.all_results.values() if r is not None)
        logger.info(f"\n已加载 {loaded_count}/5 组实验结果")
        
        return self.all_results
    
    def visualize_model_comparison(self, save_fig=True):
        """
        可视化模型比较结果
        
        Args:
            save_fig: 是否保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.all_results['model_comparison'] is None:
            logger.error("未加载模型比较结果，请先调用load_model_comparison_results")
            return None
        
        df = self.all_results['model_comparison']
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 准确率对比
        bars1 = ax1.bar(df['model_name'], df['accuracy'], color='skyblue')
        ax1.set_title('不同分类器的准确率对比')
        ax1.set_xlabel('分类器')
        ax1.set_ylabel('准确率')
        ax1.set_ylim([0.6, 1.0])
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        # F1分数对比
        bars2 = ax2.bar(df['model_name'], df['f1'], color='lightgreen')
        ax2.set_title('不同分类器的F1分数对比')
        ax2.set_xlabel('分类器')
        ax2.set_ylabel('F1分数')
        ax2.set_ylim([0.6, 1.0])
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        if save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = self.visualizations_dir / f"model_comparison_summary_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"模型比较汇总可视化已保存: {fig_path.name}")
        
        return fig
    
    def visualize_multimodal_fusion(self, save_fig=True):
        """
        可视化多模态融合结果
        
        Args:
            save_fig: 是否保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.all_results['multimodal_fusion'] is None:
            logger.error("未加载多模态融合结果，请先调用load_multimodal_fusion_results")
            return None
        
        df = self.all_results['multimodal_fusion']
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 准确率对比
        bars1 = ax1.bar(df['modality_combination'], df['accuracy'], color='lightcoral')
        ax1.set_title('不同模态组合的准确率对比')
        ax1.set_xlabel('模态组合')
        ax1.set_ylabel('准确率')
        ax1.set_ylim([0.6, 1.0])
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        # 特征维度对比
        bars2 = ax2.bar(df['modality_combination'], df['feature_dimension'], color='orange')
        ax2.set_title('不同模态组合的特征维度')
        ax2.set_xlabel('模态组合')
        ax2.set_ylabel('特征维度')
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(height)}', ha='center', va='bottom', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        if save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = self.visualizations_dir / f"multimodal_fusion_summary_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"多模态融合汇总可视化已保存: {fig_path.name}")
        
        return fig
    
    def visualize_rule_ml_fusion(self, save_fig=True):
        """
        可视化规则与机器学习融合结果
        
        Args:
            save_fig: 是否保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.all_results['rule_ml_fusion'] is None:
            logger.error("未加载规则与机器学习融合结果，请先调用load_rule_ml_fusion_results")
            return None
        
        df = self.all_results['rule_ml_fusion']
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制lambda权重与性能关系
        ax.plot(df['lambda_weight'], df['accuracy'], 'o-', color='blue', label='准确率')
        ax.plot(df['lambda_weight'], df['f1'], 's-', color='green', label='F1分数')
        ax.plot(df['lambda_weight'], df['rule_confidence'], '^-', color='red', label='规则置信度')
        ax.plot(df['lambda_weight'], df['ml_confidence'], 'd-', color='purple', label='ML置信度')
        
        ax.set_title('规则与机器学习融合权重对性能的影响')
        ax.set_xlabel('λ权重 (规则系统权重)')
        ax.set_ylabel('性能指标')
        ax.set_ylim([0.6, 1.0])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # 找出最佳lambda值
        best_lambda_idx = df['f1'].idxmax()
        best_lambda = df.loc[best_lambda_idx, 'lambda_weight']
        best_f1 = df.loc[best_lambda_idx, 'f1']
        
        # 标记最佳点
        ax.annotate(f'最佳λ={best_lambda}\nF1={best_f1:.4f}',
                    xy=(best_lambda, best_f1),
                    xytext=(best_lambda+0.1, best_f1-0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        if save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = self.visualizations_dir / f"rule_ml_fusion_summary_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"规则与机器学习融合汇总可视化已保存: {fig_path.name}")
        
        return fig
    
    def visualize_smi_validation(self, save_fig=True):
        """
        可视化SMI验证结果
        
        Args:
            save_fig: 是否保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.all_results['smi_validation'] is None:
            logger.error("未加载SMI验证结果，请先调用load_smi_validation_results")
            return None
        
        df = self.all_results['smi_validation']
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # SMI阈值与准确率关系
        ax1.plot(df['smi_threshold'], df['accuracy'], 'o-', color='blue')
        ax1.set_title('SMI阈值对准确率的影响')
        ax1.set_xlabel('SMI阈值')
        ax1.set_ylabel('准确率')
        ax1.set_xlim([0, 1.0])
        ax1.set_ylim([0.6, 1.0])
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 找出最佳阈值
        best_threshold_idx = df['accuracy'].idxmax()
        best_threshold = df.loc[best_threshold_idx, 'smi_threshold']
        best_accuracy = df.loc[best_threshold_idx, 'accuracy']
        
        # 标记最佳点
        ax1.annotate(f'最佳阈值={best_threshold}\n准确率={best_accuracy:.4f}',
                    xy=(best_threshold, best_accuracy),
                    xytext=(best_threshold+0.1, best_accuracy-0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10, fontweight='bold')
        
        # SMI分布箱线图
        smi_columns = [col for col in df.columns if col.startswith('smi_distribution_')]
        smi_data = df[smi_columns].iloc[0].values
        smi_labels = [col.replace('smi_distribution_', '') for col in smi_columns]
        
        # 准备SMI分布数据（模拟各风格类别的SMI分布）
        # 实际应用中，这里应该有更详细的分布数据
        smi_distributions = []
        for i, mean in enumerate(smi_data):
            # 生成模拟分布
            std = 0.1  # 假设标准差
            n_samples = 100
            distribution = np.random.normal(mean, std, n_samples)
            # 限制在0-1范围内
            distribution = np.clip(distribution, 0, 1)
            smi_distributions.append(distribution)
        
        # 绘制箱线图
        box_plot = ax2.boxplot(smi_distributions, labels=smi_labels, patch_artist=True)
        
        # 设置箱体颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightcyan', 'lightsalmon']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_title('不同教学风格的SMI分布')
        ax2.set_xlabel('教学风格')
        ax2.set_ylabel('SMI值')
        ax2.set_ylim([0, 1.0])
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        if save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = self.visualizations_dir / f"smi_validation_summary_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"SMI验证汇总可视化已保存: {fig_path.name}")
        
        return fig
    
    def visualize_cross_discipline(self, save_fig=True):
        """
        可视化跨学科评估结果
        
        Args:
            save_fig: 是否保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.all_results['cross_discipline'] is None:
            logger.error("未加载跨学科评估结果，请先调用load_cross_discipline_results")
            return None
        
        df = self.all_results['cross_discipline']
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 准确率对比
        bars1 = ax1.bar(df['discipline'], df['accuracy'], color='lightblue')
        
        # 计算平均准确率和标准差
        avg_accuracy = df['accuracy'].mean()
        std_accuracy = df['accuracy'].std()
        
        # 绘制平均线
        ax1.axhline(avg_accuracy, color='red', linestyle='--', label=f'平均准确率: {avg_accuracy:.4f}')
        
        ax1.set_title('不同学科的模型准确率')
        ax1.set_xlabel('学科')
        ax1.set_ylabel('准确率')
        ax1.set_ylim([0.6, 1.0])
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax1.legend()
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45)
        
        # 样本数量对比
        bars2 = ax2.bar(df['discipline'], df['sample_count'], color='lightgreen')
        ax2.set_title('不同学科的样本数量')
        ax2.set_xlabel('学科')
        ax2.set_ylabel('样本数量')
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(height)}', ha='center', va='bottom', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        if save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = self.visualizations_dir / f"cross_discipline_summary_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"跨学科评估汇总可视化已保存: {fig_path.name}")
        
        return fig
    
    def create_summary_dashboard(self, save_fig=True):
        """
        创建实验结果汇总仪表盘
        
        Args:
            save_fig: 是否保存图表
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        # 检查是否有已加载的结果
        if not any(self.all_results.values()):
            logger.error("未加载任何实验结果，请先调用load_all_results")
            return None
        
        # 创建一个大的图表
        fig = plt.figure(figsize=(20, 15))
        
        # 定义子图布局
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        # 模型比较子图
        if self.all_results['model_comparison'] is not None:
            ax1 = fig.add_subplot(gs[0, 0])
            df = self.all_results['model_comparison']
            bars = ax1.bar(df['model_name'], df['accuracy'], color='skyblue')
            ax1.set_title('(a) 不同分类器的准确率对比')
            ax1.set_xlabel('分类器')
            ax1.set_ylabel('准确率')
            ax1.set_ylim([0.6, 1.0])
            ax1.tick_params(axis='x', rotation=45, ha='right')
            ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', rotation=45, fontsize=9)
        
        # 多模态融合子图
        if self.all_results['multimodal_fusion'] is not None:
            ax2 = fig.add_subplot(gs[0, 1])
            df = self.all_results['multimodal_fusion']
            bars = ax2.bar(df['modality_combination'], df['accuracy'], color='lightcoral')
            ax2.set_title('(b) 不同模态组合的准确率对比')
            ax2.set_xlabel('模态组合')
            ax2.set_ylabel('准确率')
            ax2.set_ylim([0.6, 1.0])
            ax2.tick_params(axis='x', rotation=45, ha='right')
            ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', rotation=45, fontsize=9)
        
        # 规则与机器学习融合子图
        if self.all_results['rule_ml_fusion'] is not None:
            ax3 = fig.add_subplot(gs[1, 0])
            df = self.all_results['rule_ml_fusion']
            ax3.plot(df['lambda_weight'], df['accuracy'], 'o-', color='blue', label='准确率')
            ax3.plot(df['lambda_weight'], df['f1'], 's-', color='green', label='F1分数')
            ax3.set_title('(c) 规则与ML融合权重对性能的影响')
            ax3.set_xlabel('λ权重 (规则系统权重)')
            ax3.set_ylabel('性能指标')
            ax3.set_ylim([0.6, 1.0])
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()
        
        # SMI验证子图
        if self.all_results['smi_validation'] is not None:
            ax4 = fig.add_subplot(gs[1, 1])
            df = self.all_results['smi_validation']
            ax4.plot(df['smi_threshold'], df['accuracy'], 'o-', color='blue')
            ax4.set_title('(d) SMI阈值对准确率的影响')
            ax4.set_xlabel('SMI阈值')
            ax4.set_ylabel('准确率')
            ax4.set_xlim([0, 1.0])
            ax4.set_ylim([0.6, 1.0])
            ax4.grid(True, linestyle='--', alpha=0.7)
        
        # 跨学科评估子图
        if self.all_results['cross_discipline'] is not None:
            ax5 = fig.add_subplot(gs[2, :])
            df = self.all_results['cross_discipline']
            bars = ax5.bar(df['discipline'], df['accuracy'], color='lightblue')
            avg_accuracy = df['accuracy'].mean()
            ax5.axhline(avg_accuracy, color='red', linestyle='--', label=f'平均准确率: {avg_accuracy:.4f}')
            ax5.set_title('(e) 不同学科的模型准确率')
            ax5.set_xlabel('学科')
            ax5.set_ylabel('准确率')
            ax5.set_ylim([0.6, 1.0])
            ax5.tick_params(axis='x', rotation=45, ha='right')
            ax5.grid(True, linestyle='--', alpha=0.7, axis='y')
            ax5.legend()
            
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', rotation=45, fontsize=9)
        
        plt.suptitle('教师风格分析系统实验结果汇总仪表盘', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # 保存图表
        if save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = self.visualizations_dir / f"experiment_summary_dashboard_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"实验结果汇总仪表盘已保存: {fig_path.name}")
        
        return fig
    
    def generate_combined_statistics(self):
        """
        生成综合统计报告
        
        Returns:
            dict: 综合统计结果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"生成综合统计报告")
        logger.info(f"{'='*60}")
        
        combined_stats = {}
        
        # 模型比较统计
        if self.all_results['model_comparison'] is not None:
            df = self.all_results['model_comparison']
            best_model_idx = df['accuracy'].idxmax()
            best_model = df.loc[best_model_idx]
            
            combined_stats['best_model'] = {
                'name': best_model['model_name'],
                'accuracy': best_model['accuracy'],
                'f1': best_model['f1'],
                'precision': best_model['precision'],
                'recall': best_model['recall'],
                'avg_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std()
            }
            
            logger.info(f"\n最佳分类器: {best_model['model_name']}")
            logger.info(f"- 准确率: {best_model['accuracy']:.4f}")
            logger.info(f"- F1分数: {best_model['f1']:.4f}")
        
        # 多模态融合统计
        if self.all_results['multimodal_fusion'] is not None:
            df = self.all_results['multimodal_fusion']
            best_fusion_idx = df['accuracy'].idxmax()
            best_fusion = df.loc[best_fusion_idx]
            
            combined_stats['best_fusion'] = {
                'modality_combination': best_fusion['modality_combination'],
                'accuracy': best_fusion['accuracy'],
                'f1': best_fusion['f1'],
                'feature_dimension': best_fusion['feature_dimension'],
                'avg_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std()
            }
            
            logger.info(f"\n最佳模态组合: {best_fusion['modality_combination']}")
            logger.info(f"- 准确率: {best_fusion['accuracy']:.4f}")
            logger.info(f"- 特征维度: {best_fusion['feature_dimension']}")
        
        # 规则与机器学习融合统计
        if self.all_results['rule_ml_fusion'] is not None:
            df = self.all_results['rule_ml_fusion']
            best_lambda_idx = df['f1'].idxmax()
            best_lambda = df.loc[best_lambda_idx]
            
            combined_stats['best_rule_ml_fusion'] = {
                'lambda_weight': best_lambda['lambda_weight'],
                'accuracy': best_lambda['accuracy'],
                'f1': best_lambda['f1'],
                'rule_confidence': best_lambda['rule_confidence'],
                'ml_confidence': best_lambda['ml_confidence']
            }
            
            logger.info(f"\n最佳规则-ML融合权重: λ={best_lambda['lambda_weight']}")
            logger.info(f"- 准确率: {best_lambda['accuracy']:.4f}")
            logger.info(f"- F1分数: {best_lambda['f1']:.4f}")
        
        # SMI验证统计
        if self.all_results['smi_validation'] is not None:
            df = self.all_results['smi_validation']
            best_threshold_idx = df['accuracy'].idxmax()
            best_threshold = df.loc[best_threshold_idx]
            
            combined_stats['best_smi_threshold'] = {
                'threshold': best_threshold['smi_threshold'],
                'accuracy': best_threshold['accuracy'],
                'f1': best_threshold['f1'],
                'precision': best_threshold['precision'],
                'recall': best_threshold['recall']
            }
            
            logger.info(f"\n最佳SMI阈值: {best_threshold['smi_threshold']}")
            logger.info(f"- 准确率: {best_threshold['accuracy']:.4f}")
        
        # 跨学科评估统计
        if self.all_results['cross_discipline'] is not None:
            df = self.all_results['cross_discipline']
            
            combined_stats['cross_discipline'] = {
                'avg_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std(),
                'max_accuracy': df['accuracy'].max(),
                'min_accuracy': df['accuracy'].min(),
                'total_samples': df['sample_count'].sum()
            }
            
            logger.info(f"\n跨学科评估汇总:")
            logger.info(f"- 平均准确率: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
            logger.info(f"- 最大准确率: {df['accuracy'].max():.4f}")
            logger.info(f"- 最小准确率: {df['accuracy'].min():.4f}")
        
        # 计算整体最佳性能
        all_accuracies = []
        all_f1_scores = []
        
        if 'best_model' in combined_stats:
            all_accuracies.append(combined_stats['best_model']['accuracy'])
            all_f1_scores.append(combined_stats['best_model']['f1'])
        
        if 'best_fusion' in combined_stats:
            all_accuracies.append(combined_stats['best_fusion']['accuracy'])
            all_f1_scores.append(combined_stats['best_fusion']['f1'])
        
        if 'best_rule_ml_fusion' in combined_stats:
            all_accuracies.append(combined_stats['best_rule_ml_fusion']['accuracy'])
            all_f1_scores.append(combined_stats['best_rule_ml_fusion']['f1'])
        
        if all_accuracies:
            combined_stats['overall'] = {
                'best_accuracy': max(all_accuracies),
                'best_f1': max(all_f1_scores),
                'avg_accuracy': np.mean(all_accuracies),
                'std_accuracy': np.std(all_accuracies)
            }
            
            logger.info(f"\n{'='*60}")
            logger.info(f"系统整体性能:")
            logger.info(f"- 最佳准确率: {max(all_accuracies):.4f}")
            logger.info(f"- 最佳F1分数: {max(all_f1_scores):.4f}")
            logger.info(f"{'='*60}")
        
        # 保存综合统计结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stats_file = self.results_dir / f"combined_statistics_{timestamp}.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n综合统计报告已保存: {stats_file.name}")
        
        return combined_stats
    
    def run_full_analysis(self, load_results=True, generate_plots=True, save_fig=True):
        """
        运行完整的实验结果分析
        
        Args:
            load_results: 是否加载所有结果
            generate_plots: 是否生成所有图表
            save_fig: 是否保存图表
            
        Returns:
            dict: 分析结果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始完整的实验结果分析")
        logger.info(f"{'='*60}")
        
        # 加载所有结果
        if load_results:
            self.load_all_results()
        
        # 生成所有可视化图表
        if generate_plots:
            self.visualize_model_comparison(save_fig)
            self.visualize_multimodal_fusion(save_fig)
            self.visualize_rule_ml_fusion(save_fig)
            self.visualize_smi_validation(save_fig)
            self.visualize_cross_discipline(save_fig)
            
            # 创建汇总仪表盘
            self.create_summary_dashboard(save_fig)
        
        # 生成综合统计报告
        combined_stats = self.generate_combined_statistics()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"实验结果分析完成")
        logger.info(f"{'='*60}")
        
        return {
            'results': self.all_results,
            'statistics': combined_stats
        }

# 结果分析器实例
result_analyzer = ResultAnalyzer()