"""实验模型比较器，用于比较不同分类算法的性能"""
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from ..configs.experiment_config import EXPERIMENTS_CONFIG, STYLE_LABELS, ID_TO_STYLE
from ..data.data_generator import data_generator
from ...config.config import logger

class ModelComparator:
    """模型比较器类"""
    
    def __init__(self):
        """初始化模型比较器"""
        self.config = EXPERIMENTS_CONFIG
        self.models_config = self.config['models']
        self.results_dir = Path(self.config['paths']['results_dir'])
        self.models_dir = Path(self.config['paths']['models_dir'])
        self.visualizations_dir = Path(self.config['paths']['visualizations_dir'])
        
        # 创建目录
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.visualizations_dir.mkdir(exist_ok=True, parents=True)
        
        # 准备数据
        self.train_data, self.val_data, self.test_data = data_generator.load_dataset()
        
        # 初始化模型字典
        self.models = {
            'cmat': self._initialize_cmat_model(),
            'xgboost': XGBClassifier(random_state=42, n_jobs=-1),
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'svm': SVC(probability=True, random_state=42),
            'neural_network': MLPClassifier(random_state=42, max_iter=500),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200),
            'knn': KNeighborsClassifier(n_jobs=-1)
        }
        
        # 结果存储
        self.results = {}
    
    def _initialize_cmat_model(self):
        """
        初始化CMAT模型（从主系统导入）
        
        Returns:
            CMAT模型实例
        """
        # 导入主系统的CMAT模型
        try:
            from ..core.style_classifier import StyleClassifier
            cmat_model = StyleClassifier()
            return cmat_model
        except Exception as e:
            logger.error(f"导入CMAT模型失败: {e}")
            # 使用XGBoost作为替代
            return XGBClassifier(random_state=42, n_jobs=-1)
    
    def _extract_features(self, data, feature_type='fusion'):
        """
        从数据中提取特征
        
        Args:
            data: 数据集
            feature_type: 特征类型 ('video', 'audio', 'text', 'fusion')
            
        Returns:
            tuple: (X, y)
        """
        if feature_type == 'video':
            feature_columns = [col for col in data.columns if col.startswith('video_feature_')]
        elif feature_type == 'audio':
            feature_columns = [col for col in data.columns if col.startswith('audio_feature_')]
        elif feature_type == 'text':
            feature_columns = [col for col in data.columns if col.startswith('text_feature_')]
        elif feature_type == 'fusion':
            feature_columns = [col for col in data.columns if col.startswith('fusion_feature_')]
        else:
            raise ValueError(f"未知的特征类型: {feature_type}")
        
        X = data[feature_columns].values
        y = data['style_id'].values
        
        return X, y
    
    def compare_models(self, feature_type='fusion', save_results=True):
        """
        比较不同模型的性能
        
        Args:
            feature_type: 使用的特征类型
            save_results: 是否保存结果
            
        Returns:
            dict: 模型性能比较结果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始模型性能比较实验 (特征类型: {feature_type})")
        logger.info(f"{'='*60}")
        
        # 提取特征
        X_train, y_train = self._extract_features(self.train_data, feature_type)
        X_val, y_val = self._extract_features(self.val_data, feature_type)
        X_test, y_test = self._extract_features(self.test_data, feature_type)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # 保存scaler
        if save_results:
            joblib.dump(scaler, self.models_dir / f'scaler_{feature_type}.joblib')
        
        # 比较各个模型
        for model_name, model in self.models.items():
            logger.info(f"\n训练模型: {model_name}")
            start_time = time.time()
            
            try:
                # 训练模型
                if model_name == 'cmat':
                    # CMAT模型使用特殊的训练方法
                    model.train(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                train_time = time.time() - start_time
                logger.info(f"训练时间: {train_time:.2f}秒")
                
                # 预测
                if model_name == 'cmat':
                    y_pred_train = model.predict(X_train)
                    y_pred_val = model.predict(X_val)
                    y_pred_test = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                else:
                    y_pred_train = model.predict(X_train)
                    y_pred_val = model.predict(X_val)
                    y_pred_test = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                
                # 评估性能
                metrics = {
                    'train_accuracy': accuracy_score(y_train, y_pred_train),
                    'val_accuracy': accuracy_score(y_val, y_pred_val),
                    'test_accuracy': accuracy_score(y_test, y_pred_test),
                    'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
                    'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
                    'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
                    'train_time': train_time
                }
                
                # 计算AUC-ROC (多类别)
                try:
                    metrics['test_auc_roc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                except:
                    metrics['test_auc_roc'] = None
                
                # 保存详细报告
                metrics['classification_report'] = classification_report(
                    y_test, y_pred_test, target_names=list(STYLE_LABELS.keys()), output_dict=True
                )
                metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)
                
                # 保存预测结果
                metrics['y_pred_test'] = y_pred_test
                metrics['y_pred_proba'] = y_pred_proba
                
                # 保存模型
                if save_results:
                    model_path = self.models_dir / f"{model_name}_{feature_type}.joblib"
                    joblib.dump(model, model_path)
                    logger.info(f"模型已保存到: {model_path}")
                
                # 记录结果
                self.results[model_name] = metrics
                
                # 打印性能摘要
                logger.info(f"性能摘要:")
                logger.info(f"- 训练集准确率: {metrics['train_accuracy']:.4f}")
                logger.info(f"- 验证集准确率: {metrics['val_accuracy']:.4f}")
                logger.info(f"- 测试集准确率: {metrics['test_accuracy']:.4f}")
                logger.info(f"- 测试集F1分数: {metrics['test_f1']:.4f}")
                
            except Exception as e:
                logger.error(f"训练{model_name}时出错: {e}")
                self.results[model_name] = {'error': str(e)}
        
        # 保存比较结果
        if save_results:
            self._save_comparison_results(feature_type)
            self._visualize_comparison_results(feature_type)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"模型性能比较实验完成")
        logger.info(f"{'='*60}")
        
        return self.results
    
    def compare_feature_modalities(self, save_results=True):
        """
        比较不同特征模态的效果
        
        Args:
            save_results: 是否保存结果
            
        Returns:
            dict: 模态比较结果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始多模态特征融合效果实验")
        logger.info(f"{'='*60}")
        
        modalities = ['video', 'audio', 'text', 'fusion']
        modality_results = {}
        
        # 选择性能最好的模型（默认XGBoost）
        best_model_name = 'xgboost'
        
        for modality in modalities:
            logger.info(f"\n使用{modality}特征进行实验")
            
            # 提取特征
            X_train, y_train = self._extract_features(self.train_data, modality)
            X_test, y_test = self._extract_features(self.test_data, modality)
            
            # 标准化特征
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # 训练模型
            model = self.models[best_model_name].__class__()
            model.fit(X_train, y_train)
            
            # 预测和评估
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            try:
                metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['auc_roc'] = None
            
            modality_results[modality] = metrics
            
            logger.info(f"性能: 准确率={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        # 保存和可视化结果
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = self.results_dir / f"modality_comparison_{timestamp}.json"
            pd.Series(modality_results).to_json(results_path)
            logger.info(f"模态比较结果已保存到: {results_path}")
            
            # 可视化
            self._visualize_modality_comparison(modality_results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"多模态特征融合效果实验完成")
        logger.info(f"{'='*60}")
        
        return modality_results
    
    def run_cross_validation(self, model_name='xgboost', feature_type='fusion', cv=5):
        """
        执行交叉验证
        
        Args:
            model_name: 模型名称
            feature_type: 特征类型
            cv: 交叉验证折数
            
        Returns:
            dict: 交叉验证结果
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"对{model_name}模型执行{cv}折交叉验证 (特征: {feature_type})")
        logger.info(f"{'='*60}")
        
        # 合并训练集和验证集进行交叉验证
        combined_data = pd.concat([self.train_data, self.val_data])
        X, y = self._extract_features(combined_data, feature_type)
        
        # 标准化特征
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 获取模型
        model = self.models[model_name]
        
        # 执行交叉验证
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            logger.info(f"\n折 {fold}/{cv}:")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # 训练模型
            if model_name == 'cmat':
                model.train(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
            else:
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
            
            # 评估
            acc = accuracy_score(y_fold_val, y_pred)
            prec = precision_score(y_fold_val, y_pred, average='weighted')
            rec = recall_score(y_fold_val, y_pred, average='weighted')
            f1 = f1_score(y_fold_val, y_pred, average='weighted')
            
            cv_results['accuracy'].append(acc)
            cv_results['precision'].append(prec)
            cv_results['recall'].append(rec)
            cv_results['f1'].append(f1)
            
            logger.info(f"准确率: {acc:.4f}, F1: {f1:.4f}")
            fold += 1
        
        # 计算平均值和标准差
        for metric in cv_results:
            cv_results[f'{metric}_mean'] = np.mean(cv_results[metric])
            cv_results[f'{metric}_std'] = np.std(cv_results[metric])
        
        logger.info(f"\n交叉验证结果:")
        logger.info(f"准确率: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        logger.info(f"F1分数: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        
        return cv_results
    
    def _save_comparison_results(self, feature_type):
        """
        保存比较结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"model_comparison_{feature_type}_{timestamp}.csv"
        
        # 提取关键指标
        metrics_data = []
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                metrics_data.append({
                    'model': model_name,
                    'train_accuracy': metrics['train_accuracy'],
                    'val_accuracy': metrics['val_accuracy'],
                    'test_accuracy': metrics['test_accuracy'],
                    'test_f1': metrics['test_f1'],
                    'test_auc_roc': metrics['test_auc_roc'],
                    'train_time': metrics['train_time']
                })
        
        # 保存为CSV
        df = pd.DataFrame(metrics_data)
        df.to_csv(results_file, index=False, encoding='utf-8')
        logger.info(f"\n比较结果已保存到: {results_file}")
        
        # 保存完整结果
        full_results_file = self.results_dir / f"model_comparison_full_{feature_type}_{timestamp}.json"
        pd.Series(self.results).to_json(full_results_file)
    
    def _visualize_comparison_results(self, feature_type):
        """
        可视化比较结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 提取关键指标
        models = []
        test_accuracies = []
        test_f1_scores = []
        
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                models.append(model_name)
                test_accuracies.append(metrics['test_accuracy'])
                test_f1_scores.append(metrics['test_f1'])
        
        # 设置绘图风格
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 绘制准确率和F1对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率对比
        bars1 = ax1.bar(models, test_accuracies, color='skyblue')
        ax1.set_ylim([0.5, 1.0])
        ax1.set_title('不同模型的测试集准确率')
        ax1.set_xlabel('模型')
        ax1.set_ylabel('准确率')
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # F1分数对比
        bars2 = ax2.bar(models, test_f1_scores, color='lightgreen')
        ax2.set_ylim([0.5, 1.0])
        ax2.set_title('不同模型的测试集F1分数')
        ax2.set_xlabel('模型')
        ax2.set_ylabel('F1分数')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"model_comparison_{feature_type}_{timestamp}.png", dpi=300)
        logger.info(f"可视化结果已保存")
        plt.close()
    
    def _visualize_modality_comparison(self, modality_results):
        """
        可视化模态比较结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 准备数据
        modalities = list(modality_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # 创建数据透视表
        data = []
        for modality, results in modality_results.items():
            for metric in metrics:
                data.append({
                    'modality': modality,
                    'metric': metric,
                    'value': results[metric]
                })
        
        df = pd.DataFrame(data)
        pivot_df = df.pivot(index='metric', columns='modality', values='value')
        
        # 绘制热力图
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.4f', cbar_kws={'label': '得分'})
        plt.title('不同模态特征的性能比较')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"modality_comparison_{timestamp}.png", dpi=300)
        logger.info(f"模态比较可视化已保存")
        plt.close()

# 模型比较器实例
model_comparator = ModelComparator()