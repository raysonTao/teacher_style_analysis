"""实验数据生成器，用于生成模拟数据或处理真实数据"""
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from pathlib import Path

# 导入全局logger
from ...config.config import logger

from ..configs.experiment_config import EXPERIMENTS_CONFIG, STYLE_LABELS, SIMULATION_CONFIG, DISCIPLINES, GRADES

class ExperimentDataGenerator:
    """实验数据生成器类"""
    
    def __init__(self):
        """初始化数据生成器"""
        self.data_config = EXPERIMENTS_CONFIG['data']
        self.simulation_config = SIMULATION_CONFIG
        self.data_dir = Path(self.data_config['dataset_path'])
        self.seed = self.data_config['seed']
        
        # 设置随机种子
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # 创建数据目录
        self.data_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_synthetic_data(self, sample_size=None):
        """
        生成合成实验数据
        
        Args:
            sample_size: 样本数量，默认为配置中的值
            
        Returns:
            pd.DataFrame: 生成的合成数据
        """
        if sample_size is None:
            sample_size = self.simulation_config['sample_size']
        
        logger.info(f"正在生成{sample_size}条合成实验数据...")
        
        data = []
        style_labels = list(STYLE_LABELS.keys())
        
        for i in range(sample_size):
            # 随机选择风格标签
            style = random.choice(style_labels)
            style_id = STYLE_LABELS[style]
            
            # 随机选择学科和年级
            discipline = random.choice(DISCIPLINES)
            grade = random.choice(GRADES)
            
            # 根据风格生成相关特征
            video_features = self._generate_modality_features('video', style_id)
            audio_features = self._generate_modality_features('audio', style_id)
            text_features = self._generate_modality_features('text', style_id)
            
            # 计算融合特征
            fusion_features = self._generate_fusion_features(video_features, audio_features, text_features)
            
            # 生成教学时长和评分
            teaching_duration = random.randint(30, 120)  # 30-120分钟
            student_rating = round(random.uniform(3.0, 5.0), 1)  # 3.0-5.0分
            
            # 构建数据记录
            record = {
                'sample_id': f"sample_{i:06d}",
                'style': style,
                'style_id': style_id,
                'discipline': discipline,
                'grade': grade,
                'teaching_duration': teaching_duration,
                'student_rating': student_rating
            }
            
            # 添加视频特征
            for j, v in enumerate(video_features):
                record[f'video_feature_{j}'] = v
            
            # 添加音频特征
            for j, v in enumerate(audio_features):
                record[f'audio_feature_{j}'] = v
            
            # 添加文本特征
            for j, v in enumerate(text_features):
                record[f'text_feature_{j}'] = v
            
            # 添加融合特征
            for j, v in enumerate(fusion_features):
                record[f'fusion_feature_{j}'] = v
            
            data.append(record)
        
        df = pd.DataFrame(data)
        logger.info(f"合成数据生成完成，共{len(df)}条记录")
        return df
    
    def _generate_modality_features(self, modality, style_id):
        """
        根据模态和风格生成特征向量
        
        Args:
            modality: 模态类型 ('video', 'audio', 'text')
            style_id: 风格ID
            
        Returns:
            np.ndarray: 特征向量
        """
        dim = self.simulation_config['feature_dimensions'][modality]
        noise_level = self.simulation_config['noise_level']
        correlation = self.simulation_config['style_correlation']
        
        # 生成基础特征，与风格有一定相关性
        base_features = np.random.randn(dim)
        
        # 添加风格相关模式
        style_pattern = np.sin(np.linspace(0, 2 * np.pi, dim) + style_id * np.pi/3)
        
        # 融合基础特征和风格模式
        features = (1 - correlation) * base_features + correlation * style_pattern
        
        # 添加噪声
        noise = np.random.normal(0, noise_level, dim)
        features += noise
        
        return features.tolist()
    
    def _generate_fusion_features(self, video_features, audio_features, text_features):
        """
        生成多模态融合特征
        
        Args:
            video_features: 视频特征
            audio_features: 音频特征
            text_features: 文本特征
            
        Returns:
            np.ndarray: 融合特征向量
        """
        dim = self.simulation_config['feature_dimensions']['fusion']
        
        # 合并所有模态特征
        all_features = np.array(video_features + audio_features + text_features)
        
        # 使用PCA类似的方法降维到目标维度
        # 这里使用简单的随机投影作为示例
        projection = np.random.randn(len(all_features), dim)
        fusion_features = np.dot(all_features, projection)
        
        # 归一化
        fusion_features = fusion_features / (np.linalg.norm(fusion_features) + 1e-8)
        
        return fusion_features.tolist()
    
    def split_dataset(self, data, save=True):
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            data: 完整数据集
            save: 是否保存分割后的数据集
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        train_ratio = self.data_config['train_ratio']
        val_ratio = self.data_config['val_ratio']
        test_ratio = self.data_config['test_ratio']
        
        # 首先分割训练集和剩余数据
        train_data, temp_data = train_test_split(
            data, 
            test_size=1 - train_ratio, 
            random_state=self.seed,
            stratify=data['style_id']
        )
        
        # 然后分割验证集和测试集
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=1 - val_size, 
            random_state=self.seed,
            stratify=temp_data['style_id']
        )
        
        logger.info(f"数据集分割完成：")
        logger.info(f"- 训练集: {len(train_data)}条 ({len(train_data)/len(data)*100:.1f}%)")
        logger.info(f"- 验证集: {len(val_data)}条 ({len(val_data)/len(data)*100:.1f}%)")
        logger.info(f"- 测试集: {len(test_data)}条 ({len(test_data)/len(data)*100:.1f}%)")
        
        # 保存数据集
        if save:
            train_data.to_csv(self.data_dir / 'train_data.csv', index=False, encoding='utf-8')
            val_data.to_csv(self.data_dir / 'val_data.csv', index=False, encoding='utf-8')
            test_data.to_csv(self.data_dir / 'test_data.csv', index=False, encoding='utf-8')
            logger.info(f"数据集已保存到: {self.data_dir}")
        
        return train_data, val_data, test_data
    
    def load_dataset(self):
        """
        加载已保存的数据集
        
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        train_path = self.data_dir / 'train_data.csv'
        val_path = self.data_dir / 'val_data.csv'
        test_path = self.data_dir / 'test_data.csv'
        
        if not (train_path.exists() and val_path.exists() and test_path.exists()):
            logger.info("数据集文件不存在，正在生成新的合成数据...")
            data = self.generate_synthetic_data()
            return self.split_dataset(data)
        
        logger.info(f"正在加载数据集...")
        train_data = pd.read_csv(train_path, encoding='utf-8')
        val_data = pd.read_csv(val_path, encoding='utf-8')
        test_data = pd.read_csv(test_path, encoding='utf-8')
        
        logger.info(f"数据集加载完成：")
        logger.info(f"- 训练集: {len(train_data)}条")
        logger.info(f"- 验证集: {len(val_data)}条")
        logger.info(f"- 测试集: {len(test_data)}条")
        
        return train_data, val_data, test_data
    
    def generate_cross_discipline_data(self):
        """
        生成跨学科实验数据
        
        Returns:
            dict: 按学科分类的数据集
        """
        logger.info("生成跨学科实验数据...")
        
        discipline_data = {}
        samples_per_discipline = 200  # 每个学科的样本数
        
        for discipline in DISCIPLINES:
            logger.info(f"生成{discipline}学科数据...")
            
            data = []
            style_labels = list(STYLE_LABELS.keys())
            
            for i in range(samples_per_discipline):
                # 随机选择风格标签
                style = random.choice(style_labels)
                style_id = STYLE_LABELS[style]
                
                # 使用固定学科，随机选择年级
                grade = random.choice(GRADES)
                
                # 根据风格生成相关特征
                video_features = self._generate_modality_features('video', style_id)
                audio_features = self._generate_modality_features('audio', style_id)
                text_features = self._generate_modality_features('text', style_id)
                
                # 计算融合特征
                fusion_features = self._generate_fusion_features(video_features, audio_features, text_features)
                
                # 生成教学时长和评分
                teaching_duration = random.randint(30, 120)
                student_rating = round(random.uniform(3.0, 5.0), 1)
                
                # 构建数据记录
                record = {
                    'sample_id': f"{discipline}_{i:04d}",
                    'style': style,
                    'style_id': style_id,
                    'discipline': discipline,
                    'grade': grade,
                    'teaching_duration': teaching_duration,
                    'student_rating': student_rating
                }
                
                # 添加视频特征
                for j, v in enumerate(video_features):
                    record[f'video_feature_{j}'] = v
                
                # 添加音频特征
                for j, v in enumerate(audio_features):
                    record[f'audio_feature_{j}'] = v
                
                # 添加文本特征
                for j, v in enumerate(text_features):
                    record[f'text_feature_{j}'] = v
                
                # 添加融合特征
                for j, v in enumerate(fusion_features):
                    record[f'fusion_feature_{j}'] = v
                
                data.append(record)
            
            discipline_data[discipline] = pd.DataFrame(data)
        
        # 保存跨学科数据
        cross_discipline_dir = self.data_dir / 'cross_discipline'
        cross_discipline_dir.mkdir(exist_ok=True)
        
        for discipline, df in discipline_data.items():
            df.to_csv(cross_discipline_dir / f'{discipline}_data.csv', index=False, encoding='utf-8')
        
        logger.info(f"跨学科数据生成完成，共{DISCIPLINES}个学科")
        return discipline_data
    
    def generate_smi_validation_data(self):
        """
        生成SMI验证实验数据
        
        Returns:
            pd.DataFrame: SMI验证数据
        """
        logger.info("生成SMI验证实验数据...")
        
        data = []
        samples_per_style = 100
        
        for style, style_id in STYLE_LABELS.items():
            for i in range(samples_per_style):
                # 生成不同匹配度的样本
                match_degree = random.uniform(0.1, 1.0)  # 1.0为完全匹配
                
                # 根据匹配度生成特征
                base_video = self._generate_modality_features('video', style_id)
                base_audio = self._generate_modality_features('audio', style_id)
                base_text = self._generate_modality_features('text', style_id)
                
                # 添加干扰，降低匹配度
                interference_style = random.choice(list(STYLE_LABELS.values()))
                while interference_style == style_id:
                    interference_style = random.choice(list(STYLE_LABELS.values()))
                
                interference_video = self._generate_modality_features('video', interference_style)
                interference_audio = self._generate_modality_features('audio', interference_style)
                interference_text = self._generate_modality_features('text', interference_style)
                
                # 融合基础特征和干扰特征
                video_features = [b * match_degree + i * (1 - match_degree) for b, i in zip(base_video, interference_video)]
                audio_features = [b * match_degree + i * (1 - match_degree) for b, i in zip(base_audio, interference_audio)]
                text_features = [b * match_degree + i * (1 - match_degree) for b, i in zip(base_text, interference_text)]
                
                # 生成记录
                record = {
                    'sample_id': f"smi_{style}_{i:04d}",
                    'style': style,
                    'style_id': style_id,
                    'true_match_degree': match_degree,
                    'discipline': random.choice(DISCIPLINES),
                    'grade': random.choice(GRADES)
                }
                
                # 添加特征
                for j, v in enumerate(video_features):
                    record[f'video_feature_{j}'] = v
                for j, v in enumerate(audio_features):
                    record[f'audio_feature_{j}'] = v
                for j, v in enumerate(text_features):
                    record[f'text_feature_{j}'] = v
                
                data.append(record)
        
        df = pd.DataFrame(data)
        
        # 保存SMI验证数据
        smi_dir = self.data_dir / 'smi_validation'
        smi_dir.mkdir(exist_ok=True)
        df.to_csv(smi_dir / 'smi_validation_data.csv', index=False, encoding='utf-8')
        
        logger.info(f"SMI验证数据生成完成，共{len(df)}条记录")
        return df

# 数据生成器实例
data_generator = ExperimentDataGenerator()