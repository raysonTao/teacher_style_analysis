"""CMAT (Combined Multi-modal Attention-based Teaching style) 深度学习模型
基于PyTorch实现的多模态教师教学风格分析模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os
import json
from config.config import logger
from pathlib import Path

class MultiModalEncoder(nn.Module):
    """多模态特征编码器"""
    
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        
        # 视频特征编码器
        self.video_encoder = nn.Sequential(
            nn.Linear(input_dims.get('video', 50), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 音频特征编码器
        self.audio_encoder = nn.Sequential(
            nn.Linear(input_dims.get('audio', 30), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 文本特征编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dims.get('text', 40), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 融合特征编码器
        self.fusion_encoder = nn.Sequential(
            nn.Linear(input_dims.get('fusion', 20), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        encoded_features = {}
        
        # 编码各种模态特征
        if 'video' in features:
            encoded_features['video'] = self.video_encoder(features['video'])
        
        if 'audio' in features:
            encoded_features['audio'] = self.audio_encoder(features['audio'])
        
        if 'text' in features:
            encoded_features['text'] = self.text_encoder(features['text'])
        
        if 'fusion' in features:
            encoded_features['fusion'] = self.fusion_encoder(features['fusion'])
        
        return encoded_features


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        # Query, Key, Value投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, encoded_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        attended_features = {}
        
        # 为每个模态计算注意力
        modalities = list(encoded_features.keys())
        
        for target_modality in modalities:
            target_feature = encoded_features[target_modality]
            target_feature_norm = self.layer_norm(target_feature)
            
            # 计算query
            query = self.q_proj(target_feature_norm)
            
            # 聚合其他模态的key和value
            keys = []
            values = []
            
            for source_modality in modalities:
                if source_modality != target_modality:
                    source_feature = encoded_features[source_modality]
                    source_feature_norm = self.layer_norm(source_feature)
                    
                    key = self.k_proj(source_feature_norm)
                    value = self.v_proj(source_feature_norm)
                    
                    keys.append(key)
                    values.append(value)
            
            if keys:  # 如果有其他模态
                # 拼接keys和values
                concat_keys = torch.cat(keys, dim=1)  # [batch, seq_len*other_modalities, hidden_dim]
                concat_values = torch.cat(values, dim=1)
                
                # 重塑query, key, value for multi-head attention
                batch_size, seq_len, _ = query.shape
                query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                key = concat_keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                value = concat_values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                
                # 计算注意力分数
                scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
                attention_weights = F.softmax(scores, dim=-1)
                attention_weights = self.dropout(attention_weights)
                
                # 应用注意力
                attended = torch.matmul(attention_weights, value)
                
                # 重塑并输出投影
                attended = attended.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, self.hidden_dim
                )
                attended = self.o_proj(attended)
                
                # 残差连接
                attended = attended + target_feature
                attended_features[target_modality] = attended
            else:
                attended_features[target_modality] = target_feature
        
        return attended_features


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 应用自注意力
        attended, attention_weights = self.attention(x, x, x)
        
        # 残差连接和层归一化
        output = self.norm(x + self.dropout(attended))
        
        return output


class StyleClassificationHead(nn.Module):
    """风格分类头"""
    
    def __init__(self, hidden_dim: int = 256, num_styles: int = 7, dropout: float = 0.3):
        super().__init__()
        self.num_styles = num_styles
        
        # 风格分类器
        self.style_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_styles),
            nn.Sigmoid()  # 使用Sigmoid激活，输出0-1之间的概率
        )
        
        # SMI分数预测器
        self.smi_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1之间的概率，乘以100得到SMI分数
        )
    
    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 全局平均池化
        if len(fused_features.shape) > 2:
            pooled_features = fused_features.mean(dim=1)  # [batch, hidden_dim]
        else:
            pooled_features = fused_features
        
        # 预测风格分数
        style_scores = self.style_classifier(pooled_features)  # [batch, num_styles]
        
        # 预测SMI分数
        smi_score = self.smi_predictor(pooled_features)  # [batch, 1]
        
        return {
            'style_scores': style_scores,
            'smi_score': smi_score,
            'features': pooled_features
        }


class CMATModel(nn.Module):
    """CMAT (Combined Multi-modal Attention-based Teaching style) 模型"""
    
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256, 
                 num_heads: int = 8, num_styles: int = 7):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_styles = num_styles
        
        # 多模态编码器
        self.encoder = MultiModalEncoder(input_dims, hidden_dim)
        
        # 跨模态注意力
        self.cross_modal_attention = CrossModalAttention(hidden_dim, num_heads)
        
        # 自注意力机制
        self.self_attention = MultiHeadSelfAttention(hidden_dim, num_heads)
        
        # 风格分类头
        self.classification_head = StyleClassificationHead(hidden_dim, num_styles)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 1. 多模态编码
        encoded_features = self.encoder(features)
        
        # 2. 跨模态注意力
        attended_features = self.cross_modal_attention(encoded_features)
        
        # 3. 特征融合
        # 将所有模态的特征拼接
        modality_features = list(attended_features.values())
        if modality_features:
            fused_features = torch.cat(modality_features, dim=1)  # [batch, num_modalities*hidden_dim, 1]
            if len(fused_features.shape) == 2:
                fused_features = fused_features.unsqueeze(1)  # 添加序列维度
            
            # 4. 自注意力机制
            fused_features = self.self_attention(fused_features)  # [batch, seq_len, hidden_dim]
        else:
            # 如果没有特征，返回零张量
            batch_size = next(iter(features.values())).size(0) if features else 1
            fused_features = torch.zeros(batch_size, 1, self.hidden_dim)
        
        # 5. 风格分类
        classification_results = self.classification_head(fused_features)
        
        return {
            'style_scores': classification_results['style_scores'],  # [batch, num_styles]
            'smi_score': classification_results['smi_score'],  # [batch, 1]
            'fused_features': classification_results['features'],  # [batch, hidden_dim]
            'attention_weights': attended_features
        }
    
    def predict(self, features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """预测接口，返回详细结果"""
        self.eval()
        with torch.no_grad():
            results = self.forward(features)
            
            # 获取风格分数和SMI分数
            style_scores = results['style_scores'].cpu().numpy()[0]  # 取第一个样本
            smi_score = results['smi_score'].cpu().numpy()[0][0] * 100  # 转换为0-100分数
            
            # 转换为字典格式
            style_labels = ['理论讲授型', '启发引导型', '互动导向型', '逻辑推导型', 
                          '题目驱动型', '情感表达型', '耐心细致型']
            
            style_scores_dict = {
                style_labels[i]: float(score) 
                for i, score in enumerate(style_scores)
            }
            
            return {
                'style_scores': style_scores_dict,
                'smi_score': float(smi_score),
                'dominant_style': style_labels[np.argmax(style_scores)],
                'confidence': float(np.max(style_scores)),
                'features': results['fused_features'].cpu().numpy()[0].tolist()
            }
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dims': self.input_dims,
            'hidden_dim': self.hidden_dim,
            'num_styles': self.num_styles
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])


class CMATLoss(nn.Module):
    """CMAT模型损失函数"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3):
        super().__init__()
        self.alpha = alpha  # 风格分类损失权重
        self.beta = beta    # SMI回归损失权重
        self.gamma = gamma  # 对比学习损失权重
        
        self.style_criterion = nn.BCELoss()  # 二分类交叉熵损失
        self.smi_criterion = nn.MSELoss()    # 均方误差损失
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        # 风格分类损失
        style_loss = self.style_criterion(
            predictions['style_scores'], 
            targets['style_labels']
        )
        
        # SMI回归损失
        smi_loss = self.smi_criterion(
            predictions['smi_score'], 
            targets['smi_score'].unsqueeze(-1)
        )
        
        # 对比学习损失（可选）
        contrastive_loss = self._compute_contrastive_loss(
            predictions['fused_features'],
            targets['style_labels']
        )
        
        # 总损失
        total_loss = (self.alpha * style_loss + 
                     self.beta * smi_loss + 
                     self.gamma * contrastive_loss)
        
        return {
            'total_loss': total_loss,
            'style_loss': style_loss,
            'smi_loss': smi_loss,
            'contrastive_loss': contrastive_loss
        }
    
    def _compute_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor):
        """计算对比学习损失"""
        # 简化版本的对比学习损失
        # 在实际应用中，可以使用更复杂的对比学习策略
        
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 计算特征之间的余弦相似度
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # 创建标签矩阵
        labels_expanded = labels.unsqueeze(1)
        labels_matrix = (labels_expanded == labels_expanded.t()).float()
        
        # 对比损失：相同标签的特征应该更相似
        mask = torch.eye(batch_size, device=features.device)
        labels_matrix = labels_matrix - mask  # 排除对角线元素
        
        # 计算对比损失
        similarity_matrix = similarity_matrix - mask * 1e9  # 排除对角线
        
        exp_sim = torch.exp(similarity_matrix * 0.1)  # 温度参数
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (labels_matrix * log_prob).sum(1) / (labels_matrix.sum(1) + 1e-8)
        contrastive_loss = -mean_log_prob_pos.mean()
        
        return contrastive_loss


class CMATTrainer:
    """CMAT模型训练器"""
    
    def __init__(self, model: CMATModel, device: torch.device = None, 
                 learning_rate: float = 1e-4, config: Dict = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = CMATLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
        
        # 训练配置
        self.config = config or {}
        self.best_val_loss = float('inf')
        self.patience = self.config.get('patience', 10)
        self.early_stopping_counter = 0
        
        logger.info(f"CMATTrainer初始化完成，使用设备: {self.device}")
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_style_loss = 0.0
        total_smi_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            features, targets = batch
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(features)
            
            # 计算损失
            losses = self.criterion(predictions, targets)
            
            # 反向传播
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累计损失
            total_loss += losses['total_loss'].item()
            total_style_loss += losses['style_loss'].item()
            total_smi_loss += losses['smi_loss'].item()
        
        self.scheduler.step()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'style_loss': total_style_loss / len(dataloader),
            'smi_loss': total_smi_loss / len(dataloader)
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_style_loss = 0.0
        total_smi_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                features, targets = batch
                predictions = self.model(features)
                losses = self.criterion(predictions, targets)
                
                total_loss += losses['total_loss'].item()
                total_style_loss += losses['style_loss'].item()
                total_smi_loss += losses['smi_loss'].item()
        
        return {
            'val_loss': total_loss / len(dataloader),
            'val_style_loss': total_style_loss / len(dataloader),
            'val_smi_loss': total_smi_loss / len(dataloader)
        }
    
    def train(self, train_dataloader, val_dataloader, epochs: int = 50, 
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """完整训练过程"""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        logger.info(f"开始训练，epochs: {epochs}")
        
        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(train_dataloader)
            train_losses.append(train_metrics)
            
            # 验证
            val_metrics = self.validate(val_dataloader)
            val_losses.append(val_metrics)
            
            # 打印日志
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f} (style: {train_metrics['style_loss']:.4f}, smi: {train_metrics['smi_loss']:.4f})")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f} (style: {val_metrics['val_style_loss']:.4f}, smi: {val_metrics['val_smi_loss']:.4f})")
            
            # 早停检查
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.early_stopping_counter = 0
                
                # 保存最佳模型
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"保存最佳模型到: {save_path}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
            
            # 调整学习率
            self.scheduler.step()
        
        logger.info(f"训练完成，最终验证损失: {best_val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }
    
    def save_model(self, path: str):
        """保存模型和训练器状态"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'config': self.config
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型和训练器状态"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        self.config.update(checkpoint.get('config', {}))
        
        logger.info(f"模型已从 {path} 加载")
        
    def predict(self, features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """使用模型进行预测"""
        self.model.eval()
        with torch.no_grad():
            return self.model.predict(features)
    
    def create_dummy_data(self, batch_size: int = 32, sequence_length: int = 100) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """创建虚拟训练数据用于测试"""
        logger.info(f"创建虚拟数据: batch_size={batch_size}, sequence_length={sequence_length}")
        
        # 模拟多模态特征
        features = {
            'audio': torch.randn(batch_size, sequence_length, self.config.get('audio_dim', 128)),
            'video': torch.randn(batch_size, sequence_length, self.config.get('video_dim', 256)),
            'text': torch.randn(batch_size, sequence_length, self.config.get('text_dim', 512))
        }
        
        # 模拟标签
        style_labels = torch.randint(0, 7, (batch_size,))  # 7种风格
        smi_scores = torch.rand(batch_size, 1)  # 0-1之间的SMI分数
        
        targets = {
            'style_labels': style_labels,
            'smi_scores': smi_scores
        }
        
        return features, targets