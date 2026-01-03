"""模型组件：Transformer编码器、LSTM层、注意力机制等"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len] or [batch_size, seq_len, seq_len]

        Returns:
            output: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)

        # 线性投影并重塑为多头
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, heads, seq, seq]

        # 应用mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力到value
        output = torch.matmul(attention_weights, V)  # [batch, heads, seq, head_dim]

        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 输出投影
        output = self.out_proj(output)

        return output, attention_weights


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]

        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # 多头注意力
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # 前馈网络
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len]

        Returns:
            output: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # 自注意力 + 残差连接 + LayerNorm
        attn_output, attention_weights = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x, attention_weights


class TransformerEncoder(nn.Module):
    """多层Transformer编码器"""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len]

        Returns:
            output: [batch_size, seq_len, embed_dim]
            all_attention_weights: list of attention weights from each layer
        """
        all_attention_weights = []

        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            all_attention_weights.append(attention_weights)

        return x, all_attention_weights


class BiLSTMEncoder(nn.Module):
    """双向LSTM编码器"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            lengths: [batch_size] 每个序列的实际长度

        Returns:
            output: [batch_size, seq_len, hidden_dim * (2 if bidirectional else 1)]
            (h_n, c_n): 最后的隐藏状态和细胞状态
        """
        if lengths is not None:
            # 打包序列以处理变长输入
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (h_n, c_n) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (h_n, c_n) = self.lstm(x)

        return output, (h_n, c_n)


class AttentionPooling(nn.Module):
    """注意力池化层"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len]

        Returns:
            pooled: [batch_size, input_dim]
            attention_weights: [batch_size, seq_len]
        """
        # 计算注意力分数
        attention_scores = self.attention(x).squeeze(-1)  # [batch_size, seq_len]

        # 应用mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]

        # 加权求和
        pooled = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # [batch_size, input_dim]

        return pooled, attention_weights


class ModalityEncoder(nn.Module):
    """模态特定编码器"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            output: [batch_size, output_dim]
        """
        return self.encoder(x)
