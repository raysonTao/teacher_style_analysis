"""简化版ST-GCN模型，用于骨骼动作识别"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (30, 32)
]


def build_adjacency(num_nodes: int) -> torch.Tensor:
    """构建归一化邻接矩阵"""
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in POSE_CONNECTIONS:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    for i in range(num_nodes):
        adjacency[i, i] = 1.0
    degree = np.sum(adjacency, axis=1)
    degree_inv = np.diag(1.0 / np.maximum(degree, 1e-6))
    normalized = degree_inv @ adjacency
    return torch.tensor(normalized, dtype=torch.float32)


class GraphConv(nn.Module):
    """图卷积层"""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor):
        super().__init__()
        self.A = adjacency
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, V]
        x = self.conv(x)
        x = torch.einsum('nctv,vw->nctw', x, self.A.to(x.device))
        return x


class STGCNBlock(nn.Module):
    """时空图卷积块"""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adjacency)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels)
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + residual
        return F.relu(x)


class STGCNModel(nn.Module):
    """简化ST-GCN动作识别模型"""

    def __init__(self, num_class: int, num_nodes: int = 33, in_channels: int = 2):
        super().__init__()
        self.A = build_adjacency(num_nodes)
        self.data_bn = nn.BatchNorm1d(num_nodes * in_channels)

        self.layer1 = STGCNBlock(in_channels, 64, self.A)
        self.layer2 = STGCNBlock(64, 128, self.A)
        self.layer3 = STGCNBlock(128, 256, self.A)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, V]
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
