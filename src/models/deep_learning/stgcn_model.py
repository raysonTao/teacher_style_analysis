"""ST-GCN模型实现（兼容MMAction2权重）"""
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_hop_distance(num_node: int, edge: List[Tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    """计算图的hop距离"""
    adjacency = np.zeros((num_node, num_node))
    for i, j in edge:
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    hop_dis = np.full((num_node, num_node), np.inf)
    transfer_mat = [np.linalg.matrix_power(adjacency, d) for d in range(max_hop + 1)]
    arrive_mat = (transfer_mat[0] > 0)
    hop_dis[arrive_mat] = 0
    for d in range(1, max_hop + 1):
        arrive_mat = (transfer_mat[d] > 0) & (hop_dis > d)
        hop_dis[arrive_mat] = d
    return hop_dis


def normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    """有向图归一化"""
    Dl = np.sum(adjacency, axis=0)
    num_node = adjacency.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(adjacency, Dn)


class Graph:
    """构建NTU骨架图"""

    def __init__(self, layout: str = 'ntu-rgb+d', strategy: str = 'spatial', max_hop: int = 1, dilation: int = 1):
        self.layout = layout
        self.strategy = strategy
        self.max_hop = max_hop
        self.dilation = dilation
        self._get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.A = self._get_adjacency(strategy)

    def _get_edge(self, layout: str):
        if layout not in ['ntu-rgb+d', 'ntu_rgb+d', 'ntu']:
            raise ValueError(f"Unsupported layout: {layout}")
        self.num_node = 25
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_1base = [
            (1, 2), (2, 21), (3, 21), (4, 3),
            (5, 21), (6, 5), (7, 6), (8, 7),
            (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15),
            (17, 1), (18, 17), (19, 18), (20, 19),
            (22, 8), (23, 8), (24, 12), (25, 12)
        ]
        neighbor = [(i - 1, j - 1) for i, j in neighbor_1base]
        self.edge = self_link + neighbor
        self.center = 20  # spine shoulder

    def _get_adjacency(self, strategy: str) -> np.ndarray:
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        elif strategy == 'spatial':
            A = np.zeros((3, self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] == 0:
                        A[0, j, i] = normalize_adjacency[j, i]
                    elif self.hop_dis[j, i] == 1:
                        if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                            A[0, j, i] = normalize_adjacency[j, i]
                        elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                            A[1, j, i] = normalize_adjacency[j, i]
                        else:
                            A[2, j, i] = normalize_adjacency[j, i]
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        return A


class ConvTemporalGraphical(nn.Module):
    """图卷积"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, t_kernel_size: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=((t_kernel_size - 1) // 2, 0)
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return x.contiguous(), A


class STGCNBlock(nn.Module):
    """ST-GCN时空块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, A.size(0))
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class STGCNModel(nn.Module):
    """ST-GCN骨骼动作识别模型（兼容MMAction2权重）"""

    def __init__(
        self,
        num_class: int,
        num_point: int = 25,
        num_person: int = 1,
        in_channels: int = 3,
        graph_layout: str = 'ntu-rgb+d',
        graph_strategy: str = 'spatial',
        edge_importance_weighting: bool = True
    ):
        super().__init__()
        self.graph = Graph(layout=graph_layout, strategy=graph_strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, A, residual=False),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2),
            STGCNBlock(256, 256, A),
            STGCNBlock(256, 256, A)
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1.0] * len(self.st_gcn_networks)

        self.fc = nn.Linear(256, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, V, M]
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = x.view(n, m, 256, t, v)
        x = x.mean(dim=4).mean(dim=3).mean(dim=1)
        return self.fc(x)
