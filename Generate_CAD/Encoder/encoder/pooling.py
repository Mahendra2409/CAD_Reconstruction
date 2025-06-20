import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, AttentionalAggregation, SAGPooling, TopKPooling, Set2Set
from torch_geometric.nn.dense import dense_diff_pool  # Changed this line
# Define separate pooling classes

class MeanPooling(nn.Module):
    def __init__(self, node_dim, output_dim=256):
        super().__init__()
        self.projection = nn.Linear(node_dim, output_dim)
    
    def forward(self, x, batch):
        pooled = global_mean_pool(x, batch)
        return self.projection(pooled)

class AttentionPooling(nn.Module):
    def __init__(self, node_dim, output_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, 1))
        self.pool = AttentionalAggregation(gate_nn=self.attention)
        self.projection = nn.Linear(node_dim, output_dim)
    
    def forward(self, x, batch):
        pooled = self.pool(x, batch)
        return self.projection(pooled)

class SAGPoolingLayer(nn.Module):
    def __init__(self, node_dim, output_dim=256, ratio=0.5):
        super().__init__()
        self.pool = SAGPooling(node_dim, ratio=ratio)
        self.projection = nn.Linear(node_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
        pooled = global_mean_pool(x, batch)
        return self.projection(pooled)

class TopKPoolingLayer(nn.Module):
    def __init__(self, node_dim, output_dim=256, ratio=0.5):
        super().__init__()
        self.pool = TopKPooling(node_dim, ratio=ratio)
        self.projection = nn.Linear(node_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
        pooled = global_mean_pool(x, batch)
        return self.projection(pooled)

class DiffPoolLayer(nn.Module):
    def __init__(self, node_dim, output_dim=256, num_clusters=10):
        super().__init__()
        self.gnn_pool = nn.Linear(node_dim, num_clusters)
        self.gnn_embed = nn.Linear(node_dim, node_dim)
        self.projection = nn.Linear(node_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        num_nodes = x.size(0)
        adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        
        s = self.gnn_pool(x)
        x = self.gnn_embed(x)
        
        x, adj, reg = dense_diff_pool(x, adj, s)
        return self.projection(x)
    
class Set2SetPooling(nn.Module):
    def __init__(self, node_dim, output_dim=256):
        super().__init__()
        self.pool = Set2Set(node_dim, processing_steps=3)
        self.projection = nn.Linear(2 * node_dim, output_dim)  # Set2Set doubles dimension
    
    def forward(self, x, batch):
        pooled = self.pool(x, batch)
        return self.projection(pooled)
    