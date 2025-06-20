from typing import Tuple
from typing import Optional, Tuple
import torch.jit
import torch
from torch import nn
from torch_geometric.utils import degree

# from utils import decrease_to_max_value
import networkx as nx
from torch_geometric.utils import to_networkx
import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import networkx as nx

import torch
from torch import jit
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# return first dimnension of the edge_attr
@jit.script
def _process_edge_weights(edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_attr.dim() > 1:
        return edge_attr[:, 0]
    return edge_attr

@jit.script
def _compute_closeness_core(adjacency_matrix: torch.Tensor, node_dim: int, normalize: bool) -> torch.Tensor:
    # Compute closeness directly from edge weights
    row_sums = adjacency_matrix.sum(dim=1)
    closeness = 1.0 / row_sums  
    '''nodes which are closer to each other have smaller direct edge distance value in adjacency matrix,
      then have smaller row sums -> larger closness value
    '''    
    # Normalize if required
    if normalize:
        min_val = closeness.min()
        max_val = closeness.max()
        closeness = (closeness - min_val) / (max_val - min_val + 1e-8)
    
    # Expand to match node_dim
    return closeness.view(-1, 1).repeat(1, node_dim) # (num_nodes) -> (num_nodes, 1) -> (num_nodes, node_dim)
                                                     # closeness   -> .view(-1, 1)   -> repeat(1, node_dim) # It repeat the along dim=1(column) node_dim times
                                                     
def compute_closeness(edge_index, edge_attr, num_nodes, node_dim, normalize=False):
    """
    Compute closeness centrality for a fully connected graph where shortest paths
    are direct edge weights.
    Args:
        edge_index: Tensor [2, E] - adjacency list representation
        edge_attr: Tensor [E,] or [E, 5] or int - edge weights 
        num_nodes: int - number of nodes
        node_dim: int - feature dimensionality
        normalize: bool - whether to normalize closeness scores
    """
    device = edge_index.device
    
    # Handle edge weights
    if isinstance(edge_attr, torch.Tensor):
        edge_weights = _process_edge_weights(edge_attr, edge_index)
    else:
        edge_weights = torch.ones(edge_index.size(1), device=device)

    # Convert edge list to adjacency matrix
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=device)
    adjacency_matrix[edge_index[0], edge_index[1]] = edge_weights
    adjacency_matrix[edge_index[1], edge_index[0]] = edge_weights  # Make symmetric 
################# 0. undirected graph?
    
    # Use JIT-compiled core computation
    return _compute_closeness_core(adjacency_matrix, node_dim, normalize)

# class LearnableCentralityEncoding(nn.Module):
#     __constants__ = ['node_dim', 'normalize']
    
#     def __init__(self, node_dim: int, normalize: bool = False):
#         super().__init__()
#         self.node_dim = node_dim
#         self.normalize = normalize
#         self.node_embedding: Optional[nn.Parameter] = None

#     @torch.jit.export
#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
#                 edge_attr: torch.Tensor) -> torch.Tensor:
#         num_nodes = x.shape[0]
        
#         if self.node_embedding is None or self.node_embedding.shape[0] != num_nodes:
#             closeness_centrality = compute_closeness(
#                 edge_index,
#                 edge_attr,
#                 num_nodes,
#                 self.node_dim, 
#                 normalize=self.normalize
#             ).to(x.device)
#             self.node_embedding = nn.Parameter(closeness_centrality)
#         # print('used')
#         return x + self.node_embedding

class LearnableCentralityEncoding(nn.Module):
    __constants__ = ['node_dim', 'normalize', 'learned'] ## This says that these are constants and will not change, used for optimization

    def __init__(self, node_dim: int, normalize: bool = True, learned: bool = False):
        super().__init__()
        self.node_dim = node_dim
        self.normalize = normalize
        self.learned = learned  # Boolean flag to choose trainable vs. fixed
        
        if self.learned:
            self.node_embedding = None  # Will be assigned as a trainable parameter
        else:
            self.register_buffer("node_embedding", None)  # Non-trainable buffer

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]

        # Compute centrality if it's not already stored or if the number of nodes has changed
        if self.node_embedding is None or self.node_embedding.shape[0] != num_nodes:  
            closeness_centrality = compute_closeness(                                                                                                                   
                edge_index, edge_attr, num_nodes, self.node_dim, normalize=self.normalize
            ).to(x.device)

            ############# 1. !!! This loop will run for first batch only ,So (self.node_embedding) value depend on only first batch, Diferernt batches have different shapes of graph), 
            ############# 2. so closeness_centrality will be different but here it is using clossness_centrality of first batch only

            if self.learned:
                self.node_embedding = nn.Parameter(closeness_centrality)  # Trainable parameter
            else:
                self.node_embedding = closeness_centrality  # Fixed buffer

        return x + self.node_embedding  # Adds centrality embedding
    
class CentralityEncoding(nn.Module):
    __constants__ = ['max_in_degree', 'max_out_degree']
    
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim))) # (max_in_degree, node_dim)
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    @torch.jit.export
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        in_degree = torch.bincount(edge_index[1], minlength=num_nodes).clamp(max=self.max_in_degree-1)  # (num_nodes)
        out_degree = torch.bincount(edge_index[0], minlength=num_nodes).clamp(max=self.max_out_degree-1)
        
        return x + self.z_in[in_degree] + self.z_out[out_degree] # (num_nodes, node_dim)
    '''
    self.z_in[in_degree]  this work as a lookup tabel , vaues of in_degree are used as index to get the corresponding row(embedding) from z_in
    self.z_out[out_degree] this work as a lookup tabel , vaues of out_degree are used as index to get the corresponding row(embedding) from z_out
    '''

class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        super().__init__()
        self.max_path_distance = max_path_distance
        self.b = nn.Parameter(torch.randn(1))

    @torch.jit.export
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        #################### 3.  !!! max_path_distance is not used in this function
        return self.b[0] * weights  
        #################### 4. !!! direct edge weights, it only contain that edges into the weight_matrix which are between adjacent nodes

class AdaptiveSpatialEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.randn(1))

    @torch.jit.export
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        max_path_distance = torch.quantile(weights, 0.9)  # Adaptive threshold at 90th percentile
        weights = torch.clamp(weights, max=max_path_distance)
        return self.b[0] * weights


# class EdgeEncoding(nn.Module):
#     __constants__ = ['edge_dim']
    
#     def __init__(self, edge_dim: int, max_path_distance: int):
#         super().__init__()
#         self.edge_dim = edge_dim
#         self.edge_vector = nn.Parameter(torch.randn(1, edge_dim))

#     @torch.jit.export
#     def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, 
#                 weights: torch.Tensor) -> torch.Tensor:
#         return torch.matmul(weights.unsqueeze(-1), self.edge_vector)

class EdgeEncoding(nn.Module):
    __constants__ = ['edge_dim']
    
    def __init__(self, edge_dim: int, max_path_distance: int):
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(torch.randn(1, edge_dim))  
##################### 5. !!! why (1, edge_dim)?? why not (max_path_distance, edge_dim), edge vector for each position in the path? 
##################### 6. Its not learning the relation between the edges along path, and can not implementable on forward pass else loop
    
    
    @torch.jit.export
    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor,  
                weights: torch.Tensor) -> torch.Tensor:
        ############# 7. !!! Is weights coming from (shortest_path_distance) or (batched_shortest_path_distance) only?
        """
        Process edge weights into encoding
        
        Args:
            x: Node features tensor [num_nodes, node_dim]
            edge_attr: Edge attribute tensor [num_edges, edge_dim]
            weights: Edge weight tensor or adjacency matrix [num_nodes, num_nodes]  
            
        Returns:
            Edge encoding for attention [num_nodes, num_nodes]
        """
        device = next(self.parameters()).device
        num_nodes = x.size(0)
        cij = torch.zeros((num_nodes, num_nodes), device=device)
        
        # If weights is already a matrix of shape [num_nodes, num_nodes]


        if isinstance(weights, torch.Tensor) and weights.dim() == 2:
            # Scale weights by edge vector
            scaled_weights = torch.clamp(weights, max=self.max_path_distance)
            cij = scaled_weights * self.edge_vector.mean() 
################ 8. !!! mean() will be claculated first and then multiplied with the (scaled_weights)
        
        else:
            # Try original code path assuming weights is edge_paths dictionary
            # This is kept for backward compatibility
            try:
############### 9. !!! There is problem in edge_vector and weights initialization (They are always not suitable for this loop), here it required same weigts shape as mentioned in the paper
                for src in weights:
                    for dst in weights[src]:
                        path_ij = weights[src][dst][:self.max_path_distance]
                        weight_vec = self.edge_vector[:len(path_ij)]
                        attrs = edge_attr[path_ij]
                        cij[src, dst] = torch.mul(weight_vec, attrs).sum(dim=-1).mean() # Same as paper , dot product(elementwise multipication -> sum along dim=1 )-> mean() 
            except (TypeError, IndexError):
                # Fallback if weights is not the expected format
                print("Warning: EdgeEncoding received unexpected weights format. Using fallback.")
                # Create a simple fallback encoding
                edge_index = torch.nonzero(weights > 0, as_tuple=True)
                cij[edge_index[0], edge_index[1]] = self.edge_vector.mean()
        
        # Ensure there are no NaNs
        cij = torch.nan_to_num(cij)
        return cij



class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, 
                 edge_dim: int, max_path_distance: int):
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    @torch.jit.export
    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, 
                b: torch.Tensor, weights: torch.Tensor, 
                ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        c = self.edge_encoding(x, edge_attr, weights)
        a = query @ key.transpose(-2, -1) / (query.size(-1) ** 0.5)
        
        batch_mask = torch.ones_like(a)
        if ptr is not None:
            batch_mask = torch.zeros_like(a)
            for i in range(len(ptr) - 1):
                batch_mask[ptr[i]:ptr[i+1], ptr[i]:ptr[i+1]] = 1

        attn = torch.softmax(a + b + c.sum(-1) * batch_mask, dim=-1) 
################ 10. why c.sum(-1),, Its doing (num_nodes, num_nodes) -> (num_nodes)?? In Graphformer paper, it is not mentioned
################ 11. Mask is not applied to a and b , this will lead to nodes of a graph are influenced by nodes of other graphs (in a and b)
        
        return attn @ value
        
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, 
                 dim_k: int, edge_dim: int, max_path_distance: int):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) 
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    @torch.jit.export
    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, 
                b: torch.Tensor, weights: torch.Tensor, 
                ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
        head_outputs = [head(x, edge_attr, b, weights, ptr) for head in self.heads]
################ 12. !!! It assigning same input(x, edge_attr, b, weights, ptr) with full dimension to all heads, Every head watches the same input
        multi_head = torch.cat(head_outputs, dim=-1) ############## 13. (num_nodes, num_heads * dim_k) mostprobably dim_k = num_nodes 
        return self.linear(multi_head)

class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, n_heads: int, 
                 ff_dim: int, max_path_distance: int, dropout: float = 0.1):  #Added Dropout 
        super().__init__()
        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )

        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(node_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # Additional dropout in FF network
            nn.Linear(ff_dim, node_dim)
        )

    @torch.jit.export
    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, 
                b: torch.Tensor, weights: torch.Tensor, 
                ptr: Optional[torch.Tensor] = None) -> torch.Tensor:
    
        x1 = self.norm1(x)  # Pre-norm + attention + residual
        attn_out = self.attention(x1, edge_attr, b, weights, ptr)
        x = x + self.dropout1(attn_out)

        # Pre-norm + FF + residual
        x2 = self.norm2(x)
        ff_out = self.ff(x2)
        x = x + self.dropout2(ff_out)
        
        return x

# Orinnal
# class GraphormerEncoderLayer(nn.Module):
#     def __init__(self, node_dim, edge_dim, n_heads, ff_dim, max_path_distance):
#         super().__init__()
#         self.attention = GraphormerMultiHeadAttention(
#             dim_in=node_dim,
#             dim_k=node_dim,
#             dim_q=node_dim,
#             num_heads=n_heads,
#             edge_dim=edge_dim,
#             max_path_distance=max_path_distance,
#         )
#         self.ln_1 = nn.LayerNorm(node_dim)
#         self.ln_2 = nn.LayerNorm(node_dim)
#         self.ff = nn.Sequential(
#             nn.Linear(node_dim, ff_dim),
#             nn.GELU(),
#             nn.Linear(ff_dim, node_dim)
#         )

#     def forward(self, x, edge_attr, b, weights, ptr=None):
#         x_attn = self.attention(self.ln_1(x), edge_attr, b, weights, ptr)
#         x = x + x_attn
#         x = x + self.ff(self.ln_2(x))
#         return x
    
# import torch.nn.functional as F
# class SimpleAttentionLayer(nn.Module):
#     def __init__(self, input_dim):
#         """
#         Simple attention-based layer that computes attention scores and zeros out low-attention node features
#         for the last 200 nodes.
        
#         Args:
#         - input_dim: Dimensionality of node features.
#         """
#         super(SimpleAttentionLayer, self).__init__()
        
#         self.attention_weights = nn.Linear(input_dim, 1)
    
#     def forward(self, node_features):
#         """Compute attention scores and zero out node attributes with low attention for the last 200 nodes."""
#         scores = self.attention_weights(node_features).squeeze(-1)  # Shape: (num_nodes,)
        
#         # Normalize attention scores using softmax
#         attn_scores = F.softmax(scores, dim=0).unsqueeze(-1)  # Shape: (num_nodes, 1)
        
#         # Apply thresholding only to the last 200 nodes
#         if node_features.shape[0] > 300:
#             threshold = attn_scores[-300:].mean()
#             mask = attn_scores[-300:] >= threshold
#             node_features[-300:] *= mask.float()
        
#         return node_features, attn_scores


class SimpleAttentionLayer(nn.Module):
    def __init__(self, total_features=328, target_features=300, group_size=3):

        super(SimpleAttentionLayer, self).__init__()
        
        self.total_features = total_features
        self.target_features = target_features
        self.group_size = group_size
        self.num_groups = target_features // group_size
        self.feature_start_idx = total_features - target_features
        
        # Create a learnable projection for attention scoring
        self.group_attention = nn.Linear(group_size, 1, bias=False)
        nn.init.xavier_uniform_(self.group_attention.weight)
    
    def forward(self, node_features):
        # Get actual dimensions - handles variable number of nodes
        batch_size, actual_dim = node_features.shape
        
        # Verify feature dimension - this should be 316
        if actual_dim != self.total_features:
            print(f"Warning: Expected {self.total_features} features per node but got {actual_dim}")
            return node_features, torch.ones((batch_size, self.num_groups), device=node_features.device)
        
        # Split features: first 16 stay unchanged, last 300 get attention
        features_to_keep = node_features[:, :self.feature_start_idx]
        features_to_attend = node_features[:, self.feature_start_idx:]
        
        # Number of complete groups we can form (should be 100 groups for 300 features with group_size=3)
        actual_groups = features_to_attend.shape[1] // self.group_size
        
        # Safety check: make sure we're processing complete groups only
        if actual_groups * self.group_size != features_to_attend.shape[1]:
            remaining_features = features_to_attend[:, actual_groups * self.group_size:]
            features_to_attend = features_to_attend[:, :actual_groups * self.group_size]
        else:
            remaining_features = None
        
        # Reshape to group format - this is the critical operation that was failing
        # Use contiguous() before reshape to ensure memory layout is correct
        grouped_features = features_to_attend.contiguous().reshape(batch_size, actual_groups, self.group_size)
        
        # Compute attention scores for each group
        group_scores = self.group_attention(grouped_features).squeeze(-1)
        
        # Normalize scores with softmax
        attn_scores = F.softmax(group_scores, dim=1)
        
        # Compute threshold for each node (mean attention value)
        thresholds = attn_scores.mean(dim=1, keepdim=True)
        
        # Create binary mask: 1 for important groups, 0 for less important
        group_masks = (attn_scores >= thresholds).to(node_features.dtype)
        
        # Apply masks to zero out less important groups
        masked_grouped_features = grouped_features * group_masks.unsqueeze(-1)
        
        # Reshape back to original format
        masked_features = masked_grouped_features.reshape(batch_size, actual_groups * self.group_size)
        
        # If we had any remaining features (shouldn't happen with 300 and group_size=3), add them back
        if remaining_features is not None:
            masked_features = torch.cat([masked_features, remaining_features], dim=1)
        
        # Combine processed features with untouched features
        result_features = torch.cat([features_to_keep, masked_features], dim=1)
        
        return result_features, attn_scores