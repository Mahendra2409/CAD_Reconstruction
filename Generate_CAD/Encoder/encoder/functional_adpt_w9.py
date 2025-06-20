from __future__ import annotations
from typing import Tuple, Dict, List
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import torch
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

##################### 1. !!! direct edge weights, it only fill that edges into the weight_matrix which are between adjacent nodes
def shortest_path_distance(data: Data):
    num_nodes = data.num_nodes
    device = data.edge_index.device
    
    # Create edge weight matrix
    weights = torch.ones((num_nodes, num_nodes), device=device) 
###################### 2. why 1s? why not 0s?
    
    # Use the first dimension of edge_attr for weights
    if data.edge_attr.dim() > 1:
        # If edge_attr is [E, 5], use the first feature
        edge_weights = data.edge_attr[:, 0]
    else:
        # If edge_attr is [E], use as is
        edge_weights = data.edge_attr
        
    # Assign weights to the matrix
    weights[data.edge_index[0], data.edge_index[1]] = edge_weights
    # print("shortest_path_distance",weights/15)
    return weights

def batched_shortest_path_distance(data):
    data_list = data.to_data_list()
    weight_matrices = []
    
    for graph in data_list:
        weights = shortest_path_distance(graph)
        weight_matrices.append(weights)   # it contain one weight metrix for each graph
    
    return torch.block_diag(*weight_matrices)





# from __future__ import annotations
# from typing import Tuple, Dict, List
# import torch
# from torch_geometric.data import Data
# from torch_geometric.utils import to_dense_adj
# from numba import jit

# @jit(nopython=True)
# def _compute_weight_matrix(edge_index: torch.Tensor, 
#                          edge_attr: torch.Tensor, 
#                          num_nodes: int) -> torch.Tensor:
#     """JIT-compiled helper function to compute weight matrix"""
#     weights = torch.ones((num_nodes, num_nodes))
    
#     for i in range(edge_index.shape[1]):
#         src = edge_index[0, i]
#         dst = edge_index[1, i]
#         if edge_attr.dim() > 1:
#             weight = edge_attr[i, 0]
#         else:
#             weight = edge_attr[i]
#         weights[src, dst] = weight
    
#     return weights

# def shortest_path_distance(data: Data):
#     num_nodes = data.num_nodes
#     device = data.edge_index.device
    
#     # Move tensors to CPU for Numba compatibility
#     edge_index_cpu = data.edge_index.cpu()
#     edge_attr_cpu = data.edge_attr.cpu()
    
#     # Use JIT-compiled function for weight matrix computation
#     weights = _compute_weight_matrix(
#         edge_index_cpu,
#         edge_attr_cpu,
#         num_nodes
#     )
    
#     # Move result back to original device
#     weights = weights.to(device)
    
#     return weights

# @jit(nopython=True)
# def _compute_block_diagonal(weight_matrices: List[torch.Tensor]) -> torch.Tensor:
#     """JIT-compiled helper function for block diagonal construction"""
#     total_size = sum(mat.shape[0] for mat in weight_matrices)
#     result = torch.ones((total_size, total_size))
    
#     current_idx = 0
#     for mat in weight_matrices:
#         size = mat.shape[0]
#         result[current_idx:current_idx+size, 
#                current_idx:current_idx+size] = mat
#         current_idx += size
    
#     return result

# def batched_shortest_path_distance(data):
#     data_list = data.to_data_list()
#     weight_matrices = []
    
#     # Compute individual weight matrices
#     for graph in data_list:
#         weights = shortest_path_distance(graph)
#         weight_matrices.append(weights.cpu())  # Move to CPU for Numba
    
#     # Use JIT-compiled function for block diagonal construction
#     result = _compute_block_diagonal(weight_matrices)
    
#     # Move result back to original device
#     return result.to(data.edge_index.device)