import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
# Append custom path for local modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layers import (
    GraphormerEncoderLayer, 
    CentralityEncoding, 
    SpatialEncoding,
    LearnableCentralityEncoding,
    AdaptiveSpatialEncoding,
    SimpleAttentionLayer
)

from pooling import (
    AttentionPooling, 
    SAGPoolingLayer, 
    TopKPoolingLayer,
    DiffPoolLayer,
    MeanPooling,
    Set2SetPooling
)

from functional_adpt_w9 import shortest_path_distance, batched_shortest_path_distance

class Graphormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int = 256,
                 n_heads: int = 8,
                 ff_dim: int = 512,
                 max_in_degree: int = 99,
                 max_out_degree: int = 99,
                 max_path_distance: int = 1,
                 spatial_encoding_type: str = "fixed",
                 centrality_type: str = "degree",
                 pooling_type: str = "mean",
                 visibility_encoding: str = "learnable"):  # New parameter
        
        super().__init__()

        # Node visibility encoding
        self.visibility_encoding = visibility_encoding

        ### input_node_dim(316) -> enhanced_dim(316 - 4 + (node_dim // 8))
        if self.visibility_encoding == "learnable":
                                
            # Node visibility embedding (4 -> node_dim // 8)
            self.node_visibility_embedding = nn.Linear(4, node_dim // 8)
            
            # SimpleAttentionLayer for 316 - 4 + (node_dim // 8) features
            # This is the dimension after visibility encoding but before node_in_lin
            enhanced_dim = input_node_dim - 4 + (node_dim // 8)
            self.attention_layer = SimpleAttentionLayer(enhanced_dim) 
            
            # Node feature processing - processes attention-weighted features
            self.node_in_lin = nn.Sequential(
                nn.Linear(enhanced_dim, node_dim),
                nn.LayerNorm(node_dim)
            )
            
            # Edge feature processing
            self.edge_in_lin = nn.Sequential(
                nn.Linear(input_edge_dim, edge_dim),
                nn.LayerNorm(edge_dim)
            )
        else:  # one-hot
            self.node_visibility_embedding = None
            self.edge_visibility_embedding = None 
#################### 1. why edge_visibility_embedding here, its not used anywhere in this function??
            self.attention_layer = SimpleAttentionLayer(input_node_dim)
            self.node_in_lin = nn.Sequential(
                nn.Linear(input_node_dim, node_dim),
                nn.LayerNorm(node_dim)
            )
            self.edge_in_lin = nn.Sequential(
                nn.Linear(input_edge_dim, edge_dim),
                nn.LayerNorm(edge_dim)
            )

        if centrality_type == "degree":
            self.centrality_encoding = CentralityEncoding(max_in_degree, max_out_degree, node_dim=node_dim)
        elif centrality_type == "learnable":
            self.centrality_encoding = LearnableCentralityEncoding(node_dim)  # Handles dynamic node counts

        # Select spatial encoding method
        if spatial_encoding_type == "fixed":
            self.spatial_encoding = SpatialEncoding(max_path_distance)
        elif spatial_encoding_type == "adaptive":
            self.spatial_encoding = AdaptiveSpatialEncoding()

        self.spatial_encoding = SpatialEncoding(   
            max_path_distance=max_path_distance,
        )
######################## 2. Why (self.spatial_encoding) here again, already intialized above?????

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                max_path_distance=max_path_distance
            ) for _ in range(num_layers)
        ])

        self.node_out_lin = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim)
        )
        # Batch Normalization Before Pooling
        self.batch_norm = nn.BatchNorm1d(node_dim)

        # Select pooling method
        if pooling_type == "attention":
            self.pooling_layer = AttentionPooling(node_dim, output_dim)
        elif pooling_type == "sag":
            self.pooling_layer = SAGPoolingLayer(node_dim, output_dim)
        elif pooling_type == "topk":
            self.pooling_layer = TopKPoolingLayer(node_dim, output_dim)
        elif pooling_type == "diffpool":
            self.pooling_layer = DiffPoolLayer(node_dim, output_dim)
        elif pooling_type == "set2set":
            self.pooling_layer = Set2SetPooling(node_dim, output_dim)
        else:
            self.pooling_layer = MeanPooling(node_dim, output_dim)

        self._init_parameters()
 
 
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if "embedding" not in m.__class__.__name__.lower():  # Avoid affecting embedding layers
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> torch.Tensor:
        # 1. Process node features with visibility encoding
        if self.visibility_encoding == "learnable":
            # Efficient slicing and concatenation for node features
            
            ### input_node_dim(316) -> enhanced_dim(316 - 4 + (node_dim // 8))
            node_features = torch.cat([
                data.x[:, :12],  # First 12 features (6 view + 6 entity)
                self.node_visibility_embedding(data.x[:, 12:16]),  # Visibility vector (4 features)
                data.x[:, 16:]  # Remaining features (coordinates)
            ], dim=1)
            
            # 2. Apply attention mechanism on enhanced features
            node_features, attention_scores = self.attention_layer(node_features)
            
            # 3. Then apply node_in_lin to transform to node_dim
            x = self.node_in_lin(node_features) # enhanced_dim -> node_dim
            
            edge_attr = self.edge_in_lin(data.edge_attr.float())
        else:
            # Apply attention on original features first
            node_features, attention_scores = self.attention_layer(data.x)
            
            # Then transform to node_dim
            x = self.node_in_lin(node_features) # input_node_dim -> node_dim
            edge_attr = self.edge_in_lin(data.edge_attr.float())

        # 4. Apply centrality encoding
        x = (self.centrality_encoding(x, data.edge_index, data.edge_attr) 
            if isinstance(self.centrality_encoding, LearnableCentralityEncoding)
            else self.centrality_encoding(x, data.edge_index))

        # 5. Calculate spatial encoding
        weights = shortest_path_distance(data) if isinstance(data, Data) else batched_shortest_path_distance(data)
        b = self.spatial_encoding(x, weights)
        
        # 6. Process through transformer layers
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        ptr = getattr(data, 'ptr', None)
        
        for layer in self.layers:
            x = layer(x, edge_attr, b, weights, ptr)  # b= spatial encong, weights = shortest_path_distance!! (direct edge)matrix

        # 7. Final processing and pooling
        x = self.batch_norm(self.node_out_lin(x))
        
        # 8. Apply appropriate pooling
        return (self.pooling_layer(x, getattr(data, 'edge_index', None), batch)
                if isinstance(self.pooling_layer, (SAGPoolingLayer, TopKPoolingLayer, DiffPoolLayer))
                else self.pooling_layer(x, batch))