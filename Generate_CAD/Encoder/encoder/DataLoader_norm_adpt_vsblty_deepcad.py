import torch
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.data import Batch
import h5py
import os
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings
from pathlib import Path
import logging
import random
torch.set_printoptions(threshold=float('inf'), linewidth=1)

# Copying the entire class definitions from the original file
class CacheManager:
    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_count = {}

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None

    def put(self, key: str, value: torch.Tensor):
        if len(self.cache) >= self.max_cache_size:
            min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[min_key]
            del self.access_count[min_key]
        
        self.cache[key] = value
        self.access_count[key] = 1

class OptimizedCustomDataset(Dataset):
    def __init__(
        self,
        input_folder: str,  # DXF graph folder
        target_folder: str,  # Projection folder
        normalize_features: bool = True,
        cache_size: int = 1000,
        num_workers: int = 4,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ):
        super().__init__()
        self.input_folder = Path(input_folder)
        self.target_folder = Path(target_folder)
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.cache_manager = CacheManager(max_cache_size=cache_size)
        self.normalize_features = normalize_features
        self.sample_size = sample_size
        self.random_seed = random_seed

        # Initialize file lists
        self.input_files: List[Path] = []
        self.target_files: List[Path] = []
        self._initialize_file_lists(num_workers)
        
        # Apply random sampling if sample_size is specified
        if self.sample_size is not None and self.sample_size < len(self.input_files):
            self._apply_random_sampling()

    def _initialize_file_lists(self, num_workers: int):
        """Find matching pairs regardless of folder structure, using filenames only."""
        # First, index all available target files for quick lookup
        target_files_dict = {}
        for dirpath, _, filenames in os.walk(str(self.target_folder)):
            for filename in filenames:
                if filename.endswith('.pt'):
                    # Store by filename (without path) for easy lookup
                    target_files_dict[filename] = Path(os.path.join(dirpath, filename))
        
        print(f"Found {len(target_files_dict)} total target files")
        
        # Now walk through input directory and find matches
        matched_pairs = 0
        for dirpath, _, filenames in os.walk(str(self.input_folder)):
            for filename in filenames:
                if filename.endswith('.pt'):
                    input_file = Path(os.path.join(dirpath, filename))
                    
                    # Look for exact filename match in the target dictionary
                    if filename in target_files_dict:
                        self.input_files.append(input_file)
                        self.target_files.append(target_files_dict[filename])
                        matched_pairs += 1
                    else:
                        warnings.warn(f"No matching projection file found for {input_file}")
        
        print(f"Found {matched_pairs} matching pairs of DXF and projection files")
    
    def _apply_random_sampling(self):
        """
        Randomly select a subset of data points while maintaining paired inputs and outputs.
        """
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Create indices for sampling
        total_samples = len(self.input_files)
        all_indices = list(range(total_samples))
        
        # Randomly sample indices
        sampled_indices = random.sample(all_indices, self.sample_size)
        
        # Create new lists with only the sampled files
        sampled_input_files = [self.input_files[i] for i in sampled_indices]
        sampled_target_files = [self.target_files[i] for i in sampled_indices]
        
        # Replace the original lists with the sampled ones
        self.input_files = sampled_input_files
        self.target_files = sampled_target_files
        
        print(f"Randomly sampled {self.sample_size} out of {total_samples} data points (seed: {self.random_seed})")

    def __len__(self) -> int:
        return len(self.input_files)

    def _normalize_graph_data(self, graph_data):
        """Normalize DXF graph features"""
        if self.normalize_features:
            # Node features normalization
            categorical_features = graph_data.x[:, :16]  # Keep one-hot encoded
            continuous_features = graph_data.x[:, 16:]   # Normalize continuous
            normalized_continuous = continuous_features / 10
            graph_data.x = torch.cat([categorical_features, normalized_continuous], dim=1)

            # Edge features normalization
            first_feature = graph_data.edge_attr[:, :1]   # Last 4 are categorical
            remaining_features = graph_data.edge_attr[:, 1:]
            normalized_edge_continuous = remaining_features/10
            
            graph_data.edge_attr = torch.cat([first_feature, normalized_edge_continuous], dim=1)

        return graph_data

    def __getitem__(self, idx: int) -> Tuple[torch_geometric.data.Data, torch.Tensor]:
        # Load DXF graph data
        graph_data = self.cache_manager.get(str(self.input_files[idx]))
        if graph_data is None:
            graph_data = torch.load(self.input_files[idx], weights_only=False)
            graph_data = self._normalize_graph_data(graph_data)
            self.cache_manager.put(str(self.input_files[idx]), graph_data)

        # Add both input and projection file paths to graph_data
        graph_data.input_path = str(self.input_files[idx])
        graph_data.projection_path = str(self.target_files[idx])

        projection_data = self.cache_manager.get(str(self.target_files[idx]))
        if projection_data is None:
            projection_data = torch.load(self.target_files[idx], weights_only=True)
            if isinstance(projection_data, dict):
                projection_data = projection_data["points"]
            
            # Normalize by centering at (0,0,0)
            projection_data = projection_data / 10  # Scale down
            projection_mean = projection_data.mean(dim=0)  # Calculate mean along the first dimension
            projection_data = projection_data - projection_mean  # Center at (0,0,0)
            
            self.cache_manager.put(str(self.target_files[idx]), projection_data)

        return graph_data, projection_data

def collate_fn(batch):
    """Custom collate function for the dataloader."""
    graphs, projections = zip(*batch)
    
    # Batch the graphs
    batched_graphs = Batch.from_data_list(list(graphs))
    
    # Stack projections
    batched_projections = torch.stack(projections)
    
    return batched_graphs, batched_projections

def create_optimized_dataloader(
    dataset: OptimizedCustomDataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True if shuffle else False
    )

# Example usage
def create_dataset_with_sampling(
    input_folder: str,
    target_folder: str,
    sample_size: Optional[int] = None,
    batch_size: int = 32,
    random_seed: int = 42
):
    """
    Create a dataset with random sampling and return both dataset and dataloader.
    
    Args:
        input_folder: Path to the input DXF graph files
        target_folder: Path to the target projection files
        sample_size: Number of samples to randomly select (None to use all data)
        batch_size: Batch size for the dataloader
        random_seed: Random seed for reproducible sampling
        
    Returns:
        tuple: (dataset, dataloader)
    """
    # Create dataset with sampling
    dataset = OptimizedCustomDataset(
        input_folder=input_folder,
        target_folder=target_folder,
        normalize_features=True,
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    # Create dataloader
    dataloader = create_optimized_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataset, dataloader

# Inspection Script
def inspect_dataloader(input_folder, target_folder, batch_size=4):
    # Create dataset and dataloader
    dataset = OptimizedCustomDataset(
        input_folder=input_folder, 
        target_folder=target_folder,
        num_workers=1  # Reduce workers for debugging
    )
    dataloader = create_optimized_dataloader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False  # For consistent inspection
    )

    # Iterate through first few batches
    for batch_idx, (batched_graphs, batched_projections) in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx + 1} ---")
        
        # Graph Batch Details
        print("\nGraph Batch Details:")
        print(f"Number of graphs in batch: {batched_graphs.num_graphs}")
        print(f"Total number of nodes: {batched_graphs.num_nodes}")
        print(f"Total number of edges: {batched_graphs.num_edges}")
        
        # Node Features
        print("\nNode Features:")
        print(f"Node feature shape: {batched_graphs.x.shape}")
        print(f"Node feature example (first few values):\n{batched_graphs.x[:30]}")
        
        # # Edge Features
        # print("\nEdge Features:")
        # print(f"Edge feature shape: {batched_graphs.edge_attr.shape}")
        # print(f"Edge feature example (first few values):\n{batched_graphs.edge_attr[:]}")
        
        # # Projections
        # print("\nProjection Details:")
        # print(f"Projection tensor shape: {batched_projections.shape}")
        # print(f"Projection example (first few values):\n{batched_projections[:2]}")
        
        # # Optional: File paths
        # print("\nFile Paths:")
        # file_paths = [graph.input_path for graph in batched_graphs.to_data_list()]
        # print("Input graph paths:", file_paths)
        
        # Break after a few batches
        if batch_idx >= 2:
            break

# Example usage (commented out)
# Replace with your actual input and target folder paths
# inspect_dataloader(
#     input_folder="/path/to/input/folder", 
#     target_folder="/path/to/target/folder"
# )

# if __name__ == "__main__":
#     # Define paths here
#     INPUT_FOLDER = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/DeepCAD_Retrival_DXF/dxf_20k"
#     TARGET_FOLDER = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/DeepCAD_Retrival_DXF/dxf_labels_20k"
    
#     inspect_dataloader(INPUT_FOLDER, TARGET_FOLDER)

# import torch
# import torch_geometric
# from pathlib import Path
# import numpy as np
# import os
# import sys
# import copy

# def export_processed_dataset(
#     input_folder: str, 
#     target_folder: str, 
#     output_folder: str, 
#     num_samples: int = 5, 
#     normalize_features: bool = True,
#     center_projections: bool = True,
#     random_seed: int = 42
# ):
#     """
#     Export processed dataset samples while preserving original data structure.
    
#     Parameters:
#     -----------
#     input_folder : str
#         Path to the folder containing DXF graph files
#     target_folder : str
#         Path to the folder containing projection files
#     output_folder : str
#         Path to save the processed samples
#     num_samples : int
#         Number of samples to export
#     normalize_features : bool
#         Whether to normalize features
#     center_projections : bool
#         Whether to center projections
#     random_seed : int
#         Random seed for reproducible sample selection
#     """
#     # Ensure absolute path
#     output_path = Path(output_folder).resolve()
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     # Create dataset with random sample selection
#     dataset = OptimizedCustomDataset(
#         input_folder=input_folder, 
#         target_folder=target_folder,
#         normalize_features=normalize_features,
#         num_workers=1,
#         sample_size=num_samples,  # Use sample_size parameter for random selection
#         random_seed=random_seed    # Set random seed for reproducibility
#     )
    
#     # Export samples
#     for idx in range(len(dataset)):
#         # Get processed data
#         graph_data, projection_data = dataset[idx]
        
#         # Load original, unmodified files for structure reference
#         original_graph_data = torch.load(dataset.input_files[idx], weights_only=False)
#         original_projection_data = torch.load(dataset.target_files[idx])
        
#         # Create a deep copy of original structures
#         export_graph_data = copy.deepcopy(original_graph_data)
#         export_projection_data = copy.deepcopy(original_projection_data)
        
#         # Update the values while maintaining the original structure
#         # For graph data
#         if hasattr(export_graph_data, 'x'):
#             export_graph_data.x = graph_data.x
#         if hasattr(export_graph_data, 'edge_attr'):
#             export_graph_data.edge_attr = graph_data.edge_attr
        
#         # For projection data
#         if isinstance(export_projection_data, dict) and 'points' in export_projection_data:
#             export_projection_data['points'] = projection_data
#         elif isinstance(export_projection_data, torch.Tensor):
#             export_projection_data = projection_data
        
#         # Prepare export dictionary
#         export_data = {
#             'graph': export_graph_data,
#             'projection': export_projection_data,
#             'metadata': {
#                 'sample_index': idx,
#                 'input_path': str(dataset.input_files[idx]),
#                 'projection_path': str(dataset.target_files[idx]),
#                 'normalized_features': normalize_features,
#                 'centered_projections': center_projections,
#                 'random_seed': random_seed
#             }
#         }
        
#         # Generate save paths
#         graph_save_path = output_path / f"sample_{idx:04d}_processed_graph.pt"
#         projection_save_path = output_path / f"sample_{idx:04d}_processed_projection.pt"
#         full_save_path = output_path / f"sample_{idx:04d}_processed_full.pt"
        
#         # Save individual files
#         torch.save(export_graph_data, graph_save_path)
#         torch.save(export_projection_data, projection_save_path)
#         torch.save(export_data, full_save_path)
        
#         # Print export details
#         print(f"Exported sample {idx}:")
#         print(f"  Graph path: {graph_save_path}")
#         print(f"  Projection path: {projection_save_path}")
#         print(f"  Full data path: {full_save_path}")
#         print(f"  Graph data type: {type(export_graph_data)}")
#         print(f"  Projection data type: {type(export_projection_data)}")
        
#         # Additional details about the data
#         if hasattr(export_graph_data, 'x'):
#             print(f"  Node features shape: {export_graph_data.x.shape}")
#         if hasattr(export_graph_data, 'edge_attr'):
#             print(f"  Edge features shape: {export_graph_data.edge_attr.shape}")
        
#         if isinstance(export_projection_data, dict) and 'points' in export_projection_data:
#             print(f"  Projection points shape: {export_projection_data['points'].shape}")
#         elif isinstance(export_projection_data, torch.Tensor):
#             print(f"  Projection points shape: {export_projection_data.shape}")
        
#         print("---")

# def main():
#     # Define paths
#     INPUT_FOLDER = r"/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/DeepCAD_Retrival_DXF/20k/dxf_20k"
#     TARGET_FOLDER = r"/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/DeepCAD_Retrival_DXF/20k/dxf_labels_20k"
#     OUTPUT_FOLDER = r"/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/DeepCAD_Retrival_DXF/processed_samples"
    
#     # Export processed samples
#     export_processed_dataset(
#         input_folder=INPUT_FOLDER,
#         target_folder=TARGET_FOLDER,
#         output_folder=OUTPUT_FOLDER,
#         num_samples=5,
#         normalize_features=True,
#         center_projections=True,
#         random_seed=42  # Added for reproducibility
#     )

# if __name__ == "__main__":
#     main()